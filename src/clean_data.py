"""
Stage 2 — clean the raw Telco Customer Churn data.

WHAT THIS SCRIPT DOES AND DOES NOT DO
-------------------------------------
It does:  fix data-quality defects, correct data types, and validate the result.
It does NOT: re-encode categories, collapse category levels, scale numbers,
          impute anything, select features, or drop rows for modeling reasons.

That split is deliberate and it is the single most important idea in this file.
Anything *fitted* to the data -- a scaler's mean, an encoder's category list, an
imputer's median -- must be learned from the TRAINING SET ONLY. If we fitted it
here, on all 7043 rows, information from the test set would leak into the model
and our evaluation scores would be optimistically biased. So all of that waits
for the Stage 5 pipeline.

Everything done here is a fact about the data, not a parameter estimated from it.
The cleaned file is therefore as close to the raw file as correctness allows: it
fixes what was *wrong*, and changes nothing that was merely inconvenient.

DELIBERATELY LEFT ALONE (deferred to the Stage 5 preprocessing pipeline)
------------------------------------------------------------------------
Three representation choices are real modeling considerations, investigated in
notebooks/01_data_quality_investigation.ipynb, but NOT applied here:

  * "No internet service" (6 columns, 1526 customers) is identical to
    InternetService == "No", so one-hot encoding it produces duplicate columns
    and perfect multicollinearity.
  * "No phone service" (682 customers) is identical to PhoneService == "No".
  * SeniorCitizen is coded 0/1 while every other binary column uses Yes/No,
    so dtype-based column routing will treat it as numeric and scale it.

All three are decisions about how to REPRESENT the data for a model, not about
whether the data is correct. Handling them in the pipeline keeps them reversible
and testable; baking them into the cleaned file would not.

Reads:   data/raw/telco_customer_churn.csv   (never modified)
Writes:  data/processed/telco_clean.csv

Run from the project root:  python src/clean_data.py
"""

from pathlib import Path

import pandas as pd

# Paths are built relative to this file, so the script runs correctly no matter
# which directory you launch it from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "telco_clean.csv"

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
EXPECTED_ROWS = 7043

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

# The six add-on services. Each keeps its original three levels, including
# "No internet service" -- see the module docstring for why we do not collapse it.
INTERNET_SERVICE_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# customerID identifies a customer; it carries no information about churn.
# We keep it in the cleaned file (SQL joins in Stage 3 and customer-level risk
# scores in Stage 6 both need it) but it must never be used as a model feature.
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"


def load_raw() -> pd.DataFrame:
    """Load the raw file as text so that data problems stay visible."""
    if not RAW_PATH.exists():
        raise SystemExit(f"Raw data not found at {RAW_PATH}. See README.")

    # dtype=str on purpose. If we let pandas infer types it would quietly decide
    # TotalCharges is text (because of 11 blanks) and we would never be told.
    df = pd.read_csv(RAW_PATH, dtype=str)

    assert list(df.columns) == EXPECTED_COLUMNS, "Raw file does not match the expected IBM Telco schema"
    assert len(df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, found {len(df)}"
    return df


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Remove accidental leading/trailing spaces from every text column.

    Only TotalCharges currently has any (the 11 blanks are " "), but stray spaces
    make ' Yes' and 'Yes' into two different categories, which is the kind of bug
    that is invisible until a model behaves strangely. Cheap insurance.
    """
    return df.apply(lambda col: col.str.strip())


def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the three numeric columns, handling the TotalCharges defect.

    DECISION: the 11 rows with a blank TotalCharges are exactly the 11 rows with
    tenure == 0 (verified, and asserted below). A customer billed for zero months
    has been charged zero. So 0.0 is not an imputed guess -- it is the true value.

    We deliberately do NOT median-impute. The median is $1397, which would assert
    that a never-billed customer has paid $1397. That fabricates data.
    """
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_total = df["TotalCharges"].isna()

    # Confirm the assumption behind setting these to 0 before acting on it.
    assert (df.loc[missing_total, "tenure"] == 0).all(), \
        "A row has missing TotalCharges but nonzero tenure -- investigate before filling with 0"
    assert (df.loc[df["tenure"] == 0, "TotalCharges"].isna()).all(), \
        "A row has tenure 0 but a recorded TotalCharges -- investigate"

    df.loc[missing_total, "TotalCharges"] = 0.0

    # tenure is a whole number of months; store it as an integer.
    df["tenure"] = df["tenure"].astype("int64")
    return df


def record_representation_notes(df: pd.DataFrame) -> dict:
    """Measure -- but do NOT change -- the three representation issues.

    These are real considerations for Stage 5, so we verify them here and report
    the numbers. Deciding how to encode a column is a modeling decision, and this
    script does not make modeling decisions. Nothing below mutates the data.
    """
    notes = {}

    # "No internet service" is identical to InternetService == "No".
    no_internet = df["InternetService"] == "No"
    redundant = [c for c in INTERNET_SERVICE_COLUMNS
                 if (df[c] == "No internet service").equals(no_internet)]
    notes["no_internet_columns"] = len(redundant)
    notes["no_internet_customers"] = int(no_internet.sum())

    # "No phone service" is identical to PhoneService == "No".
    no_phone = df["PhoneService"] == "No"
    notes["no_phone_redundant"] = (df["MultipleLines"] == "No phone service").equals(no_phone)
    notes["no_phone_customers"] = int(no_phone.sum())

    notes["senior_values"] = sorted(df["SeniorCitizen"].unique())
    return notes


def validate_clean(df: pd.DataFrame) -> None:
    """Assertions that must hold before we write the file."""
    assert len(df) == EXPECTED_ROWS, "Row count changed during cleaning -- no rows should be dropped"
    assert list(df.columns) == EXPECTED_COLUMNS, "Columns were added, removed or reordered"
    assert df[ID_COLUMN].is_unique, "customerID is no longer unique"
    assert not df.isna().any().any(), "Cleaned data still contains missing values"

    # Original category levels must be PRESERVED -- we are not collapsing them.
    for col in INTERNET_SERVICE_COLUMNS:
        assert set(df[col].unique()) == {"Yes", "No", "No internet service"}, \
            f"{col} lost its original three levels: {sorted(df[col].unique())}"
    assert set(df["MultipleLines"].unique()) == {"Yes", "No", "No phone service"}, \
        "MultipleLines lost its original three levels"

    # SeniorCitizen must remain in its original 0/1 coding.
    assert set(df["SeniorCitizen"].unique()) == {"0", "1"}, \
        f"SeniorCitizen is no longer 0/1: {sorted(df['SeniorCitizen'].unique())}"

    # The genuinely binary Yes/No columns are untouched.
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        assert set(df[col].unique()) == {"Yes", "No"}, f"{col} has unexpected values"

    # Types we DID convert.
    assert df["tenure"].dtype.kind == "i", "tenure is not an integer column"
    assert df["MonthlyCharges"].dtype.kind == "f", "MonthlyCharges is not a float column"
    assert df["TotalCharges"].dtype.kind == "f", "TotalCharges is not a float column"

    # Ranges that must be true of real billing data.
    assert (df["tenure"] >= 0).all(), "Negative tenure"
    assert (df["MonthlyCharges"] > 0).all(), "Non-positive MonthlyCharges"
    assert (df["TotalCharges"] >= 0).all(), "Negative TotalCharges"

    # The 11 zero-charge customers must be exactly the zero-tenure customers.
    assert (df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all(), \
        "A tenure==0 customer does not have TotalCharges == 0"

    # The target must still be intact and untouched.
    assert set(df[TARGET_COLUMN].unique()) == {"Yes", "No"}, "Churn column was altered"


def main() -> None:
    raw = load_raw()

    df = strip_whitespace(raw)
    df = convert_numeric(df)

    notes = record_representation_notes(df)
    validate_clean(df)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    # --- Report what changed, so the run is auditable ---
    churn_rate = (df[TARGET_COLUMN] == "Yes").mean()
    new_customers = int((df["tenure"] == 0).sum())

    print(f"Read   {RAW_PATH.relative_to(PROJECT_ROOT)}  ({len(raw)} rows)")
    print(f"Wrote  {CLEAN_PATH.relative_to(PROJECT_ROOT)}  ({len(df)} rows, {df.shape[1]} columns)")
    print()
    print("Changes applied:")
    print(f"  TotalCharges : {new_customers} blank values -> 0.0 (verified: all have tenure == 0)")
    print(f"  dtypes       : tenure -> int64, MonthlyCharges/TotalCharges -> float64")
    print(f"  whitespace   : stripped from all text columns")
    print()
    print("Left unchanged (representation choices -> Stage 5 pipeline):")
    print(f"  'No internet service' kept in {notes['no_internet_columns']} columns "
          f"({notes['no_internet_customers']} customers)")
    print(f"  'No phone service'    kept in MultipleLines "
          f"({notes['no_phone_customers']} customers)")
    print(f"  SeniorCitizen kept as {'/'.join(notes['senior_values'])}")
    print()
    print("Not done here (belongs in the Stage 5 pipeline, fitted on training data")
    print("only, to avoid leakage): encoding, scaling, imputation, feature selection,")
    print("category collapsing, row exclusion")
    print()
    print(f"Rows dropped   : 0")
    print(f"Churn rate     : {churn_rate:.2%}  (unchanged from raw -- cleaning must not move the target)")
    print()
    print(f"NOTE for Stage 5: the {new_customers} customers with tenure == 0 have no observed")
    print("outcome yet -- no customer with tenure 0 can be labelled Churn=Yes. Their")
    print("'No' is definitional, not measured. Consider excluding them from training.")


if __name__ == "__main__":
    main()
