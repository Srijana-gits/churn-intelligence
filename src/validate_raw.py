"""
Stage 1 — validate the raw Telco Customer Churn file before any analysis.

Why this exists: every later result (churn rates, model scores, revenue at risk)
is only as trustworthy as the file underneath it. This script fails loudly if the
raw data is not the file we think it is, and prints a quality report we use to
plan the cleaning work in Stage 2.

Run:  python src/validate_raw.py
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_customer_churn.csv"

# The canonical IBM Telco Customer Churn schema (21 columns, in order).
# Hard-coding it turns "I hope this is the right file" into an assertion.
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
EXPECTED_ROWS = 7043


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"Raw data not found at {RAW_PATH}. See README for how to obtain it.")

    # Read everything as text first. If we let pandas infer dtypes we would not
    # SEE the data problems, we would inherit them silently.
    df = pd.read_csv(RAW_PATH, dtype=str)

    print(f"File:    {RAW_PATH.name}")
    print(f"Shape:   {df.shape[0]} rows x {df.shape[1]} columns")

    # --- Structural checks: fail fast if this isn't the expected dataset ---
    assert list(df.columns) == EXPECTED_COLUMNS, "Column names/order do not match the IBM Telco schema"
    assert len(df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, found {len(df)}"
    assert df["customerID"].is_unique, "customerID is not unique — rows are not one-per-customer"
    print("Schema:  OK (21 canonical columns, 7043 rows, customerID unique)\n")

    # --- Target variable: the number that anchors the whole project ---
    churn = df["Churn"].value_counts()
    churn_rate = churn.get("Yes", 0) / len(df)
    print("Target (Churn):")
    for label, n in churn.items():
        print(f"  {label:<4} {n:>5}  ({n / len(df):.1%})")
    print(f"  Baseline churn rate: {churn_rate:.2%}")
    print(f"  A 'nobody churns' model would be {1 - churn_rate:.2%} accurate and useless.\n")

    # --- Missingness ---
    # True nulls are rare here; the real problem is blank STRINGS, which pandas
    # does not count as missing. We check for both.
    true_nulls = df.isna().sum()
    blanks = df.apply(lambda s: s.str.strip().eq("").sum())
    issues = (true_nulls + blanks)
    issues = issues[issues > 0]
    print("Missing / blank values:")
    if issues.empty:
        print("  none")
    else:
        for col, n in issues.items():
            print(f"  {col:<16} {n} blank or null")
    print()

    # --- Numeric columns that may not actually be numeric ---
    print("Numeric column check (values that will NOT convert to a number):")
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad = coerced.isna().sum()
        status = "OK" if bad == 0 else f"{bad} non-numeric values -> must be handled in Stage 2"
        print(f"  {col:<16} {status}")
    print()

    # --- Investigate the TotalCharges problem rather than just patching it ---
    bad_total = df[pd.to_numeric(df["TotalCharges"], errors="coerce").isna()]
    if not bad_total.empty:
        print(f"Rows with unusable TotalCharges: {len(bad_total)}")
        print(f"  tenure values among them: {sorted(bad_total['tenure'].unique())}")
        print(f"  Churn values among them:  {sorted(bad_total['Churn'].unique())}")
        print("  -> These look structurally missing (never billed), not randomly missing.")
        print("     That distinction decides how we treat them in Stage 2.\n")

    # --- Cardinality of every categorical column, so nothing surprises us later ---
    print("Distinct values per column:")
    for col in df.columns:
        n = df[col].nunique()
        sample = "" if n > 6 else "  " + ", ".join(sorted(df[col].dropna().unique()))
        print(f"  {col:<18} {n:>5}{sample}")


if __name__ == "__main__":
    main()
