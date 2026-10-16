"""
Stage 4 — load the cleaned data, split it, and define the feature sets.

Everything here runs BEFORE any model is fitted. The split happens first so that
no transformation can see the test set; the test set is not used at all until
Stage 5.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "telco_clean.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.25

# The six add-on columns plus MultipleLines carry a level ("No internet service",
# "No phone service") that is identical to InternetService/PhoneService being "No".
# Collapsing removes seven duplicate dummy columns. Nothing is lost: the
# no-internet group is still identified by InternetService itself.
SERVICE_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]

# Columns are listed by name rather than selected by dtype. SeniorCitizen is
# stored as 0/1, so dtype-based selection would treat it as numeric and scale it.
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def prepare_features(data):
    """Apply the same deterministic representation rules to training and scoring."""
    data = data.copy()
    data["SeniorCitizen"] = data["SeniorCitizen"].astype(str)
    for column in SERVICE_COLUMNS:
        data[column] = data[column].replace("No internet service", "No")
    data["MultipleLines"] = data["MultipleLines"].replace("No phone service", "No")
    return data


def load_modelling_data():
    """Cleaned data, minus the rows that cannot be modelled.

    The 11 customers with tenure == 0 have never completed a billing period, so
    their Churn = "No" is true by definition rather than observed. They are
    removed here, before the split, because they are not part of the population
    the question is about.
    """
    data = pd.read_csv(CLEAN_PATH)
    excluded = (data["tenure"] == 0).sum()
    data = data[data["tenure"] > 0].copy()

    # SeniorCitizen is categorical despite its 0/1 storage; make that explicit so
    # it is never treated as a number by accident.
    data = prepare_features(data)

    data["y"] = (data[TARGET] == "Yes").astype(int)
    print(f"{len(data):,} customers for modelling ({excluded} with tenure = 0 excluded)")
    return data


def split(data):
    """Stratified train/test split. Stratifying keeps the 26.5% churn rate in both."""
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, stratify=target, random_state=RANDOM_STATE)

    print(f"train {len(X_train):,} rows, churn {y_train.mean():.3f}")
    print(f"test  {len(X_test):,} rows, churn {y_test.mean():.3f}   (untouched until Stage 5)")
    return X_train, X_test, y_train, y_test


def make_preprocessor(model_type, numeric_features=None, bin_tenure=False):
    """Preprocessing for one model type. Fitted on training data only.

    The two models need different preparation, so they get different
    preprocessors rather than sharing one:

    - logistic regression: one-hot encode, and standardise the numeric columns so
      that L2 regularisation penalises all features on a comparable scale.
    - gradient boosting: one-hot encode, leave numbers alone. Tree splits are
      scale-invariant, so scaling would change nothing.

    numeric_features lets us test feature variants (e.g. with and without
    TotalCharges) by cross-validation without editing this function.

    bin_tenure=True replaces the linear tenure term with quantile bins. The bin
    edges are learned inside the pipeline, so each CV fold computes its own from
    training data only. Binning with pd.qcut beforehand would set the edges using
    every row, including the held-out ones -- a quiet form of leakage.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES

    # drop="first" avoids the dummy-variable trap for the linear model.
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

    numeric_step = StandardScaler() if model_type == "linear" else "passthrough"

    steps = [("categorical", encoder, CATEGORICAL_FEATURES)]

    if bin_tenure:
        steps.append(("tenure_bins",
                      KBinsDiscretizer(n_bins=6, encode="onehot-dense", strategy="quantile"),
                      ["tenure"]))
        numeric_features = [f for f in numeric_features if f != "tenure"]

    if numeric_features:
        steps.append(("numeric", numeric_step, numeric_features))

    return ColumnTransformer(steps)
