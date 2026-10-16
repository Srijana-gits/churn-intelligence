"""
Stage 4 — fit the chosen model and save it.

The choices made here are justified by cross-validation in
notebooks/03_modeling.ipynb. In short:

  features   tenure + the 16 categorical columns. MonthlyCharges is excluded
             because it is a deterministic sum of the service columns (VIF 878),
             and TotalCharges because it helps neither model.
  imbalance  trained on the natural 26.6% base rate. class_weight="balanced"
             produced an identical ranking (Spearman 0.9995) while pushing the
             mean out-of-fold probability to 0.414 against a true rate of
             0.266. Stage 6 needs calibrated probabilities (checked out-of-fold:
             Brier 0.136, worst decile gap 4.4pp), so the threshold is moved in
             Stage 5 instead.
  model      logistic regression. A constrained gradient booster scored +0.006
             ROC-AUC and was inconsistent on PR-AUC, which does not buy back
             the interpretability Stage 6 requires.

Run from the project root:  python src/train_model.py
"""

import sys
import argparse
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Lets the script be run from any directory: python src/train_model.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import PROJECT_ROOT, load_modelling_data, make_preprocessor, split

MODEL_PATH = PROJECT_ROOT / "models" / "churn_logistic_regression.joblib"
SCORING_MODEL_PATH = PROJECT_ROOT / "models" / "churn_scoring.joblib"
MODEL_FEATURES = ["tenure"]   # numeric features kept; categoricals are always included


def build_model():
    """The chosen pipeline. Preprocessing is inside it, so it is fitted on
    training data only and travels with the model."""
    return Pipeline([
        ("prep", make_preprocessor("linear", MODEL_FEATURES)),
        ("model", LogisticRegression(max_iter=2000)),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-data", action="store_true",
                        help="Fit a separate scoring artifact on all eligible rows after evaluation.")
    args = parser.parse_args()
    data = load_modelling_data()
    if args.full_data:
        from evaluate_model import load_evaluation
        from features import CATEGORICAL_FEATURES
        import sklearn

        evaluation = load_evaluation()
        X = data[MODEL_FEATURES + CATEGORICAL_FEATURES]
        model = build_model().fit(X, data["y"])
        SCORING_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, SCORING_MODEL_PATH)
        metadata = {"training_rows": len(data), "clean_data_sha256": evaluation["clean_data_sha256"],
                    "threshold": evaluation["threshold"], "sklearn_version": sklearn.__version__,
                    "score_method": "full_data_fitted_model",
                    "value_cutoff": float(data.loc[data["Churn"].eq("No"), "MonthlyCharges"].median())}
        SCORING_MODEL_PATH.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved separate scoring model fitted on {len(data):,} eligible rows")
        return
    X_train, X_test, y_train, y_test = split(data)

    model = build_model().fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    n_features = model["prep"].transform(X_train).shape[1]

    print(f"\nfitted on {len(X_train):,} training rows, {n_features} encoded features")
    print(f"saved {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print("\nThe test set has not been scored. That happens once, in Stage 5.")


if __name__ == "__main__":
    main()
