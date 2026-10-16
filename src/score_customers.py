"""Score a CSV of current customers with the separate full-data fitted model.

Input requires customerID, MonthlyCharges, tenure and the categorical features.
Churn is optional; if supplied, every row must be No. Zero-tenure rows are retained
with an explicit exclusion status. This is still snapshot similarity, not a forecast.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from export_dashboard import make_priority_list
from features import CATEGORICAL_FEATURES, ID_COLUMN, prepare_features
from train_model import SCORING_MODEL_PATH


def score_customers(data, model, metadata):
    required = [ID_COLUMN, "tenure", "MonthlyCharges"] + CATEGORICAL_FEATURES
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    data = data.copy()
    for col in required:
        if data[col].isna().any() or data[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing or blank values in {col}")
    data[ID_COLUMN] = data[ID_COLUMN].astype(str).str.strip()
    if not data[ID_COLUMN].is_unique:
        raise ValueError("customerID must be unique")
    if "Churn" in data and not data.Churn.eq("No").all():
        raise ValueError("Input must contain current customers only (Churn = No)")
    for col in ["tenure", "MonthlyCharges"]:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all():
            raise ValueError(f"Non-finite values in {col}")
    if (data.tenure < 0).any() or (data.tenure % 1 != 0).any() or (data.MonthlyCharges <= 0).any():
        raise ValueError("tenure must be a nonnegative integer and MonthlyCharges must be positive")
    prepared = prepare_features(data)
    encoder = model["prep"].named_transformers_["categorical"]
    for col, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        unknown = set(prepared[col]) - set(categories)
        if unknown:
            raise ValueError(f"Unknown categories in {col}: {sorted(unknown)}")
    # Build output from input fields only; never trust pre-existing scores or ranks.
    output = data[required].copy()
    output["scoring_status"] = np.where(data.tenure > 0, "Scored", "Excluded: zero tenure")
    output["score_method"] = np.where(data.tenure > 0, metadata["score_method"], "not_scored")
    output["churn_probability"] = np.nan
    output["monthly_bill_at_risk"] = np.nan
    output["bucket"] = "Not scored"
    output["priority_rank"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    eligible = prepared.loc[prepared.tenure > 0].copy()
    if not eligible.empty:
        eligible["churn_probability"] = model.predict_proba(eligible)[:, 1]
        eligible["monthly_bill_at_risk"] = eligible.churn_probability * eligible.MonthlyCharges
        eligible["Churn"] = "No"
        ranked, _ = make_priority_list(eligible, metadata["threshold"], metadata["value_cutoff"])
        for column in ["churn_probability", "monthly_bill_at_risk", "bucket", "priority_rank"]:
            output.loc[ranked.index, column] = ranked[column]
    output["risk_cutoff"] = metadata["threshold"]
    output["value_cutoff"] = metadata["value_cutoff"]
    return output.sort_values("priority_rank", na_position="last")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output paths must differ")
    if not SCORING_MODEL_PATH.exists():
        parser.error("Run python src/train_model.py --full-data first")
    metadata = json.loads(SCORING_MODEL_PATH.with_suffix(".json").read_text())
    model = joblib.load(SCORING_MODEL_PATH)
    try:
        scores = score_customers(pd.read_csv(args.input), model, metadata)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False, float_format="%.12g")
    print(f"Saved {len(scores):,} customers to {args.output}")


if __name__ == "__main__":
    main()
