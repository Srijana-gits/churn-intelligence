"""Build full-population descriptive data and explicitly retrospective OOF scores."""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from evaluate_model import load_evaluation
from features import (
    CATEGORICAL_FEATURES, CLEAN_PATH, ID_COLUMN, NUMERIC_FEATURES, PROJECT_ROOT,
    load_modelling_data,
)
from modeling import CV
from train_model import build_model

EXPORT_PATH = PROJECT_ROOT / "dashboard" / "data"
BUCKET_ORDER = ["Contact first", "Low-cost outreach", "Monitor", "Leave alone"]
DASHBOARD_COLUMNS = [ID_COLUMN, "tenure", "Contract", "InternetService", "PaymentMethod",
                     "PaperlessBilling", "SeniorCitizen", "MonthlyCharges"]


def make_priority_list(scored, threshold, value_cutoff=None):
    current = scored.loc[scored["Churn"].eq("No")].copy()
    if current.empty:
        raise ValueError("No eligible current customers to prioritise.")
    if value_cutoff is None:
        value_cutoff = float(current["MonthlyCharges"].median())
    high_risk = current["churn_probability"] >= threshold
    high_value = current["MonthlyCharges"] >= value_cutoff
    current["bucket"] = np.where(
        high_risk, np.where(high_value, "Contact first", "Low-cost outreach"),
        np.where(high_value, "Monitor", "Leave alone"))
    current = current.sort_values(["monthly_bill_at_risk", ID_COLUMN], ascending=[False, True])
    current["priority_rank"] = np.arange(1, len(current) + 1)
    current["risk_cutoff"] = threshold
    current["value_cutoff"] = value_cutoff
    return current, value_cutoff


def ranking_comparison(data):
    """Exploratory full-population OOF comparison, not independent validation."""
    data = data.copy()
    data["simple_rule"] = (1000 * (data.Contract.eq("Month-to-month") &
                                  data.InternetService.eq("Fiber optic")) + data.MonthlyCharges)
    rows = []
    churn = data.Churn.eq("Yes")
    total_bill = data.loc[churn, "MonthlyCharges"].sum()
    for policy, column in {"Risk only": "churn_probability", "Risk × bill": "monthly_bill_at_risk",
                           "Bill only": "MonthlyCharges", "Simple rule": "simple_rule"}.items():
        ordered = data.sort_values([column, ID_COLUMN], ascending=[False, True])
        for share in (0.1, 0.2, 0.3):
            top = ordered.head(int(np.ceil(len(data) * share)))
            left = top.Churn.eq("Yes")
            rows.append({"ranking": policy, "contact_share": share,
                         "churners_caught": float(left.sum() / churn.sum()),
                         "churned_billing_reached": float(top.loc[left, "MonthlyCharges"].sum() / total_bill)})
    return pd.DataFrame(rows)


def build_exports():
    evaluation = load_evaluation()
    eligible = load_modelling_data()
    X = eligible[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    eligible["churn_probability"] = cross_val_predict(
        build_model(), X, eligible["y"], cv=CV, method="predict_proba")[:, 1]
    eligible["monthly_bill_at_risk"] = eligible["churn_probability"] * eligible["MonthlyCharges"]
    eligible["score_method"] = "retrospective_5_fold_oof"
    current, value_cutoff = make_priority_list(eligible, evaluation["threshold"])
    # Descriptive charts retain the original categories and all 7,043 customers.
    customers = pd.read_csv(CLEAN_PATH)
    customers["model_eligible"] = customers.tenure > 0
    customers["scoring_status"] = np.where(customers.model_eligible, "Scored", "Excluded: zero tenure")
    customers["churn_actual"] = customers.Churn.eq("Yes").astype(int)
    customers["customer_count"] = 1
    customers["churned_monthly_billing"] = customers.churn_actual * customers.MonthlyCharges
    customers["senior_label"] = np.where(customers.SeniorCitizen.eq(1), "Senior", "Non-senior")
    customers = customers.merge(
        eligible[[ID_COLUMN, "churn_probability", "monthly_bill_at_risk", "score_method"]],
        on=ID_COLUMN, how="left", validate="one_to_one")
    customers["score_method"] = customers.score_method.fillna("not_scored")
    priority = current[["priority_rank"] + DASHBOARD_COLUMNS + [
        "churn_probability", "monthly_bill_at_risk", "bucket", "score_method", "risk_cutoff", "value_cutoff"]]
    priority["customer_count"] = 1
    buckets = priority.groupby("bucket").agg(
        customers=(ID_COLUMN, "size"), average_score=("churn_probability", "mean"),
        monthly_bill_at_risk=("monthly_bill_at_risk", "sum")).reindex(BUCKET_ORDER).reset_index()
    buckets["share_of_bill_at_risk"] = buckets.monthly_bill_at_risk / buckets.monthly_bill_at_risk.sum()
    metadata = {
        "schema_version": 1, "clean_data_sha256": evaluation["clean_data_sha256"],
        "total_customers": len(customers), "eligible_customers": len(eligible),
        "excluded_customers": int((~customers.model_eligible).sum()),
        "current_customers": int(customers.Churn.eq("No").sum()),
        "eligible_current_customers": len(priority), "risk_cutoff": evaluation["threshold"],
        "value_cutoff": value_cutoff, "score_method": "retrospective_5_fold_oof",
        "ranking_comparison": "exploratory; model selection reused this dataset",
        "bill_at_risk": "score × monthly bill; ranking measure, not expected losses or savings",
    }
    return {"customers_scored": customers, "priority_list": priority, "bucket_summary": buckets,
            "ranking_comparison": ranking_comparison(eligible)}, metadata


def save_exports(exports, metadata):
    EXPORT_PATH.mkdir(parents=True, exist_ok=True)
    for name, frame in exports.items():
        frame.to_csv(EXPORT_PATH / f"{name}.csv", index=False, float_format="%.12g")
        print(f"{name}.csv: {len(frame):,} rows")
    (EXPORT_PATH / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    evaluation = load_evaluation()
    pd.DataFrame(evaluation["capacity"]).to_csv(EXPORT_PATH / "capacity_metrics.csv", index=False)
    pd.DataFrame([{"metric": k, "value": v} for k, v in evaluation["quality"].items()]).to_csv(
        EXPORT_PATH / "model_metrics.csv", index=False)
    predictions = pd.read_csv(CLEAN_PATH.parent / "churn_test_predictions.csv")
    predictions["decile"] = pd.qcut(predictions.churn_probability, 10, duplicates="drop")
    calibration = predictions.groupby("decile", observed=True).agg(
        customers=("churn_actual", "size"), mean_probability=("churn_probability", "mean"),
        observed_churn_rate=("churn_actual", "mean")).reset_index(drop=True)
    calibration.to_csv(EXPORT_PATH / "calibration.csv", index=False)


def main():
    save_exports(*build_exports())


if __name__ == "__main__":
    main()
