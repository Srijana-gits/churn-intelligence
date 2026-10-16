"""Reproduce Stage 5 without notebooks; freeze the training-OOF threshold.

Repeated execution reproduces the existing evaluation, not a new independent test.
Model choices must not be revised using these held-out results.
"""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from features import CLEAN_PATH, ID_COLUMN, PROJECT_ROOT, load_modelling_data, split
from modeling import CV
from train_model import build_model

EVALUATION_PATH = PROJECT_ROOT / "reports" / "evaluation.json"


def data_fingerprint():
    return hashlib.sha256(CLEAN_PATH.read_bytes()).hexdigest()


def select_threshold(actual, probabilities):
    """Maximise training OOF F1; prefer the higher threshold for exact ties."""
    precision, recall, thresholds = precision_recall_curve(actual, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def operating_metrics(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {
        "customers_flagged": int(predicted.sum()),
        "share_flagged": float(predicted.mean()),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "specificity": float(tn / (tn + fp)),
        "true_positives": int(tp), "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def capacity_metrics(actual, probabilities, share):
    count = int(np.ceil(len(actual) * share))
    selected = np.argsort(-probabilities, kind="stable")[:count]
    found = int(np.asarray(actual)[selected].sum())
    return {"contact_share": share, "customers_contacted": count,
            "precision": found / count, "recall": found / int(np.sum(actual)),
            "churners_captured": found}


def load_evaluation(path=EVALUATION_PATH):
    if not path.exists():
        raise ValueError("Run python src/evaluate_model.py before exporting or scoring.")
    report = json.loads(path.read_text())
    if report["clean_data_sha256"] != data_fingerprint():
        raise ValueError("Clean data changed; rerun evaluation before exporting or scoring.")
    return report


def evaluate():
    data = load_modelling_data()
    X_train, X_test, y_train, y_test = split(data)
    oof = cross_val_predict(build_model(), X_train, y_train, cv=CV,
                            method="predict_proba")[:, 1]
    threshold = select_threshold(y_train, oof)
    model = build_model().fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    actual = y_test.to_numpy()
    quality = {
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "average_precision": float(average_precision_score(actual, probabilities)),
        "brier_score": float(brier_score_loss(actual, probabilities)),
        "base_rate_brier": float(brier_score_loss(actual, np.full(len(actual), y_train.mean()))),
    }
    # Fixed-model bootstrap: conditions on this fitted model and held-out sample.
    rng = np.random.default_rng(42)
    boot = {"roc_auc": [], "average_precision": []}
    for _ in range(2000):
        idx = rng.integers(0, len(actual), len(actual))
        if np.unique(actual[idx]).size < 2:
            continue
        boot["roc_auc"].append(roc_auc_score(actual[idx], probabilities[idx]))
        boot["average_precision"].append(average_precision_score(actual[idx], probabilities[idx]))
    intervals = {name: np.percentile(values, [2.5, 97.5]).tolist()
                 for name, values in boot.items()}
    report = {
        "schema_version": 1, "clean_data_sha256": data_fingerprint(),
        "random_state": 42, "cv_folds": 5,
        "train_rows": len(X_train), "test_rows": len(X_test),
        "threshold": threshold, "threshold_source": "maximum F1 on training-only OOF scores",
        "quality": quality, "bootstrap_95_percent_intervals": intervals,
        "operating_points": {
            "default": {"threshold": 0.5, **operating_metrics(actual, probabilities >= 0.5)},
            "selected": {"threshold": threshold, **operating_metrics(actual, probabilities >= threshold)},
        },
        "capacity": [capacity_metrics(actual, probabilities, s) for s in (0.1, 0.2, 0.3)],
        "interpretation": "Retrospective snapshot classification; no future-month prediction or causal effect.",
    }
    predictions = pd.DataFrame({
        ID_COLUMN: data.loc[X_test.index, ID_COLUMN].to_numpy(),
        "churn_actual": actual, "churn_probability": probabilities,
        "flagged_at_oof_f1_threshold": (probabilities >= threshold).astype(int),
        "oof_f1_threshold": threshold,
    })
    return report, predictions


def main():
    report, predictions = evaluate()
    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    predictions.sort_values("churn_probability", ascending=False).to_csv(
        CLEAN_PATH.parent / "churn_test_predictions.csv", index=False)
    print(f"Saved {EVALUATION_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Threshold {report['threshold']:.8f}; test ROC-AUC {report['quality']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
