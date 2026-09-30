"""
Stage 4 — baselines and models, scored by cross-validation on the training set.

The test set is not touched anywhere in this file. Every comparison here uses
5-fold cross-validation inside the training data, so the test set stays clean
for a single honest evaluation in Stage 5.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# zero_division=0 is set explicitly because the "predict nobody churns" baseline
# makes no positive predictions, leaving precision undefined. Scoring it as 0 is
# the honest reading: a model that flags no one has no precision to speak of.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}

# Threshold-free metrics, for models that output probabilities. They rank models
# without committing to a cut-off; choosing the cut-off is Stage 5. The two
# baselines are hard rules with no probabilities, so they are scored on SCORING
# only -- an AUC for a single-threshold rule is a degenerate number.
RANKING_SCORING = {**SCORING, "roc_auc": "roc_auc", "pr_auc": "average_precision"}


class ContractRule(BaseEstimator, ClassifierMixin):
    """Predict churn for every month-to-month customer, nobody else.

    A deliberately dumb benchmark built from the strongest single association
    found in Stage 3 (Contract, Cramer's V = 0.410). If a trained model cannot
    beat one `if` statement, the complexity is not earning its place.
    """

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return (X["Contract"] == "Month-to-month").astype(int).to_numpy()


def score_model(model, X, y, label, scoring=RANKING_SCORING):
    """5-fold CV on the training set. Returns mean and std of each metric."""
    results = cross_validate(model, X, y, cv=CV, scoring=scoring)
    row = {"model": label}
    for metric in scoring:
        scores = results[f"test_{metric}"]
        row[metric] = scores.mean()
        row[f"{metric}_std"] = scores.std()
    return row
