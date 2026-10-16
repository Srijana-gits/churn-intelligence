"""Behaviour checks using synthetic customers; no raw-data download required."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clean_data import convert_numeric, strip_whitespace
from evaluate_model import capacity_metrics, load_evaluation, select_threshold
from export_dashboard import make_priority_list
from features import prepare_features
from score_customers import score_customers
from train_model import build_model


def customers():
    """Cover all category levels without using the IBM file."""
    rng = np.random.default_rng(17)
    n = 90
    levels = {
        "gender": ["Male", "Female"], "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"], "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"], "MultipleLines": ["Yes", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    }
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        levels[col] = ["Yes", "No", "No internet service"]
    data = pd.DataFrame({col: np.resize(values, n) for col, values in levels.items()})
    data["customerID"] = [f"customer-{i:03d}" for i in range(n)]
    data["tenure"] = rng.integers(1, 73, n)
    data["MonthlyCharges"] = rng.uniform(20, 120, n)
    data["TotalCharges"] = data.tenure * data.MonthlyCharges
    data["Churn"] = np.where(rng.random(n) < 0.3, "Yes", "No")
    return data


class CleaningTests(unittest.TestCase):
    def test_blank_charge_is_zero_only_for_zero_tenure(self):
        raw = pd.DataFrame({"tenure": ["0", "1"], "MonthlyCharges": ["20.0", "30.0"],
                            "TotalCharges": [" ", "30.0"]})
        clean = convert_numeric(strip_whitespace(raw))
        self.assertEqual(clean.TotalCharges.tolist(), [0, 30])
        raw.loc[0, "tenure"] = "2"
        with self.assertRaises(AssertionError):
            convert_numeric(strip_whitespace(raw))

    def test_corrupt_charge_and_fractional_tenure_fail(self):
        for tenure, total in [("0", "corrupt"), ("1.5", "30.0")]:
            with self.subTest(tenure=tenure, total=total), self.assertRaises(ValueError):
                convert_numeric(pd.DataFrame({"tenure": [tenure], "MonthlyCharges": ["20.0"], "TotalCharges": [total]}))


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = customers()
        prepared = prepare_features(cls.raw)
        cls.model = build_model().fit(prepared, cls.raw.Churn.eq("Yes").astype(int))
        cls.metadata = {"threshold": 0.3187138, "value_cutoff": 64.45, "score_method": "full_data_fitted_model"}

    def current(self):
        return self.raw.loc[self.raw.Churn.eq("No")].copy()

    def test_preparation_is_repeatable_and_does_not_mutate_input(self):
        before = self.raw.copy(deep=True)
        prepared = prepare_features(self.raw)
        pd.testing.assert_frame_equal(self.raw, before)
        pd.testing.assert_frame_equal(prepared, prepare_features(prepared))
        self.assertTrue(prepared.SeniorCitizen.map(type).eq(str).all())
        self.assertFalse(prepared.MultipleLines.eq("No phone service").any())

    def test_identifiers_target_and_charges_are_not_model_features(self):
        prepared = prepare_features(self.raw)
        before = self.model.predict_proba(prepared)
        prepared["customerID"] = "changed"
        prepared["Churn"] = "Yes"
        prepared["y"] = 1
        prepared["MonthlyCharges"] = 99999
        prepared["TotalCharges"] = 999999
        np.testing.assert_allclose(before, self.model.predict_proba(prepared))
        names = self.model["prep"].get_feature_names_out()
        self.assertFalse(any(any(x in name for x in ["customerID", "Churn", "Charges"]) for name in names))

    def test_batch_scoring_matches_training_representation(self):
        raw = self.current()
        result = score_customers(raw, self.model, self.metadata).set_index("customerID")
        expected = self.model.predict_proba(prepare_features(raw))[:, 1]
        np.testing.assert_allclose(result.loc[raw.customerID, "churn_probability"], expected)
        self.assertEqual(set(result.score_method), {"full_data_fitted_model"})
        self.assertTrue(result.monthly_bill_at_risk.is_monotonic_decreasing)
        np.testing.assert_array_equal(result.priority_rank, np.arange(1, len(result) + 1))

    def test_zero_tenure_is_kept_without_score_or_rank(self):
        raw = self.current()
        idx = raw.index[0]
        raw.loc[idx, "tenure"] = 0
        result = score_customers(raw, self.model, self.metadata)
        row = result.loc[idx]
        self.assertEqual(len(result), len(raw))
        self.assertTrue(pd.isna(row.churn_probability))
        self.assertTrue(pd.isna(row.priority_rank))
        self.assertEqual(row.score_method, "not_scored")
        raw["tenure"] = 0
        all_excluded = score_customers(raw, self.model, self.metadata)
        self.assertTrue(all_excluded.churn_probability.isna().all())

    def test_invalid_scoring_input_fails_clearly(self):
        mutations = [
            lambda d: d.drop(columns="Contract"),
            lambda d: pd.concat([d, d.iloc[[0]]], ignore_index=True),
            lambda d: d.assign(Churn="Yes"),
            lambda d: d.assign(Contract="Unexpected"),
            lambda d: d.assign(MonthlyCharges=np.inf),
            lambda d: d.assign(tenure=1.5),
            lambda d: d.assign(customerID=" "),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                score_customers(mutate(self.current()), self.model, self.metadata)

    def test_priority_boundaries_and_ties(self):
        data = pd.DataFrame({"customerID": ["b", "a", "c", "d", "e", "churned"],
                             "Churn": ["No", "No", "No", "No", "No", "Yes"],
                             "churn_probability": [.32, .32, .31, .31, .32, .9],
                             "MonthlyCharges": [65, 65, 60, 70, 64.99, 100]})
        data["monthly_bill_at_risk"] = data.churn_probability * data.MonthlyCharges
        ranked, _ = make_priority_list(data, .32, 65)
        self.assertNotIn("churned", set(ranked.customerID))
        self.assertEqual(ranked.loc[ranked.customerID.eq("a"), "bucket"].iloc[0], "Contact first")
        self.assertEqual(ranked.loc[ranked.customerID.eq("c"), "bucket"].iloc[0], "Leave alone")
        self.assertEqual(ranked.loc[ranked.customerID.eq("d"), "bucket"].iloc[0], "Monitor")
        self.assertEqual(ranked.loc[ranked.customerID.eq("e"), "bucket"].iloc[0], "Low-cost outreach")
        self.assertLess(ranked.index.get_loc(1), ranked.index.get_loc(0))


class EvaluationTests(unittest.TestCase):
    def test_threshold_and_capacity(self):
        actual = np.array([0, 0, 1, 1])
        prob = np.array([.1, .2, .3, .9])
        self.assertEqual(select_threshold(actual, prob), .3)
        metrics = capacity_metrics(actual, prob, .25)
        self.assertEqual(metrics["customers_contacted"], 1)
        self.assertEqual(metrics["recall"], .5)
        self.assertEqual(metrics["precision"], 1)

    def test_stale_evaluation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "evaluation.json"
            report.write_text(json.dumps({"clean_data_sha256": "old"}))
            with patch("evaluate_model.data_fingerprint", return_value="new"), self.assertRaisesRegex(ValueError, "changed"):
                load_evaluation(report)


if __name__ == "__main__":
    unittest.main()
