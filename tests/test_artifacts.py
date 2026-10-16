"""Check committed dashboard artifacts, including population and workbook links."""

import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dashboard" / "data"


class DashboardArtifactTests(unittest.TestCase):
    def test_population_and_exclusions(self):
        data = pd.read_csv(DATA / "customers_scored.csv")
        metadata = json.loads((DATA / "metadata.json").read_text())
        self.assertEqual(len(data), 7043)
        self.assertTrue(data.customerID.is_unique)
        self.assertEqual(int(data.churn_actual.sum()), 1869)
        self.assertEqual(metadata["total_customers"], len(data))
        eligible = data.model_eligible
        self.assertEqual(int((~eligible).sum()), 11)
        self.assertTrue((data.loc[~eligible, "tenure"] == 0).all())
        self.assertTrue(data.loc[~eligible, "churn_probability"].isna().all())
        self.assertTrue(data.loc[eligible, "churn_probability"].between(0, 1).all())
        np.testing.assert_allclose(data.loc[eligible, "monthly_bill_at_risk"],
                                   data.loc[eligible, "churn_probability"] * data.loc[eligible, "MonthlyCharges"], rtol=1e-9)

    def test_priority_list_matches_scores_thresholds_and_buckets(self):
        data = pd.read_csv(DATA / "customers_scored.csv").set_index("customerID")
        priority = pd.read_csv(DATA / "priority_list.csv")
        metadata = json.loads((DATA / "metadata.json").read_text())
        evaluation = json.loads((ROOT / "reports" / "evaluation.json").read_text())
        self.assertEqual(metadata["clean_data_sha256"], evaluation["clean_data_sha256"])
        self.assertEqual(metadata["risk_cutoff"], evaluation["threshold"])
        self.assertEqual(len(priority), 5163)
        self.assertTrue(priority.customerID.is_unique)
        self.assertFalse(priority.isna().any().any())
        self.assertEqual(set(priority.customerID), set(data.index[data.Churn.eq("No") & data.model_eligible]))
        np.testing.assert_array_equal(priority.priority_rank, np.arange(1, len(priority) + 1))
        self.assertTrue(priority.monthly_bill_at_risk.is_monotonic_decreasing)
        np.testing.assert_allclose(priority.churn_probability, data.loc[priority.customerID, "churn_probability"])
        high_risk = priority.churn_probability >= metadata["risk_cutoff"]
        high_value = priority.MonthlyCharges >= metadata["value_cutoff"]
        expected = np.where(high_risk, np.where(high_value, "Contact first", "Low-cost outreach"),
                            np.where(high_value, "Monitor", "Leave alone"))
        np.testing.assert_array_equal(priority.bucket, expected)
        buckets = pd.read_csv(DATA / "bucket_summary.csv")
        self.assertEqual(int(buckets.customers.sum()), len(priority))
        self.assertAlmostEqual(buckets.share_of_bill_at_risk.sum(), 1)

    def test_tableau_package_has_portable_connections_and_valid_sheet_references(self):
        workbook = ROOT / "dashboard" / "churn_intelligence.twb"
        root = ET.parse(workbook).getroot()
        self.assertIn("source-build", root.attrib)
        for window in root.findall("windows/window[@class='worksheet']"):
            self.assertIsNotNone(window.find("cards"))
            self.assertIsNotNone(window.find("viewpoint"))
            self.assertIsNone(window.find("view"))
        sheets = {s.get("name") for s in root.findall("worksheets/worksheet")}
        dashboards = root.findall("dashboards/dashboard")
        self.assertEqual(len(dashboards), 3)
        for dashboard in dashboards:
            for zone in dashboard.findall(".//zone[@name]"):
                self.assertIn(zone.get("name"), sheets)
        with zipfile.ZipFile(workbook.with_suffix(".twbx")) as package:
            self.assertEqual(package.read(workbook.name), workbook.read_bytes())
            connections = root.findall(".//connection[@class='hyper']")
            self.assertEqual(len(connections), 4)
            for connection in connections:
                relative = Path(connection.get("dbname"))
                self.assertFalse(relative.is_absolute())
                self.assertEqual(package.read(relative.as_posix()), (workbook.parent / relative).read_bytes())
        sources = {source.get("name"): {c.get("name") for c in source.findall("column")}
                   for source in root.findall("datasources/datasource")}
        for dependency in root.findall(".//datasource-dependencies"):
            for instance in dependency.findall("column-instance"):
                self.assertIn(instance.get("column"), sources[dependency.get("datasource")])


if __name__ == "__main__":
    unittest.main()
