"""
SmartSight AI - Backend Route Integration Test Suite
Directly validates all FastAPI route handlers, parameters, and response structures.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web_app.backend.routes import (
    datasets,
    profiling,
    cleaning,
    visualization,
    insights,
    nlp,
    ml,
    reports,
    history,
)


class TestFastAPIRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize sample dataset
        req = datasets.SampleLoadRequest(sample_id="sample-employees")
        cls.sample_res = datasets.load_sample(req)
        cls.dataset_id = cls.sample_res["dataset_id"]

    def test_01_sample_loading(self):
        """Validates dataset loading and metadata schema."""
        self.assertTrue(self.sample_res["success"])
        self.assertEqual(self.sample_res["dataset_id"], "sample-employees")
        self.assertIn("inspection", self.sample_res)
        self.assertGreater(self.sample_res["inspection"]["rows"], 50)

    def test_02_dataset_preview(self):
        """Validates paginated records preview handler."""
        res = datasets.get_preview(self.dataset_id, page=1, page_size=10)
        self.assertEqual(res["page"], 1)
        self.assertEqual(len(res["records"]), 10)
        self.assertIn("Department", res["columns"])

    def test_03_profiling_handler(self):
        """Validates data profiler handler."""
        res = profiling.get_profile(self.dataset_id)
        self.assertTrue(res["success"])
        self.assertIn("data_quality_score", res["profile"]["overview"])
        self.assertGreaterEqual(res["profile"]["overview"]["data_quality_score"], 0)

    def test_04_cleaning_and_reset(self):
        """Validates cleaning handler and reset rollback."""
        opts = cleaning.CleaningOptions(
            remove_duplicates=True,
            handle_missing_num="median",
            handle_missing_cat="mode",
            strip_whitespace=True,
        )
        clean_res = cleaning.clean_dataset(self.dataset_id, opts)
        self.assertTrue(clean_res["success"])
        self.assertIn("summary", clean_res)

        reset_res = cleaning.reset_dataset(self.dataset_id)
        self.assertTrue(reset_res["success"])

    def test_05_visualization_handler(self):
        """Validates interactive Plotly JSON generator handler."""
        req = visualization.ChartRequest(
            chart_type="Column",
            x_col="Department",
            y_col="Salary",
            agg_func="mean",
        )
        res = visualization.generate_chart(self.dataset_id, req)
        self.assertTrue(res["success"])
        self.assertIn("figure_json", res)
        self.assertIn("data", res["figure_json"])

    def test_06_insights_handler(self):
        """Validates autonomous AI insights handler."""
        req = insights.InsightsRequest(use_llm=False)
        res = insights.get_insights(self.dataset_id, req)
        self.assertTrue(res["success"])
        self.assertGreater(res["insights_count"], 0)

    def test_07_nlp_query_handler(self):
        """Validates Ask Your Data conversational NLP handler."""
        req = nlp.QueryRequest(query="What is the average salary?")
        res = nlp.ask_question(self.dataset_id, req)
        self.assertTrue(res["success"])
        self.assertEqual(res["intent"], "metric_aggregation")
        self.assertIn("average", res["answer"].lower())

    def test_08_ml_training_handler(self):
        """Validates predictive machine learning training handler."""
        req = ml.TrainModelRequest(
            target_column="Salary",
            feature_columns=["Experience_Years", "Age", "Performance_Rating"],
            algorithm="Linear Regression",
        )
        res = ml.train_model(self.dataset_id, req)
        self.assertTrue(res["success"])
        self.assertEqual(res["task_type"], "regression")
        self.assertIn("r2_score", res["metrics"])

    def test_09_reports_download_handlers(self):
        """Validates PDF, Excel, and CSV download responses."""
        # PDF Response
        pdf_resp = reports.download_pdf(self.dataset_id)
        self.assertEqual(pdf_resp.media_type, "application/pdf")
        self.assertGreater(len(pdf_resp.body), 1000)

        # Excel Response
        excel_resp = reports.download_excel(self.dataset_id)
        self.assertEqual(excel_resp.media_type, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        self.assertGreater(len(excel_resp.body), 1000)

        # CSV Response
        csv_resp = reports.download_csv(self.dataset_id)
        self.assertEqual(csv_resp.media_type, "text/csv")
        self.assertGreater(len(csv_resp.body), 500)

    def test_10_history_handler(self):
        """Validates SQLite activity log handler."""
        res = history.get_history(limit=10)
        self.assertTrue(res["success"])
        self.assertGreater(res["count"], 0)


if __name__ == "__main__":
    unittest.main()
