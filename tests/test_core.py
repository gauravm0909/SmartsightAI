"""
SmartSight AI - Core Test Suite
Validates all data processing, profiling, visualization, ML, NLP, and reporting pipelines.
"""

import os
import unittest
import pandas as pd
import numpy as np

from core.data_loader import DataLoader
from core.data_cleaner import DataCleaner
from core.data_profiler import DataProfiler
from core.data_analyzer import DataAnalyzer
from core.visualization import Visualizer
from core.ai_insights import AIInsightsEngine
from core.natural_language_query import NaturalLanguageQueryEngine
from core.ml_models import MLPredictiveEngine
from core.report_generator import ReportGenerator


class TestSmartSightCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.emp_path = "datasets/corporate_employees.csv"
        cls.sales_path = "datasets/retail_sales.csv"
        cls.churn_path = "datasets/customer_churn.csv"

        cls.df_emp = DataLoader.load_dataset(cls.emp_path)
        cls.df_sales = DataLoader.load_dataset(cls.sales_path)
        cls.df_churn = DataLoader.load_dataset(cls.churn_path)

    def test_01_data_loader(self):
        """Tests ingestion, validation, and schema inference."""
        self.assertFalse(self.df_emp.empty)
        self.assertGreater(len(self.df_emp), 50)
        self.assertIn("Salary", self.df_emp.columns)

        inspection = DataLoader.inspect_dataset(self.df_emp)
        self.assertEqual(inspection["rows"], len(self.df_emp))
        self.assertIn("Salary", inspection["numeric_columns"])
        self.assertIn("Department", inspection["categorical_columns"])

    def test_02_data_cleaner(self):
        """Tests non-destructive cleaning, missing value imputation, and duplicate removal."""
        raw_df = self.df_emp.copy()
        raw_rows = len(raw_df)

        cleaned_df, summary = DataCleaner.clean_dataset(raw_df, {
            "remove_duplicates": True,
            "handle_missing_num": "median",
            "handle_missing_cat": "mode",
            "strip_whitespace": True,
            "convert_numeric_strings": True,
        })

        # Ensure original dataframe was NOT modified
        self.assertEqual(len(raw_df), raw_rows)
        # Ensure no nulls in cleaned numeric column
        self.assertEqual(int(cleaned_df["Bonus"].isna().sum()), 0)
        # Ensure whitespace was stripped
        self.assertNotIn("  IT  ", cleaned_df["Department"].tolist())
        self.assertIn("initial_rows", summary)
        self.assertIn("actions_taken", summary)

    def test_03_data_profiler(self):
        """Tests profile generation, statistics calculation, and quality scoring."""
        profile = DataProfiler.generate_profile(self.df_emp)
        ov = profile["overview"]
        self.assertGreaterEqual(ov["data_quality_score"], 0)
        self.assertLessEqual(ov["data_quality_score"], 100)
        self.assertIn("Salary", profile["numeric_stats"])

        salary_stats = profile["numeric_stats"]["Salary"]
        self.assertGreater(salary_stats["mean"], 0)
        self.assertGreater(salary_stats["max"], salary_stats["min"])
        self.assertIn("skewness", salary_stats)

        md = DataProfiler.get_summary_markdown(profile)
        self.assertIn("# SmartSight AI - Data Profile Report", md)

    def test_04_data_analyzer(self):
        """Tests GroupBy aggregation and safe numeric conversion (Corporate Report Builder)."""
        grouped = DataAnalyzer.group_aggregate(
            self.df_emp, group_col="Department", value_col="Salary", agg_func="mean"
        )
        self.assertFalse(grouped.empty)
        self.assertEqual(len(grouped.columns), 2)
        self.assertIn("Department", grouped.columns)

        # Filtering test
        filtered = DataAnalyzer.filter_dataset(self.df_emp, "Department", "==", "it")
        self.assertTrue((filtered["Department"].str.strip().str.lower() == "it").all())

    def test_05_visualizer_plotly_and_matplotlib(self):
        """Tests creation of both Plotly and Matplotlib figures."""
        # Plotly
        fig_p = Visualizer.create_plotly_figure(
            self.df_emp, chart_type="Column", x_col="Department", y_col="Salary", agg_func="mean"
        )
        self.assertIsNotNone(fig_p)

        # Matplotlib
        fig_m = Visualizer.create_matplotlib_figure(
            self.df_emp, chart_type="Column", x_col="Department", y_col="Salary", agg_func="mean"
        )
        self.assertIsNotNone(fig_m)

    def test_06_ai_insights(self):
        """Tests deterministic statistical insight discovery."""
        res = AIInsightsEngine.generate_insights(self.df_emp, use_llm=False)
        self.assertGreater(res["insights_count"], 0)
        insights = res["insights"]
        categories = [i["category"] for i in insights]
        self.assertTrue(any(c in ["Performance", "Correlation", "Anomaly", "Distribution"] for c in categories))

    def test_07_natural_language_query(self):
        """Tests conversational query parsing without code execution."""
        # 1. Superlative lookup
        q1 = NaturalLanguageQueryEngine.process_query(self.df_emp, "Which department has the highest average salary?")
        self.assertEqual(q1["intent"], "extreme_lookup")
        self.assertIn("highest average", q1["answer"].lower())

        # 2. Metric aggregation
        q2 = NaturalLanguageQueryEngine.process_query(self.df_emp, "What is the average salary?")
        self.assertEqual(q2["intent"], "metric_aggregation")
        self.assertIn("average", q2["answer"].lower())

        # 3. Row count
        q3 = NaturalLanguageQueryEngine.process_query(self.df_emp, "How many records are there?")
        self.assertEqual(q3["intent"], "row_count")

    def test_08_ml_regression_and_classification(self):
        """Tests ML models: Regression, Classification, and Clustering."""
        # Regression on Salary
        reg_res = MLPredictiveEngine.train_and_evaluate(
            self.df_emp,
            feature_cols=["Experience_Years", "Age", "Performance_Rating"],
            target_col="Salary",
            algorithm="Random Forest Regressor",
        )
        self.assertEqual(reg_res["task_type"], "regression")
        self.assertIn("r2_score", reg_res["metrics"])
        self.assertGreater(reg_res["metrics"]["r2_score"], 0.4)

        # Classification on Churn
        cls_res = MLPredictiveEngine.train_and_evaluate(
            self.df_churn,
            feature_cols=["Contract", "Tenure_Months", "Monthly_Charges", "Payment_Method"],
            target_col="Churn",
            algorithm="Random Forest Classifier",
        )
        self.assertEqual(cls_res["task_type"], "classification")
        self.assertIn("accuracy", cls_res["metrics"])
        self.assertIn("confusion_matrix", cls_res)

        # Clustering on Retail Sales
        cluster_res = MLPredictiveEngine.train_and_evaluate(
            self.df_sales,
            feature_cols=["Quantity", "Unit_Price", "Total_Revenue", "Profit"],
            target_col=None,
            algorithm="K-Means Clustering",
            n_clusters=3,
        )
        self.assertEqual(cluster_res["task_type"], "clustering")
        self.assertEqual(cluster_res["n_clusters"], 3)
        self.assertIn("silhouette_score", cluster_res["metrics"])

    def test_09_report_generator(self):
        """Tests export of Excel, CSV, PNG, and PDF reports."""
        profile = DataProfiler.generate_profile(self.df_emp)
        insights = AIInsightsEngine.generate_insights(self.df_emp)["insights"]
        fig = Visualizer.create_matplotlib_figure(
            self.df_emp, chart_type="Column", x_col="Department", y_col="Salary", agg_func="mean"
        )

        # Test Excel bytes
        excel_bytes = ReportGenerator.export_excel(self.df_emp, profile=profile, insights=insights)
        self.assertGreater(len(excel_bytes), 1000)

        # Test CSV bytes
        csv_bytes = ReportGenerator.export_csv(self.df_emp)
        self.assertGreater(len(csv_bytes), 500)

        # Test Chart PNG bytes
        png_bytes = ReportGenerator.export_chart_png(fig)
        self.assertGreater(len(png_bytes), 1000)

        # Test PDF bytes
        pdf_bytes = ReportGenerator.export_pdf(
            self.df_emp, profile=profile, insights=insights, chart_fig=fig
        )
        self.assertGreater(len(pdf_bytes), 2000)


if __name__ == "__main__":
    unittest.main()
