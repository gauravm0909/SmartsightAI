"""
SmartSight AI - Data Analyzer Module
Preserves and advances Corporate Data Analyzer aggregation, pivot, and filtering logic.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np


class DataAnalyzer:
    """Core analytics engine powering report grouping, multi-dimensional aggregations, and filtering."""

    AGG_MAP = {
        "sum": "sum",
        "mean": "mean",
        "average": "mean",
        "median": "median",
        "max": "max",
        "min": "min",
        "count": "count",
        "std": "std",
        "var": "var",
    }

    @staticmethod
    def safe_numeric_convert(series: pd.Series) -> pd.Series:
        """
        Converts formatted currency, percentages, or comma-separated numbers safely to float64.
        Preserves compatibility with the original Corporate Data Analyzer helper.
        """
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")
        sanitized = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("£", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        return pd.to_numeric(sanitized, errors="coerce")

    @classmethod
    def group_aggregate(
        cls,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        agg_func: str = "sum",
        top_n: Optional[int] = None,
        sort_ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Groups data by a categorical dimension and aggregates a numeric metric.
        Directly fulfills and upgrades Corporate Report Builder functionality.
        """
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in dataset.")
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found in dataset.")

        func_key = agg_func.lower().strip()
        if func_key not in cls.AGG_MAP:
            raise ValueError(f"Unsupported aggregation: '{agg_func}'. Supported: {list(cls.AGG_MAP.keys())}")
        actual_func = cls.AGG_MAP[func_key]

        work_df = df[[group_col, value_col]].copy()
        # Clean group column
        work_df[group_col] = work_df[group_col].astype(str).str.strip().str.title()
        # Clean numeric value
        work_df[value_col] = cls.safe_numeric_convert(work_df[value_col])
        work_df = work_df.dropna(subset=[value_col])

        if work_df.empty:
            return pd.DataFrame(columns=[group_col, value_col])

        # Execute GroupBy
        result = (
            work_df.groupby(group_col)[value_col]
            .agg(actual_func)
            .reset_index()
            .rename(columns={value_col: f"{func_key.capitalize()} ({value_col})"})
        )

        metric_col = result.columns[1]
        result = result.sort_values(by=metric_col, ascending=sort_ascending)

        if top_n is not None and top_n > 0:
            result = result.head(top_n)

        return result

    @classmethod
    def pivot_summary(
        cls,
        df: pd.DataFrame,
        index_col: str,
        columns_col: str,
        value_col: str,
        agg_func: str = "sum"
    ) -> pd.DataFrame:
        """Builds a two-dimensional pivot table summary."""
        func_key = cls.AGG_MAP.get(agg_func.lower().strip(), "sum")
        work_df = df.copy()
        work_df[value_col] = cls.safe_numeric_convert(work_df[value_col])
        work_df = work_df.dropna(subset=[value_col])

        pivot = pd.pivot_table(
            work_df,
            index=index_col,
            columns=columns_col,
            values=value_col,
            aggfunc=func_key,
            fill_value=0
        )
        return pivot.reset_index()

    @classmethod
    def filter_dataset(
        cls,
        df: pd.DataFrame,
        column: str,
        operator: str,
        target_value: Any
    ) -> pd.DataFrame:
        """
        Safely applies allowlisted filter operations without arbitrary code execution.
        """
        if column not in df.columns:
            raise ValueError(f"Filter column '{column}' not found.")

        op = operator.lower().strip()
        series = df[column]

        if op in ["==", "equals", "eq"]:
            return df[series.astype(str).str.lower() == str(target_value).lower()]
        elif op in ["!=", "not equals", "ne"]:
            return df[series.astype(str).str.lower() != str(target_value).lower()]
        elif op in [">", "gt"]:
            num_s = cls.safe_numeric_convert(series)
            val = float(target_value)
            return df[num_s > val]
        elif op in [">=", "gte"]:
            num_s = cls.safe_numeric_convert(series)
            val = float(target_value)
            return df[num_s >= val]
        elif op in ["<", "lt"]:
            num_s = cls.safe_numeric_convert(series)
            val = float(target_value)
            return df[num_s < val]
        elif op in ["<=", "lte"]:
            num_s = cls.safe_numeric_convert(series)
            val = float(target_value)
            return df[num_s <= val]
        elif op in ["contains", "like"]:
            return df[series.astype(str).str.contains(str(target_value), case=False, na=False)]
        elif op in ["in"]:
            targets = [str(t).lower().strip() for t in target_value] if isinstance(target_value, list) else [str(target_value).lower().strip()]
            return df[series.astype(str).str.lower().isin(targets)]
        else:
            raise ValueError(f"Unsupported filter operator: '{operator}'")

    @classmethod
    def frequency_distribution(cls, df: pd.DataFrame, column: str, top_n: int = 10) -> pd.DataFrame:
        """Returns value frequencies and percentage breakdown for a column."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in DataFrame.")

        counts = df[column].value_counts(dropna=False).head(top_n).reset_index()
        counts.columns = [column, "Count"]
        total = len(df)
        counts["Percentage"] = (counts["Count"] / total * 100).round(2)
        return counts
