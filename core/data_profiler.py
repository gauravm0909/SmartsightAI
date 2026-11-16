"""
SmartSight AI - Smart Data Profiler Module
Computes statistical distributions, data quality scores, correlation matrices, and profiling reports.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


class DataProfiler:
    """Calculates comprehensive statistical profiling for numerical and categorical dimensions."""

    @classmethod
    def generate_profile(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates exhaustive summary metrics, correlations, and distributions."""
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols
        missing_count = int(df.isna().sum().sum())
        missing_pct = round((missing_count / total_cells * 100), 2) if total_cells > 0 else 0.0
        duplicates = int(df.duplicated().sum())
        dup_pct = round((duplicates / n_rows * 100), 2) if n_rows > 0 else 0.0

        mem_bytes = df.memory_usage(deep=True).sum()
        mem_formatted = (
            f"{mem_bytes / 1024:.2f} KB" if mem_bytes < 1024 * 1024 else f"{mem_bytes / (1024 * 1024):.2f} MB"
        )

        # 1. Numerical analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_stats: Dict[str, Dict[str, Any]] = {}

        for col in numeric_cols:
            s = df[col].dropna()
            col_missing = int(df[col].isna().sum())
            if len(s) == 0:
                continue

            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            low_b = q1 - 1.5 * iqr
            high_b = q3 + 1.5 * iqr
            outliers = int(((s < low_b) | (s > high_b)).sum())

            # Mode computation
            modes = s.mode()
            mode_val = float(modes.iloc[0]) if not modes.empty else float(s.mean())

            numeric_stats[col] = {
                "count": int(s.count()),
                "missing": col_missing,
                "missing_pct": round(col_missing / n_rows * 100, 2) if n_rows > 0 else 0.0,
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std(ddof=1)), 4) if len(s) > 1 else 0.0,
                "variance": round(float(s.var(ddof=1)), 4) if len(s) > 1 else 0.0,
                "min": round(float(s.min()), 4),
                "q25": round(q1, 4),
                "median": round(float(s.median()), 4),
                "q75": round(q3, 4),
                "max": round(float(s.max()), 4),
                "range": round(float(s.max() - s.min()), 4),
                "iqr": round(iqr, 4),
                "mode": round(mode_val, 4),
                "skewness": round(float(s.skew()), 4) if len(s) > 2 else 0.0,
                "kurtosis": round(float(s.kurt()), 4) if len(s) > 3 else 0.0,
                "outlier_count": outliers,
                "outlier_pct": round(outliers / len(s) * 100, 2) if len(s) > 0 else 0.0,
            }

        # 2. Categorical analysis
        cat_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
        categorical_stats: Dict[str, Dict[str, Any]] = {}

        for col in cat_cols:
            s = df[col].dropna()
            col_missing = int(df[col].isna().sum())
            n_unique = int(s.nunique())
            modes = s.mode()
            mode_str = str(modes.iloc[0]) if not modes.empty else "N/A"

            # Top 5 value frequencies
            val_counts = s.value_counts().head(5)
            top_vals = [
                {
                    "value": str(idx),
                    "count": int(cnt),
                    "pct": round(cnt / len(s) * 100, 2) if len(s) > 0 else 0.0
                }
                for idx, cnt in val_counts.items()
            ]

            categorical_stats[col] = {
                "count": int(s.count()),
                "missing": col_missing,
                "missing_pct": round(col_missing / n_rows * 100, 2) if n_rows > 0 else 0.0,
                "unique_values": n_unique,
                "mode": mode_str,
                "top_values": top_vals,
            }

        # 3. Datetime analysis
        dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        datetime_stats: Dict[str, Dict[str, Any]] = {}
        for col in dt_cols:
            s = df[col].dropna()
            if len(s) > 0:
                min_d = s.min()
                max_d = s.max()
                days_diff = (max_d - min_d).days if hasattr(max_d - min_d, "days") else 0
                datetime_stats[col] = {
                    "count": int(s.count()),
                    "missing": int(df[col].isna().sum()),
                    "min_date": str(min_d),
                    "max_date": str(max_d),
                    "span_days": days_diff,
                }

        # 4. Correlation matrix for numeric columns
        corr_data: Dict[str, Any] = {"columns": numeric_cols, "matrix": {}}
        if len(numeric_cols) > 1:
            corr_df = df[numeric_cols].corr(method="pearson").round(3)
            # Replace NaNs with 0
            corr_df = corr_df.fillna(0.0)
            corr_data["matrix"] = corr_df.to_dict()

        # 5. Data Quality Score (0 - 100)
        # Completeness (50%), Uniqueness (30%), Statistical consistency (20%)
        completeness_score = max(0.0, 100.0 - missing_pct)
        uniqueness_score = max(0.0, 100.0 - dup_pct)
        total_outliers = sum(v["outlier_count"] for v in numeric_stats.values())
        outlier_pct = (total_outliers / total_cells * 100) if total_cells > 0 else 0.0
        consistency_score = max(0.0, 100.0 - (outlier_pct * 3))

        overall_quality = round(
            (completeness_score * 0.50) + (uniqueness_score * 0.30) + (consistency_score * 0.20), 1
        )

        return {
            "overview": {
                "rows": n_rows,
                "columns": n_cols,
                "total_cells": total_cells,
                "missing_values": missing_count,
                "missing_pct": missing_pct,
                "duplicate_rows": duplicates,
                "duplicate_pct": dup_pct,
                "memory_footprint": mem_formatted,
                "numeric_columns_count": len(numeric_cols),
                "categorical_columns_count": len(cat_cols),
                "datetime_columns_count": len(dt_cols),
                "data_quality_score": overall_quality,
            },
            "numeric_columns": numeric_cols,
            "categorical_columns": cat_cols,
            "datetime_columns": dt_cols,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "datetime_stats": datetime_stats,
            "correlation": corr_data,
        }

    @classmethod
    def get_summary_markdown(cls, profile: Dict[str, Any]) -> str:
        """Renders an executive summary formatted as clean GitHub-style Markdown."""
        ov = profile["overview"]
        lines = [
            f"# SmartSight AI - Data Profile Report",
            f"**Data Quality Score:** {ov['data_quality_score']}/100",
            f"",
            f"### Dataset Snapshot",
            f"| Metric | Value |",
            f"| :--- | :--- |",
            f"| Total Records | {ov['rows']:,} |",
            f"| Total Columns | {ov['columns']} |",
            f"| Missing Cells | {ov['missing_values']:,} ({ov['missing_pct']}%) |",
            f"| Duplicate Rows | {ov['duplicate_rows']:,} ({ov['duplicate_pct']}%) |",
            f"| Memory Consumption | {ov['memory_footprint']} |",
            f"| Numeric Columns | {ov['numeric_columns_count']} |",
            f"| Categorical Columns | {ov['categorical_columns_count']} |",
            f"",
            f"### Numerical Features Analysis",
            f"| Column | Mean | Median | Min | Max | Std Dev | Outliers |",
            f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for col, st in profile["numeric_stats"].items():
            lines.append(
                f"| **{col}** | {st['mean']:,} | {st['median']:,} | {st['min']:,} | {st['max']:,} | {st['std']:,} | {st['outlier_count']} ({st['outlier_pct']}%) |"
            )

        if profile["categorical_stats"]:
            lines.extend([
                f"",
                f"### Categorical Features Analysis",
                f"| Column | Unique Values | Most Common (Mode) | Top Value Frequency |",
                f"| :--- | :--- | :--- | :--- |",
            ])
            for col, st in profile["categorical_stats"].items():
                top = st["top_values"][0] if st["top_values"] else {"value": "N/A", "pct": 0.0}
                lines.append(
                    f"| **{col}** | {st['unique_values']} | {st['mode']} | {top['value']} ({top['pct']}%) |"
                )

        return "\n".join(lines)
