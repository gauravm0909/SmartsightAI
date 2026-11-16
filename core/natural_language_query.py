"""
SmartSight AI - Ask Your Data (Natural Language Query Engine)
Provides secure, allowlisted conversational intelligence without executing arbitrary code.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from core.data_analyzer import DataAnalyzer


class NaturalLanguageQueryEngine:
    """Safe, intent-driven conversational query engine for uploaded tabular datasets."""

    AGG_KEYWORDS = {
        "average": "mean",
        "mean": "mean",
        "avg": "mean",
        "total": "sum",
        "sum": "sum",
        "highest": "max",
        "maximum": "max",
        "max": "max",
        "lowest": "min",
        "minimum": "min",
        "min": "min",
        "median": "median",
        "count": "count",
        "how many": "count",
        "number of": "count",
    }

    CHART_KEYWORDS = {
        "bar": "bar",
        "column": "column",
        "pie": "pie",
        "donut": "donut",
        "line": "line",
        "scatter": "scatter",
        "trend": "line",
        "histogram": "histogram",
        "distribution": "histogram",
        "box": "box",
        "plot": "column",
        "chart": "column",
        "graph": "column",
        "visualize": "column",
        "show": None,
    }

    @classmethod
    def process_query(cls, df: pd.DataFrame, query_text: str) -> Dict[str, Any]:
        """
        Parses a natural language question and executes safe, allowlisted computations.
        Never executes raw Python code, shell commands, or SQL strings.
        """
        q = query_text.lower().strip()
        if not q:
            return cls._fallback_response(df, "Please enter a query or question about your dataset.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()
        all_cols = list(df.columns)

        matched_cols = cls._match_columns(q, all_cols)

        # 1. Intent: "Which [category] has the highest/lowest [metric]?"
        superlative_match = re.search(r"which\s+([a-z0-9_\s]+?)\s+has\s+(?:the\s+)?(highest|lowest|maximum|minimum|top|most|least)\s+([a-z0-9_\s]+)", q)
        if superlative_match:
            cat_candidate = superlative_match.group(1).strip()
            direction = superlative_match.group(2).strip()
            metric_candidate = superlative_match.group(3).strip()

            target_cat = cls._find_best_col_match(cat_candidate, cat_cols or all_cols)
            target_metric = cls._find_best_col_match(metric_candidate, numeric_cols)

            if target_cat and target_metric:
                is_max = direction in ["highest", "maximum", "top", "most"]
                agg_func = "max" if is_max else "min"
                grouped = (
                    df.groupby(target_cat)[target_metric]
                    .mean()
                    .reset_index()
                    .sort_values(by=target_metric, ascending=not is_max)
                )
                if not grouped.empty:
                    top_row = grouped.iloc[0]
                    answer_text = (
                        f"The **{top_row[target_cat]}** {target_cat} has the "
                        f"{'highest' if is_max else 'lowest'} average {target_metric} "
                        f"at **{top_row[target_metric]:,.2f}**."
                    )
                    return {
                        "answer": answer_text,
                        "intent": "extreme_lookup",
                        "confidence": "high",
                        "data": grouped.head(5).to_dict(orient="records"),
                        "chart_spec": {
                            "chart_type": "column",
                            "x_col": target_cat,
                            "y_col": target_metric,
                            "agg_func": "mean",
                            "title": f"Average {target_metric} by {target_cat}",
                        },
                    }

        # 2. Intent: Simple Overall Aggregation ("What is the average salary?", "Total revenue?")
        detected_agg = None
        for kw, func in cls.AGG_KEYWORDS.items():
            if re.search(r"\b" + re.escape(kw) + r"\b", q):
                detected_agg = func
                break

        # Check for single metric aggregation
        if detected_agg and matched_cols:
            num_match = [c for c in matched_cols if c in numeric_cols]
            if num_match:
                target_col = num_match[0]
                s = DataAnalyzer.safe_numeric_convert(df[target_col]).dropna()
                val = 0.0
                if detected_agg == "mean":
                    val = s.mean()
                    desc = "average"
                elif detected_agg == "sum":
                    val = s.sum()
                    desc = "total"
                elif detected_agg == "max":
                    val = s.max()
                    desc = "maximum"
                elif detected_agg == "min":
                    val = s.min()
                    desc = "minimum"
                elif detected_agg == "median":
                    val = s.median()
                    desc = "median"
                elif detected_agg == "count":
                    val = float(len(s))
                    desc = "count of"

                answer_text = f"The {desc} **{target_col}** across {len(s):,} records is **{val:,.2f}**."
                return {
                    "answer": answer_text,
                    "intent": "metric_aggregation",
                    "confidence": "high",
                    "data": {"metric": target_col, "operation": desc, "result": round(val, 2)},
                    "chart_spec": {
                        "chart_type": "histogram",
                        "x_col": target_col,
                        "title": f"Distribution of {target_col}",
                    },
                }

        # 3. Intent: Grouped Query or Chart Request ("Show sales by region", "Plot salary by department")
        by_match = re.search(r"(?:by|per|across|for\s+each)\s+([a-z0-9_\s]+)", q)
        if by_match or any(k in q for k in ["plot", "chart", "visualize", "graph"]):
            target_group = None
            if by_match:
                target_group = cls._find_best_col_match(by_match.group(1).strip(), cat_cols or all_cols)

            if not target_group and matched_cols:
                # Find first categorical
                cats = [c for c in matched_cols if c in cat_cols]
                if cats:
                    target_group = cats[0]

            target_val = None
            nums = [c for c in matched_cols if c in numeric_cols]
            if nums:
                target_val = nums[0]
            elif numeric_cols:
                target_val = numeric_cols[0]

            # Detect chart type
            chosen_chart = "column"
            for kw, c_type in cls.CHART_KEYWORDS.items():
                if kw in q and c_type:
                    chosen_chart = c_type
                    break

            if target_group and target_val:
                agg_type = detected_agg or "sum"
                grouped_df = DataAnalyzer.group_aggregate(
                    df, group_col=target_group, value_col=target_val, agg_func=agg_type, top_n=10
                )
                answer_text = (
                    f"Generated visualization for **{target_val}** by **{target_group}** "
                    f"({agg_type.capitalize()}). Showing top {len(grouped_df)} categories."
                )
                return {
                    "answer": answer_text,
                    "intent": "grouped_visualization",
                    "confidence": "high",
                    "data": grouped_df.to_dict(orient="records"),
                    "chart_spec": {
                        "chart_type": chosen_chart,
                        "x_col": target_group,
                        "y_col": target_val,
                        "agg_func": agg_type,
                        "title": f"{target_val} by {target_group} ({agg_type.capitalize()})",
                    },
                }

        # 4. Intent: "How many records / rows?" or Count check
        if any(term in q for term in ["how many rows", "how many records", "dataset size", "total rows", "row count"]):
            return {
                "answer": f"The dataset contains **{len(df):,} total records** across **{len(df.columns)} columns**.",
                "intent": "row_count",
                "confidence": "high",
                "data": {"rows": len(df), "columns": len(df.columns)},
                "chart_spec": None,
            }

        # 5. Intent: Correlation check ("Is there a correlation between x and y?")
        if "correlat" in q and len(numeric_cols) >= 2:
            num_matches = [c for c in matched_cols if c in numeric_cols]
            if len(num_matches) >= 2:
                c1, c2 = num_matches[0], num_matches[1]
                s1 = DataAnalyzer.safe_numeric_convert(df[c1])
                s2 = DataAnalyzer.safe_numeric_convert(df[c2])
                r = s1.corr(s2)
                strength = "strong" if abs(r) >= 0.7 else ("moderate" if abs(r) >= 0.4 else "weak")
                dir_str = "positive" if r > 0 else "negative"
                return {
                    "answer": f"The Pearson correlation between **{c1}** and **{c2}** is **{r:.3f}** (indicating a {strength} {dir_str} correlation).",
                    "intent": "correlation",
                    "confidence": "high",
                    "data": {"col1": c1, "col2": c2, "pearson_r": round(float(r), 3)},
                    "chart_spec": {
                        "chart_type": "scatter",
                        "x_col": c1,
                        "y_col": c2,
                        "title": f"{c2} vs {c1} (Correlation: {r:.2f})",
                    },
                }

        # Fallback if no specific rule matched
        return cls._fallback_response(df, q)

    @classmethod
    def _match_columns(cls, query: str, columns: List[str]) -> List[str]:
        """Finds all dataset column names referenced inside user query."""
        found = []
        for col in columns:
            col_norm = col.lower().replace("_", " ")
            if col_norm in query or col.lower() in query:
                found.append(col)
        return found

    @classmethod
    def _find_best_col_match(cls, token: str, candidate_cols: List[str]) -> Optional[str]:
        """Fuzzy-matches a string token to the closest column name."""
        t = token.lower().replace("_", " ").strip()
        for col in candidate_cols:
            c = col.lower().replace("_", " ").strip()
            if t == c or t in c or c in t:
                return col
        return None

    @classmethod
    def _fallback_response(cls, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Constructs an informative fallback guidance with dataset-specific suggested queries."""
        sample_cats = df.select_dtypes(include=["object", "category"]).columns.tolist()[:2]
        sample_nums = df.select_dtypes(include=[np.number]).columns.tolist()[:2]

        suggestions = []
        if sample_nums:
            suggestions.append(f"What is the average {sample_nums[0]}?")
            suggestions.append(f"What is the maximum {sample_nums[0]}?")
        if sample_cats and sample_nums:
            suggestions.append(f"Which {sample_cats[0]} has the highest average {sample_nums[0]}?")
            suggestions.append(f"Show {sample_nums[0]} by {sample_cats[0]} as a bar chart")
        suggestions.append("How many records are in this dataset?")

        return {
            "answer": (
                "I couldn't identify specific calculations for that phrasing. "
                "SmartSight AI uses secure, verified statistical operations. "
                "Try asking one of the recommended queries below:"
            ),
            "intent": "unrecognized",
            "confidence": "low",
            "data": None,
            "chart_spec": None,
            "suggested_queries": suggestions,
        }
