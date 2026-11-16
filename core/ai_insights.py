"""
SmartSight AI - AI Insights Engine
Generates deterministic statistical insights and provides optional LLM-augmented strategic summaries.
"""

import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import requests

from core.data_analyzer import DataAnalyzer


class AIInsightsEngine:
    """Calculates data-backed statistical findings and optional LLM-enhanced strategic narratives."""

    @classmethod
    def generate_insights(
        cls,
        df: pd.DataFrame,
        use_llm: bool = False,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates automated insights based on mathematical facts, correlations, and distributions.
        Works 100% offline without external APIs.
        """
        insights: List[Dict[str, Any]] = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

        # 1. Group-Wise Dominance & Extremes (e.g. Highest/Lowest Average by Category)
        for cat in cat_cols[:3]:  # Check prominent categorical features
            unique_cnt = df[cat].nunique()
            if 2 <= unique_cnt <= 30:
                for num in numeric_cols[:3]:  # Check prominent numeric metrics
                    try:
                        grouped = (
                            df.groupby(cat)[num]
                            .agg(["mean", "sum", "count"])
                            .reset_index()
                            .dropna()
                        )
                        if len(grouped) >= 2:
                            # Highest average
                            top_mean = grouped.sort_values(by="mean", ascending=False).iloc[0]
                            low_mean = grouped.sort_values(by="mean", ascending=True).iloc[0]

                            if top_mean["mean"] > low_mean["mean"] * 1.15:  # At least 15% difference
                                insights.append({
                                    "category": "Performance",
                                    "type": "highest_average",
                                    "title": f"Highest Average {num}: {top_mean[cat]}",
                                    "description": (
                                        f"'{top_mean[cat]}' records the highest average {num} at {top_mean['mean']:,.2f}, "
                                        f"which is {((top_mean['mean'] / low_mean['mean']) - 1) * 100:.1f}% higher than "
                                        f"the lowest group '{low_mean[cat]}' ({low_mean['mean']:,.2f})."
                                    ),
                                    "severity": "info",
                                    "metric_involved": [cat, num],
                                    "recommended_action": f"Benchmark operational drivers in '{top_mean[cat]}' to replicate success in lower-performing segments.",
                                })

                            # Revenue/Total contribution dominance
                            total_sum = grouped["sum"].sum()
                            if total_sum > 0:
                                top_sum = grouped.sort_values(by="sum", ascending=False).iloc[0]
                                contrib_pct = (top_sum["sum"] / total_sum) * 100
                                if contrib_pct >= 35.0:
                                    insights.append({
                                        "category": "Contribution",
                                        "type": "dominant_contributor",
                                        "title": f"Concentrated Volume in {top_sum[cat]}",
                                        "description": (
                                            f"'{top_sum[cat]}' contributes {contrib_pct:.1f}% of total {num} "
                                            f"({top_sum['sum']:,.2f} of {total_sum:,.2f}), representing a significant concentration."
                                        ),
                                        "severity": "warning" if contrib_pct > 60 else "info",
                                        "metric_involved": [cat, num],
                                        "recommended_action": f"Diversify dependency or safeguard stability for '{top_sum[cat]}' given high concentration.",
                                    })
                    except Exception:
                        continue

        # 2. Pearson Correlation Insights
        if len(numeric_cols) >= 2:
            corr_df = df[numeric_cols].corr(method="pearson")
            paired = set()
            for c1 in numeric_cols:
                for c2 in numeric_cols:
                    if c1 != c2 and (c2, c1) not in paired:
                        paired.add((c1, c2))
                        r = corr_df.loc[c1, c2]
                        if not np.isnan(r):
                            if abs(r) >= 0.65:
                                direction = "positive" if r > 0 else "inverse (negative)"
                                relationship = "strong direct" if r > 0 else "strong inverse"
                                insights.append({
                                    "category": "Correlation",
                                    "type": "strong_correlation",
                                    "title": f"Strong {direction.capitalize()} Correlation: {c1} & {c2}",
                                    "description": (
                                        f"'{c1}' and '{c2}' exhibit a {relationship} statistical correlation (Pearson r = {r:.3f}). "
                                        f"{'As ' + c1 + ' rises, ' + c2 + ' predictably rises.' if r > 0 else 'An increase in ' + c1 + ' correlates with a drop in ' + c2 + '.'}"
                                    ),
                                    "severity": "success" if r > 0 else "info",
                                    "metric_involved": [c1, c2],
                                    "recommended_action": f"Incorporate both variables when modeling predictive forecasts to capture synergistic variance.",
                                })

        # 3. Distribution & Outlier Insights
        for num in numeric_cols:
            s = df[num].dropna()
            if len(s) > 10:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                outlier_rate = (outliers / len(s)) * 100

                if outlier_rate >= 4.0:
                    insights.append({
                        "category": "Anomaly",
                        "type": "outlier_alert",
                        "title": f"High Outlier Frequency in '{num}'",
                        "description": (
                            f"{outliers} anomalous data points ({outlier_rate:.1f}% of records) were detected in '{num}' "
                            f"outside normal statistical boundaries [{q1 - 1.5*iqr:,.1f}, {q3 + 1.5*iqr:,.1f}]."
                        ),
                        "severity": "warning",
                        "metric_involved": [num],
                        "recommended_action": f"Review whether extreme values in '{num}' represent legitimate high-value exceptions or data entry anomalies.",
                    })

                skew = s.skew()
                if abs(skew) >= 1.5:
                    skew_type = "right-skewed (concentrated at lower values with long positive tail)" if skew > 0 else "left-skewed (concentrated at higher values)"
                    insights.append({
                        "category": "Distribution",
                        "type": "skewness_alert",
                        "title": f"Highly Skewed Distribution: '{num}'",
                        "description": (
                            f"'{num}' is significantly {skew_type} with skewness coefficient of {skew:.2f}. "
                            f"Median ({s.median():,.2f}) provides a more robust central tendency measure than mean ({s.mean():,.2f})."
                        ),
                        "severity": "info",
                        "metric_involved": [num],
                        "recommended_action": f"Utilize median-based aggregation and log-transforms when preparing '{num}' for linear machine learning.",
                    })

        # 4. Data Quality / Completeness Warning
        missing_count = int(df.isna().sum().sum())
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = (missing_count / total_cells * 100) if total_cells > 0 else 0
        if missing_pct > 3.0:
            insights.append({
                "category": "Data Quality",
                "type": "missing_data_alert",
                "title": f"Missing Value Warning ({missing_pct:.1f}% of total cells)",
                "description": f"Dataset contains {missing_count} null or missing entries. Cleaning is advised before feeding data into ML models.",
                "severity": "warning",
                "metric_involved": [],
                "recommended_action": "Apply SmartSight AI Data Cleaner to impute numerical fields with median and categoricals with mode.",
            })

        # Optional LLM Enhancement
        llm_summary = None
        key = api_key or os.getenv("GEMINI_API_KEY")
        if use_llm and key:
            llm_summary = cls._generate_llm_summary(df, insights, key)

        return {
            "insights_count": len(insights),
            "insights": insights,
            "llm_summary": llm_summary,
            "offline_mode": not bool(use_llm and key),
        }

    @classmethod
    def _generate_llm_summary(
        cls,
        df: pd.DataFrame,
        insights: List[Dict[str, Any]],
        api_key: str
    ) -> Optional[str]:
        """Calls Google Gemini API to synthesize statistical findings into an executive brief."""
        try:
            # Prepare summary context
            context_text = f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n"
            context_text += "Key Mathematical Facts Discovered:\n"
            for idx, ins in enumerate(insights[:8], 1):
                context_text += f"{idx}. [{ins['category']}] {ins['title']}: {ins['description']}\n"

            prompt = (
                "You are SmartSight AI's Principal Data Strategist. "
                "Synthesize the following computed statistical findings into a concise 3-paragraph executive briefing:\n"
                "1. Executive Overview: Core narrative of the data.\n"
                "2. Strategic Discoveries: The most impactful relationships and anomalies.\n"
                "3. Prescriptive Next Steps: 2-3 concrete business recommendations.\n"
                "IMPORTANT: Rely exclusively on the provided statistical calculations. Do not invent numbers.\n\n"
                f"{context_text}"
            )

            # Standard Gemini REST endpoint
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600}
            }
            res = requests.post(url, json=payload, timeout=12)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass
        return None
