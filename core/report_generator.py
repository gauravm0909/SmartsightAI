"""
SmartSight AI - Report Generator Module
Generates boardroom-ready Excel spreadsheets, CSV data exports, PNG chart graphics, and executive PDF reports.
"""

import os
import io
import datetime
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.units import inch

from matplotlib.figure import Figure


class ReportGenerator:
    """Multi-format report compilation and export service."""

    # =========================================================================
    # EXCEL EXPORT (Multi-Sheet Workbook)
    # =========================================================================

    @classmethod
    def export_excel(
        cls,
        df: pd.DataFrame,
        profile: Optional[Dict[str, Any]] = None,
        cleaning_summary: Optional[Dict[str, Any]] = None,
        insights: Optional[List[Dict[str, Any]]] = None,
        ml_results: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> Union[str, bytes]:
        """Compiles a professional multi-sheet Excel workbook."""
        buffer = io.BytesIO() if output_path is None else output_path

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Sheet 1: Cleaned Data Preview (first 1000 rows for performance)
            df.head(1000).to_excel(writer, sheet_name="Data Preview", index=False)

            # Sheet 2: Executive Overview
            if profile and "overview" in profile:
                ov = profile["overview"]
                overview_df = pd.DataFrame([
                    {"Metric": "Data Quality Score", "Value": f"{ov.get('data_quality_score', 'N/A')}/100"},
                    {"Metric": "Total Records", "Value": ov.get("rows", 0)},
                    {"Metric": "Total Columns", "Value": ov.get("columns", 0)},
                    {"Metric": "Missing Cells", "Value": f"{ov.get('missing_values', 0)} ({ov.get('missing_pct', 0)}%)"},
                    {"Metric": "Duplicate Records", "Value": f"{ov.get('duplicate_rows', 0)} ({ov.get('duplicate_pct', 0)}%)"},
                    {"Metric": "Memory Footprint", "Value": ov.get("memory_footprint", "N/A")},
                ])
                overview_df.to_excel(writer, sheet_name="Dataset Overview", index=False)

            # Sheet 3: Numerical Statistics
            if profile and "numeric_stats" in profile and profile["numeric_stats"]:
                num_df = pd.DataFrame.from_dict(profile["numeric_stats"], orient="index")
                num_df.index.name = "Column"
                num_df.reset_index().to_excel(writer, sheet_name="Numeric Profiling", index=False)

            # Sheet 4: Categorical Statistics
            if profile and "categorical_stats" in profile and profile["categorical_stats"]:
                cat_rows = []
                for col, data in profile["categorical_stats"].items():
                    top = data.get("top_values", [{}])[0]
                    cat_rows.append({
                        "Column": col,
                        "Unique Values": data.get("unique_values"),
                        "Most Frequent": data.get("mode"),
                        "Top Value": top.get("value", "N/A"),
                        "Top Value Frequency": top.get("count", 0),
                        "Top Value Share (%)": top.get("pct", 0),
                    })
                pd.DataFrame(cat_rows).to_excel(writer, sheet_name="Categorical Profiling", index=False)

            # Sheet 5: AI Insights
            if insights:
                ins_df = pd.DataFrame([
                    {
                        "Category": i.get("category"),
                        "Finding": i.get("title"),
                        "Details": i.get("description"),
                        "Recommendation": i.get("recommended_action"),
                    }
                    for i in insights
                ])
                ins_df.to_excel(writer, sheet_name="AI Insights", index=False)

            # Sheet 6: Machine Learning
            if ml_results and "metrics" in ml_results:
                ml_info = [
                    {"Key": "Task Type", "Value": ml_results.get("task_type")},
                    {"Key": "Algorithm", "Value": ml_results.get("algorithm")},
                    {"Key": "Target Column", "Value": ml_results.get("target_column")},
                    {"Key": "Explanation", "Value": ml_results.get("explanation")},
                ]
                for m_k, m_v in ml_results["metrics"].items():
                    ml_info.append({"Key": f"Metric: {m_k.upper()}", "Value": m_v})
                pd.DataFrame(ml_info).to_excel(writer, sheet_name="Predictive ML", index=False)

        if output_path is not None:
            return output_path
        buffer.seek(0)
        return buffer.getvalue()

    # =========================================================================
    # CSV EXPORT
    # =========================================================================

    @classmethod
    def export_csv(cls, df: pd.DataFrame, output_path: Optional[str] = None) -> Union[str, bytes]:
        """Exports DataFrame to UTF-8 CSV."""
        if output_path is not None:
            df.to_csv(output_path, index=False)
            return output_path
        return df.to_csv(index=False).encode("utf-8")

    # =========================================================================
    # CHART PNG EXPORT
    # =========================================================================

    @classmethod
    def export_chart_png(cls, fig: Figure, output_path: Optional[str] = None, dpi: int = 200) -> Union[str, bytes]:
        """Saves a Matplotlib Figure as a high-resolution PNG image."""
        if output_path is not None:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            return output_path
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()

    # =========================================================================
    # PDF EXECUTIVE REPORT (ReportLab)
    # =========================================================================

    @classmethod
    def export_pdf(
        cls,
        df: pd.DataFrame,
        profile: Optional[Dict[str, Any]] = None,
        cleaning_summary: Optional[Dict[str, Any]] = None,
        insights: Optional[List[Dict[str, Any]]] = None,
        ml_results: Optional[Dict[str, Any]] = None,
        chart_fig: Optional[Figure] = None,
        output_path: Optional[str] = None,
    ) -> Union[str, bytes]:
        """Generates an executive-level PDF report containing statistics, tables, insights, and charts."""
        buffer = io.BytesIO() if output_path is None else output_path
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=40,
            rightMargin=40,
            topMargin=40,
            bottomMargin=40,
        )

        styles = getSampleStyleSheet()
        # Custom Typography
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=4,
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#4F46E5"),
            spaceAfter=14,
        )
        h2_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "DocBody",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#334155"),
        )
        bullet_style = ParagraphStyle(
            "DocBullet",
            parent=styles["Normal"],
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#1E293B"),
            leftIndent=12,
            spaceAfter=3,
        )

        story: List[Any] = []

        # 1. Header Banner
        now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M")
        story.append(Paragraph("SmartSight AI", title_style))
        story.append(Paragraph("Enterprise Data Analytics, Intelligence & Predictive Report", subtitle_style))
        story.append(Paragraph(f"Generated: {now_str} | Platform: SmartSight AI Engine v1.0", body_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4F46E5"), spaceBefore=6, spaceAfter=12))

        # 2. Executive Overview Table
        if profile and "overview" in profile:
            ov = profile["overview"]
            story.append(Paragraph("1. Dataset Executive Summary", h2_style))
            overview_data = [
                ["Total Rows", f"{ov.get('rows', 0):,}", "Data Quality Score", f"{ov.get('data_quality_score', 'N/A')}/100"],
                ["Total Columns", str(ov.get("columns", 0)), "Memory Usage", str(ov.get("memory_footprint", "N/A"))],
                ["Missing Values", f"{ov.get('missing_values', 0):,} ({ov.get('missing_pct', 0)}%)", "Duplicate Rows", f"{ov.get('duplicate_rows', 0):,} ({ov.get('duplicate_pct', 0)}%)"],
                ["Numeric Features", str(ov.get("numeric_columns_count", 0)), "Categorical Features", str(ov.get("categorical_columns_count", 0))],
            ]
            t_ov = Table(overview_data, colWidths=[120, 140, 130, 140])
            t_ov.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1E293B")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t_ov)
            story.append(Spacer(1, 10))

        # 3. Data Cleaning Audit (if available)
        if cleaning_summary:
            story.append(Paragraph("2. Data Cleaning & Sanitization Audit", h2_style))
            clean_text = (
                f"Initial Records: {cleaning_summary.get('initial_rows', 0):,} → Cleaned Records: {cleaning_summary.get('final_rows', 0):,} "
                f"| Duplicates Purged: {cleaning_summary.get('duplicates_removed', 0):,} "
                f"| Numerical Imputations: {cleaning_summary.get('missing_filled_numerical', 0):,} "
                f"| Categorical Imputations: {cleaning_summary.get('missing_filled_categorical', 0):,}"
            )
            story.append(Paragraph(clean_text, body_style))
            story.append(Spacer(1, 8))

        # 4. Numerical Features Table
        if profile and "numeric_stats" in profile and profile["numeric_stats"]:
            story.append(Paragraph("3. Statistical Profiling (Numerical Features)", h2_style))
            num_rows = [["Feature", "Mean", "Median", "Min", "Max", "Std Dev", "Outliers"]]
            for col, s in list(profile["numeric_stats"].items())[:8]:
                num_rows.append([
                    col[:16],
                    f"{s['mean']:,.1f}",
                    f"{s['median']:,.1f}",
                    f"{s['min']:,.1f}",
                    f"{s['max']:,.1f}",
                    f"{s['std']:,.1f}",
                    f"{s['outlier_count']} ({s['outlier_pct']}%)",
                ])
            t_num = Table(num_rows, colWidths=[110, 70, 70, 70, 70, 70, 70])
            t_num.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4F46E5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t_num)
            story.append(Spacer(1, 10))

        # 5. Visualizations (Embedded Chart)
        if chart_fig is not None:
            story.append(Paragraph("4. Visual Intelligence", h2_style))
            img_buf = io.BytesIO()
            chart_fig.savefig(img_buf, format="png", dpi=180, bbox_inches="tight")
            img_buf.seek(0)
            story.append(Image(img_buf, width=6.8 * inch, height=3.5 * inch))
            story.append(Spacer(1, 10))

        # 6. AI-Powered Insights
        if insights:
            story.append(Paragraph("5. AI-Powered Statistical Discoveries", h2_style))
            for ins in insights[:6]:
                story.append(Paragraph(f"• <b>[{ins.get('category')}] {ins.get('title')}</b>", bullet_style))
                story.append(Paragraph(f"  {ins.get('description')}", body_style))
                story.append(Spacer(1, 3))
            story.append(Spacer(1, 8))

        # 7. Predictive Analytics (ML Results)
        if ml_results and "metrics" in ml_results:
            story.append(Paragraph("6. Machine Learning Predictive Intelligence", h2_style))
            ml_p1 = f"<b>Model:</b> {ml_results.get('algorithm')} ({ml_results.get('task_type', '').capitalize()}) | <b>Target:</b> {ml_results.get('target_column')}"
            story.append(Paragraph(ml_p1, body_style))

            metric_str = " | ".join(f"<b>{k.upper()}:</b> {v}" for k, v in ml_results["metrics"].items())
            story.append(Paragraph(metric_str, body_style))
            if "explanation" in ml_results:
                story.append(Paragraph(f"<b>Interpretation:</b> {ml_results['explanation']}", body_style))

        # Build PDF document
        doc.build(story)

        if output_path is not None:
            return output_path
        buffer.seek(0)
        return buffer.getvalue()
