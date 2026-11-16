"""
SmartSight AI - Visualization Module
Provides unified charting powered by Plotly (Web/Streamlit) and Matplotlib (Desktop/PDF).
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from core.data_analyzer import DataAnalyzer


class Visualizer:
    """Unified engine generating Plotly interactive figures and Matplotlib canvas figures."""

    COLOR_PALETTES = {
        "Corporate Indigo": ["#4F46E5", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"],
        "Emerald Tech": ["#059669", "#10B981", "#34D399", "#6EE7B7", "#047857", "#065F46"],
        "Sunset Glow": ["#F43F5E", "#FB923C", "#FBBF24", "#38BDF8", "#818CF8", "#C084FC"],
        "Modern Dark": ["#6366F1", "#A855F7", "#EC4899", "#14B8A6", "#F97316", "#06B6D4"],
        "Monochrome Slate": ["#0F172A", "#334155", "#475569", "#64748B", "#94A3B8", "#CBD5E1"],
    }

    CHART_TYPES = [
        "Column",
        "Bar",
        "Line",
        "Area",
        "Pie",
        "Donut",
        "Scatter",
        "Histogram",
        "Box Plot",
        "Violin Plot",
        "Correlation Heatmap",
    ]

    # =========================================================================
    # PLOTLY INTERACTIVE CHARTS (For Streamlit & React Web Frontend)
    # =========================================================================

    @classmethod
    def create_plotly_figure(
        cls,
        df: pd.DataFrame,
        chart_type: str,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        color_col: Optional[str] = None,
        agg_func: str = "sum",
        palette_name: str = "Corporate Indigo",
        title: Optional[str] = None,
        theme: str = "plotly_white",
        top_n: Optional[int] = None,
    ) -> go.Figure:
        """Constructs an interactive Plotly Figure with responsive layout."""
        ctype = chart_type.lower().replace(" ", "").replace("_", "").replace("-", "")
        colors = cls.COLOR_PALETTES.get(palette_name, cls.COLOR_PALETTES["Corporate Indigo"])

        chart_title = title or f"{chart_type} Chart"
        if x_col and y_col:
            chart_title = title or f"{y_col} by {x_col} ({agg_func.capitalize()})"

        # 1. Correlation Matrix Heatmap
        if ctype in ["correlationheatmap", "heatmap", "correlationmatrix"]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                fig = go.Figure()
                fig.add_annotation(text="At least 2 numeric columns required for correlation matrix.", showarrow=False)
                return fig
            corr = df[numeric_cols].corr().round(2)
            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title=title or "Feature Correlation Heatmap",
                template=theme
            )
            fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
            return fig

        # 2. Aggregation processing for categorized charts
        plot_df = df.copy()
        if ctype in ["column", "bar", "pie", "donut", "area", "line"] and x_col and y_col:
            if agg_func.lower() != "none":
                plot_df = DataAnalyzer.group_aggregate(
                    df, group_col=x_col, value_col=y_col, agg_func=agg_func, top_n=top_n
                )
                metric_col = plot_df.columns[1]
                y_target = metric_col
            else:
                y_target = y_col
        else:
            y_target = y_col

        # 3. Chart builder
        if ctype == "column":
            fig = px.bar(
                plot_df,
                x=x_col,
                y=y_target,
                color=color_col or x_col,
                color_discrete_sequence=colors,
                title=chart_title,
                template=theme,
            )
            fig.update_layout(xaxis_tickangle=-30)

        elif ctype == "bar":
            fig = px.bar(
                plot_df,
                x=y_target,
                y=x_col,
                orientation="h",
                color=color_col or x_col,
                color_discrete_sequence=colors,
                title=chart_title,
                template=theme,
            )

        elif ctype in ["pie", "donut"]:
            hole = 0.45 if ctype == "donut" else 0.0
            val_col = y_target if y_col else None
            if val_col is None:
                # Count frequency
                counts = plot_df[x_col].value_counts().head(top_n or 10).reset_index()
                counts.columns = [x_col, "Count"]
                fig = px.pie(
                    counts,
                    names=x_col,
                    values="Count",
                    hole=hole,
                    color_discrete_sequence=colors,
                    title=chart_title,
                    template=theme,
                )
            else:
                fig = px.pie(
                    plot_df,
                    names=x_col,
                    values=val_col,
                    hole=hole,
                    color_discrete_sequence=colors,
                    title=chart_title,
                    template=theme,
                )

        elif ctype == "line":
            fig = px.line(
                plot_df,
                x=x_col,
                y=y_target,
                color=color_col,
                markers=True,
                color_discrete_sequence=colors,
                title=chart_title,
                template=theme,
            )

        elif ctype == "area":
            fig = px.area(
                plot_df,
                x=x_col,
                y=y_target,
                color=color_col,
                color_discrete_sequence=colors,
                title=chart_title,
                template=theme,
            )

        elif ctype == "scatter":
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col,
                size=None,
                trendline=None,
                color_discrete_sequence=colors,
                title=chart_title,
                template=theme,
            )

        elif ctype in ["histogram", "hist"]:
            fig = px.histogram(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col,
                color_discrete_sequence=colors,
                title=chart_title,
                template=theme,
            )

        elif ctype in ["boxplot", "box"]:
            fig = px.box(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col or x_col,
                color_discrete_sequence=colors,
                title=chart_title,
                template=theme,
            )

        elif ctype in ["violinplot", "violin"]:
            fig = px.violin(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col or x_col,
                box=True,
                points="all",
                color_discrete_sequence=colors,
                title=chart_title,
                template=theme,
            )

        else:
            fig = px.bar(plot_df, x=x_col, y=y_target, title=chart_title, template=theme)

        fig.update_layout(
            font=dict(family="Inter, Roboto, sans-serif"),
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="closest",
        )
        return fig

    # =========================================================================
    # MATPLOTLIB STATIC CHARTS (For CustomTkinter Desktop App & PDF Reports)
    # =========================================================================

    @classmethod
    def create_matplotlib_figure(
        cls,
        df: pd.DataFrame,
        chart_type: str,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        agg_func: str = "sum",
        palette_name: str = "Corporate Indigo",
        title: Optional[str] = None,
        figsize: Tuple[float, float] = (7.5, 4.8),
        dpi: int = 100,
        dark_mode: bool = False,
        top_n: Optional[int] = 10,
    ) -> Figure:
        """Creates a standalone Matplotlib Figure suitable for Tkinter or PDF rendering."""
        ctype = chart_type.lower().replace(" ", "").replace("_", "").replace("-", "")
        colors = cls.COLOR_PALETTES.get(palette_name, cls.COLOR_PALETTES["Corporate Indigo"])

        fig = Figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)

        # Apply dark/light styles
        if dark_mode:
            fig.patch.set_facecolor("#1E293B")
            ax.set_facecolor("#0F172A")
            ax.tick_params(colors="#94A3B8")
            ax.xaxis.label.set_color("#CBD5E1")
            ax.yaxis.label.set_color("#CBD5E1")
            ax.title.set_color("#F8FAFC")
            spine_color = "#334155"
        else:
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#F8FAFC")
            ax.tick_params(colors="#475569")
            ax.xaxis.label.set_color("#1E293B")
            ax.yaxis.label.set_color("#1E293B")
            ax.title.set_color("#0F172A")
            spine_color = "#E2E8F0"

        for s in ax.spines.values():
            s.set_color(spine_color)
        ax.grid(True, linestyle="--", alpha=0.3, color=spine_color)

        chart_title = title or f"{chart_type} Chart"

        if ctype in ["correlationheatmap", "heatmap", "correlationmatrix"]:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) >= 2:
                corr = df[num_cols].corr().values
                im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(num_cols)))
                ax.set_yticks(range(len(num_cols)))
                ax.set_xticklabels(num_cols, rotation=45, ha="right")
                ax.set_yticklabels(num_cols)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                for i in range(len(num_cols)):
                    for j in range(len(num_cols)):
                        val = corr[i, j]
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.5 else "black", fontsize=8)
                ax.set_title(chart_title, fontsize=12, fontweight="bold")
                fig.tight_layout()
                return fig

        if x_col and y_col and ctype in ["column", "bar", "pie", "donut", "line", "area"]:
            agg_df = DataAnalyzer.group_aggregate(df, group_col=x_col, value_col=y_col, agg_func=agg_func, top_n=top_n)
            labels = agg_df.iloc[:, 0].astype(str).tolist()
            values = agg_df.iloc[:, 1].astype(float).tolist()
            val_name = agg_df.columns[1]

            if ctype == "column":
                bars = ax.bar(labels, values, color=colors[0], edgecolor="none", alpha=0.9)
                ax.set_ylabel(val_name)
                ax.tick_params(axis="x", rotation=35)
                # Value labels
                for bar in bars:
                    yval = bar.get_height()
                    ax.annotate(f"{yval:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, yval),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8,
                                color="#CBD5E1" if dark_mode else "#334155")

            elif ctype == "bar":
                ax.barh(labels[::-1], values[::-1], color=colors[0], edgecolor="none", alpha=0.9)
                ax.set_xlabel(val_name)

            elif ctype in ["pie", "donut"]:
                pie_labels = labels[:7]
                pie_vals = values[:7]
                wedgeprops = dict(width=0.45) if ctype == "donut" else dict()
                ax.pie(pie_vals, labels=pie_labels, autopct="%1.1f%%", colors=colors[:len(pie_vals)], wedgeprops=wedgeprops, startangle=140)
                ax.axis("equal")

            elif ctype == "line":
                ax.plot(labels, values, marker="o", color=colors[0], linewidth=2.5, markersize=6)
                ax.set_ylabel(val_name)
                ax.tick_params(axis="x", rotation=35)

            elif ctype == "area":
                ax.fill_between(range(len(values)), values, color=colors[0], alpha=0.4)
                ax.plot(range(len(values)), values, color=colors[0], linewidth=2)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=35)
                ax.set_ylabel(val_name)

        elif ctype == "scatter" and x_col and y_col:
            x_vals = DataAnalyzer.safe_numeric_convert(df[x_col])
            y_vals = DataAnalyzer.safe_numeric_convert(df[y_col])
            valid_mask = x_vals.notna() & y_vals.notna()
            ax.scatter(x_vals[valid_mask], y_vals[valid_mask], color=colors[0], alpha=0.7, edgecolors="none", s=40)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)

        elif ctype == "histogram" and x_col:
            vals = DataAnalyzer.safe_numeric_convert(df[x_col]).dropna()
            ax.hist(vals, bins=20, color=colors[0], edgecolor="white" if dark_mode else "none", alpha=0.85)
            ax.set_xlabel(x_col)
            ax.set_ylabel("Frequency")

        elif ctype == "boxplot" and y_col:
            vals = DataAnalyzer.safe_numeric_convert(df[y_col]).dropna()
            ax.boxplot(vals, patch_artist=True, boxprops=dict(facecolor=colors[0], color=colors[0]))
            ax.set_ylabel(y_col)

        ax.set_title(chart_title, fontsize=12, fontweight="bold", pad=12)
        fig.tight_layout()
        return fig
