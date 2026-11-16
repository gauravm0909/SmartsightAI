"""
SmartSight AI - Desktop Application
Built with CustomTkinter, Matplotlib, and the SmartSight AI Shared Analytics Engine.
Tagline: "Upload. Analyze. Visualize. Predict."
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import customtkinter as ctk
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.data_loader import DataLoader
from core.data_cleaner import DataCleaner
from core.data_profiler import DataProfiler
from core.data_analyzer import DataAnalyzer
from core.visualization import Visualizer
from core.ai_insights import AIInsightsEngine
from core.natural_language_query import NaturalLanguageQueryEngine
from core.ml_models import MLPredictiveEngine
from core.report_generator import ReportGenerator


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SmartSightDesktopApp(ctk.CTk):
    """Production-grade desktop analytics and predictive intelligence suite."""

    def __init__(self):
        super().__init__()

        self.title("SmartSight AI - Enterprise Analytics & Predictive Intelligence")
        self.geometry("1280x820")
        self.minsize(1050, 700)

        # State
        self.current_filepath: Optional[str] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.active_df: Optional[pd.DataFrame] = None  # Working reference
        self.profile_data: Optional[Dict[str, Any]] = None
        self.cleaning_history: List[Dict[str, Any]] = []
        self.last_ml_results: Optional[Dict[str, Any]] = None
        self.last_chart_fig = None
        self.current_canvas = None

        self._build_layout()
        self._show_page("dashboard")

    # =========================================================================
    # UI SHELL & SIDEBAR NAVIGATION
    # =========================================================================

    def _build_layout(self):
        # Main grid: 2 columns (Sidebar, Content)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar Frame
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(11, weight=1)

        # Brand header
        brand_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="SmartSight AI",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        brand_label.grid(row=0, column=0, padx=20, pady=(20, 2))

        tagline_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Analyze. Visualize. Predict.",
            font=ctk.CTkFont(size=10, slant="italic"),
            text_color="#94A3B8",
        )
        tagline_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # Nav Buttons
        self.nav_buttons = {}
        pages = [
            ("dashboard", "🏠 Dashboard"),
            ("upload", "📁 Upload & Inspect"),
            ("profiling", "📊 Smart Profiling"),
            ("cleaning", "🧹 Data Cleaning"),
            ("visualization", "📈 Visual Studio"),
            ("insights", "💡 AI Insights"),
            ("ask_data", "💬 Ask Your Data"),
            ("predictive", "🤖 Predictive ML"),
            ("reports", "📑 Export Reports"),
        ]

        for idx, (key, label) in enumerate(pages, start=2):
            btn = ctk.CTkButton(
                self.sidebar_frame,
                text=label,
                anchor="w",
                font=ctk.CTkFont(size=12, weight="bold"),
                height=36,
                fg_color="transparent",
                hover_color="#1E293B",
                text_color="#E2E8F0",
                command=lambda k=key: self._show_page(k),
            )
            btn.grid(row=idx, column=0, padx=12, pady=3, sticky="ew")
            self.nav_buttons[key] = btn

        # Theme Selector at bottom of sidebar
        theme_lbl = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", font=ctk.CTkFont(size=11))
        theme_lbl.grid(row=12, column=0, padx=20, pady=(10, 0), sticky="w")
        self.theme_menu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["Dark", "Light"],
            command=self._change_appearance_mode,
            height=28,
        )
        self.theme_menu.set("Dark")
        self.theme_menu.grid(row=13, column=0, padx=20, pady=(4, 20), sticky="ew")

        # 2. Main Content Container
        self.content_container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.content_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.content_container.grid_rowconfigure(0, weight=1)
        self.content_container.grid_columnconfigure(0, weight=1)

        # Dictionary of instantiated pages
        self.pages: Dict[str, ctk.CTkFrame] = {}

    def _show_page(self, page_name: str):
        # Highlight active button
        for key, btn in self.nav_buttons.items():
            if key == page_name:
                btn.configure(fg_color="#3B82F6", text_color="#FFFFFF")
            else:
                btn.configure(fg_color="transparent", text_color="#E2E8F0")

        # Hide current pages
        for p in self.pages.values():
            p.grid_forget()

        # Build page on demand or refresh
        page_builder = getattr(self, f"_render_{page_name}_page", None)
        if page_builder:
            frame = page_builder()
            self.pages[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

    def _change_appearance_mode(self, mode: str):
        ctk.set_appearance_mode(mode)
        # Refresh current page
        for key, btn in self.nav_buttons.items():
            if btn.cget("fg_color") == "#3B82F6":
                self._show_page(key)
                break

    # =========================================================================
    # 1. MAIN DASHBOARD PAGE
    # =========================================================================

    def _render_dashboard_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Header Banner
        header = ctk.CTkLabel(
            page,
            text="Executive Intelligence Dashboard",
            font=ctk.CTkFont(size=22, weight="bold"),
            anchor="w",
        )
        header.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 15))

        df = self.active_df
        rows_val = f"{len(df):,}" if df is not None else "0"
        cols_val = str(len(df.columns)) if df is not None else "0"
        missing_val = f"{df.isna().sum().sum():,}" if df is not None else "0"
        dup_val = f"{df.duplicated().sum():,}" if df is not None else "0"

        # Stat Cards Row
        cards = [
            ("Total Records", rows_val, "#3B82F6"),
            ("Total Features", cols_val, "#10B981"),
            ("Missing Values", missing_val, "#F59E0B"),
            ("Duplicate Rows", dup_val, "#EF4444"),
        ]

        for i, (title, val, color) in enumerate(cards):
            card = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
            card.grid(row=1, column=i, padx=6, pady=6, sticky="ew")
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12), text_color="#94A3B8").pack(anchor="w", padx=15, pady=(12, 2))
            ctk.CTkLabel(card, text=val, font=ctk.CTkFont(size=24, weight="bold"), text_color=color).pack(anchor="w", padx=15, pady=(0, 12))

        # Dataset Status / Overview
        status_box = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        status_box.grid(row=2, column=0, columnspan=4, padx=6, pady=15, sticky="ew")

        current_file_text = f"Active Dataset: {os.path.basename(self.current_filepath)}" if self.current_filepath else "No dataset loaded currently."
        ctk.CTkLabel(status_box, text=current_file_text, font=ctk.CTkFont(size=14, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=15, pady=(12, 4))

        if df is not None:
            num_cols = len(df.select_dtypes(include=[np.number]).columns)
            cat_cols = len(df.select_dtypes(include=["object", "string", "category"]).columns)
            details = f"Numerical Columns: {num_cols}  |  Categorical Columns: {cat_cols}  |  Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB"
        else:
            details = "Click 'Upload & Inspect' to load a CSV or Excel dataset, or load a sample dataset below."
        ctk.CTkLabel(status_box, text=details, font=ctk.CTkFont(size=12), text_color="#CBD5E1").pack(anchor="w", padx=15, pady=(0, 12))

        # Quick Actions Row
        action_frame = ctk.CTkFrame(page, fg_color="transparent")
        action_frame.grid(row=3, column=0, columnspan=4, pady=10, sticky="ew")

        ctk.CTkButton(
            action_frame,
            text="Load Employee Sample",
            command=lambda: self._load_sample_file("datasets/corporate_employees.csv"),
            height=36,
            fg_color="#4F46E5",
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            action_frame,
            text="Load Retail Sales Sample",
            command=lambda: self._load_sample_file("datasets/retail_sales.csv"),
            height=36,
            fg_color="#059669",
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            action_frame,
            text="Load Customer Churn Sample",
            command=lambda: self._load_sample_file("datasets/customer_churn.csv"),
            height=36,
            fg_color="#D97706",
        ).pack(side="left", padx=6)

        return page

    # =========================================================================
    # 2. UPLOAD & INSPECT PAGE
    # =========================================================================

    def _render_upload_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)

        # Header & Upload Bar
        top_bar = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 12), padx=4)

        ctk.CTkButton(
            top_bar,
            text="Browse CSV / Excel File",
            command=self._browse_and_load,
            height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2563EB",
        ).pack(side="left", padx=15, pady=12)

        file_label_text = os.path.basename(self.current_filepath) if self.current_filepath else "No file selected"
        self.file_status_lbl = ctk.CTkLabel(top_bar, text=file_label_text, font=ctk.CTkFont(size=13), text_color="#38BDF8")
        self.file_status_lbl.pack(side="left", padx=10)

        # Meta Info Box
        self.meta_box = ctk.CTkLabel(
            page,
            text="Dataset columns and schema information will appear here once loaded.",
            font=ctk.CTkFont(size=12),
            anchor="w",
            text_color="#94A3B8",
        )
        self.meta_box.grid(row=1, column=0, sticky="w", padx=6, pady=4)

        if self.active_df is not None:
            insp = DataLoader.inspect_dataset(self.active_df)
            meta_str = (
                f"Records: {insp['rows']:,} | Columns: {insp['columns']} | "
                f"Numeric: {len(insp['numeric_columns'])} | Categorical: {len(insp['categorical_columns'])} | "
                f"Memory: {insp['memory_usage']}"
            )
            self.meta_box.configure(text=meta_str, text_color="#E2E8F0")

        # Table Preview
        table_container = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        table_container.grid(row=2, column=0, sticky="nsew", padx=4, pady=8)

        if self.active_df is not None:
            self._render_treeview(table_container, self.active_df.head(50))
        else:
            ctk.CTkLabel(
                table_container,
                text="Please select or upload a dataset to preview records.",
                font=ctk.CTkFont(size=14),
                text_color="#64748B",
            ).pack(expand=True)

        return page

    # =========================================================================
    # 3. SMART DATA PROFILING PAGE
    # =========================================================================

    def _render_profiling_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        if self.active_df is None:
            ctk.CTkLabel(page, text="Load a dataset first to view smart data profiling.", font=ctk.CTkFont(size=14)).pack(expand=True)
            return page

        if self.profile_data is None:
            self.profile_data = DataProfiler.generate_profile(self.active_df)

        # Top summary
        ov = self.profile_data["overview"]
        summary_text = (
            f"Data Quality Score: {ov['data_quality_score']}/100  |  "
            f"Total Records: {ov['rows']:,}  |  Total Columns: {ov['columns']}  |  "
            f"Missing Values: {ov['missing_values']:,} ({ov['missing_pct']}%)  |  "
            f"Duplicates: {ov['duplicate_rows']:,}"
        )
        top_bar = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        top_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(top_bar, text=summary_text, font=ctk.CTkFont(size=13, weight="bold"), text_color="#10B981").pack(padx=15, pady=10)

        # Profiling Notebook / Tabs (Numeric vs Categorical)
        tabview = ctk.CTkTabview(page)
        tabview.grid(row=1, column=0, sticky="nsew")

        tab_num = tabview.add("Numerical Dimensions")
        tab_cat = tabview.add("Categorical Dimensions")
        tab_corr = tabview.add("Correlation Matrix")

        # Numeric stats table
        if self.profile_data["numeric_stats"]:
            num_rows = []
            for col, st in self.profile_data["numeric_stats"].items():
                num_rows.append({
                    "Feature": col,
                    "Mean": f"{st['mean']:,.2f}",
                    "Median": f"{st['median']:,.2f}",
                    "Min": f"{st['min']:,.2f}",
                    "Max": f"{st['max']:,.2f}",
                    "Std Dev": f"{st['std']:,.2f}",
                    "Outliers": f"{st['outlier_count']} ({st['outlier_pct']}%)",
                    "Missing": f"{st['missing']} ({st['missing_pct']}%)",
                })
            self._render_treeview(tab_num, pd.DataFrame(num_rows))

        # Categorical stats table
        if self.profile_data["categorical_stats"]:
            cat_rows = []
            for col, st in self.profile_data["categorical_stats"].items():
                top = st["top_values"][0] if st["top_values"] else {"value": "N/A", "pct": 0}
                cat_rows.append({
                    "Feature": col,
                    "Unique Values": st["unique_values"],
                    "Mode": st["mode"],
                    "Top Value": top["value"],
                    "Top Value %": f"{top['pct']}%",
                    "Missing": f"{st['missing']} ({st['missing_pct']}%)",
                })
            self._render_treeview(tab_cat, pd.DataFrame(cat_rows))

        # Correlation table
        corr_m = self.profile_data["correlation"].get("matrix", {})
        if corr_m:
            corr_df = pd.DataFrame(corr_m)
            self._render_treeview(tab_corr, corr_df.reset_index())

        return page

    # =========================================================================
    # 4. DATA CLEANING STUDIO
    # =========================================================================

    def _render_cleaning_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.grid_columnconfigure((0, 1), weight=1)
        page.grid_rowconfigure(2, weight=1)

        if self.raw_df is None:
            ctk.CTkLabel(page, text="Please upload a dataset first to configure cleaning.", font=ctk.CTkFont(size=14)).pack(expand=True)
            return page

        # Options Controls
        ctrl = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10), padx=4)

        self.clean_dup_var = ctk.BooleanVar(value=True)
        self.clean_num_var = ctk.StringVar(value="median")
        self.clean_cat_var = ctk.StringVar(value="mode")
        self.clean_strip_var = ctk.BooleanVar(value=True)

        ctk.CTkCheckBox(ctrl, text="Remove Duplicate Rows", variable=self.clean_dup_var).pack(side="left", padx=12, pady=10)
        ctk.CTkCheckBox(ctrl, text="Trim Whitespace", variable=self.clean_strip_var).pack(side="left", padx=12, pady=10)

        ctk.CTkLabel(ctrl, text="Missing Numeric:").pack(side="left", padx=(15, 4))
        ctk.CTkOptionMenu(ctrl, values=["median", "mean", "zero", "drop", "none"], variable=self.clean_num_var, width=110).pack(side="left", padx=4)

        ctk.CTkLabel(ctrl, text="Missing Categorical:").pack(side="left", padx=(15, 4))
        ctk.CTkOptionMenu(ctrl, values=["mode", "constant", "drop", "none"], variable=self.clean_cat_var, width=110).pack(side="left", padx=4)

        # Action buttons
        btn_bar = ctk.CTkFrame(page, fg_color="transparent")
        btn_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ctk.CTkButton(btn_bar, text="Apply Cleaning Pipeline", command=self._apply_cleaning, fg_color="#10B981", height=36).pack(side="left", padx=6)
        ctk.CTkButton(btn_bar, text="Reset to Original", command=self._reset_cleaning, fg_color="#64748B", height=36).pack(side="left", padx=6)
        ctk.CTkButton(btn_bar, text="Export Cleaned CSV", command=self._export_cleaned_csv, fg_color="#3B82F6", height=36).pack(side="right", padx=6)

        # Before and After Split View
        left_box = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        left_box.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)
        ctk.CTkLabel(left_box, text=f"Raw Original ({len(self.raw_df):,} rows)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#94A3B8").pack(anchor="w", padx=10, pady=6)
        self._render_treeview(left_box, self.raw_df.head(25))

        right_box = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        right_box.grid(row=2, column=1, sticky="nsew", padx=4, pady=4)
        clean_rows = len(self.cleaned_df) if self.cleaned_df is not None else 0
        ctk.CTkLabel(right_box, text=f"Cleaned Preview ({clean_rows:,} rows)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#10B981").pack(anchor="w", padx=10, pady=6)
        if self.cleaned_df is not None:
            self._render_treeview(right_box, self.cleaned_df.head(25))
        else:
            ctk.CTkLabel(right_box, text="Click 'Apply Cleaning Pipeline' to generate cleaned dataset.", text_color="#64748B").pack(expand=True)

        return page

    def _apply_cleaning(self):
        opts = {
            "remove_duplicates": self.clean_dup_var.get(),
            "handle_missing_num": self.clean_num_var.get(),
            "handle_missing_cat": self.clean_cat_var.get(),
            "strip_whitespace": self.clean_strip_var.get(),
            "convert_numeric_strings": True,
            "convert_dates": True,
        }
        self.cleaned_df, summary = DataCleaner.clean_dataset(self.raw_df, opts)
        self.active_df = self.cleaned_df
        self.profile_data = DataProfiler.generate_profile(self.active_df)
        self.cleaning_history.append(summary)
        messagebox.showinfo(
            "Cleaning Complete",
            f"Dataset successfully cleaned!\n\n"
            f"• Duplicates Removed: {summary['duplicates_removed']}\n"
            f"• Numerical Imputations: {summary['missing_filled_numerical']}\n"
            f"• Categorical Imputations: {summary['missing_filled_categorical']}\n"
            f"• Remaining Missing: {summary['final_missing']}",
        )
        self._show_page("cleaning")

    def _reset_cleaning(self):
        self.cleaned_df = None
        self.active_df = self.raw_df
        self.profile_data = DataProfiler.generate_profile(self.active_df) if self.active_df is not None else None
        self._show_page("cleaning")

    def _export_cleaned_csv(self):
        if self.active_df is None:
            messagebox.showerror("Error", "No dataset to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Cleaned Dataset As",
        )
        if path:
            ReportGenerator.export_csv(self.active_df, path)
            messagebox.showinfo("Export Successful", f"Saved cleaned dataset to:\n{path}")

    # =========================================================================
    # 5. VISUALIZATION STUDIO (Upgraded Corporate Report Builder)
    # =========================================================================

    def _render_visualization_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        if self.active_df is None:
            ctk.CTkLabel(page, text="Please upload or load a dataset first.", font=ctk.CTkFont(size=14)).pack(expand=True)
            return page

        df = self.active_df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        # Control Bar
        bar = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 10), padx=4)

        # Chart type
        ctk.CTkLabel(bar, text="Chart:").pack(side="left", padx=(10, 4), pady=10)
        self.chart_type_cb = ctk.CTkOptionMenu(
            bar, values=Visualizer.CHART_TYPES, width=130
        )
        self.chart_type_cb.set("Column")
        self.chart_type_cb.pack(side="left", padx=4)

        # X-Axis
        ctk.CTkLabel(bar, text="X / Category:").pack(side="left", padx=(10, 4))
        self.chart_x_cb = ctk.CTkOptionMenu(
            bar, values=cat_cols or list(df.columns), width=140
        )
        if cat_cols:
            self.chart_x_cb.set(cat_cols[0])
        self.chart_x_cb.pack(side="left", padx=4)

        # Y-Axis
        ctk.CTkLabel(bar, text="Y / Metric:").pack(side="left", padx=(10, 4))
        self.chart_y_cb = ctk.CTkOptionMenu(
            bar, values=num_cols or list(df.columns), width=140
        )
        if num_cols:
            self.chart_y_cb.set(num_cols[0])
        self.chart_y_cb.pack(side="left", padx=4)

        # Aggregation
        ctk.CTkLabel(bar, text="Agg:").pack(side="left", padx=(10, 4))
        self.chart_agg_cb = ctk.CTkOptionMenu(
            bar, values=["sum", "mean", "median", "max", "min", "count"], width=90
        )
        self.chart_agg_cb.set("sum")
        self.chart_agg_cb.pack(side="left", padx=4)

        # Buttons
        ctk.CTkButton(
            bar, text="Render Chart", command=self._render_current_chart, fg_color="#2563EB", width=110
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            bar, text="Export PNG", command=self._export_current_chart, fg_color="#059669", width=100
        ).pack(side="left", padx=4)

        # Canvas Container
        self.chart_display_frame = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        self.chart_display_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # Initial render
        self._render_current_chart()

        return page

    def _render_current_chart(self):
        if self.active_df is None or not hasattr(self, "chart_display_frame"):
            return

        for w in self.chart_display_frame.winfo_children():
            w.destroy()

        c_type = self.chart_type_cb.get()
        x_col = self.chart_x_cb.get()
        y_col = self.chart_y_cb.get()
        agg = self.chart_agg_cb.get()

        is_dark = ctk.get_appearance_mode() == "Dark"

        try:
            fig = Visualizer.create_matplotlib_figure(
                self.active_df,
                chart_type=c_type,
                x_col=x_col,
                y_col=y_col,
                agg_func=agg,
                dark_mode=is_dark,
                figsize=(8.5, 5.0),
                dpi=100,
            )
            self.last_chart_fig = fig

            canvas = FigureCanvasTkAgg(fig, master=self.chart_display_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
            self.current_canvas = canvas
        except Exception as e:
            ctk.CTkLabel(
                self.chart_display_frame,
                text=f"Chart generation error:\n{str(e)}",
                font=ctk.CTkFont(size=12),
                text_color="#EF4444",
            ).pack(expand=True)

    def _export_current_chart(self):
        if self.last_chart_fig is None:
            messagebox.showerror("Error", "No chart to export. Render a chart first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png")],
            title="Save Chart Graphic As",
        )
        if path:
            ReportGenerator.export_chart_png(self.last_chart_fig, path)
            messagebox.showinfo("Export Successful", f"Saved chart graphic to:\n{path}")

    # =========================================================================
    # 6. AI INSIGHTS PAGE
    # =========================================================================

    def _render_insights_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)

        if self.active_df is None:
            ctk.CTkLabel(page, text="Please upload a dataset to uncover statistical AI insights.", font=ctk.CTkFont(size=14)).pack(expand=True)
            return page

        # Header bar
        header_bar = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        header_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10), padx=4)

        ctk.CTkLabel(
            header_bar,
            text="Autonomous Statistical Findings & Strategic Insights",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#38BDF8",
        ).pack(side="left", padx=15, pady=10)

        # Scrollable container for cards
        scroll_frame = ctk.CTkScrollableFrame(page, corner_radius=10, fg_color="#0F172A")
        scroll_frame.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        res = AIInsightsEngine.generate_insights(self.active_df)
        insights = res["insights"]

        if not insights:
            ctk.CTkLabel(scroll_frame, text="No anomalous statistical outliers detected in current sample.", font=ctk.CTkFont(size=13)).pack(pady=20)
            return page

        for ins in insights:
            card = ctk.CTkFrame(scroll_frame, corner_radius=8, fg_color="#1E293B")
            card.pack(fill="x", padx=10, pady=6)

            sev_color = "#10B981" if ins["severity"] == "success" else ("#F59E0B" if ins["severity"] == "warning" else "#3B82F6")

            # Title row
            title_box = ctk.CTkFrame(card, fg_color="transparent")
            title_box.pack(fill="x", padx=12, pady=(10, 2))

            ctk.CTkLabel(title_box, text=f"[{ins['category'].upper()}]", font=ctk.CTkFont(size=11, weight="bold"), text_color=sev_color).pack(side="left")
            ctk.CTkLabel(title_box, text=f" {ins['title']}", font=ctk.CTkFont(size=13, weight="bold"), text_color="#F8FAFC").pack(side="left")

            # Description
            ctk.CTkLabel(card, text=ins["description"], font=ctk.CTkFont(size=11), text_color="#CBD5E1", justify="left", wraplength=780).pack(anchor="w", padx=12, pady=(2, 6))

            # Recommendation
            rec_frame = ctk.CTkFrame(card, corner_radius=4, fg_color="#0F172A")
            rec_frame.pack(fill="x", padx=12, pady=(0, 10))
            ctk.CTkLabel(rec_frame, text=f"💡 Action: {ins['recommended_action']}", font=ctk.CTkFont(size=10, slant="italic"), text_color="#94A3B8", justify="left", wraplength=760).pack(anchor="w", padx=8, pady=4)

        return page

    # =========================================================================
    # 7. ASK YOUR DATA (AI CHATBOT)
    # =========================================================================

    def _render_ask_data_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(0, weight=1)

        if self.active_df is None:
            ctk.CTkLabel(page, text="Please upload a dataset to enable Ask Your Data Q&A.", font=ctk.CTkFont(size=14)).pack(expand=True)
            return page

        # Chat history log
        self.chat_history_frame = ctk.CTkScrollableFrame(page, corner_radius=10, fg_color="#1E293B")
        self.chat_history_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=(0, 10))

        # Initial assistant greeting
        ctk.CTkLabel(
            self.chat_history_frame,
            text="👋 Hello! I am SmartSight AI. Ask me anything about your dataset (e.g. 'What is the average salary?', 'Show sales by category', 'Which department has the highest bonus?').",
            font=ctk.CTkFont(size=12),
            text_color="#38BDF8",
            wraplength=750,
            justify="left",
        ).pack(anchor="w", padx=12, pady=8)

        # Input box row
        input_bar = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        input_bar.grid(row=1, column=0, sticky="ew", padx=4)

        self.query_entry = ctk.CTkEntry(
            input_bar,
            placeholder_text="Ask a question about this dataset...",
            height=38,
            font=ctk.CTkFont(size=13),
        )
        self.query_entry.pack(side="left", fill="x", expand=True, padx=(12, 6), pady=8)
        self.query_entry.bind("<Return>", lambda e: self._submit_query())

        ctk.CTkButton(
            input_bar,
            text="Send",
            command=self._submit_query,
            width=80,
            height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2563EB",
        ).pack(side="right", padx=(0, 12), pady=8)

        return page

    def _submit_query(self):
        text = self.query_entry.get().strip()
        if not text or self.active_df is None:
            return

        self.query_entry.delete(0, "end")

        # User bubble
        u_box = ctk.CTkFrame(self.chat_history_frame, corner_radius=6, fg_color="#2563EB")
        u_box.pack(anchor="e", padx=12, pady=4)
        ctk.CTkLabel(u_box, text=text, font=ctk.CTkFont(size=12, weight="bold"), text_color="white").pack(padx=10, pady=6)

        # Process through NLP engine
        res = NaturalLanguageQueryEngine.process_query(self.active_df, text)

        # Bot bubble
        b_box = ctk.CTkFrame(self.chat_history_frame, corner_radius=6, fg_color="#0F172A")
        b_box.pack(anchor="w", padx=12, pady=4)

        ans_clean = res["answer"].replace("**", "")
        ctk.CTkLabel(b_box, text=ans_clean, font=ctk.CTkFont(size=12), text_color="#E2E8F0", wraplength=700, justify="left").pack(padx=10, pady=6)

        # If query produced a chart spec, render a mini inline chart!
        spec = res.get("chart_spec")
        if spec:
            try:
                is_dark = ctk.get_appearance_mode() == "Dark"
                fig = Visualizer.create_matplotlib_figure(
                    self.active_df,
                    chart_type=spec.get("chart_type", "column"),
                    x_col=spec.get("x_col"),
                    y_col=spec.get("y_col"),
                    agg_func=spec.get("agg_func", "sum"),
                    title=spec.get("title"),
                    dark_mode=is_dark,
                    figsize=(6.5, 3.2),
                    dpi=85,
                )
                chart_holder = ctk.CTkFrame(b_box, fg_color="transparent")
                chart_holder.pack(padx=10, pady=6)
                c = FigureCanvasTkAgg(fig, master=chart_holder)
                c.draw()
                c.get_tk_widget().pack()
            except Exception:
                pass

    # =========================================================================
    # 8. PREDICTIVE ANALYTICS (MACHINE LEARNING)
    # =========================================================================

    def _render_predictive_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.grid_columnconfigure((0, 1), weight=1)
        page.grid_rowconfigure(2, weight=1)

        if self.active_df is None:
            ctk.CTkLabel(page, text="Please upload a dataset to train ML predictive models.", font=ctk.CTkFont(size=14)).pack(expand=True)
            return page

        df = self.active_df
        all_cols = list(df.columns)

        # Configuration Bar
        cfg = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        cfg.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10), padx=4)

        # Target Column
        ctk.CTkLabel(cfg, text="Target (Y):").pack(side="left", padx=(10, 4), pady=10)
        self.ml_target_cb = ctk.CTkOptionMenu(
            cfg, values=all_cols, command=self._on_ml_target_changed, width=150
        )
        self.ml_target_cb.set(all_cols[-1])
        self.ml_target_cb.pack(side="left", padx=4)

        # Algorithm
        ctk.CTkLabel(cfg, text="Algorithm:").pack(side="left", padx=(12, 4))
        all_algos = (
            MLPredictiveEngine.ALGORITHMS["regression"]
            + MLPredictiveEngine.ALGORITHMS["classification"]
            + MLPredictiveEngine.ALGORITHMS["clustering"]
        )
        self.ml_algo_cb = ctk.CTkOptionMenu(cfg, values=all_algos, width=190)
        self.ml_algo_cb.set("Random Forest Regressor")
        self.ml_algo_cb.pack(side="left", padx=4)

        # Train button
        ctk.CTkButton(
            cfg, text="Train & Evaluate", command=self._train_ml_model, fg_color="#10B981", height=34
        ).pack(side="left", padx=12)

        # Results area
        self.ml_results_frame = ctk.CTkScrollableFrame(page, corner_radius=10, fg_color="#1E293B")
        self.ml_results_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)

        if self.last_ml_results:
            self._render_ml_results_content(self.last_ml_results)
        else:
            ctk.CTkLabel(
                self.ml_results_frame,
                text="Select target and features above, then click 'Train & Evaluate'.",
                font=ctk.CTkFont(size=13),
                text_color="#64748B",
            ).pack(pady=40)

        return page

    def _on_ml_target_changed(self, target: str):
        if self.active_df is None:
            return
        task = MLPredictiveEngine.identify_task_type(self.active_df, target)
        algos = MLPredictiveEngine.ALGORITHMS.get(task, [])
        if algos:
            self.ml_algo_cb.configure(values=algos)
            self.ml_algo_cb.set(algos[0])

    def _train_ml_model(self):
        if self.active_df is None:
            return
        target = self.ml_target_cb.get()
        algo = self.ml_algo_cb.get()

        # Input features: all other columns
        features = [c for c in self.active_df.columns if c != target]

        try:
            res = MLPredictiveEngine.train_and_evaluate(
                self.active_df,
                feature_cols=features,
                target_col=target,
                algorithm=algo,
            )
            self.last_ml_results = res
            self._show_page("predictive")
        except Exception as e:
            messagebox.showerror("ML Error", f"Model training failed:\n{str(e)}")

    def _render_ml_results_content(self, res: Dict[str, Any]):
        for w in self.ml_results_frame.winfo_children():
            w.destroy()

        # Overview banner
        top = ctk.CTkFrame(self.ml_results_frame, corner_radius=6, fg_color="#0F172A")
        top.pack(fill="x", padx=10, pady=6)

        ctk.CTkLabel(
            top,
            text=f"Model: {res['algorithm']} ({res['task_type'].capitalize()}) | Target: '{res.get('target_column')}'",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#38BDF8",
        ).pack(anchor="w", padx=10, pady=(6, 2))

        ctk.CTkLabel(
            top,
            text=f"Training split: {res.get('train_samples', 0):,} records  |  Testing split: {res.get('test_samples', 0):,} records",
            font=ctk.CTkFont(size=11),
            text_color="#94A3B8",
        ).pack(anchor="w", padx=10, pady=(0, 6))

        # Metrics row
        m_row = ctk.CTkFrame(self.ml_results_frame, fg_color="transparent")
        m_row.pack(fill="x", padx=10, pady=6)

        for k, v in res["metrics"].items():
            card = ctk.CTkFrame(m_row, corner_radius=6, fg_color="#0F172A", width=120)
            card.pack(side="left", padx=4)
            ctk.CTkLabel(card, text=k.upper(), font=ctk.CTkFont(size=10), text_color="#94A3B8").pack(padx=8, pady=(4, 0))
            ctk.CTkLabel(card, text=str(v), font=ctk.CTkFont(size=16, weight="bold"), text_color="#10B981").pack(padx=8, pady=(0, 4))

        # Explanation
        if "explanation" in res:
            e_box = ctk.CTkFrame(self.ml_results_frame, corner_radius=6, fg_color="#0F172A")
            e_box.pack(fill="x", padx=10, pady=6)
            ctk.CTkLabel(e_box, text=f"📊 Diagnostic: {res['explanation']}", font=ctk.CTkFont(size=11), text_color="#E2E8F0", wraplength=740, justify="left").pack(anchor="w", padx=10, pady=6)

        # Feature Importance
        imp = res.get("feature_importance", [])
        if imp:
            imp_box = ctk.CTkFrame(self.ml_results_frame, corner_radius=6, fg_color="#0F172A")
            imp_box.pack(fill="x", padx=10, pady=6)
            ctk.CTkLabel(imp_box, text="Top Feature Importances:", font=ctk.CTkFont(size=11, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=10, pady=(6, 4))

            for item in imp[:6]:
                f_name = item["feature"]
                pct = item["importance"] * 100
                ctk.CTkLabel(imp_box, text=f"• {f_name}: {pct:.1f}%", font=ctk.CTkFont(size=10), text_color="#CBD5E1").pack(anchor="w", padx=14, pady=1)

    # =========================================================================
    # 9. REPORT GENERATION PAGE
    # =========================================================================

    def _render_reports_page(self) -> ctk.CTkFrame:
        page = ctk.CTkFrame(self.content_container, fg_color="transparent")
        page.grid_columnconfigure((0, 1), weight=1)

        if self.active_df is None:
            ctk.CTkLabel(page, text="Please upload a dataset first to compile reports.", font=ctk.CTkFont(size=14)).pack(expand=True)
            return page

        # Export Hub Card
        box = ctk.CTkFrame(page, corner_radius=10, fg_color="#1E293B")
        box.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        ctk.CTkLabel(box, text="SmartSight AI Export Center", font=ctk.CTkFont(size=18, weight="bold"), text_color="#F8FAFC").pack(anchor="w", padx=20, pady=(15, 2))
        ctk.CTkLabel(box, text="Export professional analytics artifacts in industry-standard formats.", font=ctk.CTkFont(size=12), text_color="#94A3B8").pack(anchor="w", padx=20, pady=(0, 15))

        btn_row = ctk.CTkFrame(box, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkButton(
            btn_row, text="📄 Generate PDF Executive Report", command=self._export_pdf_report, height=40, fg_color="#DC2626"
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            btn_row, text="📊 Export Multi-Sheet Excel (.xlsx)", command=self._export_excel_report, height=40, fg_color="#16A34A"
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            btn_row, text="💾 Export Cleaned CSV (.csv)", command=self._export_cleaned_csv, height=40, fg_color="#2563EB"
        ).pack(side="left", padx=8)

        return page

    def _export_pdf_report(self):
        if self.active_df is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Document", "*.pdf")],
            title="Save PDF Executive Report",
        )
        if path:
            profile = self.profile_data or DataProfiler.generate_profile(self.active_df)
            insights = AIInsightsEngine.generate_insights(self.active_df)["insights"]
            ReportGenerator.export_pdf(
                self.active_df,
                profile=profile,
                cleaning_summary=self.cleaning_history[-1] if self.cleaning_history else None,
                insights=insights,
                ml_results=self.last_ml_results,
                chart_fig=self.last_chart_fig,
                output_path=path,
            )
            messagebox.showinfo("Export Complete", f"PDF Executive Report generated:\n{path}")

    def _export_excel_report(self):
        if self.active_df is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Workbook", "*.xlsx")],
            title="Save Multi-Sheet Excel Workbook",
        )
        if path:
            profile = self.profile_data or DataProfiler.generate_profile(self.active_df)
            insights = AIInsightsEngine.generate_insights(self.active_df)["insights"]
            ReportGenerator.export_excel(
                self.active_df,
                profile=profile,
                cleaning_summary=self.cleaning_history[-1] if self.cleaning_history else None,
                insights=insights,
                ml_results=self.last_ml_results,
                output_path=path,
            )
            messagebox.showinfo("Export Complete", f"Multi-Sheet Excel Report saved:\n{path}")

    # =========================================================================
    # HELPERS & FILE ACTIONS
    # =========================================================================

    def _browse_and_load(self):
        path = filedialog.askopenfilename(
            title="Select CSV or Excel Dataset",
            filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx *.xls")],
        )
        if path:
            self._load_file(path)

    def _load_sample_file(self, relative_path: str):
        full_path = os.path.abspath(relative_path)
        if os.path.exists(full_path):
            self._load_file(full_path)
        else:
            messagebox.showerror("File Error", f"Sample file not found: {full_path}")

    def _load_file(self, file_path: str):
        try:
            df = DataLoader.load_dataset(file_path)
            self.current_filepath = file_path
            self.raw_df = df
            self.cleaned_df = None
            self.active_df = df
            self.profile_data = DataProfiler.generate_profile(df)
            self.last_ml_results = None
            self.cleaning_history = []

            # Refresh view
            self._show_page("upload")
            messagebox.showinfo("Success", f"Loaded {os.path.basename(file_path)} ({len(df):,} rows, {len(df.columns)} columns).")
        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load dataset:\n{str(e)}")

    def _render_treeview(self, parent: ctk.CTkFrame, df: pd.DataFrame):
        """Creates a modern scrollable Treeview table."""
        for w in parent.winfo_children():
            if isinstance(w, (ttk.Treeview, ttk.Scrollbar, tk.Frame)):
                w.destroy()

        frame = tk.Frame(parent, bg="#1E293B")
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        cols = list(df.columns)
        tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)

        # Style Treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#0F172A",
            foreground="#F8FAFC",
            fieldbackground="#0F172A",
            rowheight=24,
            font=("Inter", 9),
        )
        style.configure(
            "Treeview.Heading",
            background="#334155",
            foreground="#F8FAFC",
            relief="flat",
            font=("Inter", 9, "bold"),
        )
        style.map("Treeview", background=[("selected", "#3B82F6")])

        for col in cols:
            tree.heading(col, text=str(col))
            tree.column(col, width=max(100, int(700 / max(1, len(cols)))), anchor="w")

        for _, r in df.iterrows():
            vals = [str(r[c]) if pd.notna(r[c]) else "" for c in cols]
            tree.insert("", "end", values=vals)

        # Scrollbars
        y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)


if __name__ == "__main__":
    app = SmartSightDesktopApp()
    app.mainloop()
