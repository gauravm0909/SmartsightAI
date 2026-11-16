"""
SmartSight AI - Desktop Dashboard Component
Reusable dashboard widget encapsulating KPIs and dataset telemetry.
"""

from typing import Optional, Dict, Any
import customtkinter as ctk
import pandas as pd


class DashboardView(ctk.CTkFrame):
    """Encapsulates the high-level KPI and dataset status view for the desktop application."""

    def __init__(self, master, df: Optional[pd.DataFrame] = None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.df = df
        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)

        header = ctk.CTkLabel(
            self,
            text="Executive Intelligence Dashboard",
            font=ctk.CTkFont(size=22, weight="bold"),
            anchor="w",
        )
        header.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 15))

        rows_val = f"{len(self.df):,}" if self.df is not None else "0"
        cols_val = str(len(self.df.columns)) if self.df is not None else "0"
        missing_val = f"{self.df.isna().sum().sum():,}" if self.df is not None else "0"
        dup_val = f"{self.df.duplicated().sum():,}" if self.df is not None else "0"

        cards = [
            ("Total Records", rows_val, "#3B82F6"),
            ("Total Features", cols_val, "#10B981"),
            ("Missing Values", missing_val, "#F59E0B"),
            ("Duplicate Rows", dup_val, "#EF4444"),
        ]

        for i, (title, val, color) in enumerate(cards):
            card = ctk.CTkFrame(self, corner_radius=10, fg_color="#1E293B")
            card.grid(row=1, column=i, padx=6, pady=6, sticky="ew")
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12), text_color="#94A3B8").pack(anchor="w", padx=15, pady=(12, 2))
            ctk.CTkLabel(card, text=val, font=ctk.CTkFont(size=24, weight="bold"), text_color=color).pack(anchor="w", padx=15, pady=(0, 12))

    def update_data(self, df: pd.DataFrame):
        self.df = df
        for widget in self.winfo_children():
            widget.destroy()
        self._build_ui()
