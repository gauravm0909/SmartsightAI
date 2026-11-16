"""
SmartSight AI - Stat Card Widget
"""

import customtkinter as ctk


class StatCard(ctk.CTkFrame):
    """Modern SaaS stat metric card with label, value, and accent color."""

    def __init__(self, master, title: str, value: str, accent_color: str = "#3B82F6", **kwargs):
        super().__init__(master, corner_radius=10, fg_color="#1E293B", **kwargs)
        self.accent_color = accent_color

        self.title_lbl = ctk.CTkLabel(
            self, text=title, font=ctk.CTkFont(size=12), text_color="#94A3B8"
        )
        self.title_lbl.pack(anchor="w", padx=15, pady=(12, 2))

        self.val_lbl = ctk.CTkLabel(
            self, text=value, font=ctk.CTkFont(size=24, weight="bold"), text_color=accent_color
        )
        self.val_lbl.pack(anchor="w", padx=15, pady=(0, 12))

    def set_value(self, new_val: str):
        self.val_lbl.configure(text=new_val)
