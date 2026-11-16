"""
SmartSight AI - Master Application Launcher
"Upload. Analyze. Visualize. Predict."

Provides unified access to:
1. SmartSight AI Desktop Suite (CustomTkinter + Matplotlib)
2. SmartSight AI Web Application (Streamlit + Plotly)
3. SmartSight AI Full-Stack Platform (FastAPI + React 18 + SQLite)
4. Legacy Corporate Data Analyzer (Original Builder preserved)
"""

import os
import sys
import argparse
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox


def launch_desktop():
    """Launches the CustomTkinter modern desktop analytics suite."""
    from desktop_app.main import SmartSightDesktopApp
    app = SmartSightDesktopApp()
    app.mainloop()


def launch_streamlit():
    """Starts the Streamlit web dashboard."""
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "streamlit_app", "app.py"))
    print(f"Launching SmartSight AI Streamlit Suite at {app_path}...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])


def launch_fullstack():
    """Starts the FastAPI backend and serves the full-stack web application."""
    print("Launching SmartSight AI FastAPI Server on http://localhost:8000 ...")
    subprocess.run([sys.executable, "-m", "uvicorn", "web_app.backend.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])


def launch_legacy():
    """Launches the original Corporate Data Analyzer."""
    from desktop_app.legacy_builder import CorporateReportBuilder
    root = tk.Tk()
    app = CorporateReportBuilder(root)
    root.mainloop()


class LauncherDialog:
    """Modern startup dialog for selecting target SmartSight AI version."""

    def __init__(self, root):
        self.root = root
        self.root.title("SmartSight AI - Edition Launcher")
        self.root.geometry("520x460")
        self.root.resizable(False, False)
        self.root.configure(bg="#0F172A")

        # Header
        header = tk.Label(
            self.root,
            text="SmartSight AI",
            font=("Arial", 22, "bold"),
            fg="#6366F1",
            bg="#0F172A",
        )
        header.pack(pady=(25, 2))

        tagline = tk.Label(
            self.root,
            text="Upload. Analyze. Visualize. Predict.",
            font=("Arial", 11, "italic"),
            fg="#94A3B8",
            bg="#0F172A",
        )
        tagline.pack(pady=(0, 20))

        desc = tk.Label(
            self.root,
            text="Select an application edition to launch:",
            font=("Arial", 11, "bold"),
            fg="#F8FAFC",
            bg="#0F172A",
        )
        desc.pack(pady=(0, 15))

        btn_frame = tk.Frame(self.root, bg="#0F172A")
        btn_frame.pack(fill="x", padx=40)

        # 1. Desktop Suite Button
        b1 = tk.Button(
            btn_frame,
            text="1. Modern Desktop Suite (CustomTkinter)",
            font=("Arial", 11, "bold"),
            bg="#2563EB",
            fg="white",
            activebackground="#1D4ED8",
            activeforeground="white",
            relief="flat",
            height=2,
            command=self._run_desktop,
        )
        b1.pack(fill="x", pady=6)

        # 2. Streamlit Web App Button
        b2 = tk.Button(
            btn_frame,
            text="2. Streamlit Web App (Plotly)",
            font=("Arial", 11, "bold"),
            bg="#059669",
            fg="white",
            activebackground="#047857",
            activeforeground="white",
            relief="flat",
            height=2,
            command=self._run_streamlit,
        )
        b2.pack(fill="x", pady=6)

        # 3. Full-Stack Web App Button
        b3 = tk.Button(
            btn_frame,
            text="3. Full-Stack Web App (FastAPI + React)",
            font=("Arial", 11, "bold"),
            bg="#7C3AED",
            fg="white",
            activebackground="#6D28D9",
            activeforeground="white",
            relief="flat",
            height=2,
            command=self._run_fullstack,
        )
        b3.pack(fill="x", pady=6)

        # 4. Legacy Corporate Analyzer Button
        b4 = tk.Button(
            btn_frame,
            text="4. Legacy Corporate Report Builder",
            font=("Arial", 10),
            bg="#334155",
            fg="#CBD5E1",
            activebackground="#1E293B",
            activeforeground="white",
            relief="flat",
            height=2,
            command=self._run_legacy,
        )
        b4.pack(fill="x", pady=6)

        footer = tk.Label(
            self.root,
            text="Tip: Use 'python main.py --mode desktop|streamlit|web' for direct CLI launching.",
            font=("Arial", 9),
            fg="#64748B",
            bg="#0F172A",
        )
        footer.pack(side="bottom", pady=15)

    def _run_desktop(self):
        self.root.destroy()
        launch_desktop()

    def _run_streamlit(self):
        self.root.destroy()
        launch_streamlit()

    def _run_fullstack(self):
        self.root.destroy()
        launch_fullstack()

    def _run_legacy(self):
        self.root.destroy()
        launch_legacy()


def main():
    parser = argparse.ArgumentParser(description="SmartSight AI - Unified Application Launcher")
    parser.add_argument(
        "--mode",
        choices=["desktop", "streamlit", "web", "legacy", "gui"],
        default="gui",
        help="Specify which SmartSight AI version to execute",
    )
    args = parser.parse_args()

    if args.mode == "desktop":
        launch_desktop()
    elif args.mode == "streamlit":
        launch_streamlit()
    elif args.mode == "web":
        launch_fullstack()
    elif args.mode == "legacy":
        launch_legacy()
    else:
        root = tk.Tk()
        LauncherDialog(root)
        root.mainloop()


if __name__ == "__main__":
    main()
