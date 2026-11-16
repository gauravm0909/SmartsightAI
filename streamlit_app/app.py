"""
SmartSight AI - Streamlit Web Application
Intelligent Data Analytics, Visualization & Predictive Intelligence Platform
Tagline: "Upload. Analyze. Visualize. Predict."
"""

import os
import sys
import io
import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.data_loader import DataLoader
from core.data_cleaner import DataCleaner
from core.data_profiler import DataProfiler
from core.data_analyzer import DataAnalyzer
from core.visualization import Visualizer
from core.ai_insights import AIInsightsEngine
from core.natural_language_query import NaturalLanguageQueryEngine
from core.ml_models import MLPredictiveEngine
from core.report_generator import ReportGenerator

# Page Config
st.set_page_config(
    page_title="SmartSight AI | Intelligent Data Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Corporate CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .stMetric label {
        font-size: 0.85rem !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        color: #38BDF8 !important;
    }

    .card-insight {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(79, 70, 229, 0.25);
        border-left: 4px solid #4F46E5;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "active_df" not in st.session_state:
    st.session_state.active_df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "None"
if "cleaning_history" not in st.session_state:
    st.session_state.cleaning_history = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "last_ml_results" not in st.session_state:
    st.session_state.last_ml_results = None


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## ⚡ SmartSight AI")
    st.markdown("*Upload. Analyze. Visualize. Predict.*")
    st.divider()

    # Pre-loaded Sample Datasets
    sample_choice = st.selectbox(
        "Load Sample Dataset",
        ["-- Select Sample --", "Corporate Employees", "Retail Sales", "Customer Churn"]
    )
    if sample_choice != "-- Select Sample --":
        sample_map = {
            "Corporate Employees": "datasets/corporate_employees.csv",
            "Retail Sales": "datasets/retail_sales.csv",
            "Customer Churn": "datasets/customer_churn.csv",
        }
        path = sample_map[sample_choice]
        if os.path.exists(path):
            df_loaded = DataLoader.load_dataset(path)
            st.session_state.raw_df = df_loaded
            st.session_state.cleaned_df = None
            st.session_state.active_df = df_loaded
            st.session_state.dataset_name = sample_choice
            st.session_state.cleaning_history = []
            st.session_state.last_ml_results = None
            st.success(f"Loaded {sample_choice}!")

    # File Uploader
    uploaded_file = st.file_uploader(
        "Or Upload CSV / Excel",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV and Excel files up to 100MB"
    )
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.read()
            df_up = DataLoader.load_dataset(bytes_data, filename=uploaded_file.name)
            st.session_state.raw_df = df_up
            st.session_state.cleaned_df = None
            st.session_state.active_df = df_up
            st.session_state.dataset_name = uploaded_file.name
            st.session_state.cleaning_history = []
            st.session_state.last_ml_results = None
            st.success(f"Successfully loaded '{uploaded_file.name}'!")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    st.divider()

    # Navigation Menu
    nav_option = st.radio(
        "Navigation",
        [
            "🏠 Dashboard Overview",
            "📁 Upload & Inspection",
            "📊 Smart Profiling",
            "🧹 Data Cleaning Studio",
            "📈 Visualization Studio",
            "💡 AI Insights",
            "💬 Ask Your Data",
            "🤖 Predictive Analytics",
            "📑 Report Center",
        ],
        index=0
    )

    st.divider()
    st.markdown("#### ⚙️ Settings")
    api_key_input = st.text_input(
        "Gemini API Key (Optional)",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Optional: Enables strategic LLM briefings. Without this key, SmartSight uses its offline statistical rule engine."
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input


# Active DataFrame Reference
df = st.session_state.active_df


# =============================================================================
# 1. DASHBOARD OVERVIEW
# =============================================================================
if nav_option == "🏠 Dashboard Overview":
    st.title("Executive Intelligence Dashboard")
    st.markdown(f"**Active Dataset:** `{st.session_state.dataset_name}`")

    if df is None:
        st.info("👋 Welcome to **SmartSight AI**. Select a sample dataset from the sidebar or upload your own CSV/Excel file to start analyzing!")
    else:
        inspection = DataLoader.inspect_dataset(df)

        # 4 Main KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{inspection['rows']:,}")
        c2.metric("Features", f"{inspection['columns']}")
        c3.metric("Missing Values", f"{inspection['missing_values_total']:,}", f"{inspection['missing_cells_pct']}% cells")
        c4.metric("Duplicates", f"{inspection['duplicate_rows']:,}")

        # Secondary telemetry
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Numeric Features", len(inspection["numeric_columns"]))
        c6.metric("Categorical Features", len(inspection["categorical_columns"]))
        c7.metric("Memory Usage", inspection["memory_usage"])
        
        # Profile calculation for quality score
        profile = DataProfiler.generate_profile(df)
        c8.metric("Data Quality Score", f"{profile['overview']['data_quality_score']}/100")

        st.markdown("### Quick Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### Feature Distribution Quick View")
        num_cols = inspection["numeric_columns"]
        cat_cols = inspection["categorical_columns"]

        if cat_cols and num_cols:
            col_l, col_r = st.columns(2)
            with col_l:
                fig1 = Visualizer.create_plotly_figure(
                    df, chart_type="Column", x_col=cat_cols[0], y_col=num_cols[0], agg_func="mean",
                    title=f"Mean {num_cols[0]} by {cat_cols[0]}"
                )
                st.plotly_chart(fig1, use_container_width=True)
            with col_r:
                fig2 = Visualizer.create_plotly_figure(
                    df, chart_type="Donut", x_col=cat_cols[0], y_col=num_cols[0], agg_func="sum",
                    title=f"Total {num_cols[0]} Share by {cat_cols[0]}"
                )
                st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# 2. UPLOAD & INSPECTION
# =============================================================================
elif nav_option == "📁 Upload & Inspection":
    st.title("Data Inspection & Schema Detection")
    if df is None:
        st.warning("No dataset currently loaded. Please upload or load a sample from the sidebar.")
    else:
        insp = DataLoader.inspect_dataset(df)
        st.markdown(f"**Records:** {insp['rows']:,} | **Columns:** {insp['columns']} | **Memory:** {insp['memory_usage']}")

        st.markdown("### Full Dataset Explorer")
        st.dataframe(df, use_container_width=True, height=420)

        st.markdown("### Inferred Column Classifications")
        schema_df = pd.DataFrame([
            {
                "Column": col,
                "Inferred Type": insp["column_types"].get(col, "unknown"),
                "Pandas Dtype": str(df[col].dtype),
                "Missing Values": int(df[col].isna().sum()),
                "Unique Values": int(df[col].nunique()),
                "Sample Value": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "None",
            }
            for col in df.columns
        ])
        st.dataframe(schema_df, use_container_width=True)


# =============================================================================
# 3. SMART DATA PROFILING
# =============================================================================
elif nav_option == "📊 Smart Profiling":
    st.title("Smart Statistical Data Profiling")
    if df is None:
        st.warning("Please upload a dataset to generate profiling reports.")
    else:
        profile = DataProfiler.generate_profile(df)
        ov = profile["overview"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Data Quality Score", f"{ov['data_quality_score']}/100")
        col2.metric("Total Cells", f"{ov['total_cells']:,}")
        col3.metric("Missing Rate", f"{ov['missing_pct']}%")
        col4.metric("Duplicate Rate", f"{ov['duplicate_pct']}%")

        tab1, tab2, tab3 = st.tabs(["Numerical Features", "Categorical Features", "Correlation Heatmap"])

        with tab1:
            if profile["numeric_stats"]:
                num_df = pd.DataFrame.from_dict(profile["numeric_stats"], orient="index")
                st.dataframe(num_df, use_container_width=True)
            else:
                st.info("No numerical features detected.")

        with tab2:
            if profile["categorical_stats"]:
                cat_rows = []
                for col, st_data in profile["categorical_stats"].items():
                    top = st_data.get("top_values", [{}])[0]
                    cat_rows.append({
                        "Feature": col,
                        "Unique Count": st_data.get("unique_values"),
                        "Mode": st_data.get("mode"),
                        "Top Value": top.get("value", "N/A"),
                        "Top Value Share": f"{top.get('pct', 0)}%",
                        "Missing %": f"{st_data.get('missing_pct')}%",
                    })
                st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)
            else:
                st.info("No categorical features detected.")

        with tab3:
            fig_corr = Visualizer.create_plotly_figure(df, chart_type="Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)


# =============================================================================
# 4. DATA CLEANING STUDIO
# =============================================================================
elif nav_option == "🧹 Data Cleaning Studio":
    st.title("Data Sanitization & Cleaning Studio")
    if st.session_state.raw_df is None:
        st.warning("Upload or load a dataset first.")
    else:
        st.markdown("Configure non-destructive cleaning transformations with before/after audit previews.")

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            clean_dup = st.checkbox("Remove Duplicate Rows", value=True)
            clean_strip = st.checkbox("Trim Whitespace in Text", value=True)
        with col_opt2:
            clean_num = st.selectbox("Missing Numeric Strategy", ["median", "mean", "zero", "drop", "none"], index=0)
            clean_outliers = st.selectbox("Outlier Handling (IQR)", ["none", "clip", "remove"], index=0)
        with col_opt3:
            clean_cat = st.selectbox("Missing Categorical Strategy", ["mode", "constant", "drop", "none"], index=0)
            convert_types = st.checkbox("Convert Formatted Currency/Dates", value=True)

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("Apply Cleaning Pipeline", type="primary"):
                options = {
                    "remove_duplicates": clean_dup,
                    "handle_missing_num": clean_num,
                    "handle_missing_cat": clean_cat,
                    "strip_whitespace": clean_strip,
                    "convert_numeric_strings": convert_types,
                    "convert_dates": convert_types,
                    "handle_outliers": clean_outliers,
                }
                cleaned, summary = DataCleaner.clean_dataset(st.session_state.raw_df, options)
                st.session_state.cleaned_df = cleaned
                st.session_state.active_df = cleaned
                st.session_state.cleaning_history.append(summary)
                st.success("Cleaning completed successfully!")
                st.rerun()

        with col_btn2:
            if st.session_state.cleaned_df is not None:
                if st.button("Undo / Reset to Original"):
                    st.session_state.cleaned_df = None
                    st.session_state.active_df = st.session_state.raw_df
                    st.info("Reset dataset to raw original.")
                    st.rerun()

        # Side by side Before and After Preview
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(f"#### 📄 Original Dataset ({len(st.session_state.raw_df):,} rows)")
            st.dataframe(st.session_state.raw_df.head(15), use_container_width=True)

        with c_right:
            if st.session_state.cleaned_df is not None:
                st.markdown(f"#### ✨ Cleaned Dataset ({len(st.session_state.cleaned_df):,} rows)")
                st.dataframe(st.session_state.cleaned_df.head(15), use_container_width=True)
            else:
                st.info("Click 'Apply Cleaning Pipeline' above to generate cleaned preview.")

        if st.session_state.cleaning_history:
            st.markdown("### Cleaning Audit Log")
            latest_sum = st.session_state.cleaning_history[-1]
            for act in latest_sum.get("actions_taken", []):
                st.markdown(f"- {act}")


# =============================================================================
# 5. VISUALIZATION STUDIO
# =============================================================================
elif nav_option == "📈 Visualization Studio":
    st.title("Interactive Visualization Studio")
    if df is None:
        st.warning("Upload or load a dataset first.")
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        all_cols = list(df.columns)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            chart_type = st.selectbox("Chart Type", Visualizer.CHART_TYPES, index=0)
        with col2:
            x_col = st.selectbox("X-Axis / Category", cat_cols or all_cols, index=0)
        with col3:
            y_col = st.selectbox("Y-Axis / Metric", num_cols or all_cols, index=0)
        with col4:
            agg_func = st.selectbox("Aggregation", ["sum", "mean", "median", "max", "min", "count", "none"], index=0)
        with col5:
            palette = st.selectbox("Color Palette", list(Visualizer.COLOR_PALETTES.keys()), index=0)

        fig = Visualizer.create_plotly_figure(
            df,
            chart_type=chart_type,
            x_col=x_col,
            y_col=y_col,
            agg_func=agg_func,
            palette_name=palette,
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 6. AI INSIGHTS
# =============================================================================
elif nav_option == "💡 AI Insights":
    st.title("Autonomous AI Insights & Discoveries")
    if df is None:
        st.warning("Upload or load a dataset first.")
    else:
        st.markdown("Real-time mathematical findings and strategic anomalies automatically derived from actual statistical data.")
        use_gemini = st.checkbox("Generate Strategic Executive Narrative via Gemini API", value=bool(os.getenv("GEMINI_API_KEY")))

        with st.spinner("Analyzing statistical distributions and relationships..."):
            res = AIInsightsEngine.generate_insights(df, use_llm=use_gemini)

        if res.get("llm_summary"):
            st.markdown("### 🤖 Executive Briefing (AI-Synthesized)")
            st.info(res["llm_summary"])

        st.markdown(f"### Discovered Findings ({res['insights_count']})")
        for ins in res["insights"]:
            sev_border = "#10B981" if ins["severity"] == "success" else ("#F59E0B" if ins["severity"] == "warning" else "#3B82F6")
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid {sev_border}; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
                <div style="font-size: 0.8rem; font-weight: 700; color: {sev_border}; text-transform: uppercase;">{ins['category']}</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-top: 2px;">{ins['title']}</div>
                <div style="font-size: 0.9rem; color: #CBD5E1; margin-top: 6px;">{ins['description']}</div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 8px; font-style: italic;">💡 Recommendation: {ins['recommended_action']}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# 7. ASK YOUR DATA
# =============================================================================
elif nav_option == "💬 Ask Your Data":
    st.title("Ask Your Data – Conversational Intelligence")
    if df is None:
        st.warning("Upload or load a dataset first.")
    else:
        st.markdown("Query your dataset in natural language. Powered by allowlisted, secure statistical operations.")

        # Display history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "chart_spec" in msg and msg["chart_spec"]:
                    spec = msg["chart_spec"]
                    fig = Visualizer.create_plotly_figure(
                        df,
                        chart_type=spec.get("chart_type", "column"),
                        x_col=spec.get("x_col"),
                        y_col=spec.get("y_col"),
                        agg_func=spec.get("agg_func", "sum"),
                        title=spec.get("title"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        user_query = st.chat_input("Ask a question (e.g. 'What is the average salary?', 'Show sales by product')")
        if user_query:
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            res = NaturalLanguageQueryEngine.process_query(df, user_query)
            ans = res["answer"]
            spec = res.get("chart_spec")

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": ans,
                "chart_spec": spec,
            })

            with st.chat_message("assistant"):
                st.markdown(ans)
                if spec:
                    fig = Visualizer.create_plotly_figure(
                        df,
                        chart_type=spec.get("chart_type", "column"),
                        x_col=spec.get("x_col"),
                        y_col=spec.get("y_col"),
                        agg_func=spec.get("agg_func", "sum"),
                        title=spec.get("title"),
                    )
                    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# 8. PREDICTIVE ANALYTICS
# =============================================================================
elif nav_option == "🤖 Predictive Analytics":
    st.title("Predictive Intelligence & Machine Learning")
    if df is None:
        st.warning("Upload or load a dataset first.")
    else:
        all_cols = list(df.columns)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        c_t1, c_t2, c_t3 = st.columns(3)
        with c_t1:
            target_col = st.selectbox("Target Column (Y)", ["None (Clustering)"] + all_cols, index=1 if len(all_cols) > 1 else 0)
        
        target_param = None if target_col == "None (Clustering)" else target_col
        detected_task = MLPredictiveEngine.identify_task_type(df, target_param)

        with c_t2:
            st.markdown(f"**Identified Task:** `{detected_task.upper()}`")
            algos = MLPredictiveEngine.ALGORITHMS.get(detected_task, [])
            selected_algo = st.selectbox("Algorithm", algos, index=0)

        with c_t3:
            avail_features = [c for c in all_cols if c != target_param]
            selected_features = st.multiselect("Input Features (X)", avail_features, default=avail_features)

        if st.button("Train Predictive Model", type="primary"):
            try:
                with st.spinner("Training model with leak-free standardization..."):
                    res = MLPredictiveEngine.train_and_evaluate(
                        df,
                        feature_cols=selected_features,
                        target_col=target_param,
                        algorithm=selected_algo,
                    )
                st.session_state.last_ml_results = res
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"Training failed: {e}")

        # Display Results
        if st.session_state.last_ml_results:
            ml_res = st.session_state.last_ml_results
            st.divider()
            st.markdown(f"### Performance Report: `{ml_res['algorithm']}`")
            st.markdown(f"*{ml_res.get('explanation', '')}*")

            # Metrics row
            metric_cols = st.columns(len(ml_res["metrics"]))
            for idx, (m_key, m_val) in enumerate(ml_res["metrics"].items()):
                metric_cols[idx].metric(m_key.upper(), str(m_val))

            # Actual vs Predicted or Confusion Matrix
            if ml_res.get("actual_vs_predicted"):
                avp = ml_res["actual_vs_predicted"]
                comp_df = pd.DataFrame({"Actual": avp["actual"], "Predicted": avp["predicted"]})
                st.markdown("#### Actual vs Predicted Comparison")
                st.line_chart(comp_df)

            if ml_res.get("confusion_matrix"):
                cm_data = ml_res["confusion_matrix"]
                cm_df = pd.DataFrame(cm_data["matrix"], index=cm_data["labels"], columns=cm_data["labels"])
                st.markdown("#### Confusion Matrix")
                st.dataframe(cm_df, use_container_width=True)

            # Feature Importance
            if ml_res.get("feature_importance"):
                st.markdown("#### Feature Importance")
                imp_df = pd.DataFrame(ml_res["feature_importance"])
                st.bar_chart(imp_df.set_index("feature"))


# =============================================================================
# 9. REPORT CENTER
# =============================================================================
elif nav_option == "📑 Report Center":
    st.title("Boardroom Report Center")
    if df is None:
        st.warning("Upload or load a dataset first.")
    else:
        st.markdown("Download publication-quality reports containing complete audits, statistics, visualizations, and predictions.")

        profile = DataProfiler.generate_profile(df)
        insights = AIInsightsEngine.generate_insights(df)["insights"]
        cleaning_sum = st.session_state.cleaning_history[-1] if st.session_state.cleaning_history else None

        c_rep1, c_rep2, c_rep3 = st.columns(3)

        with c_rep1:
            st.markdown("#### 📄 Executive PDF Report")
            st.markdown("Complete multi-page report with executive summary, tables, and AI findings.")
            pdf_bytes = ReportGenerator.export_pdf(
                df,
                profile=profile,
                cleaning_summary=cleaning_sum,
                insights=insights,
                ml_results=st.session_state.last_ml_results,
            )
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"SmartSight_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                type="primary"
            )

        with c_rep2:
            st.markdown("#### 📊 Multi-Sheet Excel (.xlsx)")
            st.markdown("Formatted workbook with Data Preview, Profiling, Insights, and ML tabs.")
            excel_bytes = ReportGenerator.export_excel(
                df,
                profile=profile,
                cleaning_summary=cleaning_sum,
                insights=insights,
                ml_results=st.session_state.last_ml_results,
            )
            st.download_button(
                label="Download Excel Workbook",
                data=excel_bytes,
                file_name=f"SmartSight_Workbook_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        with c_rep3:
            st.markdown("#### 💾 Cleaned CSV Dataset")
            st.markdown("Normalized, sanitized tabular dataset ready for data science pipelines.")
            csv_bytes = ReportGenerator.export_csv(df)
            st.download_button(
                label="Download Cleaned CSV",
                data=csv_bytes,
                file_name=f"SmartSight_Cleaned_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
