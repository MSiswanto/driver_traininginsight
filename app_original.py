# ------------------------------------------------------------
# app_original.py — MAIN APPLICATION
# ------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Pages
from pages.ai_insight import show_ai_insight
# from pages.compare_drivers import show_compare_drivers

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
st.set_page_config(
    page_title="Driver Telemetry Dashboard (FAST MODE)",
    page_icon="🏎️",
    layout="wide"
)

CSV_URL = "https://github.com/MSiswanto/driver_traininginsight/releases/download/csv/telemetry_filtered_v2.csv"

# ------------------------------------------------------------
# FAST CSV LOADER
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_telemetry_wide(csv_path):
    """FAST MODE loader supporting local & URL CSV."""
    if isinstance(csv_path, pd.DataFrame):
        raise TypeError("csv_path must be a path or URL string.")

    is_url = csv_path.startswith("http://") or csv_path.startswith("https://")

    try:
        if is_url:
            df = pd.read_csv(csv_path)
        else:
            df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None

    return df


# ------------------------------------------------------------
# MAIN APP LOGIC
# ------------------------------------------------------------
def main():

    st.title("🏎️ Driver Telemetry Dashboard — FAST MODE")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Page:", ["AI Insight"])

    # Load Data
    with st.spinner("Loading telemetry data..."):
        df = load_telemetry_wide(CSV_URL)

    if df is None:
        st.error("❌ Failed to load telemetry data.")
        return

    # Routing
    if page == "AI Insight":
        show_ai_insight(df)

    # Future pages:
    # elif page == "Compare Drivers":
    #     show_compare_drivers(df)


# Entry point
if __name__ == "__main__":
    main()
