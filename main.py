# ============================================================
# main.py — Driver Telemetry Dashboard (FINAL RELEASE)
# Compatible with Streamlit Cloud & your repo structure
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

from sklearn.ensemble import IsolationForest

# Import pages
from pages.ai_insight import show_ai_insight
from pages.compare_drivers import page_compare_drivers
from pages.lap_analysis import page_lap_analysis   # jika ada

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Driver Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide"
)

# ------------------------------------------------------------
# CSV Source (GitHub Release)
# ------------------------------------------------------------
DEFAULT_TELEMETRY_CSV = (
    "https://github.com/MSiswanto/driver_traininginsight/"
    "releases/download/csv/telemetry_filtered_v2.csv"
)

# ------------------------------------------------------------
# FAST Telemetry Loader
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_telemetry_wide(csv_path):
    """Load long-format telemetry and convert to wide format safely."""

    if csv_path.startswith("http://") or csv_path.startswith("https://"):
        df = pd.read_csv(csv_path)
    else:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")
        df = pd.read_csv(csv_path)

    required_cols = [
        "vehicle_id",
        "vehicle_number",
        "lap",
        "telemetry_name",
        "telemetry_value"
    ]

    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Create sample index
    df["sample_id"] = np.arange(len(df))

    # Pivot to wide format
    df_wide = df.pivot(
        index=["vehicle_id", "vehicle_number", "lap", "sample_id"],
        columns="telemetry_name",
        values="telemetry_value"
    ).reset_index()

    # Normalize columns
    df_wide.columns = [c.lower().strip() for c in df_wide.columns]

    # Renaming for consistency
    rename_map = {
        "accx_can": "acc_x",
        "accy_can": "acc_y",
        "pbrake_f": "brake_front",
        "pbrake_r": "brake_rear",
        "steering_angle": "steering"
    }
    df_wide.rename(columns=rename_map, inplace=True)

    # Ensure required fields exist
    for col in ["speed", "throttle", "brake_front", "brake_rear", "steering"]:
        if col not in df_wide:
            df_wide[col] = np.nan

    # Auto throttle if missing
    if df_wide["throttle"].isna().all() or df_wide["throttle"].max() == 0:
        acc_val = df_wide["acc_x"].fillna(0).clip(lower=0)
        df_wide["throttle"] = (
            acc_val / acc_val.max() if acc_val.max() > 0 else acc_val
        ).round(3)

    return df_wide


# ------------------------------------------------------------
# MAIN APPLICATION
# ------------------------------------------------------------
def main():
    st.title("🏎️ Driver Telemetry Dashboard")
    st.caption("Fast, stable, and optimized telemetry analytics.")

    # Load data
    try:
        df = load_telemetry_wide(DEFAULT_TELEMETRY_CSV)
    except Exception as e:
        st.error(f"Error loading telemetry file:\n\n{e}")
        st.stop()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.image("assets/team_logo.png", width=160)

    menu = st.sidebar.radio(
        "Go to:",
        ["Lap Analysis", "Compare Drivers", "AI Insight"]
    )

    # Connect pages
    if menu == "Lap Analysis":
        page_lap_analysis(df)

    elif menu == "Compare Drivers":
        page_compare_drivers(df)

    elif menu == "AI Insight":
        show_ai_insight(df)

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")


# ------------------------------------------------------------
# Launch app
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
