# main.py — FINAL VERSION
import streamlit as st
import pandas as pd
import os

# Import pages
from pages.ai_insight import show_ai_insight
from pages.compare_drivers import show_compare_drivers
from pages.lap_analysis import show_lap_analysis

st.set_page_config(page_title="Driver Telemetry Dashboard", layout="wide")

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to page:",
    [
        "🏎 AI Insight",
        "🆚 Compare Drivers",
        "📊 Lap Analysis"
    ]
)

# Default telemetry CSV
DEFAULT_CSV = "https://github.com/MSiswanto/driver_traininginsight/releases/download/csv/telemetry_filtered_v2.csv"

@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)

df = load_data(DEFAULT_CSV)

# Route pages
if page == "🏎 AI Insight":
    show_ai_insight(df)

elif page == "🆚 Compare Drivers":
    show_compare_drivers(df)

#elif page == "📊 Lap Analysis":
    #show_lap_analysis(df)
