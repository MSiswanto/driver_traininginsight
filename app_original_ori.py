import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Pages
from pages.ai_insight import show_ai_insight

st.set_page_config(
    page_title="Driver Telemetry Dashboard (FAST MODE)",
    page_icon="🏎️",
    layout="wide"
)

DEFAULT_TELEMETRY_CSV = "https://github.com/MSiswanto/driver_traininginsight/releases/download/csv/telemetry_filtered_v2.csv"


@st.cache_data(show_spinner=True)
def load_telemetry_wide(csv_path):
    if isinstance(csv_path, pd.DataFrame):
        raise TypeError("csv_path must be a path or URL, not a DataFrame.")

    is_url = csv_path.startswith("http://") or csv_path.startswith("https://")

    try:
        if is_url:
            df = pd.read_csv(csv_path)
        else:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Telemetry file not found: {csv_path}")
            df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")

    required = ["vehicle_id", "vehicle_number", "lap",
                "telemetry_name", "telemetry_value"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["sample_id"] = np.arange(len(df))

    df_wide = df.pivot(
        index=["vehicle_id", "vehicle_number", "lap", "sample_id"],
        columns="telemetry_name",
        values="telemetry_value"
    ).reset_index()

    df_wide.columns = [c.lower().strip() for c in df_wide.columns]

    rename_map = {
        "accx_can": "acc_x",
        "accy_can": "acc_y",
        "pbrake_f": "brake_front",
        "pbrake_r": "brake_rear",
        "steering_angle": "steering"
    }

    df_wide = df_wide.rename(columns=rename_map)

    for col in ["speed", "throttle", "brake_front", "brake_rear", "steering"]:
        if col not in df_wide.columns:
            df_wide[col] = np.nan

    if df_wide["throttle"].isna().all() or df_wide["throttle"].max() == 0:
        acc = df_wide["acc_x"].fillna(0).clip(lower=0)
        df_wide["throttle"] = (
            acc / acc.max() if acc.max() > 0 else acc
        ).round(3)

    return df_wide


def page_lap_analysis(df):
    st.header("📊 Lap Analysis")

    possible = ["speed", "throttle", "steering",
                "brake_front", "brake_rear", "acc_x", "acc_y"]

    avail = [m for m in possible if m in df.columns and df[m].notna().sum() > 0]

    if not avail:
        st.warning("No telemetry channels available for lap analysis.")
        return

    metric = st.selectbox("Select metric", avail)

    df_plot = df.groupby("lap")[metric].mean().reset_index()

    if metric in ["speed", "acc_x"]:
        best_idx = df_plot[metric].idxmax()
    else:
        best_idx = df_plot[metric].idxmin()

    st.success(
        f"Best lap: {int(df_plot.loc[best_idx,'lap'])} "
        f"({df_plot.loc[best_idx,metric]:.3f})"
    )

    fig = px.line(df_plot, x="lap", y=metric, markers=True,
                  title=f"{metric} per lap")
    st.plotly_chart(fig, use_container_width=True)


import matplotlib.pyplot as plt

def page_compare_drivers(df):
    st.header("🏎️ Compare Drivers")

    metrics = ["speed", "throttle", "steering", "brake_front", "brake_rear"]
    metrics = [m for m in metrics if m in df.columns]

    if len(metrics) == 0:
        st.error("❌ No valid telemetry metrics found.")
        return

    drivers = sorted(df["vehicle_number"].unique())

    colA, colB = st.columns([1.5, 1])
    driverA = colA.selectbox("Driver A", drivers, key="drvA")
    driverB = colB.selectbox("Driver B", drivers, key="drvB")

    dfA = df[df["vehicle_number"] == driverA].copy()
    dfB = df[df["vehicle_number"] == driverB].copy()

    dfA["lap"] = pd.to_numeric(dfA["lap"], errors="ignore")
    dfB["lap"] = pd.to_numeric(dfB["lap"], errors="ignore")

    laps_common = sorted(list(set(dfA["lap"]).intersection(dfB["lap"])))

    if len(laps_common) == 0:
        st.error("❌ These two drivers have no overlapping laps.")
        return

    lap_selected = st.selectbox("Select Lap", laps_common)
    lapA = dfA[dfA["lap"] == lap_selected].reset_index(drop=True)
    lapB = dfB[dfB["lap"] == lap_selected].reset_index(drop=True)

    metric = st.selectbox("Select telemetry metric", metrics)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(lapA[metric].values, label=f"Driver {driverA}", linewidth=2)
    ax.plot(lapB[metric].values, label=f"Driver {driverB}", linewidth=2)

    ax.set_title(f"Lap {lap_selected} — {metric} Comparison", fontsize=16)
    ax.set_xlabel("Telemetry Sample Index", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    st.pyplot(fig)

    st.subheader("📊 Lap Summary (Averages)")
    summary_df = pd.DataFrame({
        "Driver": [driverA, driverB],
        "Mean": [lapA[metric].mean(), lapB[metric].mean()],
        "Max": [lapA[metric].max(), lapB[metric].max()],
        "Min": [lapA[metric].min(), lapB[metric].min()],
    })
    st.table(summary_df)


def main():
    st.title("🏎️ Driver Telemetry Dashboard")

    try:
        df_wide = load_telemetry_wide(DEFAULT_TELEMETRY_CSV)
    except Exception as e:
        st.error(f"Error loading telemetry: {e}")
        st.stop()

    st.sidebar.title("🏎️ Driver AI Dashboard")
    st.sidebar.image("assets/team_logo.png", width=180)

    st.sidebar.header("Navigation")
    menu = st.sidebar.radio("Go to:", [
        "Lap Analysis",
        "Compare Drivers",
        "AI Insight"
    ])

    if menu == "Lap Analysis":
        page_lap_analysis(df_wide)
    elif menu == "Compare Drivers":
        page_compare_drivers(df_wide)
    elif menu == "AI Insight":
        show_ai_insight(df_wide)

    st.markdown("---")
    st.caption("Built with ❤️ • FAST MODE")


if __name__ == "__main__":
    main()
