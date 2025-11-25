# ------------------------------------------------------------
# app.py — FAST MODE (Super Optimized)
# ------------------------------------------------------------
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

# Pages
from pages.ai_insight import show_ai_insight
#from pages.compare_drivers import show_compare_drivers


st.set_page_config(page_title="Driver Telemetry Dashboard (FAST MODE)",
                   page_icon="🏎️", layout="wide")

DEFAULT_TELEMETRY_CSV = "data/telemetry_filtered_v2.csv"


# =====================================================================
# 🔥 SUPER FAST LOADER (NO TIMESTAMP, NO EXPENSIVE PIVOT)
# =====================================================================
@st.cache_data(show_spinner=True)
def load_telemetry_wide(csv_path):
    """
    FAST MODE loader for long-format telemetry:
    vehicle_id, vehicle_number, lap, telemetry_name, telemetry_value
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Telemetry file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = ["vehicle_id", "vehicle_number", "lap",
                "telemetry_name", "telemetry_value"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Sample index (replaces timestamp)
    df["sample_id"] = np.arange(len(df))

    # SUPER FAST PIVOT: no timestamp, no large index
    df_wide = df.pivot(
        index=["vehicle_id", "vehicle_number", "lap", "sample_id"],
        columns="telemetry_name",
        values="telemetry_value"
    ).reset_index()

    df_wide.columns = [c.lower().strip() for c in df_wide.columns]

    # Rename some common channels
    rename_map = {
        "accx_can": "acc_x",
        "accy_can": "acc_y",
        "pbrake_f": "brake_front",
        "pbrake_r": "brake_rear",
        "steering_angle": "steering"
    }
    df_wide = df_wide.rename(columns=rename_map)

    # Ensure main numeric fields exist
    for col in ["speed", "throttle", "brake_front", "brake_rear", "steering"]:
        if col not in df_wide.columns:
            df_wide[col] = np.nan

    # Fallback throttle estimate from acc_x
    if df_wide["throttle"].isna().all() or df_wide["throttle"].max() == 0:
        acc = df_wide["acc_x"].fillna(0)
        acc = acc.clip(lower=0)
        df_wide["throttle"] = (
            acc / acc.max() if acc.max() > 0 else acc
        ).round(3)

    return df_wide


# =====================================================================
# 🧠 ANOMALY DETECTION (Lazy load)
# =====================================================================
def run_anomaly_detection(df):
    cols = ["speed", "throttle", "brake_front", "steering", "acc_x", "acc_y"]
    features = [c for c in cols if c in df.columns and df[c].notna().sum() > 0]

    if len(features) < 2:
        return None, "Not enough numeric fields for anomaly detection."

    df_clean = df.dropna(subset=features).copy()
    if len(df_clean) < 30:
        return None, "Dataset too small for anomaly detection."

    iso = IsolationForest(contamination=0.03, random_state=42)
    df_clean["iso_score"] = -iso.fit(df_clean[features]).decision_function(df_clean[features])
    df_clean["is_anomaly"] = iso.predict(df_clean[features]) == -1

    lap_summary = df_clean.groupby(["vehicle_number", "lap"])["is_anomaly"].mean()
    lap_summary = lap_summary.reset_index().rename(columns={"is_anomaly": "anomaly_ratio"})

    return lap_summary.sort_values("anomaly_ratio", ascending=False), None


# =====================================================================
# 📊 LAP ANALYSIS
# =====================================================================
def page_lap_analysis(df):
    st.header("📊 Lap Analysis")

    possible = ["speed", "throttle", "steering", "brake_front", "brake_rear", "acc_x", "acc_y"]
    avail = [m for m in possible if m in df.columns and df[m].notna().sum() > 0]

    if not avail:
        st.warning("No telemetry channels available for lap analysis.")
        return

    metric = st.selectbox("Select metric", avail)
    df_plot = df.groupby("lap")[metric].mean().reset_index()

    # Best lap logic
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


# =====================================================================
# 🆚 COMPARE DRIVERS
# =====================================================================
# =========================================================
# 📌 Compare Drivers — FINAL FIXED VERSION (Fast + Big Plots)
# =========================================================
import matplotlib.pyplot as plt

def page_compare_drivers(df):
    st.header("🏎️ Compare Drivers")
    #st.container().markdown("### 📈 Driver Telemetry Comparison")

    # Metrics available
    metrics = ["speed", "throttle", "steering", "brake_front", "brake_rear"]
    metrics = [m for m in metrics if m in df.columns]

    if len(metrics) == 0:
        st.error("❌ No valid telemetry metrics found.")
        return

    # Choose drivers
    drivers = sorted(df["vehicle_number"].unique())

    colA, colB = st.columns([1.5, 1])
    driverA = colA.selectbox("Driver A", drivers, key="drvA")
    driverB = colB.selectbox("Driver B", drivers, key="drvB")

    dfA = df[df["vehicle_number"] == driverA].copy()
    dfB = df[df["vehicle_number"] == driverB].copy()

    # Fix lap dtype
    dfA["lap"] = pd.to_numeric(dfA["lap"], errors="ignore")
    dfB["lap"] = pd.to_numeric(dfB["lap"], errors="ignore")

    laps_common = sorted(list(set(dfA["lap"]).intersection(dfB["lap"])))
    if len(laps_common) == 0:
        st.error("❌ These two drivers have no overlapping laps.")
        return

    lap_selected = st.selectbox("Select Lap", laps_common)

    lapA = dfA[dfA["lap"] == lap_selected].reset_index(drop=True)
    lapB = dfB[dfB["lap"] == lap_selected].reset_index(drop=True)

    # Dropdown metric selection
    metric = st.selectbox("Select telemetry metric", metrics)

    # ---------- BIG, CLEAR RESPONSIVE PLOT ----------
    fig, ax = plt.subplots(figsize=(12, 5))  # Large plot for clarity

    ax.plot(lapA[metric].values, label=f"Driver {driverA}", linewidth=2)
    ax.plot(lapB[metric].values, label=f"Driver {driverB}", linewidth=2)

    ax.set_title(f"Lap {lap_selected} — {metric} Comparison", fontsize=16)
    ax.set_xlabel("Telemetry Sample Index", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    st.pyplot(fig)


    # ---------- Optional: Show averages ----------
    st.subheader("📊 Lap Summary (Averages)")
    summary_df = pd.DataFrame({
        "Driver": [driverA, driverB],
        "Mean": [lapA[metric].mean(), lapB[metric].mean()],
        "Max": [lapA[metric].max(), lapB[metric].max()],
        "Min": [lapA[metric].min(), lapB[metric].min()],
    })
    st.table(summary_df)

# =====================================================================
# 🤖 AI INSIGHT
# =====================================================================

def page_ai_insight(df_lap):

    st.subheader("🤖 AI Insights (Hybrid Mode)")
    st.info("Model analyzes lap performance to highlight unusual patterns.")

    if df_lap is None or len(df_lap) < 5:
        st.warning("Dataset too small. No AI insights available.")
        return

    df_lap = df_lap.copy()

    numeric_cols = ["speed", "acc_x", "acc_y", "brake_front",
                    "brake_rear", "steering", "throttle"]
    numeric_cols = [c for c in numeric_cols if c in df_lap.columns]

    if len(numeric_cols) == 0:
        st.error("No numeric telemetry columns available.")
        return

    df = df_lap[numeric_cols].fillna(0)

    # =============================
    # FAST MODE for small dataset
    # =============================
    if len(df) < 80:
        st.info("Using FAST mode (Z-score) due to small dataset.")

        z = (df - df.mean()) / (df.std() + 1e-6)
        df_lap["anomaly_score"] = z.abs().mean(axis=1)

    # =============================
    # ML MODE (Memory Safe)
    # =============================
    else:
        st.info("Using ML mode with Mini-Batch Isolation Forest.")

        from sklearn.ensemble import IsolationForest
        import math

        # Windowing: process in chunks to avoid huge allocation
        WINDOW = 50000       # safe memory chunk
        N = len(df)
        chunks = math.ceil(N / WINDOW)

        scores = []

        try:
            for i in range(chunks):
                start = i * WINDOW
                end = min((i + 1) * WINDOW, N)

                batch = df.iloc[start:end]

                model = IsolationForest(
                    n_estimators=30,
                    max_samples=min(2000, len(batch)),
                    contamination=0.05,
                    random_state=42,
                    n_jobs=-1
                )

                model.fit(batch)
                batch_score = -model.decision_function(batch)
                scores.extend(batch_score)

            df_lap["anomaly_score"] = np.array(scores)

        except Exception as e:
            st.error(f"ML detection failed safely (HYBRID MODE fallback): {e}")
            # fallback ke z-score
            z = (df - df.mean()) / (df.std() + 1e-6)
            df_lap["anomaly_score"] = z.abs().mean(axis=1)

    # =============================
    # OUTPUT
    # =============================
    st.write("Model analyzes lap performance to highlight unusual patterns.")

    #st.line_chart(df_lap[["anomaly_score"]], width="stretch")
    # ========== SAFE DOWNSAMPLE UNTUK CHART BESAR ==========
    MAX_POINTS = 5000

    if len(df_lap) > MAX_POINTS:
        step = len(df_lap) // MAX_POINTS
        df_plot = df_lap.iloc[::step][["anomaly_score"]]
    else:
        df_plot = df_lap[["anomaly_score"]]

    st.line_chart(df_plot, width="stretch")


    # Top unusual segments
    # ======== PREVENT INDEX EXPLOSION ========
    df_lap = df_lap.reset_index(drop=True)

    # ======== GET TOP ANOMALY ROWS SAFELY ========
    top_k = 10
    top_idx = df_lap["anomaly_score"].nlargest(top_k).index.tolist()

    df_top = df_lap.loc[top_idx, safe_cols].copy()

    # ======== SAFE DISPLAY ========
    st.dataframe(df_top, width="stretch")

    #top_idx = df_lap["anomaly_score"].nlargest(5).index

    safe_cols = ["anomaly_score"] + numeric_cols
    safe_cols = [c for c in safe_cols if c in df_lap.columns]

    st.success("Top unusual telemetry segments:")
    #st.dataframe(df_lap.loc[top_idx, safe_cols], width="stretch")

# =====================================================================
# MAIN APP
# =====================================================================
def main():
    st.title("🏎️ Driver Telemetry Dashboard")
    st.caption("Optimized build: ultra-fast loading and processing.")

    try:
        df_wide = load_telemetry_wide(DEFAULT_TELEMETRY_CSV)
    except Exception as e:
        st.error(f"Error loading telemetry: {e}")
        st.stop()

    st.sidebar.title("🏎️ Driver AI Dashboard")
    st.sidebar.image("assets/team_logo.png", width=180)

    st.sidebar.header("Navigation")
    menu = st.sidebar.radio("Go to:", ["Lap Analysis", "Compare Drivers", "AI Insight"])
    
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
