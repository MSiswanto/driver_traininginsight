# pages/ai_insight.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.anomaly import run_full_anomaly_pipeline, format_anomaly_summary


def show_ai_insight(df_lap=None):

    st.title("🤖 AI Driver Insight (Anomaly Detection — Safety + Performance)")
    st.markdown("This panel runs a memory-safe anomaly pipeline (chunked aggregation + IsolationForest + robust z-score).")

    if df_lap is None or len(df_lap) == 0:
        st.error("No telemetry data received from main app.")
        return

    # 🔥 FIX: pass DataFrame, not URL
    telemetry_df = df_lap   
    
    st.caption("Using in-memory telemetry from main dashboard (FAST MODE).")

    # create a progress bar
    prog = st.progress(0)
    def _progress_cb(f):
        try:
            prog.progress(min(100, int(f * 100)))
        except Exception:
            pass

    with st.spinner("Running anomaly pipeline (safe mode)..."):
        try:
            # pass df_lap (DataFrame) directly, sample_max_rows to limit memory if needed
            pivot, anomalies = run_full_anomaly_pipeline(
                df_lap,
                metrics_keep=None,
                contamination=0.02,
                chunksize=200_000,
                progress=_progress_cb,
                sample_max_rows=200_000,
                n_jobs=1
            )
            prog.progress(100)
        except Exception as e:
            st.error(f"An error occurred while running anomaly detection: {e}")
            prog.empty()
            return

    if pivot is None or pivot.empty:
        st.warning("No telemetry pivot produced — missing required columns.")
        return
    if anomalies is None or anomalies.empty:
        st.info("No anomalies found or not enough data to run detector.")
        return

    st.success("Anomaly pipeline complete (memory-safe).")

    merged = pd.concat([pivot.reset_index(drop=True)[
        ["vehicle_id", "vehicle_number", "lap"]
    ], anomalies.reset_index(drop=True)], axis=1)

    st.subheader("🚨 Top Anomalous Laps (per vehicle+lap)")
    top_k = st.slider("Show top K anomalies", min_value=3, max_value=50, value=10)
    top = merged.sort_values("final_score", ascending=False).head(top_k)

    safe_cols = [
        "vehicle_number", "vehicle_id", "lap",
        "final_score", "iso_score", "max_z", "is_anomaly"
    ]
    safe_cols = [c for c in safe_cols if c in merged.columns]

    st.dataframe(top[safe_cols], width='stretch')

    st.subheader("Final anomaly score distribution")
    fig = px.histogram(merged, x="final_score", nbins=40)
    st.plotly_chart(fig, width='stretch')

    st.subheader("Anomaly counts (per vehicle)")
    counts = merged.groupby("vehicle_number")["is_anomaly"].sum()
    st.bar_chart(counts)

    st.subheader("Detailed inspect")
    if not top.empty:
        idx_map = top.reset_index().index.tolist()
        sel_idx = st.selectbox(
            "Select anomaly",
            idx_map,
            format_func=lambda i: f"{top.iloc[i]['vehicle_number']} — Lap {int(top.iloc[i]['lap'])}"
        )
        row = top.reset_index().iloc[sel_idx]

        st.markdown(f"**Selected:** Vehicle {row['vehicle_number']} → Lap {int(row['lap'])}")

        radar_cols = [c for c in [
            "speed","throttle","brake","steer_abs","accel_mag","throttle_brake_diff"
        ] if c in anomalies.columns]

        if radar_cols:
            vals = [row[c] if c in row.index else 0 for c in radar_cols]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(r=vals, theta=radar_cols, fill='toself'))
            fig_r.update_layout(showlegend=False)
            st.plotly_chart(fig_r)

        st.markdown("### Summary")
        st.text(format_anomaly_summary(pivot, anomalies, top_n=10))
