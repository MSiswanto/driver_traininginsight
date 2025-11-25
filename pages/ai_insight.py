# pages/ai_insight.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.anomaly import run_full_anomaly_pipeline, format_anomaly_summary  # we will add a small helper below


def show_ai_insight(df_lap=None):

    st.title("🤖 AI Driver Insight (Anomaly Detection — Safety + Performance)")
    st.markdown("This panel runs a memory-safe anomaly pipeline (chunked aggregation + IsolationForest + robust z-score).")

    telemetry_path = "data/telemetry_filtered_v2.csv"
    st.caption(f"Telemetry path used: `{telemetry_path}`")

    with st.spinner("Running anomaly pipeline (safe mode)..."):
        try:
            # Run pipeline; this does chunked aggregation and returns per-vehicle-lap pivot and anomalies (small)
            pivot, anomalies = run_full_anomaly_pipeline(telemetry_path)
        except Exception as e:
            st.error(f"An error occurred while running anomaly detection: {e}")
            return

    if pivot is None or pivot.empty:
        st.warning("No telemetry pivot produced — check telemetry file format (vehicle_id,vehicle_number,lap,telemetry_name,telemetry_value).")
        return
    if anomalies is None or anomalies.empty:
        st.info("No anomalies found or not enough data to run detector.")
        return

    st.success("Anomaly pipeline complete (memory-safe).")

    # merge for display (small table)
    merged = pd.concat([pivot.reset_index(drop=True)[["vehicle_id","vehicle_number","lap"]], anomalies.reset_index(drop=True)], axis=1)

    st.subheader("🚨 Top Anomalous Laps (per vehicle+lap)")
    top_k = st.slider("Show top K anomalies", min_value=3, max_value=50, value=10)
    top = merged.sort_values("final_score", ascending=False).head(top_k)

    # SAFELY display top rows only (no reindexing of giant DF)
    safe_cols = ["vehicle_number", "vehicle_id", "lap", "final_score", "iso_score", "max_z", "is_anomaly"]
    safe_cols = [c for c in safe_cols if c in merged.columns]
    st.dataframe(top[safe_cols], width='stretch')

    st.subheader("Final anomaly score distribution")
    fig = px.histogram(merged, x="final_score", nbins=40, title="Distribution of final anomaly score")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Anomaly counts (per vehicle)")
    counts = merged.groupby("vehicle_number")["is_anomaly"].sum().reset_index().sort_values("is_anomaly", ascending=False)
    if not counts.empty:
        st.bar_chart(counts.set_index("vehicle_number")["is_anomaly"], width='stretch')

    st.subheader("Detailed inspect (choose one of top list)")
    if not top.empty:
        idx_map = top.reset_index().index.tolist()
        sel_idx = st.selectbox("Select top anomaly item (index)", idx_map, format_func=lambda i: f"{top.iloc[i]['vehicle_number']} — lap {int(top.iloc[i]['lap'])} ({top.iloc[i]['final_score']:.3f})")
        row = top.reset_index().iloc[sel_idx]
        st.markdown(f"**Selected:** Vehicle {row['vehicle_number']} (id {row['vehicle_id']}), Lap {int(row['lap'])}")
        # show radar-like chart if available
        radar_cols = [c for c in ["speed","throttle","brake","steer_abs","accel_mag","throttle_brake_diff"] if c in anomalies.columns]
        if radar_cols:
            vals = [row[c] if c in row.index else 0.0 for c in radar_cols]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(r=vals, theta=radar_cols, fill='toself', name='Selected anomaly'))
            fig_r.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
            st.plotly_chart(fig_r, width='stretch')

        st.markdown("### Heuristic explanation")
        expl = f"- Final score: {row['final_score']:.3f}\n- Isolation score: {row['iso_score']:.3f}\n- Max per-feature z: {row['max_z']:.2f}\n"
        reasons = []
        if row.get("iso_score", 0) > 0.6:
            reasons.append("Global pattern anomaly (unusual combo of features)")
        if row.get("max_z", 0) > 3.0:
            reasons.append("Large deviation on at least one metric (per-feature outlier)")
        if reasons:
            expl += "- Likely causes: " + "; ".join(reasons)
        else:
            expl += "- No clear high-level cause detected."
        st.text(expl)

    st.markdown("### Summary (top anomalies)")
    st.text(format_anomaly_summary(pivot, anomalies, top_n=10))
