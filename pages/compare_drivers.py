import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# helper to safe-get column
def _get_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def show_compare_drivers(df):
    st.title("🆚 Compare Drivers — Full Race Comparison")

    # normalize basic column names if needed
    if "vehicle_number" not in df.columns and "vehicle_id" in df.columns:
        df["vehicle_number"] = df["vehicle_id"]

    # Ensure numeric types where applicable
    for col in ["lap", "lap_time", "speed", "s1", "s2", "s3", "top_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Driver selectors (choose by vehicle_number or vehicle id)
    drivers = sorted(df["vehicle_number"].dropna().unique().tolist())
    if len(drivers) < 2:
        st.warning("Not enough drivers available in dataset to compare.")
        return

    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Driver A", drivers, index=0)
    with c2:
        b = st.selectbox("Driver B", drivers, index=1 if len(drivers) > 1 else 0)

    df_a = df[df["vehicle_number"] == a].copy()
    df_b = df[df["vehicle_number"] == b].copy()

    if df_a.empty or df_b.empty:
        st.error("Selected drivers have no data.")
        return

    # ===== SUMMARY CARDS =====
    def summary_card(dfx, title):
        fastest = dfx["lap_time"].min() if "lap_time" in dfx.columns else np.nan
        avg = dfx["lap_time"].mean() if "lap_time" in dfx.columns else np.nan
        top_speed = dfx["top_speed"].max() if "top_speed" in dfx.columns else np.nan
        consistency = dfx["lap_time"].std() if "lap_time" in dfx.columns else np.nan

        st.markdown(f"""
        <div style='padding:12px;border-radius:10px;background:#0b0b0b;color:#fff;'>
          <h3 style='margin:0;color:#E10600'>{title}</h3>
          <p>Fastest lap: <b>{fastest:.3f if not np.isnan(fastest) else 'N/A'}</b></p>
          <p>Average lap: <b>{avg:.3f if not np.isnan(avg) else 'N/A'}</b></p>
          <p>Top speed: <b>{top_speed:.1f if not np.isnan(top_speed) else 'N/A'}</b></p>
          <p>Consistency (std): <b>{consistency:.3f if not np.isnan(consistency) else 'N/A'}</b></p>
        </div>
        """, unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        summary_card(df_a, f"Driver A — {a}")
    with colB:
        summary_card(df_b, f"Driver B — {b}")

    st.markdown("---")

    # ===== Lap time comparison =====
    if "lap_time" in df.columns:
        st.subheader("📊 Lap Time Comparison")
        lap_a = df_a.groupby("lap")["lap_time"].min().reset_index()
        lap_b = df_b.groupby("lap")["lap_time"].min().reset_index()
        lap_comp = pd.merge(lap_a, lap_b, on="lap", how="inner", suffixes=(f"_{a}", f"_{b}"))
        if not lap_comp.empty:
            lap_comp["delta"] = lap_comp[f"lap_time_{a}"] - lap_comp[f"lap_time_{b}"]
            fig = px.line(lap_comp, x="lap", y=[f"lap_time_{a}", f"lap_time_{b}"],
                          labels={"value": "Lap Time (s)", "lap": "Lap"},
                          title=f"Lap Times: {a} vs {b}")
            st.plotly_chart(fig, use_container_width=True)

            # gap bar
            fig_gap = px.bar(lap_comp, x="lap", y="delta",
                             color=lap_comp["delta"] > 0,
                             color_discrete_map={True: "red", False: "green"},
                             labels={"delta": "Time gap (s)"},
                             title="Lap Delta (positive = A slower than B)")
            st.plotly_chart(fig_gap, use_container_width=True)

    # ===== Sector comparison (S1/S2/S3) =====
    if all(x in df.columns for x in ["S1_SECONDS", "S2_SECONDS", "S3_SECONDS"]):
        st.subheader("🏁 Sector Times Comparison (per lap)")
        sa = df_a.groupby("lap")[["S1_SECONDS","S2_SECONDS","S3_SECONDS"]].mean().reset_index()
        sb = df_b.groupby("lap")[["S1_SECONDS","S2_SECONDS","S3_SECONDS"]].mean().reset_index()
        # melt for plotting
        ma = sa.melt(id_vars="lap", value_vars=["S1_SECONDS","S2_SECONDS","S3_SECONDS"], var_name="sector", value_name="seconds")
        mb = sb.melt(id_vars="lap", value_vars=["S1_SECONDS","S2_SECONDS","S3_SECONDS"], var_name="sector", value_name="seconds")
        ma["driver"]="A"
        mb["driver"]="B"
        m = pd.concat([ma, mb], axis=0)
        fig_sector = px.line(m, x="lap", y="seconds", color="driver", facet_col="sector", title="Sector times comparison")
        st.plotly_chart(fig_sector, use_container_width=True)

    # ===== Speed profile overlay (per lap or distance) =====
    speed_col = _get_col(df, ["speed","KPH","kph","TOP_SPEED"])
    if speed_col:
        st.subheader("🚀 Speed Profile Overlay")
        # choose to show per lap or aggregated by lap distance
        option = st.radio("Show speed by", ["Lap index", "Lap distance"], horizontal=True)
        if option == "Lap index":
            fig_speed = go.Figure()
            # plot average lap speed per lap
            mean_a = df_a.groupby("lap")[speed_col].mean().reset_index()
            mean_b = df_b.groupby("lap")[speed_col].mean().reset_index()
            fig_speed.add_trace(go.Scatter(x=mean_a["lap"], y=mean_a[speed_col], mode="lines+markers", name=f"A {a}"))
            fig_speed.add_trace(go.Scatter(x=mean_b["lap"], y=mean_b[speed_col], mode="lines+markers", name=f"B {b}"))
            fig_speed.update_layout(title="Avg Speed per Lap", xaxis_title="Lap", yaxis_title="Speed")
            st.plotly_chart(fig_speed, use_container_width=True)
        else:
            # overlay by lap distance if lap_distance exists
            dist_col = _get_col(df, ["lap_distance","LAP_DISTANCE"])
            if dist_col:
                seg = pd.concat([
                    df_a[[dist_col, speed_col]].assign(driver=f"A_{a}"),
                    df_b[[dist_col, speed_col]].assign(driver=f"B_{b}")
                ], axis=0)
                fig = px.line(seg, x=dist_col, y=speed_col, color="driver", title="Speed vs Lap Distance (overlay)")
                st.plotly_chart(fig, use_container_width=True)

    # ===== Throttle / Brake overlay (if present) =====
    throttle_col = _get_col(df, ["aps","APS","throttle","THROTTLE"])
    brake_col = _get_col(df, ["pbrake_f","PBRk","brake","BRAKE","pbrake_f"])
    if throttle_col or brake_col:
        st.subheader("⚙️ Throttle & Brake Comparison")
        if throttle_col:
            ta = df_a[[throttle_col,"lap"]].groupby("lap").mean().reset_index()
            tb = df_b[[throttle_col,"lap"]].groupby("lap").mean().reset_index()
            tcomp = pd.merge(ta, tb, on="lap", how="inner", suffixes=(f"_{a}", f"_{b}"))
            fig_t = px.line(tcomp, x="lap", y=[f"{throttle_col}_{a}", f"{throttle_col}_{b}"], labels={"value":"Throttle"}, title="Avg Throttle per Lap")
            st.plotly_chart(fig_t, use_container_width=True)
        if brake_col:
            ba = df_a[[brake_col,"lap"]].groupby("lap").mean().reset_index()
            bb = df_b[[brake_col,"lap"]].groupby("lap").mean().reset_index()
            bcomp = pd.merge(ba, bb, on="lap", how="inner", suffixes=(f"_{a}", f"_{b}"))
            fig_b = px.line(bcomp, x="lap", y=[f"{brake_col}_{a}", f"{brake_col}_{b}"], labels={"value":"Brake"}, title="Avg Brake per Lap")
            st.plotly_chart(fig_b, use_container_width=True)

    # ===== Radar chart (driving style) =====
    st.subheader("📡 Driving Style Radar")
    metrics = []
    mnames = []
    if speed_col:
        metrics.append(df_a[speed_col].mean()); mnames.append("Speed")
    if throttle_col:
        metrics.append(df_a[throttle_col].mean()); mnames.append("Throttle")
    if brake_col:
        metrics.append(df_a[brake_col].mean()); mnames.append("Brake")
    # driver B
    metrics_b = []
    if speed_col:
        metrics_b.append(df_b[speed_col].mean())
    if throttle_col:
        metrics_b.append(df_b[throttle_col].mean())
    if brake_col:
        metrics_b.append(df_b[brake_col].mean())

    if metrics and metrics_b:
        radar_df = pd.DataFrame({
            "metric": mnames,
            f"A_{a}": metrics,
            f"B_{b}": metrics_b
        })
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=radar_df[f"A_{a}"].values, theta=radar_df["metric"], fill='toself', name=f"A {a}"))
        fig_r.add_trace(go.Scatterpolar(r=radar_df[f"B_{b}"].values, theta=radar_df["metric"], fill='toself', name=f"B {b}"))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="Driving Style Radar")
        st.plotly_chart(fig_r, use_container_width=True)

    # ===== Auto summary using simple heuristics =====
    st.subheader("🧾 Auto Summary")
    lines = []
    if "lap_time" in df.columns:
        avg_a = df_a["lap_time"].mean()
        avg_b = df_b["lap_time"].mean()
        winner = a if avg_a < avg_b else b
        lines.append(f"Average lap: {a} = {avg_a:.3f}s, {b} = {avg_b:.3f}s → **{winner}** is faster on average.")
    if speed_col:
        lines.append(f"Average speed: {a} = {df_a[speed_col].mean():.2f}, {b} = {df_b[speed_col].mean():.2f}.")
    if throttle_col and brake_col:
        lines.append(f"Throttle mean: {a} = {df_a[throttle_col].mean():.2f}, {b} = {df_b[throttle_col].mean():.2f}. Brake mean: {a} = {df_a[brake_col].mean():.2f}, {b} = {df_b[brake_col].mean():.2f}.")
    for l in lines:
        st.markdown(f"- {l}")
