import streamlit as st
import pandas as pd
import plotly.express as px

def show_lap_analysis(df):
    st.title("📊 Lap Analysis")

    # Check data
    if "lap" not in df.columns:
        st.error("Dataset missing 'lap' column.")
        return
    if "lap_time" not in df.columns:
        st.error("Dataset missing 'lap_time' column.")
        return

    # Driver selector
    if "vehicle_number" in df.columns:
        drivers = sorted(df["vehicle_number"].dropna().unique().tolist())
    elif "vehicle_id" in df.columns:
        drivers = sorted(df["vehicle_id"].dropna().unique().tolist())
    else:
        st.error("No driver column found.")
        return

    driver = st.selectbox("Select Driver", drivers)

    df_driver = df[df["vehicle_number"] == driver] if "vehicle_number" in df.columns else df[df["vehicle_id"] == driver]

    if df_driver.empty:
        st.warning("Selected driver has no data.")
        return

    # Lap time trend
    st.subheader("⏱ Lap Time Trend")
    lap_times = df_driver.groupby("lap")["lap_time"].min().reset_index()

    fig = px.line(
        lap_times,
        x="lap",
        y="lap_time",
        markers=True,
        title=f"Lap Times for Driver {driver}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Speed profile optional
    speed_col = None
    for c in ["speed", "kph", "KPH", "TOP_SPEED"]:
        if c in df.columns:
            speed_col = c
            break

    if speed_col:
        st.subheader("🚀 Speed Trend")
        speed_df = df_driver.groupby("lap")[speed_col].mean().reset_index()
        fig2 = px.line(
            speed_df,
            x="lap",
            y=speed_col,
            markers=True,
            title=f"Avg Speed per Lap ({driver})"
        )
        st.plotly_chart(fig2, use_container_width=True)
