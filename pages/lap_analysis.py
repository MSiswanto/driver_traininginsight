import streamlit as st
import plotly.express as px

def show_lap_analysis(df_vehicle):
    st.header("Lap Analysis")

    # Filtering valid laps
    df_lap = df_vehicle[df_vehicle["lap"].notna()]

    # Kolom metric yang mau divisualisasikan
    lap_metrics = [
        "speed_median",
        "throttle_median",
        "steering_median",
        "brake_median",
        "speed_max",
        "speed_min",
        "lap_time",
    ]

    # Pastikan kolomnya ada
    available_metrics = [c for c in lap_metrics if c in df_vehicle.columns]

    if len(available_metrics) == 0:
        st.error("No lap metrics available in dataframe.")
        return

    st.write("Available metrics:", available_metrics)

    # Pilih metric
    metric = st.selectbox("Select Metric", available_metrics)

    # Filter NaN on selected metric
    df_lap = df_lap[df_lap[metric].notna()]

    # Plot line chart
    fig = px.line(
        df_lap,
        x="lap",
        y=metric,
        title=f"{metric} per Lap",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
