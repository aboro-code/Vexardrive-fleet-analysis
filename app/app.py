import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="VexarDrive Fleet Dashboards", layout="wide")


@st.cache_data
def load_data():
    driver_df = pd.read_csv("outputs/driver_features_final.csv")
    vehicle_df = pd.read_csv("outputs/vehicle_health_features_final.csv")
    return driver_df, vehicle_df


driver_df, vehicle_df = load_data()

page = st.sidebar.radio("Dashboard", ["Driver Behaviour", "Vehicle Health"])

if page == "Driver Behaviour":
    st.title("Driver Behaviour Dashboard")
    st.caption(
        "Risk score combines harsh-event frequency and magnitude, validated against independent K-means clustering (silhouette score 0.516 at k=3)."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("High Risk Drivers", (driver_df["risk_tier"] == "High Risk").sum())
    col2.metric("Medium Risk Drivers", (driver_df["risk_tier"] == "Medium Risk").sum())
    col3.metric("Low Risk Drivers", (driver_df["risk_tier"] == "Low Risk").sum())

    fig = px.bar(
        driver_df.sort_values("risk_score", ascending=False),
        x="Driver_ID",
        y="risk_score",
        color="risk_tier",
        color_discrete_map={
            "High Risk": "#d62728",
            "Medium Risk": "#ff7f0e",
            "Low Risk": "#2ca02c",
        },
        title="Driver Risk Score by Tier",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Driver Detail Table")
    st.dataframe(
        driver_df[
            [
                "Driver_ID",
                "risk_score",
                "risk_tier",
                "mean_harsh_events_per_min",
                "max_harsh_events_per_min",
                "accel_x_std",
                "gyro_z_std",
            ]
        ].sort_values("risk_score", ascending=False),
        use_container_width=True,
    )

elif page == "Vehicle Health":
    st.title("Vehicle Health Dashboard")
    st.caption(
        "Health score weighted 60% vertical vibration (accel_z_std), 40% days since last service — validated via Isolation Forest anomaly detection (contamination=0.1)."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Vehicles", len(vehicle_df))
    col2.metric("Flagged Anomalies", (vehicle_df["anomaly_flag"] == "Anomaly").sum())
    col3.metric(
        "Avg Days Since Service", f"{vehicle_df['days_since_service'].mean():.0f}"
    )

    fig = px.scatter(
        vehicle_df,
        x="days_since_service",
        y="accel_z_std",
        color="anomaly_flag",
        size="health_score",
        color_discrete_map={"Anomaly": "#d62728", "Normal": "#1f77b4"},
        hover_data=["Vehicle_ID", "health_score"],
        title="Vibration vs Days Since Service (flagged anomalies in red)",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(
        vehicle_df.sort_values("health_score", ascending=False),
        x="Vehicle_ID",
        y="health_score",
        color="anomaly_flag",
        color_discrete_map={"Anomaly": "#d62728", "Normal": "#1f77b4"},
        title="Vehicle Health Score",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Vehicle Detail Table")
    st.dataframe(
        vehicle_df[
            [
                "Vehicle_ID",
                "health_score",
                "anomaly_flag",
                "accel_z_std",
                "days_since_service",
                "Make",
                "Model",
            ]
        ].sort_values("health_score", ascending=False),
        use_container_width=True,
    )
