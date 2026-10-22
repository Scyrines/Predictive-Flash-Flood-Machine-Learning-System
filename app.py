import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import time
from datetime import datetime, timedelta
from flood_model import FloodPredictorModel, NORMAL_WATER_LEVEL, ALERT_WATER_LEVEL, CRITICAL_WATER_LEVEL

# Set Streamlit Page Config
st.set_page_config(
    page_title="Smart Lake Flash-Flood AI Monitor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and status indicators
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.88rem;
        color: #94a3b8;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Status Badge Styling */
    .status-badge {
        padding: 10px 18px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
        width: 100%;
        justify-content: center;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-normal {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
    }
    .status-alert {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid #F59E0B;
    }
    .status-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.85; transform: scale(1.01); }
    }
    .alert-card {
        padding: 12px 16px;
        border-radius: 8px;
        background: #0f172a;
        border-left: 4px solid #38bdf8;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache model instance
@st.cache_resource
def load_model():
    model = FloodPredictorModel()
    model.train()
    return model

model = load_model()

# Title Header
st.title("🌊 Smart Environmental IT System — 4-Acre Malaysian Lake")
st.caption("Predictive Flash-Flood Machine Learning System")

# Top Banner Context
col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    st.info("📍 **Target Site**: 4-Acre Campus / Municipal Lake")
with col_b2:
    st.info("🌧️ **Climate Zone**: Peninsular Malaysia (High Monsoon Runoff)")
with col_b3:
    st.info("📶 **IoT Gateway**: ESP32 Edge Node + Ultrasonic Hydro-Sensor")
with col_b4:
    st.info("⚡ **Architecture**: Store-and-Forward + MQTT Cloud Sync")

st.markdown("---")

# Sidebar Controls
st.sidebar.header("🎛️ Simulation & Telemetry Controls")

scenario = st.sidebar.selectbox(
    "🌦️ Select Malaysian Weather Scenario",
    [
        "Custom Parameters",
        "☀️ Normal Sunny Afternoon (Calm Lake)",
        "🌦️ Moderate Convective Shower (Typical)",
        "⛈️ Monsoon Downpour (Amaran Lebat)",
        "🚨 Severe Tropical Flash-Flood Deluge (Bah Merah)"
    ]
)

# Preset scenario handling
if scenario == "☀️ Normal Sunny Afternoon (Calm Lake)":
    default_water = 1.95
    default_rain = 0.0
    default_sat = 0.35
    default_culvert = "Normal Discharge (100% Open)"
elif scenario == "🌦️ Moderate Convective Shower (Typical)":
    default_water = 2.25
    default_rain = 28.0
    default_sat = 0.60
    default_culvert = "Normal Discharge (100% Open)"
elif scenario == "⛈️ Monsoon Downpour (Amaran Lebat)":
    default_water = 2.85
    default_rain = 75.0
    default_sat = 0.85
    default_culvert = "Normal Discharge (100% Open)"
elif scenario == "🚨 Severe Tropical Flash-Flood Deluge (Bah Merah)":
    default_water = 3.20
    default_rain = 115.0
    default_sat = 0.95
    default_culvert = "Culvert Partially Blocked (50%)"
else:
    default_water = 2.30
    default_rain = 45.0
    default_sat = 0.70
    default_culvert = "Normal Discharge (100% Open)"

current_water_level = st.sidebar.slider(
    "Current Lake Water Level (m)",
    min_value=1.50,
    max_value=3.80,
    value=float(default_water),
    step=0.05,
    help="Baseline depth is ~2.00m. Embankment warning at 3.00m, flood overflow at 3.50m."
)

rainfall_intensity = st.sidebar.slider(
    "Rainfall Intensity (mm/h)",
    min_value=0.0,
    max_value=130.0,
    value=float(default_rain),
    step=2.5,
    help="Meteorological Department Malaysia standard: >60 mm/h is heavy monsoon downpour."
)

soil_saturation = st.sidebar.slider(
    "Catchment Soil Saturation Factor",
    min_value=0.10,
    max_value=1.00,
    value=float(default_sat),
    step=0.05,
    help="Higher soil saturation means less ground absorption and faster surface runoff into the 4-acre lake."
)

drainage_gate = st.sidebar.selectbox(
    "Lake Spillway / Culvert Drainage Status",
    ["Normal Discharge (100% Open)", "Culvert Partially Blocked (50%)", "Emergency Bypass Active"],
    index=0 if default_culvert.startswith("Normal") else 1
)

# Drainage modifier calculation
drainage_penalty = 0.08 if "50%" in drainage_gate else (-0.05 if "Bypass" in drainage_gate else 0.0)

# Real-time simulation mode toggle
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Live Dynamic Stream")
live_mode = st.sidebar.toggle("Enable Live Auto-Simulated Influx", value=False)
sim_speed = st.sidebar.select_slider("Simulation Clock Speed", options=["1x (Realistic)", "5x (Fast Demo)", "10x (Sprint)"], value="5x (Fast Demo)")

# Compute ML Prediction
pred_result = model.predict(
    current_level=current_water_level,
    rainfall_intensity=rainfall_intensity,
    soil_saturation=soil_saturation
)

# Apply drainage penalty/adjustment
adjusted_prediction = round(pred_result['predicted_level'] + drainage_penalty, 2)
adjusted_delta = round(adjusted_prediction - current_water_level, 2)

if adjusted_prediction >= CRITICAL_WATER_LEVEL:
    status_class = "status-critical"
    status_icon = "🚨"
    risk_label = "CRITICAL FLASH FLOOD RISK"
    rec_action = "IMMEDIATE: Trigger audible park sirens, automated Department of Environment (DOE) webhook alert, and dispatch municipal flood-gate team."
elif adjusted_prediction >= ALERT_WATER_LEVEL:
    status_class = "status-alert"
    status_icon = "⚠️"
    risk_label = "ALERT — WATER SURGE IMMINENT"
    rec_action = "ADVISORY: Open auxiliary drainage sluice gate, deploy SMS notice to park rangers and campus security."
else:
    status_class = "status-normal"
    status_icon = "✅"
    risk_label = "NORMAL — CONDITIONS SAFE"
    rec_action = "ROUTINE: Normal reservoir retention capacity. Edge sensors logging every 60s."

# --- MAIN DASHBOARD VIEW ---

# Status Light Banner
st.markdown(f"""
<div class="status-badge {status_class}">
    <span>{status_icon} Status: {risk_label}</span>
</div>
""", unsafe_allow_html=True)

# 4 Key Metric Cards
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.metric(
        label="Current Lake Water Level",
        value=f"{current_water_level:.2f} m",
        delta=f"Normal: {NORMAL_WATER_LEVEL:.2f}m",
        delta_color="off"
    )

with m_col2:
    st.metric(
        label="Rainfall Intensity",
        value=f"{rainfall_intensity:.1f} mm/h",
        delta="Monsoon Surge" if rainfall_intensity >= 60 else ("Moderate" if rainfall_intensity > 15 else "Dry"),
        delta_color="inverse" if rainfall_intensity >= 60 else "normal"
    )

with m_col3:
    st.metric(
        label="Predicted Level (+45 min)",
        value=f"{adjusted_prediction:.2f} m",
        delta=f"{'+' if adjusted_delta >= 0 else ''}{adjusted_delta:.2f} m",
        delta_color="inverse" if adjusted_delta > 0.20 else "normal"
    )

with m_col4:
    if pred_result['time_to_flood_min'] > 0:
        time_text = f"~{pred_result['time_to_flood_min']} mins"
    elif pred_result['time_to_flood_min'] == 0:
        time_text = "OVERTOPPING NOW"
    else:
        time_text = "Safe (> 3 hours)"
        
    st.metric(
        label="Est. Time to Overbank (3.50m)",
        value=time_text,
        delta="Critical Window" if (pred_result['time_to_flood_min'] >= 0 and pred_result['time_to_flood_min'] <= 45) else "Within Safe Limits",
        delta_color="inverse" if (pred_result['time_to_flood_min'] >= 0 and pred_result['time_to_flood_min'] <= 45) else "normal"
    )

st.markdown("")

# Visual reservoir cross-section & charts
c_left, c_right = st.columns([1.3, 1])

with c_left:
    st.subheader("📈 Hydrological Forecast & Hydrograph")
    
    # Generate past 4 hours + future 1 hour trajectory
    now = datetime.now()
    timestamps = [now - timedelta(minutes=15 * i) for i in range(16, 0, -1)]
    timestamps.append(now)
    future_timestamps = [now + timedelta(minutes=15 * i) for i in range(1, 5)]
    all_times = timestamps + future_timestamps
    
    # Simulate past hydrograph leading to current state
    history_water = []
    base = current_water_level - (rainfall_intensity * 0.004)
    for i in range(len(timestamps)):
        t_frac = i / len(timestamps)
        w = base + (current_water_level - base) * (t_frac ** 1.3) + np.random.normal(0, 0.01)
        history_water.append(round(w, 2))
    history_water[-1] = current_water_level
    
    # Future projections
    future_water = []
    for i in range(1, 5):
        interp_val = current_water_level + (adjusted_prediction - current_water_level) * (i / 4.0)
        future_water.append(round(interp_val, 2))
        
    combined_water = history_water + future_water
    types = ["Observed (IoT Telemetry)"] * len(timestamps) + ["ML Forecast (+45m)"] * len(future_timestamps)
    
    chart_df = pd.DataFrame({
        "Timestamp": all_times,
        "Water Level (m)": combined_water,
        "Type": types
    })
    
    # Altair time-series chart with threshold lines
    base_chart = alt.Chart(chart_df).encode(
        x=alt.X("Timestamp:T", title="Time (MYT)", axis=alt.Axis(format="%H:%M")),
        y=alt.Y("Water Level (m):Q", scale=alt.Scale(domain=[1.5, 4.0]), title="Water Level (meters)")
    )
    
    line = base_chart.mark_line(point=True, strokeWidth=3).encode(
        color=alt.Color("Type:N", scale=alt.Scale(domain=["Observed (IoT Telemetry)", "ML Forecast (+45m)"], range=["#38bdf8", "#f43f5e"]))
    )
    
    # Warning Threshold (3.0m)
    rule_alert = alt.Chart(pd.DataFrame({'y': [ALERT_WATER_LEVEL]})).mark_rule(
        color='#F59E0B', strokeDash=[4, 4], strokeWidth=2
    ).encode(y='y:Q')
    
    # Critical Flood Threshold (3.5m)
    rule_critical = alt.Chart(pd.DataFrame({'y': [CRITICAL_WATER_LEVEL]})).mark_rule(
        color='#EF4444', strokeDash=[6, 3], strokeWidth=2
    ).encode(y='y:Q')
    
    final_chart = (line + rule_alert + rule_critical).properties(height=340).interactive()
    st.altair_chart(final_chart, use_container_width=True)
    
    st.caption("🟡 Yellow Dashed: Sluice Gate Alert (3.00m) | 🔴 Red Dashed: Overbank Flash-Flood Embankment Breach (3.50m)")

with c_right:
    st.subheader("🏞️ 4-Acre Lake Reservoir Gauge")
    
    # Visual depth gauge progress bar
    fill_pct = min(max((current_water_level - 1.5) / (CRITICAL_WATER_LEVEL - 1.5), 0.0), 1.0)
    
    st.markdown(f"""
    <div style="background: #1e293b; padding: 18px; border-radius: 12px; border: 1px solid #334155;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-weight: 600;">Lake Retention Capacity</span>
            <span style="font-weight: 700; color: {'#EF4444' if fill_pct > 0.85 else ('#F59E0B' if fill_pct > 0.7 else '#10B981')};">{int(fill_pct*100)}% Used</span>
        </div>
        <div style="width: 100%; background: #334155; border-radius: 8px; height: 26px; overflow: hidden; position: relative;">
            <div style="width: {fill_pct*100}%; background: {'linear-gradient(90deg, #10B981, #EF4444)' if fill_pct > 0.7 else '#0284c7'}; height: 100%; transition: width 0.5s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">
            <span>1.50m (Low)</span>
            <span>2.00m (Normal)</span>
            <span>3.00m (Alert)</span>
            <span>3.50m (Critical)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🛡️ Autonomous Response Protocol")
    st.markdown(f"""
    <div class="alert-card">
        <b>System Recommendation:</b><br/>
        {rec_action}
    </div>
    """, unsafe_allow_html=True)
    
    # Store-and-forward indicator for resilient edge telemetry
    st.markdown("#### 📡 Edge Connectivity & Store-and-Forward")
    c_edge1, c_edge2 = st.columns(2)
    with c_edge1:
        st.write("• **Local SQLite Cache**: Active (0 backlog)")
    with c_edge2:
        st.write("• **Cloud MQTT Link**: Connected (QoS 1)")

st.markdown("---")

# Bottom Section: Active Alert Feed & Model Explainability
col_feed, col_explain = st.columns([1.1, 0.9])

with col_feed:
    st.subheader("🔔 Real-Time Incident & Alert Feed")
    
    now_str = datetime.now().strftime("%H:%M:%S")
    alerts_data = []
    
    if adjusted_prediction >= CRITICAL_WATER_LEVEL:
        alerts_data.append((now_str, "CRITICAL", f"Projected water level of {adjusted_prediction:.2f}m breaches safe embankment (3.50m). Automated Webhook triggered to Department of Environment (JAS)."))
        alerts_data.append(((datetime.now() - timedelta(minutes=3)).strftime("%H:%M:%S"), "ACTION", "Edge Actuator: Emergency spillway relief valve opened automatically via MQTT command."))
    elif adjusted_prediction >= ALERT_WATER_LEVEL:
        alerts_data.append((now_str, "WARNING", f"Predictive threshold warning: Inflow surge detected ({rainfall_intensity:.1f} mm/h). Projected lake depth {adjusted_prediction:.2f}m."))
        alerts_data.append(((datetime.now() - timedelta(minutes=6)).strftime("%H:%M:%S"), "ADVISORY", "SMS dispatch transmitted to Municipal Park Management and Facilities Team."))
    else:
        alerts_data.append((now_str, "INFO", f"Lake telemetry nominal: Level at {current_water_level:.2f}m. IoT ultrasonic echo quality 99.4%."))
        alerts_data.append(((datetime.now() - timedelta(minutes=15)).strftime("%H:%M:%S"), "HEARTBEAT", "ESP32 edge node heartbeat synced with cloud time-series database."))

    for a_time, a_type, a_msg in alerts_data:
        b_color = "#ef4444" if a_type == "CRITICAL" else ("#f59e0b" if a_type in ["WARNING", "ACTION"] else "#38bdf8")
        st.markdown(f"""
        <div style="background: #1e293b; border-left: 4px solid {b_color}; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">
            <span style="font-size: 0.8rem; color: #94a3b8;">[{a_time} MYT] <b>{a_type}</b></span>
            <div style="font-size: 0.92rem; margin-top: 3px;">{a_msg}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Download sample simulated data button
    export_df = pd.DataFrame({
        "timestamp": all_times,
        "water_level_m": combined_water,
        "rainfall_intensity_mm_h": [rainfall_intensity] * len(all_times),
        "risk_status": [risk_label] * len(all_times)
    })
    st.download_button(
        label="📥 Download Telemetry Log (CSV)",
        data=export_df.to_csv(index=False),
        file_name="malaysian_lake_flood_telemetry.csv",
        mime="text/csv"
    )

with col_explain:
    st.subheader("🧠 Machine Learning Model Evaluation")
    
    st.markdown("""
    **Model Architecture**: Random Forest Regressor trained on 2,500 tropical hydrology readings with 5 features:
    - *Current Lake Level*, *Rainfall Intensity*, *30-min Rolling Rain*, *Rate of Change*, *Soil Moisture*.
    """)
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    perf_col1.metric("Model R² Score", f"{model.metrics['r2_score']:.3f}", "High Accuracy")
    perf_col2.metric("RMSE Error", f"{model.metrics['rmse']:.3f} m", "< 4cm Precision")
    perf_col3.metric("Baseline R² (Linear)", f"{model.metrics['baseline_r2']:.3f}", "RF vs Linear")
    
    st.markdown("##### 📊 Feature Importance Breakdown")
    feat_df = pd.DataFrame({
        "Feature": [
            "Current Lake Depth",
            "Rainfall Intensity",
            "Rate of Change (Δh/Δt)",
            "30m Rolling Rain",
            "Catchment Soil Moisture"
        ],
        "Importance (%)": [
            round(model.metrics['feature_importances']['current_water_level'] * 100, 1),
            round(model.metrics['feature_importances']['rainfall_intensity'] * 100, 1),
            round(model.metrics['feature_importances']['water_level_roc'] * 100, 1),
            round(model.metrics['feature_importances']['rolling_rain_30m'] * 100, 1),
            round(model.metrics['feature_importances']['soil_saturation'] * 100, 1),
        ]
    })
    
    feat_chart = alt.Chart(feat_df).mark_bar(cornerRadius=4, color="#38bdf8").encode(
        x=alt.X("Importance (%):Q", title="Relative Weight (%)"),
        y=alt.Y("Feature:N", sort="-x", title="Feature")
    ).properties(height=160)
    
    st.altair_chart(feat_chart, use_container_width=True)

# Auto refresh in live simulation mode
if live_mode:
    sleep_time = 3 if "1x" in sim_speed else (1.2 if "5x" in sim_speed else 0.6)
    time.sleep(sleep_time)
    st.rerun()
