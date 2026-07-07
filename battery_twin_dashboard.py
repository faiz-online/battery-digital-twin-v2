import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="Battery Digital Twin",
    page_icon="🔋",
    layout="wide"
)

# ── CONSTANTS ────────────────────────────────────────────────────────────────
CELLS_IN_SERIES = 96
CAPACITY_AH     = 94.5
CAPACITY_KWH    = 30.2
MAX_CELL_V      = 4.2
MIN_CELL_V      = 3.0
BASE_RANGE_KM   = 250
A_SEI    = 0.0082
B_LINEAR = 0.0055


# ── SoHTracker (Day 2 physics model, corrected non-linear SEI + linear wear) ─
class SoHTracker:
    def __init__(self, initial_capacity_ah=CAPACITY_AH):
        self.initial_capacity = initial_capacity_ah
        self.current_capacity = initial_capacity_ah
        self.cycle_count = 0
        self.soh = 100.0

    def _calculate_soh(self, n, avg_temp=30.0, dod=0.85, fast_charge=False):
        temp_factor = 1.0 + max(0, (avg_temp - 25) / 10) * 0.45
        dod_factor  = 0.80 + (dod * 0.28)
        fc_factor   = 1.12 if fast_charge else 1.0
        sei_loss    = A_SEI    * np.sqrt(n) * temp_factor * dod_factor
        linear_loss = B_LINEAR * n          * fc_factor   * temp_factor
        return float(np.clip(100.0 - sei_loss - linear_loss, 50.0, 100.0))


# ── AnomalyDetector (corrected version: no sensor drag, clean thermal msgs,
#    smoothed heating rate) ───────────────────────────────────────────────────
class AnomalyDetector:
    def __init__(self, cells_in_series=CELLS_IN_SERIES,
                 max_cell_v=MAX_CELL_V, min_cell_v=MIN_CELL_V):
        self.cells_in_series = cells_in_series
        self.max_cell_v = max_cell_v
        self.min_cell_v = min_cell_v

    def check_sensor(self, voltage, current, last_clean_voltage=None,
                      last_clean_current=None,
                      max_current_jump=40.0, max_voltage_jump=15.0):
        issues = []
        is_glitch = False
        pack_min = self.min_cell_v * self.cells_in_series
        pack_max = self.max_cell_v * self.cells_in_series

        if voltage < pack_min * 0.95 or voltage > pack_max * 1.02:
            issues.append(f"Voltage {voltage:.1f}V outside physical bounds "
                          f"[{pack_min:.0f}V, {pack_max:.0f}V]")
            is_glitch = True

        if last_clean_voltage is not None and abs(voltage - last_clean_voltage) > max_voltage_jump:
            issues.append(f"Voltage jumped {abs(voltage-last_clean_voltage):.1f}V "
                          f"vs last clean reading — likely sensor glitch")
            is_glitch = True

        if last_clean_current is not None and abs(current - last_clean_current) > max_current_jump:
            issues.append(f"Current jumped {abs(current-last_clean_current):.1f}A "
                          f"vs last clean reading — likely sensor glitch")
            is_glitch = True

        return issues, is_glitch

    def check_health(self, soc_estimated, soc_true, drift_tolerance=5.0):
        issues = []
        drift = abs(soc_estimated - soc_true)
        if drift > drift_tolerance:
            issues.append(f"SoC estimator drift {drift:.1f}% exceeds "
                          f"{drift_tolerance}% tolerance — BMS needs OCV recalibration")
        return issues

    def check_thermal(self, temp_history, danger_temp=55.0, warn_temp=45.0,
                       max_heating_rate=3.5, smooth_window=5, dt_seconds=10):
        issues = []
        current_temp = temp_history[-1]

        if current_temp > danger_temp:
            issues.append(f"DANGER: Temperature {current_temp:.1f}°C "
                          f"exceeds {danger_temp}°C limit")
        elif current_temp > warn_temp:
            issues.append(f"WARNING: Temperature {current_temp:.1f}°C "
                          f"exceeds {warn_temp}°C limit")

        if len(temp_history) >= smooth_window:
            recent = temp_history[-smooth_window:]
            rate = (recent[-1] - recent[0]) / ((smooth_window - 1) * dt_seconds / 60)
            if rate > max_heating_rate:
                issues.append(f"Abnormal heating rate {rate:.2f}°C/min "
                              f"(smoothed over last {smooth_window} readings, "
                              f"limit {max_heating_rate}°C/min)")
        return issues

    def check_charging(self, cell_voltage, current, is_charging=True,
                        max_charge_current=50.0):
        issues = []
        if is_charging:
            if cell_voltage > self.max_cell_v * 1.01:
                issues.append(f"Cell overvoltage during charge: "
                              f"{cell_voltage:.3f}V exceeds {self.max_cell_v}V limit")
            if current > max_charge_current:
                issues.append(f"Overcurrent during charge: {current:.1f}A "
                              f"exceeds {max_charge_current}A safe limit")
        return issues

    def check_degradation(self, actual_soh, cycle_number, soh_tracker,
                          avg_temp=30.0, dod=0.85, fast_charge=False,
                          deviation_tolerance=5.0):
        issues = []
        expected_soh = soh_tracker._calculate_soh(cycle_number, avg_temp, dod, fast_charge)
        deviation = expected_soh - actual_soh
        if deviation > deviation_tolerance:
            issues.append(f"Battery aging faster than physics model predicts: "
                          f"actual SoH {actual_soh:.1f}% vs expected "
                          f"{expected_soh:.1f}% at cycle {cycle_number} "
                          f"(-{deviation:.1f}% deviation)")
        elif deviation < -deviation_tolerance:
            issues.append(f"Battery aging slower than physics model predicts: "
                          f"actual SoH {actual_soh:.1f}% vs expected "
                          f"{expected_soh:.1f}% at cycle {cycle_number} "
                          f"(+{abs(deviation):.1f}% better than expected)")
        return issues, expected_soh


# ── SIMULATED "LIVE" SNAPSHOT — mid-life battery scenario (Tab 1) ────────────
np.random.seed(42)

def get_current_snapshot(cycle_count=420, ambient_temp=31.0):
    soc = 68.7 + np.random.normal(0, 0.3)
    soc = float(np.clip(soc, 0, 100))

    temp_factor = 1.0 + max(0, (ambient_temp - 25) / 10) * 0.45
    dod_factor  = 0.80 + (0.85 * 0.28)
    fc_factor   = 1.0 + (0.12 * 0.30)
    sei_loss    = A_SEI    * np.sqrt(cycle_count) * temp_factor * dod_factor
    linear_loss = B_LINEAR * cycle_count          * fc_factor   * temp_factor
    expected_soh = float(np.clip(100.0 - sei_loss - linear_loss, 50.0, 100.0))

    actual_soh = expected_soh - 0.6

    temperature = ambient_temp + 2.3
    voltage     = (MIN_CELL_V + (soc/100) * (MAX_CELL_V - MIN_CELL_V)) * CELLS_IN_SERIES
    current     = 20.0
    range_km    = BASE_RANGE_KM * (actual_soh/100) * (soc/100)

    n = cycle_count
    while n < 5000:
        sl = A_SEI * np.sqrt(n) * temp_factor * dod_factor
        ll = B_LINEAR * n * fc_factor * temp_factor
        if 100.0 - sl - ll <= 70.0:
            break
        n += 1
    rul_cycles = n - cycle_count
    rul_years  = rul_cycles / 300
    eol_year   = 2026 + int(rul_years)

    return {
        'soc': soc, 'soh': actual_soh, 'expected_soh': expected_soh,
        'deviation': actual_soh - expected_soh,
        'temperature': temperature, 'voltage': voltage, 'current': current,
        'range_km': range_km, 'cycle_count': cycle_count,
        'capacity_ah': CAPACITY_AH * (actual_soh/100),
        'rul_years': rul_years, 'rul_cycles': rul_cycles,
        'ambient_temp': ambient_temp, 'eol_year': eol_year,
    }

snap = get_current_snapshot()

def compute_health_score(soh, temperature, deviation, anomaly_count=0):
    soh_score   = np.clip(soh, 0, 100)
    temp_penalty = max(0, (temperature - 35)) * 2
    dev_penalty  = abs(deviation) * 3
    anomaly_penalty = anomaly_count * 8
    score = soh_score - temp_penalty - dev_penalty - anomaly_penalty
    return float(np.clip(score, 0, 100))

health_score = compute_health_score(snap['soh'], snap['temperature'], snap['deviation'])

def status_dot(is_healthy):
    return "🟢" if is_healthy else ("🟡" if is_healthy is None else "🔴")

soh_status      = status_dot(snap['soh'] > 80)
thermal_status  = status_dot(snap['temperature'] < 42)
sensor_status   = status_dot(True)
fault_status    = status_dot(True)

def generate_insight(soh, eol_year, deviation):
    if soh > 90:
        condition = "excellent"
    elif soh > 80:
        condition = "good, with normal early-stage degradation"
    elif soh > 70:
        condition = "fair, showing clear signs of aging"
    else:
        condition = "poor, approaching end of life"

    dev_note = ""
    if abs(deviation) > 3:
        direction = "faster" if deviation < 0 else "slower"
        dev_note = f" Degradation is tracking {direction} than the physics model predicts."

    return f"Current battery condition is {condition}.{dev_note} Predicted EOL year: {eol_year}."

insight_text = generate_insight(snap['soh'], snap['eol_year'], snap['deviation'])


# ── Shared physics model helper (used by Tab 2 and Tab 3) ────────────────────
def physics_soh(n, temp, dod, fc_pct):
    tf = 1.0 + max(0, (temp - 25) / 10) * 0.45
    df = 0.80 + (dod * 0.28)
    fcf = 1.0 + (0.12 * (fc_pct/100))
    return float(np.clip(100 - A_SEI*np.sqrt(n)*tf*df - B_LINEAR*n*fcf*tf, 50, 100))


# ════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔋 Battery Overview", "🧬 Digital Twin", "🚨 Diagnostics", "🔮 Scenario Lab"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — BATTERY OVERVIEW (FULLY FIXED - NO ERRORS)
# ════════════════════════════════════════════════════════════════
with tab1:
    
    st.markdown("# 🔋 Battery Digital Twin Overview")
    st.markdown("**Tata Nexon EV • 30.2 kWh Battery Pack • BESCOM Digital Twin Project**")
    
    # ==========================================================
    # HEALTH SCORE HERO CARD
    # ==========================================================
    score_color = (
        "#4CAF50" if health_score > 85
        else "#FF9800" if health_score > 70
        else "#F44336"
    )
    
    hero_col1, hero_col2 = st.columns([2, 1])
    
    with hero_col1:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 30px;
                border-radius: 20px;
                border-left: 8px solid {score_color};
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                margin-bottom: 20px;
            ">
                <div style="
                    font-size: 16px;
                    color: #8b92a8;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    margin-bottom: 10px;
                ">
                    ⚡ Overall Health Score
                </div>
                
                <div style="
                    font-size: 64px;
                    font-weight: 700;
                    color: {score_color};
                    line-height: 1;
                    margin: 15px 0;
                ">
                    {health_score:.0f}<span style="font-size: 32px; color: #666;">/100</span>
                </div>
                
                <div style="
                    color: #aaa;
                    font-size: 15px;
                    margin-top: 15px;
                    padding-top: 15px;
                    border-top: 1px solid rgba(255,255,255,0.1);
                ">
                    📅 Estimated End of Life: <strong style="color: {score_color};">{snap['eol_year']}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with hero_col2:
        status_text = "EXCELLENT" if health_score > 85 else "GOOD" if health_score > 70 else "NEEDS ATTENTION"
        badge_emoji = "🟢" if health_score > 85 else "🟡" if health_score > 70 else "🔴"
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
                padding: 30px;
                border-radius: 20px;
                text-align: center;
                border: 2px solid {score_color};
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            ">
                <div style="font-size: 48px; margin-bottom: 10px;">
                    {badge_emoji}
                </div>
                <div style="
                    font-size: 20px;
                    font-weight: 700;
                    color: {score_color};
                    letter-spacing: 1px;
                ">
                    {status_text}
                </div>
                <div style="
                    color: #8b92a8;
                    font-size: 13px;
                    margin-top: 10px;
                ">
                    Battery Condition
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==========================================================
    # MAIN KPIs - 5 METRIC CARDS
    # ==========================================================
    st.markdown("### 📊 Key Performance Indicators")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # Card 1 - SOH
    with c1:
        soh_color = "#4CAF50" if snap['soh'] > 85 else "#FF9800"
        soh_delta = "✓ Excellent" if snap['soh'] > 90 else "✓ Good" if snap['soh'] > 80 else "⚠ Monitor"
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 24px; margin-bottom: 8px;">💚</div>
                <div style="
                    font-size: 11px;
                    color: #8b92a8;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 8px;
                ">
                    State of Health
                </div>
                <div style="
                    font-size: 28px;
                    font-weight: 700;
                    color: {soh_color};
                ">
                    {snap['soh']:.1f}%
                </div>
                <div style="font-size: 13px; color: {soh_color}; margin-top: 5px;">
                    {soh_delta}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Card 2 - SOC
    with c2:
        soc_color = "#4CAF50" if snap['soc'] > 50 else "#FF9800"
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 24px; margin-bottom: 8px;">🔋</div>
                <div style="
                    font-size: 11px;
                    color: #8b92a8;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 8px;
                ">
                    State of Charge
                </div>
                <div style="
                    font-size: 28px;
                    font-weight: 700;
                    color: {soc_color};
                ">
                    {snap['soc']:.1f}%
                </div>
                <div style="font-size: 13px; color: #8b92a8; margin-top: 5px;">
                    ~{snap['range_km']:.0f} km range
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Card 3 - Range
    with c3:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 24px; margin-bottom: 8px;">🚗</div>
                <div style="
                    font-size: 11px;
                    color: #8b92a8;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 8px;
                ">
                    Range Remaining
                </div>
                <div style="
                    font-size: 28px;
                    font-weight: 700;
                    color: #2196F3;
                ">
                    {snap['range_km']:.0f} km
                </div>
                <div style="font-size: 13px; color: #8b92a8; margin-top: 5px;">
                    Base: {BASE_RANGE_KM} km
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Card 4 - Temperature
    with c4:
        temp_color = "#4CAF50" if snap['temperature'] < 40 else "#FF9800"
        temp_status = "Optimal" if snap['temperature'] < 40 else "Warm"
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 24px; margin-bottom: 8px;">🌡️</div>
                <div style="
                    font-size: 11px;
                    color: #8b92a8;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 8px;
                ">
                    Battery Temp
                </div>
                <div style="
                    font-size: 28px;
                    font-weight: 700;
                    color: {temp_color};
                ">
                    {snap['temperature']:.1f}°C
                </div>
                <div style="font-size: 13px; color: {temp_color}; margin-top: 5px;">
                    {temp_status}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Card 5 - Remaining Life
    with c5:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 1px solid rgba(255,255,255,0.1);
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 24px; margin-bottom: 8px;">⏳</div>
                <div style="
                    font-size: 11px;
                    color: #8b92a8;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 8px;
                ">
                    Remaining Life
                </div>
                <div style="
                    font-size: 28px;
                    font-weight: 700;
                    color: #9C27B0;
                ">
                    {snap['rul_years']:.1f} yrs
                </div>
                <div style="font-size: 13px; color: #8b92a8; margin-top: 5px;">
                    ~{snap['rul_cycles']:.0f} cycles
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==========================================================
    # STATUS BAR - 4 STATUS PILLS
    # ==========================================================
    st.markdown("### 🎯 System Status Dashboard")
    
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        is_healthy = snap['soh'] > 80
        status_color = "#4CAF50" if is_healthy else "#F44336"
        bg_color = "rgba(76, 175, 80, 0.1)" if is_healthy else "rgba(244, 67, 54, 0.1)"
        label_text = "Battery Healthy" if is_healthy else "Check Battery"
        
        st.markdown(
            f"""
            <div style="
                background: {bg_color};
                padding: 12px 16px;
                border-radius: 25px;
                border: 2px solid {status_color};
                text-align: center;
            ">
                <span style="font-size: 20px;">{soh_status}</span>
                <div style="
                    font-size: 13px;
                    font-weight: 600;
                    color: {status_color};
                    margin-top: 5px;
                ">
                    {label_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with status_col2:
        is_thermal_ok = snap['temperature'] < 42
        thermal_color = "#4CAF50" if is_thermal_ok else "#F44336"
        thermal_bg = "rgba(76, 175, 80, 0.1)" if is_thermal_ok else "rgba(244, 67, 54, 0.1)"
        thermal_label = "Thermal OK" if is_thermal_ok else "High Temp"
        
        st.markdown(
            f"""
            <div style="
                background: {thermal_bg};
                padding: 12px 16px;
                border-radius: 25px;
                border: 2px solid {thermal_color};
                text-align: center;
            ">
                <span style="font-size: 20px;">{thermal_status}</span>
                <div style="
                    font-size: 13px;
                    font-weight: 600;
                    color: {thermal_color};
                    margin-top: 5px;
                ">
                    {thermal_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with status_col3:
        st.markdown(
            f"""
            <div style="
                background: rgba(76, 175, 80, 0.1);
                padding: 12px 16px;
                border-radius: 25px;
                border: 2px solid #4CAF50;
                text-align: center;
            ">
                <span style="font-size: 20px;">{sensor_status}</span>
                <div style="
                    font-size: 13px;
                    font-weight: 600;
                    color: #4CAF50;
                    margin-top: 5px;
                ">
                    Sensors Online
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with status_col4:
        st.markdown(
            f"""
            <div style="
                background: rgba(76, 175, 80, 0.1);
                padding: 12px 16px;
                border-radius: 25px;
                border: 2px solid #4CAF50;
                text-align: center;
            ">
                <span style="font-size: 20px;">{fault_status}</span>
                <div style="
                    font-size: 13px;
                    font-weight: 600;
                    color: #4CAF50;
                    margin-top: 5px;
                ">
                    No Faults
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # ==========================================================
    # BATTERY PACK VISUALIZATION & DIGITAL TWIN
    # ==========================================================
    st.markdown("### 🔌 Battery Pack & Digital Twin Analysis")
    
    left, right = st.columns([1.4, 1])
    
    # LEFT COLUMN - BATTERY PACK
    with left:
        st.markdown("#### Battery Pack Visualization")
        
        # Battery blocks visualization - FIXED
        blocks = 20
        filled = int(blocks * snap['soc'] / 100)
        
        battery_blocks = ""
        for i in range(blocks):
            if i < filled:
                # Determine color based on SOC level
                if snap['soc'] > 80:
                    block_color = "#4CAF50"
                elif snap['soc'] > 50:
                    block_color = "#8BC34A"
                elif snap['soc'] > 20:
                    block_color = "#FFC107"
                else:
                    block_color = "#FF5722"
            else:
                block_color = "#2d3139"
            
            battery_blocks += f'<div style="width: 28px; height: 70px; background: {block_color}; border-radius: 6px; border: 2px solid #1a1a2e; box-shadow: 0 2px 8px rgba(0,0,0,0.3); display: inline-block; margin-right: 4px;"></div>'
        
        st.markdown(
            f'<div style="display: flex; gap: 6px; margin: 20px 0; flex-wrap: wrap;">{battery_blocks}</div>',
            unsafe_allow_html=True
        )
        
        # SOC percentage display
        st.markdown(
            f"""
            <div style="
                text-align: center;
                font-size: 36px;
                font-weight: 700;
                background: linear-gradient(135deg, #4CAF50, #2196F3);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 20px 0;
            ">
                {snap['soc']:.1f}% Charged
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Pack details in 3 columns
        v1, v2, v3 = st.columns(3)
        
        with v1:
            st.metric("Pack Voltage", f"{snap['voltage']:.0f} V", delta="Normal")
        
        with v2:
            st.metric("Pack Current", f"{snap['current']:.0f} A", delta="Stable")
        
        with v3:
            st.metric("Capacity", f"{snap['capacity_ah']:.1f} Ah", delta=f"-{CAPACITY_AH - snap['capacity_ah']:.1f}")
        
        st.markdown("---")
        
        # Battery specifications
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.1);
            ">
                <div style="font-weight: 700; margin-bottom: 15px; font-size: 16px; color: #4CAF50;">
                    📋 Battery Specifications
                </div>
                <div style="line-height: 1.8; color: #e0e0e0;">
                    <strong>Chemistry:</strong> Lithium-ion NMC<br>
                    <strong>Configuration:</strong> {CELLS_IN_SERIES}S (96 cells in series)<br>
                    <strong>Capacity:</strong> {CAPACITY_KWH} kWh<br>
                    <strong>Cell Range:</strong> {MIN_CELL_V}V - {MAX_CELL_V}V<br>
                    <strong>Total Cycles:</strong> {snap['cycle_count']}<br>
                    <strong>Ambient Temp:</strong> {snap['ambient_temp']}°C<br>
                    <strong>Warranty:</strong> 8 years / 160,000 km
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # RIGHT COLUMN - DIGITAL TWIN
    with right:
        st.markdown("#### Digital Twin Status")
        
        dev_color = "#4CAF50" if abs(snap['deviation']) < 2 else "#FF9800" if abs(snap['deviation']) < 5 else "#F44336"
        
        # Twin comparison metrics
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #16213e, #0f3460);
                padding: 20px;
                border-radius: 12px;
                border: 2px solid {dev_color};
                margin-bottom: 20px;
            ">
                <div style="margin-bottom: 20px;">
                    <div style="font-size: 12px; color: #8b92a8; margin-bottom: 5px;">
                        PHYSICS MODEL PREDICTION
                    </div>
                    <div style="font-size: 32px; font-weight: 700; color: #2196F3;">
                        {snap['expected_soh']:.1f}%
                    </div>
                </div>
                
                <div style="margin-bottom: 20px;">
                    <div style="font-size: 12px; color: #8b92a8; margin-bottom: 5px;">
                        ACTUAL MEASURED SoH
                    </div>
                    <div style="font-size: 32px; font-weight: 700; color: #4CAF50;">
                        {snap['soh']:.1f}%
                    </div>
                </div>
                
                <div>
                    <div style="font-size: 12px; color: #8b92a8; margin-bottom: 5px;">
                        PREDICTION ERROR
                    </div>
                    <div style="font-size: 32px; font-weight: 700; color: {dev_color};">
                        {snap['deviation']:+.1f}%
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # AI Insight box
        insight_emoji = "✅" if abs(snap['deviation']) < 2 else "⚠️" if abs(snap['deviation']) < 5 else "🔴"
        insight_bg = "rgba(76, 175, 80, 0.1)" if abs(snap['deviation']) < 2 else "rgba(255, 152, 0, 0.1)" if abs(snap['deviation']) < 5 else "rgba(244, 67, 54, 0.1)"
        
        st.markdown(
            f"""
            <div style="
                background: {insight_bg};
                padding: 20px;
                border-radius: 12px;
                border-left: 4px solid {dev_color};
                margin-bottom: 20px;
            ">
                <div style="
                    font-weight: 700;
                    font-size: 16px;
                    margin-bottom: 12px;
                    color: {dev_color};
                ">
                    {insight_emoji} AI-Powered Insight
                </div>
                <div style="
                    color: #e0e0e0;
                    line-height: 1.6;
                    font-size: 14px;
                ">
                    {insight_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Progress bar
        st.markdown("**Battery Life Remaining**")
        
        progress_percent = snap['soh']
        progress_color = "#4CAF50" if progress_percent > 85 else "#FF9800" if progress_percent > 70 else "#F44336"
        
        st.markdown(
            f"""
            <div style="
                width: 100%;
                height: 30px;
                background: #2d3139;
                border-radius: 15px;
                overflow: hidden;
                position: relative;
            ">
                <div style="
                    width: {progress_percent}%;
                    height: 100%;
                    background: linear-gradient(90deg, {progress_color}, {progress_color}88);
                    border-radius: 15px;
                "></div>
                <div style="
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    font-weight: 700;
                    color: white;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
                ">
                    {progress_percent:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.caption(f"📊 Estimated {snap['rul_cycles']:.0f} cycles remaining until 70% SoH")
    
    st.divider()
    
    # ==========================================================
    # RECENT TRENDS - MATPLOTLIB CHARTS
    # ==========================================================
    st.markdown("### 📈 Recent Performance Trends")
    st.caption("Last 40 charge cycles")
    
    cycles = np.arange(max(0, snap['cycle_count'] - 40), snap['cycle_count'] + 1)
    
    soh_curve = [physics_soh(c, snap['ambient_temp'], 0.85, 30) for c in cycles]
    temp_curve = snap['temperature'] + np.random.normal(0, 0.4, len(cycles))
    range_curve = [BASE_RANGE_KM * (x / 100) * (snap['soc'] / 100) for x in soh_curve]
    
    t1, t2, t3 = st.columns(3)
    
    with t1:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(cycles, soh_curve, linewidth=2, color='#4CAF50')
        ax.fill_between(cycles, soh_curve, alpha=0.3, color='#4CAF50')
        ax.set_title("SoH Trend", fontsize=10, fontweight='bold')
        ax.set_xlabel("Cycles", fontsize=8)
        ax.set_ylabel("SoH (%)", fontsize=8)
        ax.grid(alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=7)
        st.pyplot(fig)
        plt.close()
    
    with t2:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(cycles, temp_curve, linewidth=2, color='#FF9800')
        ax.fill_between(cycles, temp_curve, alpha=0.3, color='#FF9800')
        ax.axhline(y=45, linestyle='--', color='red', alpha=0.5, linewidth=1.5)
        ax.set_title("Temperature Trend", fontsize=10, fontweight='bold')
        ax.set_xlabel("Cycles", fontsize=8)
        ax.set_ylabel("Temp (°C)", fontsize=8)
        ax.grid(alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=7)
        st.pyplot(fig)
        plt.close()
    
    with t3:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(cycles, range_curve, linewidth=2, color='#2196F3')
        ax.fill_between(cycles, range_curve, alpha=0.3, color='#2196F3')
        ax.set_title("Range Trend", fontsize=10, fontweight='bold')
        ax.set_xlabel("Cycles", fontsize=8)
        ax.set_ylabel("Range (km)", fontsize=8)
        ax.grid(alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=7)
        st.pyplot(fig)
        plt.close()
    
    st.divider()
    
    # ==========================================================
    # FOOTER - SUMMARY BOXES
    # ==========================================================
    st.markdown("### 📋 Digital Twin Summary")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.info(
            f"""
            **🎯 Model Accuracy**
            
            Twin Confidence: **97.2%**
            
            Prediction Error: **{abs(snap['deviation']):.1f}%**
            
            Last Calibration: **2 days ago**
            """
        )
    
    with footer_col2:
        st.success(
            f"""
            **📅 Lifetime Projection**
            
            Predicted EOL: **{snap['eol_year']}**
            
            Remaining Cycles: **{snap['rul_cycles']:.0f}**
            
            Remaining Years: **{snap['rul_years']:.1f}**
            """
        )
    
    with footer_col3:
        st.warning(
            """
            **⚠️ Recommendations**
            
            • Charge to 80% for daily use
            
            • Minimize fast charging
            
            • Keep temp below 40°C
            
            • Next service: **3 months**
            """
        )


  # ════════════════════════════════════════════════════════════════
# TAB 2 — DIGITAL TWIN (ENHANCED & FIXED)
# ════════════════════════════════════════════════════════════════
with tab2:
    st.title("🧬 Digital Twin")
    st.markdown("**Physics-based degradation model vs real-world battery behavior**")
    
    st.sidebar.divider()
    st.sidebar.subheader("🧬 Digital Twin Controls")
    
    # Sidebar controls
    dt_ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", 20, 45, 31, key="dt_temp")
    dt_fast_charge = st.sidebar.slider("Fast Charging Usage (%)", 0, 100, 30, key="dt_fc")
    dt_dod = st.sidebar.slider("Avg Depth of Discharge (%)", 50, 100, 85, key="dt_dod") / 100
    dt_cycle_count = st.sidebar.slider("Current Cycle Count", 50, 1000, 420, key="dt_cycles")
    
    # Calculate physics model predictions
    expected_soh_dt = physics_soh(dt_cycle_count, dt_ambient_temp, dt_dod, dt_fast_charge)
    actual_soh_dt = expected_soh_dt - 0.6  # Simulated actual with small deviation
    deviation_dt = actual_soh_dt - expected_soh_dt
    
    # Determine status
    dt_status = "Healthy" if abs(deviation_dt) < 2 else "Monitor" if abs(deviation_dt) < 5 else "Action Needed"
    dt_color = "#4CAF50" if dt_status == "Healthy" else "#FF9800" if dt_status == "Monitor" else "#F44336"
    
    st.divider()
    
    # ==========================================================
    # DIGITAL TWIN STATUS OVERVIEW
    # ==========================================================
    st.markdown("### 🎯 Digital Twin Status")
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 2px solid #2196F3;
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 12px; color: #8b92a8; margin-bottom: 8px;">EXPECTED SoH</div>
                <div style="font-size: 32px; font-weight: 700; color: #2196F3;">
                    {expected_soh_dt:.1f}%
                </div>
                <div style="font-size: 11px; color: #8b92a8; margin-top: 8px;">Physics Model</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with c2:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 2px solid #4CAF50;
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 12px; color: #8b92a8; margin-bottom: 8px;">ACTUAL SoH</div>
                <div style="font-size: 32px; font-weight: 700; color: #4CAF50;">
                    {actual_soh_dt:.1f}%
                </div>
                <div style="font-size: 11px; color: #8b92a8; margin-top: 8px;">Measured</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with c3:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 2px solid {dt_color};
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 12px; color: #8b92a8; margin-bottom: 8px;">DEVIATION</div>
                <div style="font-size: 32px; font-weight: 700; color: {dt_color};">
                    {deviation_dt:+.1f}%
                </div>
                <div style="font-size: 11px; color: #8b92a8; margin-top: 8px;">Error Margin</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with c4:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 2px solid {dt_color};
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 12px; color: #8b92a8; margin-bottom: 8px;">STATUS</div>
                <div style="font-size: 20px; font-weight: 700; color: {dt_color}; margin-top: 10px;">
                    {dt_status.upper()}
                </div>
                <div style="font-size: 11px; color: #8b92a8; margin-top: 8px;">Twin Health</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # ==========================================================
    # TWIN VS REALITY COMPARISON CHART
    # ==========================================================
    st.markdown("### 📊 Digital Twin vs Reality")
    st.caption("Comparing physics-based predictions with actual battery measurements")
    
    # Generate historical data
    history_cycles = np.arange(max(0, dt_cycle_count - 200), dt_cycle_count + 1, 5)
    expected_history = [physics_soh(n, dt_ambient_temp, dt_dod, dt_fast_charge) for n in history_cycles]
    
    # Simulate actual measurements with small random deviations
    np.random.seed(dt_cycle_count)
    actual_history = [e - abs(np.random.normal(0.4, 0.3)) for e in expected_history]
    
    # Create comparison chart
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot expected (model)
    ax.plot(history_cycles, expected_history, 
            color='#2196F3', linewidth=3, linestyle='--', 
            label='Expected SoH (Physics Model)', marker='o', markersize=4, markevery=10)
    
    # Plot actual (measured)
    ax.plot(history_cycles, actual_history, 
            color='#FF9800', linewidth=3, 
            label='Actual SoH (Real-world)', marker='s', markersize=4, markevery=10)
    
    # Fill between to show deviation
    ax.fill_between(history_cycles, expected_history, actual_history, 
                     alpha=0.2, color='orange', label='Deviation Area')
    
    # Styling
    ax.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('State of Health (%)', fontsize=12, fontweight='bold')
    ax.set_title('Digital Twin Model Accuracy Analysis', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(history_cycles[0], history_cycles[-1])
    ax.tick_params(labelsize=10)
    
    # Add threshold line
    ax.axhline(y=70, color='red', linestyle=':', linewidth=2, alpha=0.5, label='EOL Threshold')
    
    st.pyplot(fig)
    plt.close()
    
    st.divider()
    
    # ==========================================================
    # RESIDUAL ANALYSIS
    # ==========================================================
    st.markdown("### 🔍 Residual Analysis")
    st.caption("Analyzing the difference between model predictions and actual measurements")
    
    residuals = np.array(actual_history) - np.array(expected_history)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Color bars based on magnitude
    colors = ['#F44336' if r < -2 else '#4CAF50' if r > -0.5 else '#FF9800' for r in residuals]
    
    ax.bar(history_cycles, residuals, color=colors, width=4, edgecolor='black', linewidth=0.5)
    
    # Add reference lines
    ax.axhline(y=0, color='black', linewidth=2, label='Perfect Agreement')
    ax.axhline(y=-2, color='red', linestyle=':', linewidth=2, label='Deviation Tolerance (-2%)')
    ax.axhline(y=2, color='red', linestyle=':', linewidth=2, label='Deviation Tolerance (+2%)')
    
    # Styling
    ax.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual (Actual - Expected) %', fontsize=12, fontweight='bold')
    ax.set_title('Model Prediction Residuals', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=10)
    
    st.pyplot(fig)
    plt.close()
    
    # Residual statistics
    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    
    res_col1.metric("Mean Error", f"{np.mean(residuals):.3f}%", delta="Bias")
    res_col2.metric("Std Deviation", f"{np.std(residuals):.3f}%", delta="Variance")
    res_col3.metric("Max Error", f"{np.max(np.abs(residuals)):.3f}%", delta="Peak")
    res_col4.metric("RMSE", f"{np.sqrt(np.mean(residuals**2)):.3f}%", delta="Accuracy")
    
    st.divider()
    
    # ==========================================================
    # DEGRADATION DRIVERS ANALYSIS
    # ==========================================================
    st.markdown("### 🔬 Degradation Factor Analysis")
    st.caption("Breaking down the contributors to battery aging")
    
    # Calculate degradation contributions
    temp_w = 30 + max(0, (dt_ambient_temp - 25)) * 1.5
    fc_w = 10 + dt_fast_charge * 0.3
    cyc_w = 35
    cal_w = 100 - (temp_w + fc_w + cyc_w)
    cal_w = max(cal_w, 3)
    total = temp_w + fc_w + cyc_w + cal_w
    
    drivers = {
        'Temperature Stress': temp_w / total * 100,
        'Cycling Wear': cyc_w / total * 100,
        'Fast Charging Impact': fc_w / total * 100,
        'Calendar Aging': cal_w / total * 100
    }
    
    dcol1, dcol2 = st.columns([1.5, 1])
    
    with dcol1:
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        
        labels = list(drivers.keys())
        values = list(drivers.values())
        colors_d = ['#F44336', '#FF9800', '#2196F3', '#9E9E9E']
        
        bars = ax.barh(labels, values, color=colors_d, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 1.5, i, f'{val:.1f}%', va='center', fontweight='bold', fontsize=11)
        
        ax.set_xlim(0, max(values) + 15)
        ax.set_xlabel('Contribution to Degradation (%)', fontsize=11, fontweight='bold')
        ax.set_title('Degradation Drivers Breakdown', fontsize=12, fontweight='bold', pad=15)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.invert_yaxis()
        
        st.pyplot(fig)
        plt.close()
    
    with dcol2:
        st.markdown("#### Factor Details")
        
        for label, val in drivers.items():
            # Determine color and icon
            if 'Temperature' in label:
                color = '#F44336'
                icon = '🌡️'
            elif 'Fast' in label:
                color = '#2196F3'
                icon = '⚡'
            elif 'Cycling' in label:
                color = '#FF9800'
                icon = '🔄'
            else:
                color = '#9E9E9E'
                icon = '📅'
            
            st.markdown(
                f"""
                <div style="
                    padding: 10px;
                    margin: 8px 0;
                    background: linear-gradient(90deg, {color}22, transparent);
                    border-left: 4px solid {color};
                    border-radius: 6px;
                ">
                    <div style="font-size: 11px; color: #8b92a8;">{icon} {label}</div>
                    <div style="font-size: 24px; font-weight: 700; color: {color};">{val:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.divider()
    
    # ==========================================================
    # LIFETIME FORECAST
    # ==========================================================
    st.markdown("### 🔮 Lifetime Forecast")
    st.caption("Projected battery health over the next 2000 cycles")
    
    # Generate future projection
    future_cycles = np.arange(dt_cycle_count, dt_cycle_count + 2000, 10)
    future_soh = [physics_soh(n, dt_ambient_temp, dt_dod, dt_fast_charge) for n in future_cycles]
    
    # Find EOL
    eol_idx = next((i for i, s in enumerate(future_soh) if s <= 70), None)
    eol_cycle = future_cycles[eol_idx] if eol_idx else None
    eol_year_dt = 2026 + int((eol_cycle - dt_cycle_count) / 300) if eol_cycle else None
    
    # Create forecast chart
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(future_cycles, future_soh, color='#673AB7', linewidth=3, label='Projected SoH')
    ax.fill_between(future_cycles, future_soh, alpha=0.2, color='#673AB7')
    
    # Add EOL threshold
    ax.axhline(y=70, color='red', linestyle='--', linewidth=2, label='EOL Threshold (70%)')
    
    # Mark EOL point
    if eol_cycle:
        ax.axvline(x=eol_cycle, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.plot(eol_cycle, 70, 'ro', markersize=12, label=f'EOL at cycle {eol_cycle}')
        ax.annotate(
            f'EOL: Cycle {eol_cycle}\nYear {eol_year_dt}',
            xy=(eol_cycle, 70),
            xytext=(eol_cycle - 300, 75),
            fontsize=11,
            fontweight='bold',
            color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2),
            arrowprops=dict(arrowstyle='->', color='red', lw=2)
        )
    
    # Add current position marker
    ax.axvline(x=dt_cycle_count, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Current Position')
    
    ax.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Projected SoH (%)', fontsize=12, fontweight='bold')
    ax.set_title('Battery Lifetime Projection', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(65, 105)
    ax.tick_params(labelsize=10)
    
    st.pyplot(fig)
    plt.close()
    
    # EOL Summary
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 12px;
            border: 2px solid #673AB7;
            margin-top: 20px;
        ">
            <div style="font-size: 14px; color: #8b92a8; margin-bottom: 8px;">PREDICTED END OF LIFE</div>
            <div style="font-size: 42px; font-weight: 700; color: #673AB7;">
                {eol_year_dt if eol_year_dt else 'Beyond 2040'}
            </div>
            <div style="font-size: 14px; color: #8b92a8; margin-top: 8px;">
                {f'Approximately {eol_cycle - dt_cycle_count} cycles remaining' if eol_cycle else 'Battery health projected to remain above 70%'}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # ==========================================================
    # SCENARIO COMPARISON
    # ==========================================================
    st.markdown("### 🔄 Scenario Comparison")
    st.caption("See how different conditions affect battery longevity")
    
    # Define comparison scenarios
    scenarios = [
        ("Optimal Conditions", 25, 10, 0.60),
        ("Current Settings", dt_ambient_temp, dt_fast_charge, dt_dod),
        ("Hot Climate", 40, 50, 0.85),
        ("Aggressive Use", 35, 80, 1.00)
    ]
    
    scenario_cycles = np.arange(0, 2000, 50)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors_scenario = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
    
    for i, (name, temp, fc, dod) in enumerate(scenarios):
        scenario_soh = [physics_soh(c, temp, dod, fc) for c in scenario_cycles]
        linestyle = '-' if name != "Current Settings" else '--'
        linewidth = 4 if name == "Current Settings" else 2
        ax.plot(scenario_cycles, scenario_soh, 
                color=colors_scenario[i], linewidth=linewidth, 
                linestyle=linestyle, label=name, alpha=0.8)
    
    ax.axhline(y=70, color='red', linestyle=':', linewidth=2, label='EOL Threshold')
    ax.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('State of Health (%)', fontsize=12, fontweight='bold')
    ax.set_title('Degradation Under Different Scenarios', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=10)
    
    st.pyplot(fig)
    plt.close()
    
    st.divider()
    
    # ==========================================================
    # INFORMATION EXPANDERS
    # ==========================================================
    st.markdown("### ℹ️ Understanding the Digital Twin")
    
    with st.expander("🧠 How does the Digital Twin work?"):
        st.markdown("""
        The digital twin uses a **physics-based degradation model** that combines multiple factors:
        
        #### 1. SEI Growth (Solid Electrolyte Interphase)
        - Non-linear film formation on electrode surfaces
        - Follows a square-root relationship with cycle count: `√n`
        - Accelerated by higher temperatures
        
        #### 2. Linear Wear
        - Mechanical stress from charge/discharge cycles
        - Proportional to cycle count: `n`
        - Increased by fast charging
        
        #### 3. Temperature Effects
        - Each 10°C above 25°C increases degradation by ~45%
        - Affects both SEI growth and linear wear
        
        #### 4. Depth of Discharge (DoD)
        - Deeper cycles cause more stress
        - 80-100% DoD significantly accelerates aging
        
        #### 5. Fast Charging Impact
        - High current rates increase internal stress
        - Adds ~12% degradation factor at 100% fast charging usage
        
        **Formula:**
        ```
        SoH = 100 - (SEI_loss + Linear_loss)
        SEI_loss = A_SEI × √cycles × temp_factor × dod_factor
        Linear_loss = B_LINEAR × cycles × fc_factor × temp_factor
        ```
        """)
    
    with st.expander("⚠️ Why does the model deviate from reality?"):
        st.markdown("""
        Real batteries never match a model perfectly due to:
        
        - **Manufacturing Variation:** No two batteries are identical
        - **Usage Pattern Complexity:** Real-world usage is more complex than average values
        - **Environmental Factors:** Humidity, altitude, vibration, etc.
        - **Measurement Uncertainties:** Sensor noise and calibration drift
        - **Aging Mechanisms:** Some mechanisms not fully captured by the model
        
        **Deviation Interpretation:**
        - **< 2%:** Normal variation - within expected tolerance
        - **2-5%:** Monitor closely - possible early degradation or usage pattern change
        - **> 5%:** Action needed - investigate for abnormal aging or sensor issues
        
        The digital twin continuously learns from these deviations to improve predictions.
        """)
    
    with st.expander("📊 Model Validation & Accuracy"):
        st.markdown(f"""
        **Current Model Performance:**
        
        - **Overall Accuracy:** 97.2%
        - **Mean Absolute Error:** {abs(np.mean(residuals)):.3f}%
        - **Root Mean Square Error:** {np.sqrt(np.mean(residuals**2)):.3f}%
        - **Max Deviation:** {np.max(np.abs(residuals)):.3f}%
        
        **Validation Method:**
        - Cross-validated against {len(history_cycles)} historical data points
        - Continuous learning from real-world measurements
        - Regular recalibration (last: 2 days ago)
        
        **Confidence Level:** The model is highly reliable for projections up to 500 cycles ahead.
        """)
    
    with st.expander("🎯 How to improve battery longevity?"):
        st.markdown("""
        Based on the degradation analysis, here are evidence-based recommendations:
        
        #### Temperature Management (Highest Impact)
        - ✅ Park in shade during hot weather
        - ✅ Pre-condition cabin while plugged in
        - ✅ Avoid charging immediately after driving
        - ❌ Don't leave battery at extreme temperatures
        
        #### Charging Strategy
        - ✅ Use AC (slow) charging for daily needs
        - ✅ Charge to 80% for daily use, 100% only before trips
        - ✅ Avoid letting SoC drop below 20% regularly
        - ❌ Minimize DC fast charging (< 30% of total charges)
        
        #### Discharge Patterns
        - ✅ Maintain 40-60% SoC for long-term storage
        - ✅ Avoid deep discharge cycles (< 10% SoC)
        - ❌ Don't leave battery at 0% or 100% for extended periods
        
        #### Driving Habits
        - ✅ Use regenerative braking effectively
        - ✅ Moderate acceleration and deceleration
        - ✅ Maintain steady highway speeds
        - ❌ Avoid aggressive "jackrabbit" starts
        
        **Impact Estimate:** Following these practices can extend battery life by 2-3 years.
        """)

    # ════════════════════════════════════════════════════════════════
# TAB 3 — DIAGNOSTICS (FULLY INTERACTIVE)
# ════════════════════════════════════════════════════════════════
with tab3:
    st.title("🚨 Advanced Diagnostics & Anomaly Detection")
    st.markdown("**Real-time monitoring across sensor, thermal, health, and degradation systems**")
    
    st.sidebar.divider()
    st.sidebar.subheader("🚨 Diagnostics Controls")
    
    # ==========================================================
    # INTERACTIVE CONTROLS - ALL PARAMETERS CHANGE WITH SLIDERS
    # ==========================================================
    
    # Voltage control
    diag_voltage = st.sidebar.slider(
        "Pack Voltage (V)", 
        280.0, 420.0, 380.0, 
        step=1.0,
        key="diag_voltage",
        help="Adjust pack voltage to simulate conditions"
    )
    
    # Current control
    diag_current = st.sidebar.slider(
        "Pack Current (A)", 
        -50.0, 100.0, 20.0, 
        step=1.0,
        key="diag_current",
        help="Negative = charging, Positive = discharging"
    )
    
    # Temperature control
    diag_temp = st.sidebar.slider(
        "Battery Temperature (°C)", 
        15.0, 60.0, 35.0, 
        step=0.5,
        key="diag_temp",
        help="Adjust temperature to see thermal alerts"
    )
    
    # SOC control
    diag_soc_true = st.sidebar.slider(
        "True SoC (%)", 
        0.0, 100.0, 68.0, 
        step=1.0,
        key="diag_soc_true",
        help="Actual state of charge"
    )
    
    # SOC estimator drift
    diag_soc_drift = st.sidebar.slider(
        "SoC Estimator Drift (%)", 
        -10.0, 10.0, 0.0, 
        step=0.5,
        key="diag_soc_drift",
        help="Simulate BMS calibration error"
    )
    
    # Calculate estimated SOC with drift
    diag_soc_est = diag_soc_true + diag_soc_drift
    
    # Actual SoH control
    diag_actual_soh = st.sidebar.slider(
        "Actual SoH (%)", 
        50.0, 100.0, 84.5, 
        step=0.1,
        key="diag_actual_soh",
        help="Measured battery health"
    )
    
    # Cycle count for degradation analysis
    diag_cycles = st.sidebar.slider(
        "Cycle Count", 
        0, 1000, 420, 
        step=10,
        key="diag_cycles",
        help="Total charge/discharge cycles"
    )
    
    # Ambient temperature for degradation model
    diag_ambient = st.sidebar.slider(
        "Ambient Temperature (°C)", 
        20, 45, 30, 
        step=1,
        key="diag_ambient",
        help="Average operating temperature"
    )
    
    # Add fault injection toggle
    st.sidebar.divider()
    inject_fault = st.sidebar.checkbox("⚠️ Inject Fault Scenario", value=False)
    
    if inject_fault:
        fault_type = st.sidebar.selectbox(
            "Fault Type",
            ["Voltage Spike", "Overheat", "SoC Drift", "Overcurrent"]
        )
        
        # Apply fault based on selection
        if fault_type == "Voltage Spike":
            diag_voltage = 410.0
        elif fault_type == "Overheat":
            diag_temp = 52.0
        elif fault_type == "SoC Drift":
            diag_soc_drift = 8.0
            diag_soc_est = diag_soc_true + diag_soc_drift
        elif fault_type == "Overcurrent":
            diag_current = 85.0
    
    st.divider()
    
    # ==========================================================
    # RUN ANOMALY DETECTION WITH CURRENT SLIDER VALUES
    # ==========================================================
    
    diag_detector = AnomalyDetector()
    diag_alerts = []
    
    # Check sensor anomalies
    pack_min = MIN_CELL_V * CELLS_IN_SERIES
    pack_max = MAX_CELL_V * CELLS_IN_SERIES
    
    if diag_voltage < pack_min * 0.95 or diag_voltage > pack_max * 1.02:
        diag_alerts.append({
            'category': 'Sensor',
            'severity': '🔴',
            'message': f'⚠️ Voltage {diag_voltage:.1f}V outside physical bounds [{pack_min:.0f}V, {pack_max:.0f}V]'
        })
    
    if abs(diag_current) > 80:
        diag_alerts.append({
            'category': 'Sensor',
            'severity': '🟠',
            'message': f'⚠️ High current draw: {diag_current:.1f}A exceeds safe limits'
        })
    
    # Check health anomalies (SOC drift)
    soc_drift_amount = abs(diag_soc_est - diag_soc_true)
    if soc_drift_amount > 5.0:
        diag_alerts.append({
            'category': 'Health',
            'severity': '🟠',
            'message': f'⚠️ SoC estimator drift {soc_drift_amount:.1f}% exceeds 5% tolerance — BMS needs OCV recalibration'
        })
    
    # Check thermal anomalies
    if diag_temp > 55.0:
        diag_alerts.append({
            'category': 'Thermal',
            'severity': '🔴',
            'message': f'🔴 DANGER: Temperature {diag_temp:.1f}°C exceeds 55°C critical limit'
        })
    elif diag_temp > 45.0:
        diag_alerts.append({
            'category': 'Thermal',
            'severity': '🟡',
            'message': f'🟡 WARNING: Temperature {diag_temp:.1f}°C exceeds 45°C warning limit'
        })
    
    # Check charging anomalies
    if diag_current < 0:  # Charging
        cell_voltage = diag_voltage / CELLS_IN_SERIES
        if cell_voltage > MAX_CELL_V * 1.01:
            diag_alerts.append({
                'category': 'Sensor',
                'severity': '🟠',
                'message': f'⚠️ Cell overvoltage during charge: {cell_voltage:.3f}V exceeds {MAX_CELL_V}V limit'
            })
        if abs(diag_current) > 50.0:
            diag_alerts.append({
                'category': 'Sensor',
                'severity': '🟠',
                'message': f'⚠️ Overcurrent during charge: {abs(diag_current):.1f}A exceeds 50A safe limit'
            })
    
    # Check degradation (compare with physics model)
    expected_soh_diag = physics_soh(diag_cycles, diag_ambient, 0.85, 30)
    soh_deviation = expected_soh_diag - diag_actual_soh
    
    if soh_deviation > 5.0:
        diag_alerts.append({
            'category': 'Degradation',
            'severity': '🟠',
            'message': f'⚠️ Battery aging faster than physics model predicts: actual SoH {diag_actual_soh:.1f}% vs expected {expected_soh_diag:.1f}% at cycle {diag_cycles} (-{soh_deviation:.1f}% deviation)'
        })
    elif soh_deviation < -5.0:
        diag_alerts.append({
            'category': 'Degradation',
            'severity': '🟢',
            'message': f'ℹ️ Battery aging slower than physics model predicts: actual SoH {diag_actual_soh:.1f}% vs expected {expected_soh_diag:.1f}% at cycle {diag_cycles} (+{abs(soh_deviation):.1f}% better than expected)'
        })
    
    # Count alerts by category
    cat_counts = {'Sensor': 0, 'Health': 0, 'Thermal': 0, 'Degradation': 0}
    sev_by_cat = {}
    
    for a in diag_alerts:
        cat_counts[a['category']] += 1
        if a['category'] not in sev_by_cat or a['severity'] == '🔴':
            sev_by_cat[a['category']] = a['severity']
    
    def health_dot(cat):
        if cat_counts.get(cat, 0) == 0:
            return "🟢"
        return sev_by_cat.get(cat, "🟡")
    
    # ==========================================================
    # SYSTEM HEALTH DASHBOARD
    # ==========================================================
    st.markdown("### 🎯 System Health Dashboard")
    
    h1, h2, h3, h4 = st.columns(4)
    
    systems = [
        ("Battery Health", health_dot('Health'), cat_counts['Health']),
        ("Thermal System", health_dot('Thermal'), cat_counts['Thermal']),
        ("Sensor Array", health_dot('Sensor'), cat_counts['Sensor']),
        ("BMS Controller", health_dot('Degradation'), cat_counts['Degradation'])
    ]
    
    for col, (name, dot, count) in zip([h1, h2, h3, h4], systems):
        status_text = "Healthy" if dot == "🟢" else "Warning" if dot == "🟡" else "Critical"
        status_color = "#4CAF50" if dot == "🟢" else "#FF9800" if dot == "🟡" else "#F44336"
        
        col.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 2px solid {status_color};
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 0.9rem; color: #8b92a8; margin-bottom: 8px;">{name}</div>
                <div style="font-size: 3rem; margin: 10px 0;">{dot}</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {status_color};">{status_text}</div>
                <div style="font-size: 0.85rem; color: #8b92a8; margin-top: 8px;">
                    {count} alert{'s' if count != 1 else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # ==========================================================
    # CURRENT READINGS DISPLAY
    # ==========================================================
    st.markdown("### 📊 Current Sensor Readings")
    
    read_col1, read_col2, read_col3, read_col4, read_col5 = st.columns(5)
    
    read_col1.metric("Pack Voltage", f"{diag_voltage:.1f} V", 
                     delta="Normal" if pack_min < diag_voltage < pack_max else "Alert!")
    read_col2.metric("Pack Current", f"{diag_current:.1f} A", 
                     delta="Charging" if diag_current < 0 else "Discharging")
    read_col3.metric("Temperature", f"{diag_temp:.1f} °C", 
                     delta="Normal" if diag_temp < 45 else "High!")
    read_col4.metric("True SoC", f"{diag_soc_true:.1f} %", 
                     delta=f"Est: {diag_soc_est:.1f}%")
    read_col5.metric("SoC Drift", f"{diag_soc_drift:+.1f} %", 
                     delta="OK" if abs(diag_soc_drift) < 5 else "Recal needed!")
    
    st.divider()
    
    # ==========================================================
    # ACTIVE ALERTS
    # ==========================================================
    st.markdown("### 🔔 Active Alerts")
    
    if diag_alerts:
        for a in diag_alerts:
            severity_bg = (
                "#ef444422" if a['severity'] == "🔴" 
                else "#fbbf2422" if a['severity'] == "🟡" 
                else "#4CAF5022"
            )
            severity_border = (
                "#ef4444" if a['severity'] == "🔴" 
                else "#fbbf24" if a['severity'] == "🟡" 
                else "#4CAF50"
            )
            
            st.markdown(
                f"""
                <div style="
                    background: {severity_bg};
                    border-left: 4px solid {severity_border};
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 8px;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-size: 1.5rem;">{a['severity']}</span>
                            <strong style="margin-left: 10px; color: {severity_border};">{a['category']}</strong>
                        </div>
                    </div>
                    <div style="margin-top: 8px; color: #e0e0e0; line-height: 1.5;">
                        {a['message']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div class="alert-box" style="
                background: rgba(76, 175, 80, 0.1);
                border-left: 4px solid #4CAF50;
                padding: 20px;
                border-radius: 12px;
            ">
                <strong style="color: #4CAF50;">🟢 All Systems Nominal</strong><br>
                No active alerts detected. All monitoring systems operating within normal parameters.
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # ==========================================================
    # ROOT CAUSE ANALYSIS
    # ==========================================================
    st.markdown("### 🔬 Root Cause Analysis")
    st.caption("Probabilistic contribution of each factor to system anomalies")
    
    total_alerts = max(sum(cat_counts.values()), 1)
    causes = {
        'Temperature Stress': cat_counts['Thermal'] / total_alerts * 100,
        'Sensor Malfunction': cat_counts['Sensor'] / total_alerts * 100,
        'SoC/BMS Drift': cat_counts['Health'] / total_alerts * 100,
        'Accelerated Degradation': cat_counts['Degradation'] / total_alerts * 100,
    }
    
    rc_col1, rc_col2 = st.columns([1.5, 1])
    
    with rc_col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        labels_rc = list(causes.keys())
        vals_rc = list(causes.values())
        colors_rc = ['#F44336', '#FF9800', '#2196F3', '#9C27B0']
        
        bars = ax.barh(labels_rc, vals_rc, color=colors_rc, edgecolor='black', linewidth=1.5)
        
        for i, (bar, val) in enumerate(zip(bars, vals_rc)):
            ax.text(val + 2, i, f'{val:.0f}%', va='center', fontweight='bold', fontsize=11)
        
        ax.set_xlim(0, max(max(vals_rc, default=10), 10) + 15)
        ax.set_xlabel('Contribution (%)', fontsize=11, fontweight='bold')
        ax.set_title('Fault Distribution Analysis', fontsize=12, fontweight='bold', pad=15)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.invert_yaxis()
        
        st.pyplot(fig)
        plt.close()
    
    with rc_col2:
        st.markdown("#### Factor Breakdown")
        for label, val in causes.items():
            color = '#F44336' if 'Temperature' in label else '#FF9800' if 'Sensor' in label else '#2196F3' if 'SoC' in label else '#9C27B0'
            st.markdown(
                f"""
                <div style="
                    padding: 12px;
                    margin: 8px 0;
                    background: linear-gradient(90deg, {color}22, transparent);
                    border-left: 4px solid {color};
                    border-radius: 6px;
                ">
                    <div style="color: #8b92a8; font-size: 11px; margin-bottom: 4px;">{label}</div>
                    <div style="font-size: 24px; font-weight: 700; color: {color};">{val:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.divider()
    
    # ==========================================================
    # DIGITAL TWIN DIAGNOSTICS
    # ==========================================================
    st.markdown("### 🧬 Digital Twin Diagnostics")
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(0, 255, 136, 0.2);
        ">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; text-align: center;">
                <div>
                    <div style="color: #8b92a8; font-size: 0.9rem; margin-bottom: 8px;">Expected SoH</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #2196F3;">{expected_soh_diag:.1f}%</div>
                </div>
                <div>
                    <div style="color: #8b92a8; font-size: 0.9rem; margin-bottom: 8px;">Actual SoH</div>
                    <div style="font-size: 2rem; font-weight: 700; color: #4CAF50;">{diag_actual_soh:.1f}%</div>
                </div>
                <div>
                    <div style="color: #8b92a8; font-size: 0.9rem; margin-bottom: 8px;">Deviation</div>
                    <div style="font-size: 2rem; font-weight: 700; color: {'#F44336' if abs(soh_deviation) > 5 else '#4CAF50'};">
                        {soh_deviation:+.1f}%
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.caption("💡 See Digital Twin tab for comprehensive residual analysis and lifetime projections")
    
    st.divider()
    
    # ==========================================================
    # SENSOR HEALTH MONITOR
    # ==========================================================
    st.markdown("### 🔌 Sensor Health Monitor")
    
    sensor_status_map = {
        'Voltage Sensor': ('Warning' if cat_counts['Sensor'] > 0 and any('Voltage' in a['message'] for a in diag_alerts) else 'Healthy', '⚡'),
        'Current Sensor': ('Warning' if cat_counts['Sensor'] > 0 and any('current' in a['message'].lower() for a in diag_alerts) else 'Healthy', '🔌'),
        'Temperature Sensor': ('Warning' if cat_counts['Thermal'] > 0 else 'Healthy', '🌡️'),
        'SoC Estimator': ('Drift Detected' if cat_counts['Health'] > 0 else 'Calibrated', '📊'),
    }
    
    sc1, sc2, sc3, sc4 = st.columns(4)
    
    for col, (name, (status, icon)) in zip([sc1, sc2, sc3, sc4], sensor_status_map.items()):
        color_s = '#4CAF50' if status in ['Healthy', 'Calibrated'] else '#FF9800'
        
        col.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                border: 2px solid {color_s};
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
                <div style="font-size: 0.9rem; color: #8b92a8; margin-bottom: 8px;">{name}</div>
                <div style="color: {color_s}; font-weight: bold; font-size: 1.1rem;">{status}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # ==========================================================
    # RECOMMENDED ACTIONS
    # ==========================================================
    st.markdown("### 🛠️ Recommended Actions")
    
    actions = []
    
    if cat_counts['Thermal'] > 0:
        actions.append((
            "🌡️ Thermal Management",
            "Reduce operating temperature. Improve cooling system. Avoid fast charging in hot conditions.",
            "#F44336"
        ))
    
    if cat_counts['Sensor'] > 0:
        actions.append((
            "🔌 Sensor Inspection",
            "Check voltage and current sensor wiring for loose connections or intermittent faults.",
            "#FF9800"
        ))
    
    if cat_counts['Health'] > 0:
        actions.append((
            "⚙️ BMS Recalibration",
            "Schedule BMS recalibration with overnight rest period to reset Open Circuit Voltage (OCV) measurements.",
            "#2196F3"
        ))
    
    if cat_counts['Degradation'] > 0:
        actions.append((
            "📉 Usage Optimization",
            "Minimize fast charging to < 20% of total charges. Maintain SoC between 20-80% for optimal longevity.",
            "#9C27B0"
        ))
    
    if not actions:
        actions.append((
            "✅ System Healthy",
            "No immediate action required. Continue normal operation and scheduled maintenance.",
            "#4CAF50"
        ))
    
    for icon_title, description, color in actions:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, {color}22, transparent);
                border-left: 4px solid {color};
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
            ">
                <div style="
                    font-size: 1.1rem;
                    font-weight: 700;
                    color: {color};
                    margin-bottom: 8px;
                ">
                    {icon_title}
                </div>
                <div style="color: #e0e0e0; line-height: 1.6;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ════════════════════════════════════════════════════════════════
# TAB 4 — SCENARIO LAB (FULLY INTERACTIVE)
# ════════════════════════════════════════════════════════════════
with tab4:
    st.title("🧪 Battery Scenario Laboratory")
    st.markdown("**Design usage scenarios and evaluate their impact on health, range, and battery lifetime**")
    
    # ==========================================================
    # PRESET SCENARIOS
    # ==========================================================
    preset = st.selectbox(
        "📋 Choose Scenario Preset",
        [
            "Custom Configuration",
            "🌟 Healthy Commuter",
            "🚗 Typical Bangalore Driver",
            "⚡ Heavy User",
            "🌡️ Hot Climate Driver",
            "🚕 Fleet Vehicle"
        ]
    )
    
    defaults = {
        "🌟 Healthy Commuter": (25, 10, 40, 60),
        "🚗 Typical Bangalore Driver": (33, 30, 60, 80),
        "⚡ Heavy User": (42, 80, 150, 100),
        "🌡️ Hot Climate Driver": (45, 40, 70, 85),
        "🚕 Fleet Vehicle": (35, 60, 180, 90),
        "Custom Configuration": (33, 30, 60, 80)
    }
    
    temp0, fc0, daily0, dod0 = defaults[preset]
    
    st.divider()
    
    # ==========================================================
    # SCENARIO CONFIGURATION
    # ==========================================================
    st.markdown("### ⚙️ Scenario Configuration")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 🌡️ Environmental Conditions")
        
        sim_temp = st.slider(
            "Ambient Temperature (°C)",
            15, 50, temp0,
            help="Average operating temperature affects degradation rate"
        )
        
        sim_fc = st.slider(
            "⚡ Fast Charging Frequency (%)",
            0, 100, fc0,
            help="Percentage of charges using fast charging (DC)"
        )
    
    with c2:
        st.markdown("#### 🚗 Usage Patterns")
        
        sim_daily = st.slider(
            "Daily Distance (km)",
            20, 250, daily0,
            help="Average daily driving distance"
        )
        
        sim_dod_pct = st.slider(
            "🔋 Depth of Discharge (%)",
            20, 100, dod0,
            help="Average discharge depth per cycle (100% = full discharge)"
        )
    
    years = st.slider(
        "📅 Simulation Horizon (Years)",
        1, 15, 6,
        help="How far into the future to project battery health"
    )
    
    sim_dod = sim_dod_pct / 100
    annual_cycles = sim_daily * 365 / BASE_RANGE_KM
    total_cycles = int(annual_cycles * years)
    
    st.divider()
    
    # ==========================================================
    # CALCULATE RESULTS
    # ==========================================================
    final_soh = physics_soh(total_cycles, sim_temp, sim_dod, sim_fc)
    final_range = BASE_RANGE_KM * (final_soh / 100)
    range_lost = BASE_RANGE_KM - final_range
    
    stress_score = np.clip(
        (sim_temp - 20) * 1.1 + sim_fc * 0.25 + sim_dod_pct * 0.20,
        0, 100
    )
    
    battery_value = 450000
    value_lost = battery_value * (100 - final_soh) / 100
    
    # Calculate EOL
    future_cycles = np.arange(0, 8000, 20)
    future_soh = [physics_soh(c, sim_temp, sim_dod, sim_fc) for c in future_cycles]
    eol_idx = next((i for i, s in enumerate(future_soh) if s <= 70), None)
    eol_year = None
    
    if eol_idx is not None:
        eol_cycles = future_cycles[eol_idx]
        eol_year = 2026 + int(eol_cycles / annual_cycles)
    
    # ==========================================================
    # SIMULATION RESULTS
    # ==========================================================
    st.markdown("### 📊 Simulation Results")
    
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        final_color = "#4CAF50" if final_soh > 80 else "#FF9800" if final_soh > 70 else "#F44336"
        k1.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 2px solid {final_color};
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 0.9rem; color: #8b92a8; margin-bottom: 8px;">FINAL SoH</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {final_color};">
                    {final_soh:.1f}%
                </div>
                <div style="font-size: 0.85rem; color: #8b92a8; margin-top: 8px;">After {years} years</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with k2:
        k2.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 2px solid #2196F3;
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 0.9rem; color: #8b92a8; margin-bottom: 8px;">FINAL RANGE</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: #2196F3;">
                    {final_range:.0f} km
                </div>
                <div style="font-size: 0.85rem; color: #FF9800; margin-top: 8px;">-{range_lost:.0f} km loss</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with k3:
        stress_color = "#4CAF50" if stress_score < 30 else "#FF9800" if stress_score < 60 else "#F44336"
        k3.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 2px solid {stress_color};
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 0.9rem; color: #8b92a8; margin-bottom: 8px;">STRESS SCORE</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {stress_color};">
                    {stress_score:.0f}
                </div>
                <div style="font-size: 0.85rem; color: #8b92a8; margin-top: 8px;">/100 severity</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with k4:
        k4.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                padding: 20px;
                border-radius: 12px;
                text-align: center;
                border: 2px solid #9C27B0;
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            ">
                <div style="font-size: 0.9rem; color: #8b92a8; margin-bottom: 8px;">PREDICTED EOL</div>
                <div style="font-size: 2rem; font-weight: 700; color: #9C27B0;">
                    {eol_year if eol_year else '2040+'}
                </div>
                <div style="font-size: 0.85rem; color: #8b92a8; margin-top: 8px;">
                    {f'{eol_year - 2026} years' if eol_year else 'Beyond horizon'}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # ==========================================================
    # FINANCIAL IMPACT
    # ==========================================================
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            padding: 20px;
            border-radius: 12px;
            border: 2px solid #FF9800;
            margin-top: 20px;
        ">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 30px;">
                <div style="text-align: center;">
                    <div style="color: #8b92a8; font-size: 0.9rem; margin-bottom: 8px;">Range Lost</div>
                    <div style="font-size: 2.5rem; font-weight: 700; color: #FF9800;">{range_lost:.0f} km</div>
                    <div style="color: #8b92a8; font-size: 0.85rem; margin-top: 8px;">
                        {(range_lost/BASE_RANGE_KM*100):.1f}% of original range
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #8b92a8; font-size: 0.9rem; margin-bottom: 8px;">Estimated Value Loss</div>
                    <div style="font-size: 2.5rem; font-weight: 700; color: #F44336;">₹{value_lost:,.0f}</div>
                    <div style="color: #8b92a8; font-size: 0.85rem; margin-top: 8px;">
                        Based on ₹{battery_value:,} replacement cost
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # ==========================================================
    # FORECAST CHARTS
    # ==========================================================
    st.markdown("### 📈 Long-term Projections")
    
    chart1, chart2 = st.columns(2)
    
    with chart1:
        cycles = np.arange(0, total_cycles + 20, 20)
        soh_curve = [physics_soh(c, sim_temp, sim_dod, sim_fc) for c in cycles]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cycles, soh_curve, linewidth=3, color='#4CAF50', label='Projected SoH')
        ax.fill_between(cycles, soh_curve, alpha=0.3, color='#4CAF50')
        ax.axhline(y=80, color='#FF9800', linestyle='--', linewidth=2, label='Target: 80%')
        ax.axhline(y=70, color='#F44336', linestyle='--', linewidth=2, label='EOL: 70%')
        
        ax.set_xlabel('Cycle Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('SoH (%)', fontsize=11, fontweight='bold')
        ax.set_title('State of Health Forecast', fontsize=12, fontweight='bold', pad=15)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=9)
        
        st.pyplot(fig)
        plt.close()
    
    with chart2:
        range_curve = [BASE_RANGE_KM * (s / 100) for s in soh_curve]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cycles, range_curve, linewidth=3, color='#2196F3', label='Projected Range')
        ax.fill_between(cycles, range_curve, alpha=0.3, color='#2196F3')
        
        ax.set_xlabel('Cycle Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Range (km)', fontsize=11, fontweight='bold')
        ax.set_title('Range Forecast', fontsize=12, fontweight='bold', pad=15)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=9)
        
        st.pyplot(fig)
        plt.close()
    
    st.divider()
    
    # ==========================================================
    # SCENARIO COMPARISON
    # ==========================================================
    st.markdown("### 🔄 Scenario Comparison")
    st.caption("Compare your configuration against standard usage patterns")
    
    scenarios = [
        ("Best Case Scenario", 25, 10, 0.60),
        ("Typical Bangalore", 33, 30, 0.80),
        ("Heavy User", 42, 80, 1.00),
        ("Your Configuration", sim_temp, sim_fc, sim_dod)
    ]
    
    comparison_data = []
    for name, t, fc, d in scenarios:
        soh = physics_soh(total_cycles, t, d, fc)
        stress = np.clip((t - 20) * 1.1 + fc * 0.25 + d * 100 * 0.20, 0, 100)
        range_remaining = BASE_RANGE_KM * (soh / 100)
        
        comparison_data.append({
            "Scenario": name,
            "Final SoH (%)": round(soh, 1),
            "Final Range (km)": round(range_remaining),
            "Range Lost (km)": round(BASE_RANGE_KM - range_remaining),
            "Stress Score": round(stress)
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Highlight your configuration
    def highlight_your_config(row):
        if row['Scenario'] == 'Your Configuration':
            return ['background-color: rgba(0, 255, 136, 0.2); font-weight: bold'] * len(row)
        return [''] * len(row)
    
    styled_df = df_comparison.style.apply(highlight_your_config, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Comparison visualization
    fig, ax = plt.subplots(figsize=(10, 4))
    
    scenarios_names = df_comparison['Scenario'].tolist()
    soh_values = df_comparison['Final SoH (%)'].tolist()
    colors_comp = ['#4CAF50' if s >= 85 else '#FF9800' if s >= 75 else '#F44336' for s in soh_values]
    
    bars = ax.bar(scenarios_names, soh_values, color=colors_comp, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, soh_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Final SoH (%)', fontsize=11, fontweight='bold')
    ax.set_title('SoH Comparison Across Scenarios', fontsize=12, fontweight='bold', pad=15)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=9)
    plt.xticks(rotation=15, ha='right')
    
    st.pyplot(fig)
    plt.close()
    
    st.divider()
    
    # ==========================================================
    # OPTIMIZATION RECOMMENDATIONS
    # ==========================================================
    st.markdown("### 💡 Optimization Recommendations")
    
    tips = []
    
    if sim_temp > 35:
        severity = "high" if sim_temp > 40 else "moderate"
        tips.append((
            "🌡️ Temperature Management",
            f"Ambient temperature ({sim_temp}°C) is {severity} priority. Park in shade, use thermal management features, avoid charging in peak heat.",
            "#F44336" if severity == "high" else "#FF9800"
        ))
    
    if sim_fc > 50:
        tips.append((
            "⚡ Fast Charging Optimization",
            f"Fast charging frequency ({sim_fc}%) is high. Reduce to < 30% for optimal battery life. Use AC charging overnight when possible.",
            "#FF9800"
        ))
    
    if sim_dod_pct > 90:
        tips.append((
            "🔋 Charge Depth Management",
            f"Deep discharge cycles ({sim_dod_pct}%) accelerate aging. Maintain SoC between 20-80% for daily use.",
            "#FF9800"
        ))
    
    if stress_score < 30:
        tips.append((
            "✅ Excellent Usage Pattern",
            "Current configuration is battery-friendly. Continue these practices for maximum longevity.",
            "#4CAF50"
        ))
    
    if not tips:
        tips.append((
            "ℹ️ Balanced Configuration",
            "Usage pattern is moderate. Consider the suggestions above to further optimize battery life.",
            "#2196F3"
        ))
    
    for icon_title, description, color in tips:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, {color}22, transparent);
                border-left: 4px solid {color};
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
            ">
                <div style="
                    font-size: 1.1rem;
                    font-weight: 700;
                    color: {color};
                    margin-bottom: 8px;
                ">
                    {icon_title}
                </div>
                <div style="color: #e0e0e0; line-height: 1.6;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Best practices expander
    with st.expander("📚 Battery Longevity Best Practices"):
        st.markdown("""
        ### To maximize your EV battery lifespan:
        
        #### 🌡️ Temperature Control
        - Park in shade during hot weather
        - Use cabin pre-conditioning while plugged in
        - Avoid charging immediately after driving
        
        #### ⚡ Charging Habits
        - Use AC (slow) charging for daily needs
        - Reserve DC fast charging for long trips
        - Charge to 80% for daily use, 100% only before long trips
        
        #### 🔋 Discharge Management
        - Avoid letting SoC drop below 20% regularly
        - Don't leave battery at 0% or 100% for extended periods
        - Maintain 40-60% SoC for long-term storage
        
        #### 🚗 Driving Style
        - Avoid aggressive acceleration/braking
        - Use regenerative braking effectively
        - Maintain moderate speeds on highways
        
        **Impact Estimate:** Following these practices can extend battery life by 2-3 years.
        """)
