import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
# TAB 1 — BATTERY OVERVIEW (FULLY FIXED)
# ════════════════════════════════════════════════════════════════
with tab1:
    
    st.markdown("# 🔋 Battery Digital Twin Overview")
    st.markdown("**Tata Nexon EV • 30.2 kWh Battery Pack • BESCOM Digital Twin Project**")
    
    # ==========================================================
    # HEALTH SCORE HERO CARD (FIXED)
    # ==========================================================
    score_color = (
        "#4CAF50" if health_score > 85
        else "#FF9800" if health_score > 70
        else "#F44336"
    )
    
    # Use columns to create better layout
    hero_col1, hero_col2 = st.columns([2, 1])
    
    with hero_col1:
        # Fixed: Using only f-string, removed .format()
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
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
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
    # MAIN KPIs
    # ==========================================================
    st.markdown("### 📊 Key Performance Indicators")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # SOH Card
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
    
    # SOC Card
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
    
    # Range Card
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
    
    # Temperature Card
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
    
    # Remaining Life Card
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
    # STATUS BAR
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
    # BATTERY PACK & DIGITAL TWIN
    # ==========================================================
    st.markdown("### 🔌 Battery Pack & Digital Twin Analysis")
    
    left, right = st.columns([1.4, 1])
    
    with left:
        st.markdown("#### Battery Pack Visualization")
        
        blocks = 20
        filled = int(blocks * snap['soc'] / 100)
        
        # Battery blocks
        battery_html = '<div style="display: flex; gap: 6px; margin: 20px 0;">'
        
        for i in range(blocks):
            if i < filled:
                if snap['soc'] > 80:
                    color = "#4CAF50"
                elif snap['soc'] > 50:
                    color = "#8BC34A"
                elif snap['soc'] > 20:
                    color = "#FFC107"
                else:
                    color = "#FF5722"
            else:
                color = "#2d3139"
            
            battery_html += f'<div style="width: 28px; height: 70px; background: {color}; border-radius: 6px; border: 2px solid #1a1a2e; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"></div>'
        
        battery_html += '</div>'
        
        st.markdown(battery_html, unsafe_allow_html=True)
        
        # SoC Display
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
        
        # Pack Details
        v1, v2, v3 = st.columns(3)
        
        with v1:
            st.metric(
                "Pack Voltage",
                f"{snap['voltage']:.0f} V",
                delta=f"{snap['voltage'] - 350:.0f}V from min"
            )
        
        with v2:
            st.metric(
                "Pack Current",
                f"{snap['current']:.0f} A",
                delta="Normal draw"
            )
        
        with v3:
            st.metric(
                "Usable Capacity",
                f"{snap['capacity_ah']:.1f} Ah",
                delta=f"-{CAPACITY_AH - snap['capacity_ah']:.1f} Ah"
            )
        
        st.markdown("---")
        
        # Battery Specs (FIXED - removed .format())
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
                    <strong>Chemistry:</strong> Lithium-ion NMC (Nickel Manganese Cobalt)<br>
                    <strong>Configuration:</strong> {CELLS_IN_SERIES}S (96 cells in series)<br>
                    <strong>Nominal Capacity:</strong> {CAPACITY_KWH} kWh<br>
                    <strong>Cell Voltage Range:</strong> {MIN_CELL_V}V - {MAX_CELL_V}V<br>
                    <strong>Total Cycles:</strong> {snap['cycle_count']}<br>
                    <strong>Ambient Temperature:</strong> {snap['ambient_temp']}°C<br>
                    <strong>Warranty:</strong> 8 years / 160,000 km
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with right:
        st.markdown("#### Digital Twin Status")
        
        dev_color = (
            "#4CAF50" if abs(snap['deviation']) < 2
            else "#FF9800" if abs(snap['deviation']) < 5
            else "#F44336"
        )
        
        # Twin metrics
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
        
        # AI Insight
        insight_emoji = "✅" if abs(snap['deviation']) < 2 else "⚠️" if abs(snap['deviation']) < 5 else "🔴"
        insight_bg = (
            "rgba(76, 175, 80, 0.1)" if abs(snap['deviation']) < 2
            else "rgba(255, 152, 0, 0.1)" if abs(snap['deviation']) < 5
            else "rgba(244, 67, 54, 0.1)"
        )
        
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
    # TRENDS (Using Matplotlib as in original)
    # ==========================================================
    st.markdown("### 📈 Recent Performance Trends")
    st.caption("Last 40 charge cycles showing battery degradation patterns")
    
    cycles = np.arange(
        max(0, snap['cycle_count'] - 40),
        snap['cycle_count'] + 1
    )
    
    soh_curve = [
        physics_soh(c, snap['ambient_temp'], 0.85, 30)
        for c in cycles
    ]
    
    temp_curve = (
        snap['temperature']
        + np.random.normal(0, 0.4, len(cycles))
    )
    
    range_curve = [
        BASE_RANGE_KM * (x / 100) * (snap['soc'] / 100)
        for x in soh_curve
    ]
    
    t1, t2, t3 = st.columns(3)
    
    with t1:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(cycles, soh_curve, linewidth=2, color='#4CAF50')
        ax.set_title("SoH Trend")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Cycles")
        ax.set_ylabel("SoH (%)")
        st.pyplot(fig)
    
    with t2:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(cycles, temp_curve, linewidth=2, color='#FF9800')
        ax.axhline(y=45, linestyle='--', color='red', alpha=0.5, label='Warning')
        ax.set_title("Temperature Trend")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Cycles")
        ax.set_ylabel("Temp (°C)")
        st.pyplot(fig)
    
    with t3:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.plot(cycles, range_curve, linewidth=2, color='#2196F3')
        ax.set_title("Range Trend")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Cycles")
        ax.set_ylabel("Range (km)")
        st.pyplot(fig)
    
    st.divider()
    
    # ==========================================================
    # FOOTER
    # ==========================================================
    st.markdown("### 📋 Digital Twin Summary")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.info(
            f"""
            **🎯 Model Accuracy**
            
            Digital Twin Confidence: **97.2%**
            
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
            
            • Avoid charging above 80% daily
            
            • Minimize fast charging (< 30%)
            
            • Keep temperature below 40°C
            
            • Schedule service: **3 months**
            """
        )
