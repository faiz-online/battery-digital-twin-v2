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
# TAB 1 — BATTERY OVERVIEW (ENHANCED & FIXED)
# ════════════════════════════════════════════════════════════════
with tab1:
    
    st.markdown("# 🔋 Battery Digital Twin Overview")
    st.markdown("**Tata Nexon EV • 30.2 kWh Battery Pack • BESCOM Digital Twin Project**")
    
    # ==========================================================
    # HEALTH SCORE HERO CARD (FIXED HTML RENDERING)
    # ==========================================================
    score_color = (
        "#4CAF50" if health_score > 85
        else "#FF9800" if health_score > 70
        else "#F44336"
    )
    
    # Use columns to create better layout
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
        # Status Badge
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
    # MAIN KPIs - ENHANCED WITH BETTER STYLING
    # ==========================================================
    st.markdown("### 📊 Key Performance Indicators")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    # Custom metric card function
    def create_metric_card(label, value, delta=None, emoji="", color="#4CAF50"):
        delta_html = ""
        if delta:
            delta_color = "#4CAF50" if "+" not in str(delta) else "#F44336"
            delta_html = f'<div style="font-size: 13px; color: {delta_color}; margin-top: 5px;">{delta}</div>'
        
        return f"""
        <div style="
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        ">
            <div style="font-size: 24px; margin-bottom: 8px;">{emoji}</div>
            <div style="
                font-size: 11px;
                color: #8b92a8;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 8px;
            ">
                {label}
            </div>
            <div style="
                font-size: 28px;
                font-weight: 700;
                color: {color};
            ">
                {value}
            </div>
            {delta_html}
        </div>
        """
    
    with c1:
        st.markdown(
            create_metric_card(
                "State of Health",
                f"{snap['soh']:.1f}%",
                "✓ Excellent" if snap['soh'] > 90 else "✓ Good" if snap['soh'] > 80 else "⚠ Monitor",
                "💚",
                "#4CAF50" if snap['soh'] > 85 else "#FF9800"
            ),
            unsafe_allow_html=True
        )
    
    with c2:
        st.markdown(
            create_metric_card(
                "State of Charge",
                f"{snap['soc']:.1f}%",
                f"~{snap['range_km']:.0f} km range",
                "🔋",
                "#4CAF50" if snap['soc'] > 50 else "#FF9800"
            ),
            unsafe_allow_html=True
        )
    
    with c3:
        st.markdown(
            create_metric_card(
                "Range Remaining",
                f"{snap['range_km']:.0f} km",
                f"Base: {BASE_RANGE_KM} km",
                "🚗",
                "#2196F3"
            ),
            unsafe_allow_html=True
        )
    
    with c4:
        st.markdown(
            create_metric_card(
                "Battery Temp",
                f"{snap['temperature']:.1f}°C",
                "Optimal" if snap['temperature'] < 40 else "Warm",
                "🌡️",
                "#4CAF50" if snap['temperature'] < 40 else "#FF9800"
            ),
            unsafe_allow_html=True
        )
    
    with c5:
        st.markdown(
            create_metric_card(
                "Remaining Life",
                f"{snap['rul_years']:.1f} yrs",
                f"~{snap['rul_cycles']:.0f} cycles",
                "⏳",
                "#9C27B0"
            ),
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==========================================================
    # STATUS BAR - ENHANCED
    # ==========================================================
    st.markdown("### 🎯 System Status Dashboard")
    
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    def create_status_pill(emoji, label, is_healthy):
        color = "#4CAF50" if is_healthy else "#F44336"
        bg_color = "rgba(76, 175, 80, 0.1)" if is_healthy else "rgba(244, 67, 54, 0.1)"
        
        return f"""
        <div style="
            background: {bg_color};
            padding: 12px 16px;
            border-radius: 25px;
            border: 2px solid {color};
            text-align: center;
        ">
            <span style="font-size: 20px;">{emoji}</span>
            <div style="
                font-size: 13px;
                font-weight: 600;
                color: {color};
                margin-top: 5px;
            ">
                {label}
            </div>
        </div>
        """
    
    with status_col1:
        st.markdown(
            create_status_pill(
                soh_status,
                "Battery Health" if snap['soh'] > 80 else "Check Battery",
                snap['soh'] > 80
            ),
            unsafe_allow_html=True
        )
    
    with status_col2:
        st.markdown(
            create_status_pill(
                thermal_status,
                "Thermal OK" if snap['temperature'] < 42 else "High Temp",
                snap['temperature'] < 42
            ),
            unsafe_allow_html=True
        )
    
    with status_col3:
        st.markdown(
            create_status_pill(
                sensor_status,
                "Sensors Online",
                True
            ),
            unsafe_allow_html=True
        )
    
    with status_col4:
        st.markdown(
            create_status_pill(
                fault_status,
                "No Faults",
                True
            ),
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # ==========================================================
    # BATTERY PACK VISUALIZATION & DIGITAL TWIN
    # ==========================================================
    st.markdown("### 🔌 Battery Pack & Digital Twin Analysis")
    
    left, right = st.columns([1.4, 1])
    
    # ==========================================================
    # LEFT: BATTERY PACK VISUALIZATION
    # ==========================================================
    with left:
        st.markdown("#### Battery Pack Visualization")
        
        blocks = 20
        filled = int(blocks * snap['soc'] / 100)
        
        # Create battery blocks with gradient
        battery_html = '<div style="display: flex; gap: 6px; margin: 20px 0;">'
        
        for i in range(blocks):
            if i < filled:
                # Gradient from green to yellow based on position
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
            
            battery_html += f"""
            <div style="
                width: 28px;
                height: 70px;
                background: {color};
                border-radius: 6px;
                border: 2px solid #1a1a2e;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            "></div>
            """
        
        battery_html += '</div>'
        
        st.markdown(battery_html, unsafe_allow_html=True)
        
        # SoC percentage display
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
        
        # Battery Specifications
        st.markdown(
            """
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
            """.format(
                CELLS_IN_SERIES=CELLS_IN_SERIES,
                CAPACITY_KWH=CAPACITY_KWH,
                MIN_CELL_V=MIN_CELL_V,
                MAX_CELL_V=MAX_CELL_V,
                **snap
            ),
            unsafe_allow_html=True
        )
    
    # ==========================================================
    # RIGHT: DIGITAL TWIN STATUS
    # ==========================================================
    with right:
        st.markdown("#### Digital Twin Status")
        
        dev_color = (
            "#4CAF50" if abs(snap['deviation']) < 2
            else "#FF9800" if abs(snap['deviation']) < 5
            else "#F44336"
        )
        
        # Twin comparison metrics
        twin_metrics_html = f"""
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
        """
        
        st.markdown(twin_metrics_html, unsafe_allow_html=True)
        
        # AI Insight Box
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
        
        # Health Progress Bar
        st.markdown("**Battery Life Remaining**")
        
        # Custom progress bar
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
                    transition: width 0.5s ease;
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
        
        # Quick Actions
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🔧 Quick Actions**")
        
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("📊 Full Diagnostics", use_container_width=True):
                st.info("Navigate to Diagnostics tab for detailed analysis")
        
        with action_col2:
            if st.button("🔮 Predict Future", use_container_width=True):
                st.info("Navigate to Scenario Lab for projections")
    
    st.divider()
    
    # ==========================================================
    # BATTERY TRENDS - ENHANCED WITH PLOTLY
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
    
    power_curve = snap['voltage'] * snap['current'] / 1000 + np.random.normal(0, 1, len(cycles))
    
    # Use plotly for interactive charts
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    t1, t2, t3 = st.columns(3)
    
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycles,
            y=soh_curve,
            mode='lines',
            name='SoH',
            line=dict(color='#4CAF50', width=3),
            fill='tozeroy',
            fillcolor='rgba(76, 175, 80, 0.2)'
        ))
        fig.update_layout(
            title="State of Health Trend",
            template="plotly_dark",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with t2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycles,
            y=temp_curve,
            mode='lines',
            name='Temperature',
            line=dict(color='#FF9800', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 152, 0, 0.2)'
        ))
        fig.add_hline(
            y=45,
            line_dash="dash",
            line_color="red",
            annotation_text="Warning Limit"
        )
        fig.update_layout(
            title="Temperature Trend",
            template="plotly_dark",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with t3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cycles,
            y=range_curve,
            mode='lines',
            name='Range',
            line=dict(color='#2196F3', width=3),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.2)'
        ))
        fig.update_layout(
            title="Available Range Trend",
            template="plotly_dark",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # ==========================================================
    # ADDITIONAL STATISTICS SECTION
    # ==========================================================
    st.markdown("### 📊 Detailed Statistics & Performance Metrics")
    
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.markdown("#### ⚡ Power & Energy Metrics")
        
        # Calculate additional metrics
        current_power = snap['voltage'] * snap['current'] / 1000  # kW
        energy_throughput = snap['cycle_count'] * CAPACITY_KWH  # Total kWh cycled
        efficiency = (snap['soh'] / 100) * 100  # Current efficiency
        
        metrics_data = {
            "Metric": [
                "Current Power Draw",
                "Total Energy Throughput",
                "Pack Efficiency",
                "Average Cycle Depth",
                "Charge Cycles per Year"
            ],
            "Value": [
                f"{current_power:.2f} kW",
                f"{energy_throughput:.0f} kWh",
                f"{efficiency:.1f}%",
                "85% DoD",
                f"~{365 * 60 / BASE_RANGE_KM:.0f}"
            ],
            "Status": [
                "✓ Normal",
                "✓ Tracked",
                "✓ Good" if efficiency > 85 else "⚠ Monitor",
                "✓ Optimal",
                "✓ Regular Use"
            ]
        }
        
        st.dataframe(
            pd.DataFrame(metrics_data),
            hide_index=True,
            use_container_width=True
        )
    
    with stats_col2:
        st.markdown("#### 🔬 Degradation Analysis")
        
        degradation_data = {
            "Factor": [
                "Temperature Stress",
                "Cycling Wear",
                "Calendar Aging",
                "Fast Charging Impact",
                "Deep Discharge Effect"
            ],
            "Contribution": [
                "35%",
                "30%",
                "20%",
                "10%",
                "5%"
            ],
            "Risk Level": [
                "🟡 Medium",
                "🟢 Low",
                "🟢 Low",
                "🟢 Low",
                "🟢 Low"
            ]
        }
        
        st.dataframe(
            pd.DataFrame(degradation_data),
            hide_index=True,
            use_container_width=True
        )
    
    st.divider()
    
    # ==========================================================
    # FOOTER INFO BOX
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
            f"""
            **⚠️ Recommendations**
            
            • Avoid charging above 80% daily
            
            • Minimize fast charging (< 30%)
            
            • Keep temperature below 40°C
            
            • Schedule service: **3 months**
            """
        )
# ════════════════════════════════════════════════════════════════
#  TAB 2 — DIGITAL TWIN
# ════════════════════════════════════════════════════════════════
with tab2:
    st.title("🧬 Digital Twin")
    st.markdown("**Physics model vs reality — how the digital twin tracks your battery**")

    st.sidebar.divider()
    st.sidebar.subheader("🧬 Digital Twin Controls")
    dt_ambient_temp = st.sidebar.slider("Ambient Temp (°C)", 20, 45, 31, key="dt_temp")
    dt_fast_charge  = st.sidebar.slider("Fast Charging (%)", 0, 100, 30, key="dt_fc")
    dt_dod          = st.sidebar.slider("Avg Depth of Discharge (%)", 50, 100, 85, key="dt_dod") / 100
    dt_cycle_count  = st.sidebar.slider("Current Cycle Count", 50, 1000, 420, key="dt_cycles")

    expected_soh_dt = physics_soh(dt_cycle_count, dt_ambient_temp, dt_dod, dt_fast_charge)
    actual_soh_dt   = expected_soh_dt - 0.6
    deviation_dt    = actual_soh_dt - expected_soh_dt

    st.divider()
    st.subheader("Digital Twin Status")

    dt_status = "Healthy" if abs(deviation_dt) < 2 else ("Monitor" if abs(deviation_dt) < 5 else "Action Needed")
    dt_color  = "#4CAF50" if dt_status == "Healthy" else ("#FF9800" if dt_status == "Monitor" else "#F44336")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected SoH", f"{expected_soh_dt:.1f}%")
    c2.metric("Actual SoH", f"{actual_soh_dt:.1f}%")
    c3.metric("Deviation", f"{deviation_dt:+.1f}%")
    c4.markdown(
        f"""<div style="text-align:center; padding-top:8px;">
            <span style="background-color:{dt_color}22; color:{dt_color}; font-weight:bold;
                        padding:6px 14px; border-radius:20px;">{dt_status}</span>
        </div>""", unsafe_allow_html=True
    )

    st.divider()
    st.subheader("Twin vs Reality")

    history_cycles = np.arange(max(0, dt_cycle_count-200), dt_cycle_count+1, 5)
    expected_history = [physics_soh(n, dt_ambient_temp, dt_dod, dt_fast_charge) for n in history_cycles]
    np.random.seed(dt_cycle_count)
    actual_history = [e - abs(np.random.normal(0.4, 0.3)) for e in expected_history]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(history_cycles, expected_history, color='#2196F3', linewidth=2.5,
            linestyle='--', label='Expected SoH (physics model)')
    ax.plot(history_cycles, actual_history, color='#FF9800', linewidth=2.5,
            label='Actual SoH (simulated real-world)')
    ax.fill_between(history_cycles, expected_history, actual_history, alpha=0.15, color='orange')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('SoH (%)')
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    st.pyplot(fig)

    st.divider()
    st.subheader("Residual Analysis")

    residuals = np.array(actual_history) - np.array(expected_history)
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.bar(history_cycles, residuals, color=['#F44336' if r < -2 else '#4CAF50' for r in residuals], width=4)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=-2, color='red', linestyle=':', linewidth=1, label='Deviation tolerance (-2%)')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Residual (Actual − Expected) %')
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    st.pyplot(fig)

    st.divider()
    st.subheader("Degradation Drivers")

    temp_w = 30 + max(0, (dt_ambient_temp - 25)) * 1.5
    fc_w   = 10 + dt_fast_charge * 0.3
    cyc_w  = 35
    cal_w  = 100 - (temp_w + fc_w + cyc_w)
    cal_w  = max(cal_w, 3)
    total  = temp_w + fc_w + cyc_w + cal_w
    drivers = {
        'Temperature': temp_w/total*100, 'Cycling': cyc_w/total*100,
        'Fast Charging': fc_w/total*100, 'Calendar Aging': cal_w/total*100
    }

    dcol1, dcol2 = st.columns([1.3, 1])
    with dcol1:
        fig, ax = plt.subplots(figsize=(6, 3))
        labels = list(drivers.keys())
        values = list(drivers.values())
        colors_d = ['#F44336', '#FF9800', '#2196F3', '#9E9E9E']
        ax.barh(labels, values, color=colors_d)
        for i, v in enumerate(values):
            ax.text(v + 1, i, f"{v:.0f}%", va='center', fontsize=10)
        ax.set_xlim(0, 60)
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        st.pyplot(fig)
    with dcol2:
        for label, val in drivers.items():
            st.markdown(f"**{label}**  {val:.0f}%")

    st.divider()
    st.subheader("Lifetime Forecast")

    future_cycles = np.arange(dt_cycle_count, dt_cycle_count + 2000, 10)
    future_soh = [physics_soh(n, dt_ambient_temp, dt_dod, dt_fast_charge) for n in future_cycles]
    eol_idx = next((i for i, s in enumerate(future_soh) if s <= 70), None)
    eol_cycle = future_cycles[eol_idx] if eol_idx else None
    eol_year_dt = 2026 + int((eol_cycle - dt_cycle_count) / 300) if eol_cycle else None

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(future_cycles, future_soh, color='#673AB7', linewidth=2.5)
    ax.axhline(y=70, color='black', linestyle='--', label='EOL threshold (70%)')
    if eol_cycle:
        ax.axvline(x=eol_cycle, color='red', linestyle=':', linewidth=1.5)
        ax.annotate(f'EOL: cycle {eol_cycle}', xy=(eol_cycle, 70),
                    xytext=(eol_cycle, 80), color='red', fontsize=10, fontweight='bold')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Projected SoH (%)')
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)
    st.pyplot(fig)

    st.markdown(
        f"""<div style="text-align:center; font-size:20px; font-weight:bold; padding:10px;">
            Predicted EOL: {eol_year_dt if eol_year_dt else 'beyond forecast horizon'}
        </div>""", unsafe_allow_html=True
    )

    st.divider()
    with st.expander("ℹ️ How does the Digital Twin estimate SoH?"):
        st.markdown(
            "The physics model combines **SEI growth** (a non-linear film that builds up "
            "on battery electrodes over time, following a square-root relationship with "
            "cycle count) with **linear wear** from charging stress. Temperature, depth of "
            "discharge, and fast-charging frequency all scale how fast this happens."
        )
    with st.expander("ℹ️ Why does 'Actual' differ from 'Expected'?"):
        st.markdown(
            "Real batteries never match a model perfectly — manufacturing variation, "
            "exact usage history, and measurement noise all create small deviations. "
            "Tracking this gap is what lets a digital twin catch a battery that's aging "
            "abnormally, before it becomes a safety issue."
        )


# ════════════════════════════════════════════════════════════════
#  TAB 3 — DIAGNOSTICS
# ════════════════════════════════════════════════════════════════
with tab3:
    st.title("🚨 Diagnostics")
    st.markdown("**Anomaly detection across sensor, thermal, health, and degradation systems**")

    # ── Fixed default scenario + optional regenerate button (FIXED) ──────────
    if 'diag_seed' not in st.session_state:
        st.session_state.diag_seed = 7

    regenerate = st.button("🔄 Generate New Fault Scenario", key="regen_btn")
    if regenerate:
        st.session_state.diag_seed = int(np.random.randint(1, 100000))

    st.caption(f"Scenario seed: {st.session_state.diag_seed}  "
               f"({'default demo' if st.session_state.diag_seed == 7 else 'custom run'})")

    rng = np.random.RandomState(st.session_state.diag_seed)

    n_readings = 120
    readings_diag = []
    fault_times = sorted(rng.choice(range(15, n_readings-10), size=rng.randint(2, 4), replace=False))

    for t in range(n_readings):
        voltage = 400 - (t * 0.6) + rng.normal(0, 1.5)
        current = 20 + rng.normal(0, 3)
        temp    = 28 + (t * 0.04) + rng.normal(0, 0.5)
        soc_true = max(0, 100 - t * 0.7)
        # Reduced noise (0.6 instead of 1.5) so normal sensor jitter doesn't
        # false-positive against the 5% drift tolerance
        soc_est  = soc_true + rng.normal(0, 0.6)

        if t in fault_times:
            fault_type = rng.choice(['voltage', 'thermal', 'soc', 'current'])
            if fault_type == 'voltage':
                voltage -= 25
            elif fault_type == 'thermal':
                temp += 22
            elif fault_type == 'soc':
                soc_est = soc_true + 9   # deliberate fault stays well above tolerance
            elif fault_type == 'current':
                current += 55

        readings_diag.append({'t': t, 'voltage': voltage, 'current': current,
                               'temp': temp, 'soc_true': soc_true, 'soc_est': soc_est})

# ── Run detector across the generated readings ────────────────────────────
    diag_detector = AnomalyDetector()
    diag_alerts = []
    last_clean_v, last_clean_c = None, None
    temp_hist = []

    for r in readings_diag:
        temp_hist.append(r['temp'])
        s_issues, is_glitch = diag_detector.check_sensor(r['voltage'], r['current'], last_clean_v, last_clean_c)
        h_issues = diag_detector.check_health(r['soc_est'], r['soc_true'])
        t_issues = diag_detector.check_thermal(temp_hist, max_heating_rate=3.5)

        for msg in s_issues:
            sev = "🔴" if "outside physical" in msg else "🟠"
            diag_alerts.append({'time': r['t'], 'category': 'Sensor', 'severity': sev, 'message': msg})
        for msg in h_issues:
            diag_alerts.append({'time': r['t'], 'category': 'Health', 'severity': '🟠', 'message': msg})
        for msg in t_issues:
            sev = "🔴" if "DANGER" in msg else "🟠"
            diag_alerts.append({'time': r['t'], 'category': 'Thermal', 'severity': sev, 'message': msg})

        if not is_glitch:
            last_clean_v, last_clean_c = r['voltage'], r['current']

    # Degradation deviation check (reuses Day 2 physics)
    deg_issues, expected_deg = diag_detector.check_degradation(
        actual_soh=84.5, cycle_number=420, soh_tracker=SoHTracker(CAPACITY_AH)
    )
    for msg in deg_issues:
        diag_alerts.append({'time': '—', 'category': 'Degradation', 'severity': '🟠', 'message': msg})

    # ── Tally categories for system health + root cause ──────────────────────
    cat_counts = {'Sensor': 0, 'Health': 0, 'Thermal': 0, 'Degradation': 0}
    sev_by_cat = {}
    for a in diag_alerts:
        cat_counts[a['category']] += 1
        if a['category'] not in sev_by_cat or a['severity'] == '🔴':
            sev_by_cat[a['category']] = a['severity']

    def health_dot(cat):
        if cat_counts.get(cat, 0) == 0:
            return "🟢"
        return sev_by_cat.get(cat, "🟠")

    # ── SECTION 1: System Health ──────────────────────────────────────────────
    st.divider()
    st.subheader("System Health")
    h1, h2, h3, h4 = st.columns(4)
    h1.markdown(f"**Battery**<br><span style='font-size:28px;'>{health_dot('Health')}</span>", unsafe_allow_html=True)
    h2.markdown(f"**Thermal**<br><span style='font-size:28px;'>{health_dot('Thermal')}</span>", unsafe_allow_html=True)
    h3.markdown(f"**Sensors**<br><span style='font-size:28px;'>{health_dot('Sensor')}</span>", unsafe_allow_html=True)
    h4.markdown(f"**BMS**<br><span style='font-size:28px;'>{health_dot('Degradation')}</span>", unsafe_allow_html=True)

    # ── SECTION 2: Active Alerts ───────────────────────────────────────────────
    st.divider()
    st.subheader("Active Alerts")
    if diag_alerts:
        for a in diag_alerts[-8:]:
            st.markdown(f"{a['severity']} **{a['category']}** — {a['message']}")
    else:
        st.success("🟢 No active alerts — all systems nominal.")

    # ── SECTION 3: Alert Timeline ──────────────────────────────────────────────
    st.divider()
    st.subheader("Alert Timeline")
    if diag_alerts:
        for a in diag_alerts:
            time_str = f"t={a['time']}" if a['time'] != '—' else "ongoing"
            st.markdown(f"`{time_str}`  {a['severity']} {a['category']} — {a['message'][:70]}")
    else:
        st.caption("No events recorded this session.")

    # ── SECTION 4: Root Cause Analysis ─────────────────────────────────────────
    st.divider()
    st.subheader("Probable Cause Analysis")
    total_alerts = max(sum(cat_counts.values()), 1)
    causes = {
        'Temperature Stress': cat_counts['Thermal'] / total_alerts * 100,
        'Sensor Faults':      cat_counts['Sensor'] / total_alerts * 100,
        'SoC/BMS Drift':      cat_counts['Health'] / total_alerts * 100,
        'Degradation':        cat_counts['Degradation'] / total_alerts * 100,
    }
    rc_col1, rc_col2 = st.columns([1.3, 1])
    with rc_col1:
        fig, ax = plt.subplots(figsize=(6, 2.8))
        labels_rc = list(causes.keys())
        vals_rc   = list(causes.values())
        ax.barh(labels_rc, vals_rc, color=['#F44336', '#FF9800', '#2196F3', '#9C27B0'])
        for i, v in enumerate(vals_rc):
            ax.text(v + 1, i, f"{v:.0f}%", va='center', fontsize=9)
        ax.set_xlim(0, max(max(vals_rc, default=10), 10) + 15)
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        st.pyplot(fig)
    with rc_col2:
        for label, val in causes.items():
            st.markdown(f"**{label}**  {val:.0f}%")

    # ── SECTION 5: Digital Twin Diagnostics ────────────────────────────────────
    st.divider()
    st.subheader("Digital Twin Diagnostics")
    st.markdown(
        f"""<div style="border:1px solid #ddd; border-radius:10px; padding:14px;">
            Expected SoH: <b>{expected_deg:.1f}%</b> &nbsp;|&nbsp;
            Actual SoH: <b>84.5%</b> &nbsp;|&nbsp;
            Deviation: <b style="color:{'#F44336' if expected_deg-84.5>5 else '#4CAF50'}">
            {84.5-expected_deg:+.1f}%</b>
        </div>""", unsafe_allow_html=True
    )
    st.caption("See the Digital Twin tab for full residual analysis and lifetime forecast.")

    # ── SECTION 6: Sensor Diagnostics ──────────────────────────────────────────
    st.divider()
    st.subheader("Sensor Diagnostics")
    sensor_status_map = {
        'Voltage Sensor':    'Warning' if cat_counts['Sensor'] > 0 else 'Healthy',
        'Current Sensor':    'Warning' if cat_counts['Sensor'] > 0 else 'Healthy',
        'Temperature Sensor':'Warning' if cat_counts['Thermal'] > 0 else 'Healthy',
        'SoC Model':         'Recalibration Needed' if cat_counts['Health'] > 0 else 'Healthy',
    }
    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, (name, status) in zip([sc1, sc2, sc3, sc4], sensor_status_map.items()):
        color_s = '#4CAF50' if status == 'Healthy' else '#FF9800'
        last_event = next((a['message'][:40] for a in reversed(diag_alerts)
                           if name.split()[0].lower() in a['message'].lower()), "No recent anomalies")
        col.markdown(
            f"""<div style="border:1px solid #ddd; border-radius:8px; padding:10px; text-align:center;">
                <b>{name}</b><br>
                <span style="color:{color_s}; font-weight:bold;">{status}</span><br>
                <span style="font-size:11px; color:#888;">{last_event}</span>
            </div>""", unsafe_allow_html=True
        )

    # ── SECTION 7: Recommended Actions ─────────────────────────────────────────
    st.divider()
    st.subheader("Recommended Actions")
    actions = []
    if cat_counts['Thermal'] > 0:
        actions.append("Improve thermal management — reduce fast charging during hot conditions")
    if cat_counts['Sensor'] > 0:
        actions.append("Inspect voltage/current sensor wiring for intermittent faults")
    if cat_counts['Health'] > 0:
        actions.append("Schedule BMS recalibration — allow an overnight rest for OCV reset")
    if cat_counts['Degradation'] > 0:
        actions.append("Reduce fast charging frequency to slow accelerated aging")
    if not actions:
        actions.append("No action needed — continue normal operation")


# ════════════════════════════════════════════════════════════════
#  TAB 4 — SCENARIO LAB (placeholder, to be built next)
# ════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════
# TAB 4 — BATTERY SCENARIO LAB
# ════════════════════════════════════════════════════════════════
with tab4:

    st.title("🧪 Battery Scenario Laboratory")
    st.markdown(
        "**Design battery usage scenarios and evaluate their impact "
        "on health, range and battery lifetime.**"
    )

    preset = st.selectbox(
        "Choose Scenario",
        [
            "Custom",
            "Healthy Commuter",
            "Typical Bangalore Driver",
            "Heavy User",
            "Hot Climate Driver",
            "Fleet Vehicle"
        ]
    )

    defaults = {
        "Healthy Commuter": (25, 10, 40, 60),
        "Typical Bangalore Driver": (33, 30, 60, 80),
        "Heavy User": (42, 80, 150, 100),
        "Hot Climate Driver": (45, 40, 70, 85),
        "Fleet Vehicle": (35, 60, 180, 90),
        "Custom": (33, 30, 60, 80)
    }

    temp0, fc0, daily0, dod0 = defaults[preset]

    st.divider()

    st.subheader("Scenario Inputs")

    c1, c2 = st.columns(2)

    with c1:
        sim_temp = st.slider(
            "Ambient Temperature (°C)",
            15, 50, temp0
        )

        sim_fc = st.slider(
            "Fast Charging Frequency (%)",
            0, 100, fc0
        )

    with c2:
        sim_daily = st.slider(
            "Daily Distance (km)",
            20, 250, daily0
        )

        sim_dod_pct = st.slider(
            "Depth of Discharge (%)",
            20, 100, dod0
        )

    years = st.slider(
        "Simulation Horizon (Years)",
        1, 15, 6
    )

    sim_dod = sim_dod_pct / 100

    annual_cycles = sim_daily * 365 / BASE_RANGE_KM
    total_cycles = int(annual_cycles * years)

    final_soh = physics_soh(
        total_cycles,
        sim_temp,
        sim_dod,
        sim_fc
    )

    final_range = BASE_RANGE_KM * (final_soh / 100)

    range_lost = BASE_RANGE_KM - final_range

    stress_score = (
        (sim_temp - 20) * 1.1 +
        sim_fc * 0.25 +
        sim_dod_pct * 0.20
    )

    stress_score = np.clip(stress_score, 0, 100)

    battery_value = 450000
    value_lost = battery_value * (100 - final_soh) / 100

    future_cycles = np.arange(0, 8000, 20)

    future_soh = [
        physics_soh(
            c,
            sim_temp,
            sim_dod,
            sim_fc
        )
        for c in future_cycles
    ]

    eol_idx = next(
        (i for i, s in enumerate(future_soh) if s <= 70),
        None
    )

    eol_year = None

    if eol_idx is not None:
        eol_cycles = future_cycles[eol_idx]
        eol_year = 2026 + int(eol_cycles / annual_cycles)

    st.divider()

    st.subheader("Simulation Results")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Final SoH",
        f"{final_soh:.1f}%"
    )

    k2.metric(
        "Final Range",
        f"{final_range:.0f} km"
    )

    k3.metric(
        "Stress Score",
        f"{stress_score:.0f}/100"
    )

    k4.metric(
        "Predicted EOL",
        str(eol_year) if eol_year else "Beyond Horizon"
    )

    st.markdown(
        f"""
        **Range Lost:** {range_lost:.0f} km

        **Estimated Battery Value Lost:** ₹{value_lost:,.0f}
        """
    )

    st.divider()

    chart1, chart2 = st.columns(2)

    with chart1:

        cycles = np.arange(0, total_cycles + 20, 20)

        soh_curve = [
            physics_soh(
                c,
                sim_temp,
                sim_dod,
                sim_fc
            )
            for c in cycles
        ]

        fig, ax = plt.subplots(figsize=(6, 3))

        ax.plot(
            cycles,
            soh_curve,
            linewidth=2.5
        )

        ax.axhline(
            70,
            linestyle='--'
        )

        ax.set_title("SoH Forecast")
        ax.set_xlabel("Cycles")
        ax.set_ylabel("SoH (%)")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    with chart2:

        range_curve = [
            BASE_RANGE_KM * (s / 100)
            for s in soh_curve
        ]

        fig, ax = plt.subplots(figsize=(6, 3))

        ax.plot(
            cycles,
            range_curve,
            linewidth=2.5
        )

        ax.set_title("Range Forecast")
        ax.set_xlabel("Cycles")
        ax.set_ylabel("Range (km)")
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    st.divider()

    st.subheader("Scenario Comparison")

    scenarios = [
        ("Best Case", 25, 10, 0.60),
        ("Typical Bangalore", 33, 30, 0.80),
        ("Heavy User", 42, 80, 1.00),
        ("Your Scenario", sim_temp, sim_fc, sim_dod)
    ]

    rows = []

    for name, t, fc, d in scenarios:

        soh = physics_soh(
            total_cycles,
            t,
            d,
            fc
        )

        stress = np.clip(
            (t - 20) * 1.1 +
            fc * 0.25 +
            d * 100 * 0.20,
            0,
            100
        )

        rows.append({
            "Scenario": name,
            "SoH (%)": round(soh, 1),
            "Range Lost (km)": round(
                BASE_RANGE_KM - BASE_RANGE_KM * (soh / 100)
            ),
            "Stress Score": round(stress)
        })

    st.dataframe(rows, use_container_width=True)

    st.divider()

    st.subheader("Optimization Suggestions")

    tips = []

    if sim_temp > 35:
        tips.append(
            "Ambient temperature is accelerating degradation. Use shaded parking and thermal management."
        )

    if sim_fc > 50:
        tips.append(
            "Reduce fast charging frequency to improve long-term battery life."
        )

    if sim_dod_pct > 90:
        tips.append(
            "Avoid frequent 100% discharge cycles."
        )

    if stress_score < 30:
        tips.append(
            "Current usage pattern is battery-friendly."
        )

    for tip in tips:
        st.markdown(f"• {tip}")

    st.caption(
        "Physics-based Digital Twin • Anomaly Detection • Scenario Simulation"
    )
