
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

# ── SIMULATED "LIVE" SNAPSHOT — mid-life battery scenario ────────────────────
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

    # Actual SoH deviates slightly from the physics-expected curve
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

# ── HEALTH SCORE (composite 0-100) ───────────────────────────────────────────
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
sensor_status   = status_dot(True)   # no sensor faults in this static snapshot
fault_status    = status_dot(True)   # no active faults in this static snapshot

# ── AI INSIGHT TEXT ──────────────────────────────────────────────────────────
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

# ════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔋 Battery Overview", "🧬 Digital Twin", "🚨 Diagnostics", "🔮 Scenario Lab"
])

with tab1:
    st.title("🔋 Battery Overview")
    st.markdown("**Tata Nexon EV (30.2 kWh) | BESCOM Digital Twin Project**")

    # ── Health Score card ─────────────────────────────────────────────────────
    score_color = '#4CAF50' if health_score > 85 else '#FF9800' if health_score > 70 else '#F44336'
    st.markdown(
        f"""
        <div style="background-color:{score_color}15; border: 2px solid {score_color};
                    border-radius: 10px; padding: 14px 20px; margin-bottom: 14px;
                    display: flex; align-items: center; justify-content: space-between;">
            <span style="font-size: 18px; font-weight: 600; color: #333;">Health Score</span>
            <span style="font-size: 32px; font-weight: bold; color:{score_color};">
                {health_score:.0f}/100
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Top metric row ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("State of Health (SoH)", f"{snap['soh']:.1f}%")
    col2.metric("State of Charge (SoC)", f"{snap['soc']:.1f}%")
    col3.metric("Range Remaining", f"{snap['range_km']:.0f} km")
    col4.metric("Remaining Useful Life", f"{snap['rul_years']:.1f} yrs")

    # ── Status indicator row ─────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="display: flex; gap: 28px; padding: 10px 4px; font-size: 15px;">
            <span>{soh_status} SoH Excellent</span>
            <span>{thermal_status} Thermal Stable</span>
            <span>{sensor_status} Sensors Healthy</span>
            <span>{fault_status} No Active Faults</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()
    col_left, col_right = st.columns([1, 1])

    # ── Redesigned battery pack visualization ────────────────────────────────
    with col_left:
        st.subheader("Battery Pack")
        n_blocks = 14
        filled = int(round(n_blocks * snap['soc'] / 100))
        bar = "█" * filled + "░" * (n_blocks - filled)

        pack_color = '#4CAF50' if snap['soh'] > 85 else '#FF9800' if snap['soh'] > 75 else '#F44336'

        st.markdown(
            f"""
            <div style="border: 3px solid #333; border-radius: 12px; padding: 18px;
                        background: linear-gradient(135deg, #fafafa, #f0f0f0);">
                <div style="font-family: monospace; font-size: 26px; letter-spacing: 1px;
                            color: {pack_color}; text-align:center;">
                    {bar}
                </div>
                <div style="text-align:center; font-size: 22px; font-weight:bold; margin-top:4px;">
                    {snap['soc']:.0f}%
                </div>
                <hr style="margin: 12px 0;">
                <div style="display: flex; justify-content: space-around; text-align:center;">
                    <div>
                        <div style="font-size:13px; color:#777;">Pack Voltage</div>
                        <div style="font-size:20px; font-weight:bold;">{snap['voltage']:.0f}V</div>
                    </div>
                    <div>
                        <div style="font-size:13px; color:#777;">Current</div>
                        <div style="font-size:20px; font-weight:bold;">{snap['current']:.0f}A</div>
                    </div>
                    <div>
                        <div style="font-size:13px; color:#777;">Temperature</div>
                        <div style="font-size:20px; font-weight:bold;">{snap['temperature']:.0f}°C</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── Digital Twin status card ──────────────────────────────────────────────
    with col_right:
        st.subheader("Digital Twin Status")
        dev_color = '#4CAF50' if abs(snap['deviation']) < 2 else '#FF9800' if abs(snap['deviation']) < 5 else '#F44336'
        st.markdown(
            f"""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 16px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:#666;">Expected SoH</span>
                    <span style="font-weight:bold;">{snap['expected_soh']:.1f}%</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:#666;">Actual SoH</span>
                    <span style="font-weight:bold;">{snap['soh']:.1f}%</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#666;">Deviation</span>
                    <span style="font-weight:bold; color:{dev_color};">{snap['deviation']:+.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background-color:#E3F2FD; border-radius: 10px; padding: 14px;">
                <span style="font-weight:bold;">💡 AI Insight</span><br>
                <span style="font-size:14px;">{insight_text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # ── Mini trend charts ─────────────────────────────────────────────────────
    st.subheader("Recent Trends")
    trend_cycles = np.arange(max(0, snap['cycle_count']-30), snap['cycle_count']+1)

    def soh_at(n):
        tf = 1.0 + max(0, (snap['ambient_temp'] - 25) / 10) * 0.45
        df = 0.80 + (0.85 * 0.28)
        fc = 1.0 + (0.12 * 0.30)
        return float(np.clip(100 - A_SEI*np.sqrt(n)*tf*df - B_LINEAR*n*fc*tf, 50, 100))

    soh_trend   = [soh_at(n) for n in trend_cycles]
    temp_trend  = snap['ambient_temp'] + 2.3 + np.random.normal(0, 0.4, len(trend_cycles))
    range_trend = [BASE_RANGE_KM * (s/100) * (snap['soc']/100) for s in soh_trend]

    tcol1, tcol2, tcol3 = st.columns(3)

    with tcol1:
        fig, ax = plt.subplots(figsize=(3.2, 1.6))
        ax.plot(trend_cycles, soh_trend, color='#2196F3', linewidth=2)
        ax.set_title("SoH Trend", fontsize=10)
        ax.set_xticks([]); ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)

    with tcol2:
        fig, ax = plt.subplots(figsize=(3.2, 1.6))
        ax.plot(trend_cycles, temp_trend, color='#F44336', linewidth=2)
        ax.set_title("Temperature Trend", fontsize=10)
        ax.set_xticks([]); ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)

    with tcol3:
        fig, ax = plt.subplots(figsize=(3.2, 1.6))
        ax.plot(trend_cycles, range_trend, color='#4CAF50', linewidth=2)
        ax.set_title("Range Trend", fontsize=10)
        ax.set_xticks([]); ax.spines[['top','right']].set_visible(False)
        st.pyplot(fig)

with tab2:
    st.title("🧬 Digital Twin")
    st.info("Coming next — Day 1 simulator + Day 2 SoC/SoH model")

with tab3:
    st.title("🚨 Diagnostics")
    st.info("Coming next — anomaly detection")

with tab4:
    st.title("🔮 Scenario Lab")
    st.info("Coming next — what-if simulator")
