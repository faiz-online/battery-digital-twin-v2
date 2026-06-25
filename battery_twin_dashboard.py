
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Battery Digital Twin",
    page_icon="🔋",
    layout="wide"
)

st.error("✅ THIS IS THE NEW VERSION - if you see this, the deploy worked - timestamp: v2.1")

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
sensor_status   = status_dot(True)
fault_status    = status_dot(True)

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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("State of Health (SoH)", f"{snap['soh']:.1f}%")
    col2.metric("State of Charge (SoC)", f"{snap['soc']:.1f}%")
    col3.metric("Range Remaining", f"{snap['range_km']:.0f} km")
    col4.metric("Remaining Useful Life", f"{snap['rul_years']:.1f} yrs")

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
    st.markdown("**Physics model vs reality — how the digital twin tracks your battery**")

    # ── INTERACTIVE CONTROLS (sidebar within tab) ───────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("🧬 Digital Twin Controls")
    dt_ambient_temp = st.sidebar.slider("Ambient Temp (°C)", 20, 45, 31, key="dt_temp")
    dt_fast_charge  = st.sidebar.slider("Fast Charging (%)", 0, 100, 30, key="dt_fc")
    dt_dod          = st.sidebar.slider("Avg Depth of Discharge (%)", 50, 100, 85, key="dt_dod") / 100
    dt_cycle_count  = st.sidebar.slider("Current Cycle Count", 50, 1000, 420, key="dt_cycles")

    # ── Shared physics model ────────────────────────────────────────────────
    def physics_soh(n, temp, dod, fc_pct):
        tf = 1.0 + max(0, (temp - 25) / 10) * 0.45
        df = 0.80 + (dod * 0.28)
        fcf = 1.0 + (0.12 * (fc_pct/100))
        return float(np.clip(100 - A_SEI*np.sqrt(n)*tf*df - B_LINEAR*n*fcf*tf, 50, 100))

    expected_soh_dt = physics_soh(dt_cycle_count, dt_ambient_temp, dt_dod, dt_fast_charge)
    actual_soh_dt   = expected_soh_dt - 0.6   # simulated small real-world deviation
    deviation_dt    = actual_soh_dt - expected_soh_dt

    # ── SECTION 1: Digital Twin Status ──────────────────────────────────────
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

    # ── SECTION 2: Twin vs Reality ───────────────────────────────────────────
    st.divider()
    st.subheader("Twin vs Reality")

    history_cycles = np.arange(max(0, dt_cycle_count-200), dt_cycle_count+1, 5)
    expected_history = [physics_soh(n, dt_ambient_temp, dt_dod, dt_fast_charge) for n in history_cycles]
    np.random.seed(dt_cycle_count)  # deterministic "actual" noise per slider state
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

    # ── SECTION 3: Residual Analysis ─────────────────────────────────────────
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

    # ── SECTION 4: Degradation Drivers ───────────────────────────────────────
    st.divider()
    st.subheader("Degradation Drivers")

    # Approximate contribution weighting, shifts slightly based on current sliders
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

    # ── SECTION 5: Lifetime Forecast ─────────────────────────────────────────
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

    # ── How this works (explainers) ─────────────────────────────────────────
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

with tab3:
    st.title("🚨 Diagnostics")
    st.info("Coming next — anomaly detection")

with tab4:
    st.title("🔮 Scenario Lab")
    st.info("Coming next — what-if simulator")
