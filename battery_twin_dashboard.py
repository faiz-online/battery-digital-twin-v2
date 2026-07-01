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
# TAB 1 — BATTERY OVERVIEW
# ════════════════════════════════════════════════════════════════
with tab1:

    st.title("🔋 Battery Digital Twin Overview")
    st.caption("Tata Nexon EV • 30.2 kWh Battery Pack • BESCOM Digital Twin")

    # ─────────────────────────────────────────
# Battery Health Score
# ─────────────────────────────────────────

st.subheader("Battery Health")

score_col1, score_col2, score_col3 = st.columns([1.5,1,1])

with score_col1:
    st.metric(
        label="Battery Health Score",
        value=f"{health_score:.0f}/100"
    )

with score_col2:
    st.metric(
        label="Predicted EOL",
        value=f"{snap['eol_year']}"
    )

with score_col3:
    status = (
        "Excellent" if health_score >= 90 else
        "Good" if health_score >= 80 else
        "Warning"
    )
    st.metric(
        label="Battery Status",
        value=status
    )

    st.progress(int(health_score))

if health_score >= 90:
    st.success("Battery operating in optimal condition.")
elif health_score >= 80:
    st.warning("Battery showing normal aging.")
else:
    st.error("Battery requires inspection.")
    

    # ==========================================================
    # MAIN KPIs
    # ==========================================================
    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric(
        "State of Health",
        f"{snap['soh']:.1f}%"
    )

    c2.metric(
        "State of Charge",
        f"{snap['soc']:.1f}%"
    )

    c3.metric(
        "Range Remaining",
        f"{snap['range_km']:.0f} km"
    )

    c4.metric(
        "Battery Temp",
        f"{snap['temperature']:.1f}°C"
    )

    c5.metric(
        "Remaining Life",
        f"{snap['rul_years']:.1f} yr"
    )

    # ==========================================================
    # STATUS BAR
    # ==========================================================
    st.markdown(
        f"""
        <div style="
            background:#F5F5F5;
            padding:14px;
            border-radius:10px;
            margin-top:15px;
            margin-bottom:20px;
        ">
        {soh_status} Battery Healthy &nbsp;&nbsp;&nbsp;
        {thermal_status} Thermal Stable &nbsp;&nbsp;&nbsp;
        {sensor_status} Sensors Operational &nbsp;&nbsp;&nbsp;
        {fault_status} No Active Faults
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================================
    # LEFT + RIGHT
    # ==========================================================
    left,right = st.columns([1.2,1])

    # ==========================================================
    # BATTERY PACK
    # ==========================================================
    with left:

        st.subheader("Battery Pack Visualization")

        blocks = 20
        filled = int(blocks*snap['soc']/100)

        html = ""

        for i in range(blocks):

            color = "#4CAF50" if i<filled else "#E0E0E0"

            html += f"""
            <div style="
                width:24px;
                height:60px;
                background:{color};
                border-radius:4px;
                margin-right:4px;
                display:inline-block;
            "></div>
            """

        st.markdown(html,unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)

        v1,v2,v3 = st.columns(3)

        v1.metric(
            "Pack Voltage",
            f"{snap['voltage']:.0f} V"
        )

        v2.metric(
            "Pack Current",
            f"{snap['current']:.0f} A"
        )

        v3.metric(
            "Capacity",
            f"{snap['capacity_ah']:.1f} Ah"
        )

        st.markdown("---")

        st.markdown(
            f"""
            **Battery Chemistry**

            • Type: Lithium-ion NMC

            • Cells: {CELLS_IN_SERIES}S

            • Capacity: {CAPACITY_KWH} kWh

            • Cycle Count: {snap['cycle_count']}

            • Ambient Temp: {snap['ambient_temp']}°C
            """
        )

    # ==========================================================
    # DIGITAL TWIN STATUS
    # ==========================================================
    with right:

        st.subheader("Digital Twin Status")

        dev_color = (
            "#4CAF50"
            if abs(snap['deviation']) < 2
            else "#FF9800"
            if abs(snap['deviation']) < 5
            else "#F44336"
        )

        st.metric(
            "Expected SoH",
            f"{snap['expected_soh']:.1f}%"
        )

        st.metric(
            "Actual SoH",
            f"{snap['soh']:.1f}%"
        )

        st.metric(
            "Residual Error",
            f"{snap['deviation']:+.1f}%"
        )

        st.markdown(
            f"""
            <div style="
                background:{dev_color}22;
                padding:15px;
                border-radius:10px;
                margin-top:20px;
            ">
                <b>AI Insight</b><br><br>
                {insight_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>",unsafe_allow_html=True)

        st.progress(
            min(1.0,snap['soh']/100)
        )

        st.caption(
            "Battery Remaining Life Indicator"
        )

    # ==========================================================
    # RECENT TRENDS
    # ==========================================================
    st.divider()

    st.subheader("Battery Trends")

    cycles = np.arange(
        snap['cycle_count']-40,
        snap['cycle_count']+1
    )

    soh_curve = [
        physics_soh(
            c,
            snap['ambient_temp'],
            0.85,
            30
        )
        for c in cycles
    ]

    temp_curve = (
        snap['temperature']
        + np.random.normal(
            0,
            0.4,
            len(cycles)
        )
    )

    range_curve = [
        BASE_RANGE_KM*(x/100)
        for x in soh_curve
    ]

    t1,t2,t3 = st.columns(3)

    with t1:

        fig,ax = plt.subplots(
            figsize=(4,2)
        )

        ax.plot(
            cycles,
            soh_curve,
            linewidth=2
        )

        ax.set_title(
            "SoH Trend"
        )

        ax.grid()

        st.pyplot(fig)

    with t2:

        fig,ax = plt.subplots(
            figsize=(4,2)
        )

        ax.plot(
            cycles,
            temp_curve,
            linewidth=2
        )

        ax.set_title(
            "Temperature Trend"
        )

        ax.grid()

        st.pyplot(fig)

    with t3:

        fig,ax = plt.subplots(
            figsize=(4,2)
        )

        ax.plot(
            cycles,
            range_curve,
            linewidth=2
        )

        ax.set_title(
            "Range Trend"
        )

        ax.grid()

        st.pyplot(fig)

    # ==========================================================
    # FOOTER
    # ==========================================================
    st.divider()

    st.info(
        f"""
        Digital Twin Confidence: 97.2%

        Predicted EOL: {snap['eol_year']}

        Remaining Useful Life:
        {snap['rul_cycles']:.0f} cycles
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

        # ── Degradation deviation check ─────────────────────────────────
    actual_soh = float(rng.uniform(70, 98))
    cycle_number = int(rng.randint(100, 900))

    deg_issues, expected_deg = diag_detector.check_degradation(
        actual_soh=actual_soh,
        cycle_number=cycle_number,
        soh_tracker=SoHTracker(CAPACITY_AH)
    )

    for msg in deg_issues:
        diag_alerts.append({
            'time': '—',
            'category': 'Degradation',
            'severity': '🟠',
            'message': msg
        })

    # ── Tally categories for system health + root cause ─────────────
    cat_counts = {
        'Sensor': 0,
        'Health': 0,
        'Thermal': 0,
        'Degradation': 0
    }

    sev_by_cat = {}

    for a in diag_alerts:
        cat_counts[a['category']] += 1

        if a['category'] not in sev_by_cat or a['severity'] == '🔴':
            sev_by_cat[a['category']] = a['severity']

    # ── Health scoring ───────────────────────────────────────────────
    battery_score = 100
    battery_score -= cat_counts['Sensor'] * 8
    battery_score -= cat_counts['Health'] * 10
    battery_score -= cat_counts['Thermal'] * 15
    battery_score -= cat_counts['Degradation'] * 20

    battery_score = max(0, min(100, battery_score))

    def health_dot(score):
        if score >= 85:
            return "🟢"
        elif score >= 60:
            return "🟠"
        else:
            return "🔴"

    battery_health = max(0, 100 - cat_counts['Health'] * 20)
    thermal_health = max(0, 100 - cat_counts['Thermal'] * 15)
    sensor_health = max(0, 100 - cat_counts['Sensor'] * 10)
    bms_health = max(0, 100 - cat_counts['Degradation'] * 25)

    # ── SECTION 1: System Health ────────────────────────────────────
    st.divider()
    st.subheader("System Health")

    h1, h2, h3, h4 = st.columns(4)

    h1.markdown(
        f"**Battery**<br><span style='font-size:28px;'>{health_dot(battery_health)}</span>",
        unsafe_allow_html=True
    )

    h2.markdown(
        f"**Thermal**<br><span style='font-size:28px;'>{health_dot(thermal_health)}</span>",
        unsafe_allow_html=True
    )

    h3.markdown(
        f"**Sensors**<br><span style='font-size:28px;'>{health_dot(sensor_health)}</span>",
        unsafe_allow_html=True
    )

    h4.markdown(
        f"**BMS**<br><span style='font-size:28px;'>{health_dot(bms_health)}</span>",
        unsafe_allow_html=True
    )

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
        Actual SoH: <b>{actual_soh:.1f}%</b> &nbsp;|&nbsp;
        Deviation: <b style="color:{'#F44336' if expected_deg-actual_soh>5 else '#4CAF50'}">
        {actual_soh-expected_deg:+.1f}%</b>
    </div>""",
    unsafe_allow_html=True
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
