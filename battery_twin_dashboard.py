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

    # ==========================================================
    # HEALTH SCORE HERO CARD
    # ==========================================================
    score_color = (
        "#4CAF50" if health_score > 85
        else "#FF9800" if health_score > 70
        else "#F44336"
    )

    st.markdown(
    f"""
    <div style="
        background-color:#1E1E1E;
        border-left:8px solid {score_color};
        border-radius:15px;
        padding:20px;
        margin-bottom:20px;
        box-shadow:0 2px 8px rgba(0,0,0,0.2);
    ">

        <div style="
            font-size:18px;
            color:#BBBBBB;
            margin-bottom:10px;
        ">
            🔋 Battery Health Score
        </div>

        <div style="
            font-size:52px;
            font-weight:bold;
            color:{score_color};
            line-height:1;
        ">
            {health_score:.0f}/100
        </div>

        <div style="
            color:#AAAAAA;
            font-size:15px;
            margin-top:12px;
        ">
            Estimated End of Life: {snap['eol_year']}
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

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
#  TAB 2 — DIGITAL TWIN (Enhanced)
#  Features: health card, confidence score, physics vs reality,
#  residual heatmap, degradation drivers, RUL gauge, failure risk,
#  aging trajectory, AI insight, physics explainer
# ════════════════════════════════════════════════════════════════
with tab2:

    # ── Compute core twin values from sidebar sliders ────────────────────────
    exp_soh  = physics_soh(dt_cycles, dt_temp, dt_dod, dt_fc)
    act_soh  = exp_soh - 0.6   # simulated real-world small deviation
    dev      = act_soh - exp_soh
    abs_dev  = abs(dev)

    # ── Twin Confidence Score (0-100) ────────────────────────────────────────
    # Based on how closely actual tracks expected + data quality proxy
    confidence = float(np.clip(100 - abs_dev * 8 - max(0,(dt_temp-35))*0.5, 0, 100))

    # ── Failure Risk Score (0-100) ───────────────────────────────────────────
    # Based on SoH, temperature stress, and fast-charge frequency
    risk_soh  = max(0, (85 - act_soh) * 2.5)    # rises as SoH drops below 85
    risk_temp = max(0, (dt_temp - 30) * 1.5)     # rises above 30°C
    risk_fc   = dt_fc * 0.3                       # scales with fast-charge %
    risk_score = float(np.clip(risk_soh + risk_temp + risk_fc, 0, 100))

    # ── Twin status ──────────────────────────────────────────────────────────
    if abs_dev < 2 and risk_score < 30:
        twin_status = "Tracking Well"
        twin_color  = ACCENT_GREEN
    elif abs_dev < 5 or risk_score < 60:
        twin_status = "Minor Deviation"
        twin_color  = ACCENT_ORANGE
    else:
        twin_status = "Significant Divergence"
        twin_color  = ACCENT_RED

    # ── SECTION 1: Digital Twin Health Status ────────────────────────────────
    st.markdown('<div class="section-header">Digital Twin Health Status</div>',
                unsafe_allow_html=True)

    h1, h2, h3, h4, h5 = st.columns(5)
    for col, label, val, color, sub in [
        (h1, "Expected SoH",      f"{exp_soh:.1f}%",     ACCENT_BLUE,   "Physics model"),
        (h2, "Actual SoH",        f"{act_soh:.1f}%",     ACCENT_GREEN,  "Real-world est."),
        (h3, "Deviation",         f"{dev:+.1f}%",        twin_color,    "Actual − Expected"),
        (h4, "Twin Confidence",   f"{confidence:.0f}%",  ACCENT_PURPLE, "Model reliability"),
        (h5, "Failure Risk",      f"{risk_score:.0f}/100", ACCENT_RED if risk_score>60 else ACCENT_ORANGE if risk_score>30 else ACCENT_GREEN, "Composite risk"),
    ]:
        col.markdown(f"""
        <div class='metric-card' style='border-top:3px solid {color};'>
            <div class='label'>{label}</div>
            <div class='value' style='color:{color}; font-size:22px;'>{val}</div>
            <div class='sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    # Twin status pill
    st.markdown(f"""
    <div style='margin:10px 0 18px 0;'>
        <span style='font-size:12px; color:#8B9DC3;'>Twin Status: </span>
        <span class='pill {"pill-green" if twin_color==ACCENT_GREEN else "pill-orange" if twin_color==ACCENT_ORANGE else "pill-red"}'>{twin_status}</span>
        <span style='font-size:12px; color:#8B9DC3; margin-left:16px;'>Cycle: {dt_cycles} &nbsp;|&nbsp; Temp: {dt_temp}°C &nbsp;|&nbsp; FC: {dt_fc}% &nbsp;|&nbsp; DoD: {dt_dod*100:.0f}%</span>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 2: Physics vs Reality with Confidence Band ───────────────────
    st.markdown('<div class="section-header">Physics Model vs Reality</div>',
                unsafe_allow_html=True)

    hist_n    = np.arange(max(0, dt_cycles-200), dt_cycles+1, 5)
    exp_hist  = np.array([physics_soh(n, dt_temp, dt_dod, dt_fc) for n in hist_n])
    np.random.seed(dt_cycles)
    noise     = np.random.normal(0.4, 0.3, len(hist_n))
    act_hist  = exp_hist - np.abs(noise)

    # Confidence band (±1σ around the physics model)
    band_upper = exp_hist + 1.5
    band_lower = exp_hist - 1.5

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=list(hist_n) + list(hist_n[::-1]),
        y=list(band_upper) + list(band_lower[::-1]),
        fill='toself', fillcolor=ACCENT_BLUE+'18',
        line=dict(color='rgba(0,0,0,0)'),
        name='±1σ Confidence Band', showlegend=True
    ))

    # Physics model
    fig.add_trace(go.Scatter(
        x=list(hist_n), y=list(exp_hist),
        mode='lines', name='Physics Model (Expected)',
        line=dict(color=ACCENT_BLUE, width=2.5, dash='dash'),
        hovertemplate='Cycle %{x}<br>Expected SoH: %{y:.2f}%<extra></extra>'
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=list(hist_n), y=list(act_hist),
        mode='lines', name='Real-World (Actual)',
        line=dict(color=ACCENT_ORANGE, width=2.5),
        hovertemplate='Cycle %{x}<br>Actual SoH: %{y:.2f}%<extra></extra>'
    ))

    # Current position marker
    fig.add_vline(x=dt_cycles, line_dash='dot', line_color='#FFFFFF', opacity=0.4,
                  annotation_text='Now', annotation_font_color='#8B9DC3',
                  annotation_position='top')

    fig.update_layout(**PLOT_LAYOUT, height=320,
                      xaxis_title='Cycle Number',
                      yaxis_title='State of Health (%)',
                      legend=dict(bgcolor='#1C2333', bordercolor='#2A3550',
                                  font=dict(color='#8B9DC3')))
    st.plotly_chart(fig, use_container_width=True)

    # ── SECTION 3: Residual Heatmap + Degradation Drivers (side by side) ────
    col_res, col_drv = st.columns(2)

    with col_res:
        st.markdown('<div class="section-header">Residual Analysis</div>',
                    unsafe_allow_html=True)

        residuals = act_hist - exp_hist

        # Heatmap — reshape residuals into a 2D grid for visual richness
        n_cols_hm = 10
        pad       = (-len(residuals)) % n_cols_hm
        res_pad   = np.concatenate([residuals, np.full(pad, np.nan)])
        res_grid  = res_pad.reshape(-1, n_cols_hm)

        fig = go.Figure(go.Heatmap(
            z=res_grid.tolist(),
            colorscale=[
                [0.0, ACCENT_RED],
                [0.4, ACCENT_ORANGE],
                [0.5, '#1C2333'],
                [0.6, ACCENT_BLUE],
                [1.0, ACCENT_GREEN],
            ],
            zmid=0,
            colorbar=dict(
                title='Residual %',
                titlefont=dict(color='#8B9DC3'),
                tickfont=dict(color='#8B9DC3'),
                bgcolor='#1C2333',
                bordercolor='#2A3550'
            ),
            hovertemplate='Residual: %{z:.2f}%<extra></extra>'
        ))
        fig.update_layout(
            **PLOT_LAYOUT, height=240,
            xaxis=dict(title='', showticklabels=False, showgrid=False),
            yaxis=dict(title='', showticklabels=False, showgrid=False),
            margin=dict(l=20, r=60, t=10, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Red = actual aging faster than model | Green = slower | "
                   f"Mean residual: {np.nanmean(residuals):.2f}% | "
                   f"Max deviation: {np.nanmin(residuals):.2f}%")

    with col_drv:
        st.markdown('<div class="section-header">Degradation Driver Contribution</div>',
                    unsafe_allow_html=True)

        tw  = 30 + max(0, (dt_temp - 25)) * 1.5
        fcw = 10 + dt_fc * 0.3
        cw  = 35
        clw = max(100 - tw - fcw - cw, 3)
        tot_w = tw + fcw + cw + clw
        drv_labels = ['Temperature', 'Cycling', 'Fast Charging', 'Calendar Aging']
        drv_vals   = [tw/tot_w*100, cw/tot_w*100, fcw/tot_w*100, clw/tot_w*100]
        drv_colors = [ACCENT_RED, ACCENT_ORANGE, ACCENT_BLUE, '#8B9DC3']

        fig = go.Figure(go.Bar(
            x=drv_vals, y=drv_labels, orientation='h',
            marker=dict(color=drv_colors, line=dict(width=0)),
            text=[f'  {v:.0f}%' for v in drv_vals],
            textposition='inside',
            textfont=dict(color='#FFFFFF', size=12),
            hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
        ))
        fig.update_layout(
            **PLOT_LAYOUT, height=240, showlegend=False,
            xaxis=dict(range=[0, max(drv_vals)+5], gridcolor='#2A3550',
                       tickfont=dict(color='#8B9DC3')),
            yaxis=dict(tickfont=dict(color='#FFFFFF', size=12)),
            margin=dict(l=120, r=20, t=10, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Dominant driver callout
        dom = drv_labels[np.argmax(drv_vals)]
        dom_c = drv_colors[np.argmax(drv_vals)]
        st.markdown(f"""
        <div class='info-card' style='border-left:3px solid {dom_c}; padding:8px 14px;'>
            <span style='font-size:11px; color:#8B9DC3;'>Dominant Factor</span><br>
            <span style='font-size:14px; font-weight:700; color:{dom_c};'>{dom}</span>
            <span style='font-size:12px; color:#8B9DC3;'> — {max(drv_vals):.0f}% of total degradation</span>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION 4: RUL Gauge + Failure Risk (side by side) ──────────────────
    col_rul, col_risk = st.columns(2)

    with col_rul:
        st.markdown('<div class="section-header">Remaining Useful Life</div>',
                    unsafe_allow_html=True)

        # Find RUL
        n_rul = dt_cycles
        while n_rul < 10000:
            if physics_soh(n_rul, dt_temp, dt_dod, dt_fc) <= 70:
                break
            n_rul += 1
        rul_cycles = n_rul - dt_cycles
        rul_years  = rul_cycles / 300
        rul_pct    = float(np.clip(rul_years / 10 * 100, 0, 100))
        rul_color  = ACCENT_GREEN if rul_years > 5 else ACCENT_ORANGE if rul_years > 2 else ACCENT_RED

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rul_years,
            delta=dict(reference=5, valueformat='.1f',
                       increasing=dict(color=ACCENT_GREEN),
                       decreasing=dict(color=ACCENT_RED)),
            title=dict(text="Years to EOL (70% SoH)", font=dict(color='#8B9DC3', size=12)),
            number=dict(suffix=" yrs", font=dict(color='#FFFFFF', size=28)),
            gauge=dict(
                axis=dict(range=[0, 10], tickcolor='#8B9DC3',
                          tickfont=dict(color='#8B9DC3')),
                bar=dict(color=rul_color),
                bgcolor='#151B2D',
                bordercolor='#2A3550',
                steps=[
                    dict(range=[0, 2],  color='#FF3B3022'),
                    dict(range=[2, 5],  color='#FF950022'),
                    dict(range=[5, 10], color='#00FF8722'),
                ],
                threshold=dict(line=dict(color=ACCENT_RED, width=3), value=2)
            )
        ))
        fig.update_layout(
            paper_bgcolor='#1C2333', font=dict(color='#8B9DC3'),
            height=260, margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div style='display:flex; gap:10px; margin-top:-10px;'>
            <div class='info-card' style='flex:1; text-align:center; padding:10px;'>
                <div style='font-size:11px; color:#8B9DC3;'>Cycles Remaining</div>
                <div style='font-size:20px; font-weight:700; color:{rul_color};'>{rul_cycles}</div>
            </div>
            <div class='info-card' style='flex:1; text-align:center; padding:10px;'>
                <div style='font-size:11px; color:#8B9DC3;'>Est. EOL Year</div>
                <div style='font-size:20px; font-weight:700; color:{rul_color};'>{2026+int(rul_years)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_risk:
        st.markdown('<div class="section-header">Failure Risk Assessment</div>',
                    unsafe_allow_html=True)

        risk_c = ACCENT_GREEN if risk_score < 30 else ACCENT_ORANGE if risk_score < 60 else ACCENT_RED
        risk_l = "Low" if risk_score < 30 else "Moderate" if risk_score < 60 else "High"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title=dict(text="Failure Risk Score", font=dict(color='#8B9DC3', size=12)),
            number=dict(suffix="/100", font=dict(color='#FFFFFF', size=28)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor='#8B9DC3',
                          tickfont=dict(color='#8B9DC3')),
                bar=dict(color=risk_c),
                bgcolor='#151B2D',
                bordercolor='#2A3550',
                steps=[
                    dict(range=[0,  30], color='#00FF8722'),
                    dict(range=[30, 60], color='#FF950022'),
                    dict(range=[60,100], color='#FF3B3022'),
                ],
                threshold=dict(line=dict(color=ACCENT_RED, width=3), value=60)
            )
        ))
        fig.update_layout(
            paper_bgcolor='#1C2333', font=dict(color='#8B9DC3'),
            height=260, margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Risk breakdown
        risk_factors = [
            ("SoH Degradation", risk_soh, ACCENT_RED),
            ("Temperature Stress", risk_temp, ACCENT_ORANGE),
            ("Fast Charge Stress", risk_fc, ACCENT_BLUE),
        ]
        for fname, fval, fc_color in risk_factors:
            pct = min(fval / 60 * 100, 100)
            st.markdown(f"""
            <div style='margin-bottom:8px;'>
                <div style='display:flex; justify-content:space-between; font-size:12px; margin-bottom:3px;'>
                    <span style='color:#8B9DC3;'>{fname}</span>
                    <span style='color:{fc_color}; font-weight:600;'>{fval:.1f}</span>
                </div>
                <div style='background:#2A3550; border-radius:4px; height:6px;'>
                    <div style='background:{fc_color}; width:{pct:.0f}%; height:6px; border-radius:4px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── SECTION 5: Battery Aging Trajectory Forecast ─────────────────────────
    st.markdown('<div class="section-header">Battery Aging Trajectory Forecast</div>',
                unsafe_allow_html=True)

    fut_n   = np.arange(dt_cycles, dt_cycles + 2000, 10)
    fut_soh = np.array([physics_soh(n, dt_temp, dt_dod, dt_fc) for n in fut_n])
    fut_yrs = (fut_n - dt_cycles) / 300

    # Best/worst case bands (±15% variation in degradation parameters)
    fut_best  = np.array([physics_soh(n, max(20, dt_temp-4), max(0.5, dt_dod-0.1), 0) for n in fut_n])
    fut_worst = np.array([physics_soh(n, min(45, dt_temp+4), min(1.0, dt_dod+0.1), True) for n in fut_n])

    eol_idx  = next((i for i, s in enumerate(fut_soh) if s <= 70), None)
    eol_n    = fut_n[eol_idx] if eol_idx else None
    eol_yr_f = 2026 + int((eol_n - dt_cycles)/300) if eol_n else None

    fig = go.Figure()

    # Best/worst band
    fig.add_trace(go.Scatter(
        x=list(fut_yrs) + list(fut_yrs[::-1]),
        y=list(fut_best) + list(fut_worst[::-1]),
        fill='toself', fillcolor=ACCENT_BLUE+'12',
        line=dict(color='rgba(0,0,0,0)'),
        name='Best/Worst Case Range',
        hoverinfo='skip'
    ))

    # Worst case
    fig.add_trace(go.Scatter(
        x=list(fut_yrs), y=list(fut_worst),
        mode='lines', name='Worst Case',
        line=dict(color=ACCENT_RED, width=1, dash='dot'),
        hovertemplate='Year +%{x:.1f}<br>Worst SoH: %{y:.1f}%<extra></extra>'
    ))

    # Best case
    fig.add_trace(go.Scatter(
        x=list(fut_yrs), y=list(fut_best),
        mode='lines', name='Best Case',
        line=dict(color=ACCENT_GREEN, width=1, dash='dot'),
        hovertemplate='Year +%{x:.1f}<br>Best SoH: %{y:.1f}%<extra></extra>'
    ))

    # Central trajectory
    fig.add_trace(go.Scatter(
        x=list(fut_yrs), y=list(fut_soh),
        mode='lines', name='Expected Trajectory',
        line=dict(color=ACCENT_PURPLE, width=3),
        hovertemplate='Year +%{x:.1f}<br>SoH: %{y:.1f}%<extra></extra>'
    ))

    fig.add_hline(y=70, line_dash='dash', line_color=ACCENT_RED, opacity=0.7,
                  annotation_text='EOL Threshold (70%)',
                  annotation_font_color=ACCENT_RED)

    if eol_n:
        fig.add_vline(x=(eol_n-dt_cycles)/300, line_dash='dot',
                      line_color=ACCENT_ORANGE,
                      annotation_text=f'Predicted EOL ~{eol_yr_f}',
                      annotation_font_color=ACCENT_ORANGE)

    fig.update_layout(**PLOT_LAYOUT, height=320,
                      xaxis_title='Years from Now',
                      yaxis_title='State of Health (%)',
                      legend=dict(bgcolor='#1C2333', bordercolor='#2A3550',
                                  font=dict(color='#8B9DC3')))
    st.plotly_chart(fig, use_container_width=True)

    # ── SECTION 6: AI Insight Box ────────────────────────────────────────────
    st.markdown('<div class="section-header">AI Insight</div>', unsafe_allow_html=True)

    if act_soh > 90:
        soh_txt = "excellent condition"
    elif act_soh > 80:
        soh_txt = "good condition with normal early degradation"
    elif act_soh > 70:
        soh_txt = "fair condition — aging is becoming noticeable"
    else:
        soh_txt = "poor condition — approaching end of life"

    dom_factor = drv_labels[int(np.argmax(drv_vals))]
    risk_desc  = "low" if risk_score < 30 else "moderate" if risk_score < 60 else "high"
    conf_desc  = "high" if confidence > 80 else "moderate" if confidence > 60 else "lower"

    insight = (
        f"At cycle {dt_cycles}, the battery is in **{soh_txt}** "
        f"(actual SoH {act_soh:.1f}% vs physics-predicted {exp_soh:.1f}%). "
        f"The digital twin is tracking with **{confidence:.0f}% confidence**, "
        f"suggesting {conf_desc} reliability in these predictions. "
        f"The dominant degradation driver is **{dom_factor}** ({max(drv_vals):.0f}% contribution). "
        f"Overall failure risk is **{risk_desc}** ({risk_score:.0f}/100). "
        f"At current usage rates, the battery is projected to reach end-of-life "
        f"(70% SoH) around **{eol_yr_f if eol_yr_f else 'beyond the forecast horizon'}**."
    )

    st.markdown(f"""
    <div class='info-card' style='background:#00D4FF08; border:1px solid #00D4FF33;
                border-left:4px solid {ACCENT_BLUE}; padding:16px 20px;'>
        <div style='font-size:13px; font-weight:700; color:{ACCENT_BLUE}; margin-bottom:8px;'>
            💡 Digital Twin Analysis
        </div>
        <div style='font-size:14px; color:#C8D6E5; line-height:1.7;'>{insight}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION 7: Physics Equation Explainer ────────────────────────────────
    st.markdown('<div class="section-header">Physics Model Explainer</div>',
                unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        with st.expander("📐 The Degradation Equation"):
            st.markdown("""
**SoH(n) = 100 − SEI\_loss(n) − Linear\_loss(n)**

Where:
- `SEI_loss = A × √n × temp_factor × dod_factor`
- `Linear_loss = B × n × fc_factor × temp_factor`
- `A = 0.0082` (SEI growth coefficient, from NMC cell data)
- `B = 0.0055` (linear wear coefficient)
- `n` = cycle count

The **√n term** captures SEI (Solid Electrolyte Interphase) growth — a protective but resistive film that forms on the anode. It grows fast early, then slows down (hence square-root, not linear).
            """)

        with st.expander("🌡️ Temperature Factor"):
            st.markdown(f"""
**temp\_factor = 1 + max(0, (T − 25) / 10) × 0.45**

At {dt_temp}°C → temp\_factor = **{1.0 + max(0,(dt_temp-25)/10)*0.45:.3f}**

Based on the **Arrhenius equation**: reaction rates approximately double every 10°C. Every degree above 25°C accelerates degradation. Bangalore's 30–35°C average means Indian EV batteries degrade noticeably faster than lab-tested European benchmarks.
            """)

    with exp_col2:
        with st.expander("⚡ Fast Charging Factor"):
            st.markdown(f"""
**fc\_factor = 1 + 0.12 × fc\_fraction**

At {dt_fc}% DC fast charging → fc\_factor = **{1.0 + 0.12*(dt_fc/100):.3f}**

DC fast charging forces high current through the cell, increasing lithium plating risk and electrolyte decomposition. At 100% fast charging, degradation is 12% faster than AC-only charging — consistent with published NMC degradation studies.
            """)

        with st.expander("🔋 Depth of Discharge Factor"):
            st.markdown(f"""
**dod\_factor = 0.80 + (DoD × 0.28)**

At {dt_dod*100:.0f}% DoD → dod\_factor = **{0.80 + dt_dod*0.28:.3f}**

Deeper discharges stress the electrode lattice more per cycle. Charging to 80% and discharging to 20% (60% DoD) reduces stress significantly compared to full 0–100% cycles. This is why manufacturers often recommend staying between 20–80% for daily use.
            """)


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
