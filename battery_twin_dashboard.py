import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# =======================
# PAGE CONFIGURATION
# =======================
st.set_page_config(
    page_title="EV Battery Digital Twin",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# CUSTOM CSS STYLING
# =======================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #00ff88;
        --secondary-color: #0066cc;
        --background-dark: #0e1117;
        --card-background: #1e2127;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00ff88, #0066cc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2127 0%, #2d3139 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #00ff88;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00ff88;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8b92a8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-delta {
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .status-excellent {
        background: rgba(0, 255, 136, 0.2);
        color: #00ff88;
        border: 1px solid #00ff88;
    }
    
    .status-good {
        background: rgba(52, 211, 153, 0.2);
        color: #34d399;
        border: 1px solid #34d399;
    }
    
    .status-warning {
        background: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
        border: 1px solid #fbbf24;
    }
    
    .status-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #1e2127;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: transparent;
        border-radius: 8px;
        color: #8b92a8;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00ff88, #0066cc);
        color: white;
    }
    
    /* Alert boxes */
    .alert-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-info {
        background: rgba(59, 130, 246, 0.1);
        border-color: #3b82f6;
        color: #93c5fd;
    }
    
    .alert-warning {
        background: rgba(251, 191, 36, 0.1);
        border-color: #fbbf24;
        color: #fcd34d;
    }
    
    .alert-success {
        background: rgba(0, 255, 136, 0.1);
        border-color: #00ff88;
        color: #00ff88;
    }
</style>
""", unsafe_allow_html=True)

# =======================
# DATA GENERATION FUNCTIONS
# =======================

@st.cache_data
def generate_battery_data():
    """Generate simulated battery data"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'voltage': np.random.normal(380, 10, 100),
        'current': np.random.normal(50, 15, 100),
        'temperature': np.random.normal(35, 5, 100),
        'soc': np.linspace(100, 20, 100) + np.random.normal(0, 2, 100),
        'soh': np.linspace(98, 96, 100) + np.random.normal(0, 0.5, 100),
        'power': np.random.normal(19, 3, 100),
        'cycles': np.linspace(150, 250, 100)
    })
    return data

def generate_cell_data(num_cells=96):
    """Generate individual cell data"""
    return pd.DataFrame({
        'cell_id': range(1, num_cells + 1),
        'voltage': np.random.normal(3.7, 0.1, num_cells),
        'temperature': np.random.normal(35, 3, num_cells),
        'resistance': np.random.normal(0.05, 0.01, num_cells),
        'capacity': np.random.normal(50, 2, num_cells)
    })

def get_battery_status(soh, temp, voltage):
    """Determine battery status based on parameters"""
    if soh > 95 and temp < 40 and 360 < voltage < 400:
        return "Excellent", "status-excellent"
    elif soh > 85 and temp < 45 and 350 < voltage < 410:
        return "Good", "status-good"
    elif soh > 70 and temp < 50:
        return "Warning", "status-warning"
    else:
        return "Critical", "status-critical"

# =======================
# SIDEBAR
# =======================
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/0e1117/00ff88?text=EV+Battery", use_container_width=True)
    st.markdown("## ⚙️ Settings")
    
    # Simulation controls
    st.markdown("### Simulation Controls")
    auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)
    refresh_rate = st.slider("Refresh Rate (sec)", 1, 10, 5)
    
    # Battery parameters
    st.markdown("### Battery Configuration")
    battery_capacity = st.number_input("Capacity (kWh)", 50, 150, 75)
    num_cells = st.number_input("Number of Cells", 48, 200, 96)
    
    # Temperature unit
    temp_unit = st.radio("Temperature Unit", ["°C", "°F"])
    
    # Export data
    st.markdown("### Data Export")
    if st.button("📥 Download Report", use_container_width=True):
        st.success("Report downloaded!")
    
    # System info
    st.markdown("---")
    st.markdown("### 📊 System Info")
    st.markdown(f"""
    - **Last Update:** {datetime.now().strftime('%H:%M:%S')}
    - **Data Points:** 100
    - **Battery ID:** BT-2024-001
    """)

# =======================
# MAIN HEADER
# =======================
st.markdown('<h1 class="main-header">🔋 EV Battery Digital Twin Dashboard</h1>', unsafe_allow_html=True)

# Load data
battery_data = generate_battery_data()
cell_data = generate_cell_data(num_cells)
latest_data = battery_data.iloc[-1]

# =======================
# KEY METRICS ROW
# =======================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">State of Charge</div>
        <div class="metric-value">{latest_data['soc']:.1f}%</div>
        <div class="metric-delta" style="color: {'#00ff88' if latest_data['soc'] > 50 else '#fbbf24'}">
            {'🔋 Good' if latest_data['soc'] > 50 else '⚠️ Low'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">State of Health</div>
        <div class="metric-value">{latest_data['soh']:.1f}%</div>
        <div class="metric-delta" style="color: #00ff88">
            ✓ Healthy
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Voltage</div>
        <div class="metric-value">{latest_data['voltage']:.1f}V</div>
        <div class="metric-delta" style="color: #34d399">
            Normal Range
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Temperature</div>
        <div class="metric-value">{latest_data['temperature']:.1f}°C</div>
        <div class="metric-delta" style="color: {'#00ff88' if latest_data['temperature'] < 40 else '#fbbf24'}">
            {'✓ Optimal' if latest_data['temperature'] < 40 else '⚠️ Warm'}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    status, status_class = get_battery_status(
        latest_data['soh'], 
        latest_data['temperature'], 
        latest_data['voltage']
    )
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Overall Status</div>
        <div class="metric-value" style="font-size: 1.5rem; margin-top: 1rem;">
            <span class="status-badge {status_class}">{status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =======================
# TABS
# =======================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Real-Time Monitor", 
    "🔬 Cell Analytics", 
    "📈 Historical Trends", 
    "🤖 Predictive Insights"
])

# =======================
# TAB 1: Real-Time Monitor
# =======================
with tab1:
    st.markdown("### ⚡ Live Battery Metrics")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Voltage and Current over time
        fig_vc = go.Figure()
        fig_vc.add_trace(go.Scatter(
            x=battery_data['timestamp'], 
            y=battery_data['voltage'],
            name='Voltage (V)',
            line=dict(color='#00ff88', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ))
        fig_vc.update_layout(
            title="Voltage Over Time",
            template="plotly_dark",
            height=300,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_vc, use_container_width=True)
        
        # Power consumption
        fig_power = go.Figure()
        fig_power.add_trace(go.Bar(
            x=battery_data['timestamp'].tail(20),
            y=battery_data['power'].tail(20),
            marker=dict(
                color=battery_data['power'].tail(20),
                colorscale='Viridis',
                showscale=True
            ),
            name='Power (kW)'
        ))
        fig_power.update_layout(
            title="Power Consumption (Last 20 Hours)",
            template="plotly_dark",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_power, use_container_width=True)
    
    with col2:
        # Temperature monitoring
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=battery_data['timestamp'],
            y=battery_data['temperature'],
            mode='lines',
            name='Temperature',
            line=dict(color='#ff6b6b', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.1)'
        ))
        # Add warning threshold line
        fig_temp.add_hline(y=45, line_dash="dash", line_color="yellow", 
                          annotation_text="Warning Threshold")
        fig_temp.update_layout(
            title="Temperature Monitoring",
            template="plotly_dark",
            height=300,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # SOC Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_data['soc'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "State of Charge", 'font': {'size': 24}},
            delta={'reference': 80, 'increasing': {'color': "#00ff88"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00ff88"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': '#ff6b6b'},
                    {'range': [20, 50], 'color': '#fbbf24'},
                    {'range': [50, 100], 'color': '#4ade80'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Alerts section
    st.markdown("### 🔔 Active Alerts & Notifications")
    
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    
    with alert_col1:
        if latest_data['temperature'] > 40:
            st.markdown("""
            <div class="alert-box alert-warning">
                <strong>⚠️ Temperature Alert</strong><br>
                Battery temperature is elevated. Monitor closely.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box alert-success">
                <strong>✓ Temperature Normal</strong><br>
                All temperature readings within range.
            </div>
            """, unsafe_allow_html=True)
    
    with alert_col2:
        if latest_data['soc'] < 30:
            st.markdown("""
            <div class="alert-box alert-warning">
                <strong>⚠️ Low Charge</strong><br>
                Battery charge below 30%. Consider charging.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box alert-success">
                <strong>✓ Charge Level Good</strong><br>
                Sufficient battery charge available.
            </div>
            """, unsafe_allow_html=True)
    
    with alert_col3:
        st.markdown("""
        <div class="alert-box alert-info">
            <strong>ℹ️ System Status</strong><br>
            All systems operating normally.
        </div>
        """, unsafe_allow_html=True)

# =======================
# TAB 2: Cell Analytics
# =======================
with tab2:
    st.markdown("### 🔬 Individual Cell Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Cell voltage heatmap
        cell_matrix = cell_data['voltage'].values.reshape(12, 8)
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=cell_matrix,
            colorscale='RdYlGn',
            text=cell_matrix,
            texttemplate='%{text:.2f}V',
            textfont={"size": 10},
            colorbar=dict(title="Voltage (V)")
        ))
        fig_heatmap.update_layout(
            title="Cell Voltage Distribution (96 Cells)",
            template="plotly_dark",
            height=400,
            xaxis_title="Column",
            yaxis_title="Row",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Cell Statistics")
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Voltage</div>
            <div class="metric-value" style="font-size: 1.8rem;">
                {cell_data['voltage'].mean():.3f}V
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Voltage Std Dev</div>
            <div class="metric-value" style="font-size: 1.8rem;">
                {cell_data['voltage'].std():.3f}V
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max Temp Diff</div>
            <div class="metric-value" style="font-size: 1.8rem;">
                {cell_data['temperature'].max() - cell_data['temperature'].min():.1f}°C
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Cell comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature distribution
        fig_temp_dist = px.histogram(
            cell_data, 
            x='temperature',
            nbins=20,
            title="Cell Temperature Distribution",
            labels={'temperature': 'Temperature (°C)', 'count': 'Number of Cells'},
            color_discrete_sequence=['#ff6b6b']
        )
        fig_temp_dist.update_layout(
            template="plotly_dark",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_temp_dist, use_container_width=True)
    
    with col2:
        # Resistance vs Capacity scatter
        fig_scatter = px.scatter(
            cell_data,
            x='resistance',
            y='capacity',
            title="Cell Resistance vs Capacity",
            labels={'resistance': 'Internal Resistance (Ω)', 'capacity': 'Capacity (Ah)'},
            color='temperature',
            color_continuous_scale='Turbo',
            size='voltage',
            hover_data=['cell_id']
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Cell data table with conditional formatting
    st.markdown("#### 📋 Detailed Cell Data")
    
    # Highlight problematic cells
    def highlight_cells(row):
        if row['voltage'] < 3.5 or row['voltage'] > 3.9:
            return ['background-color: rgba(255, 107, 107, 0.3)'] * len(row)
        elif row['temperature'] > 40:
            return ['background-color: rgba(251, 191, 36, 0.3)'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = cell_data.style.apply(highlight_cells, axis=1).format({
        'voltage': '{:.3f}V',
        'temperature': '{:.1f}°C',
        'resistance': '{:.4f}Ω',
        'capacity': '{:.2f}Ah'
    })
    
    st.dataframe(styled_df, use_container_width=True, height=300)

# =======================
# TAB 3: Historical Trends
# =======================
with tab3:
    st.markdown("### 📈 Historical Performance Analysis")
    
    # Date range selector
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    with col3:
        metric_choice = st.selectbox("Metric", ["SOC", "SOH", "Temperature", "Voltage"])
    
    # SOC and SOH trends
    fig_trends = go.Figure()
    fig_trends.add_trace(go.Scatter(
        x=battery_data['timestamp'],
        y=battery_data['soc'],
        name='State of Charge (%)',
        line=dict(color='#00ff88', width=2),
        yaxis='y'
    ))
    fig_trends.add_trace(go.Scatter(
        x=battery_data['timestamp'],
        y=battery_data['soh'],
        name='State of Health (%)',
        line=dict(color='#3b82f6', width=2),
        yaxis='y2'
    ))
    
    fig_trends.update_layout(
        title="SOC & SOH Trends Over Time",
        template="plotly_dark",
        height=400,
        hovermode='x unified',
        yaxis=dict(title="SOC (%)", titlefont=dict(color="#00ff88"), tickfont=dict(color="#00ff88")),
        yaxis2=dict(title="SOH (%)", titlefont=dict(color="#3b82f6"), tickfont=dict(color="#3b82f6"), 
                    overlaying='y', side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Cycling and degradation
    col1, col2 = st.columns(2)
    
    with col1:
        # Charge cycles
        fig_cycles = go.Figure()
        fig_cycles.add_trace(go.Scatter(
            x=battery_data['timestamp'],
            y=battery_data['cycles'],
            mode='lines+markers',
            name='Charge Cycles',
            line=dict(color='#a78bfa', width=3),
            marker=dict(size=6)
        ))
        fig_cycles.update_layout(
            title="Cumulative Charge Cycles",
            template="plotly_dark",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_cycles, use_container_width=True)
    
    with col2:
        # Energy throughput
        battery_data['energy'] = battery_data['voltage'] * battery_data['current'] / 1000
        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(
            x=battery_data['timestamp'],
            y=battery_data['energy'].cumsum(),
            mode='lines',
            name='Cumulative Energy',
            line=dict(color='#fbbf24', width=3),
            fill='tozeroy',
            fillcolor='rgba(251, 191, 36, 0.1)'
        ))
        fig_energy.update_layout(
            title="Cumulative Energy Throughput (kWh)",
            template="plotly_dark",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_energy, use_container_width=True)
    
    # Performance metrics table
    st.markdown("#### 📊 Performance Summary")
    
    summary_data = {
        "Metric": ["Average SOC", "Average SOH", "Avg Temperature", "Total Cycles", "Peak Power"],
        "Value": [
            f"{battery_data['soc'].mean():.1f}%",
            f"{battery_data['soh'].mean():.1f}%",
            f"{battery_data['temperature'].mean():.1f}°C",
            f"{int(battery_data['cycles'].iloc[-1])}",
            f"{battery_data['power'].max():.1f} kW"
        ],
        "Status": ["✓ Good", "✓ Excellent", "✓ Normal", "ℹ️ Moderate", "✓ Normal"]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# =======================
# TAB 4: Predictive Insights
# =======================
with tab4:
    st.markdown("### 🤖 AI-Powered Predictions & Insights")
    
    # Prediction metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Predicted EOL</div>
            <div class="metric-value" style="font-size: 1.8rem;">
                2.3 Years
            </div>
            <div class="metric-delta" style="color: #00ff88">
                Based on current usage
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Remaining Cycles</div>
            <div class="metric-value" style="font-size: 1.8rem;">
                ~1,250
            </div>
            <div class="metric-delta" style="color: #3b82f6">
                Estimated capacity
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Health Score</div>
            <div class="metric-value" style="font-size: 1.8rem;">
                94/100
            </div>
            <div class="metric-delta" style="color: #00ff88">
                ✓ Excellent condition
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # SOH Prediction chart
    future_dates = pd.date_range(start=datetime.now(), periods=50, freq='W')
    predicted_soh = np.linspace(latest_data['soh'], latest_data['soh'] - 10, 50)
    predicted_soh += np.random.normal(0, 0.5, 50)
    
    fig_prediction = go.Figure()
    
    # Historical data
    fig_prediction.add_trace(go.Scatter(
        x=battery_data['timestamp'],
        y=battery_data['soh'],
        name='Historical SOH',
        mode='lines',
        line=dict(color='#00ff88', width=3)
    ))
    
    # Predicted data
    fig_prediction.add_trace(go.Scatter(
        x=future_dates,
        y=predicted_soh,
        name='Predicted SOH',
        mode='lines',
        line=dict(color='#3b82f6', width=3, dash='dash')
    ))
    
    # Confidence interval
    fig_prediction.add_trace(go.Scatter(
        x=future_dates.tolist() + future_dates.tolist()[::-1],
        y=(predicted_soh + 2).tolist() + (predicted_soh - 2).tolist()[::-1],
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=True
    ))
    
    fig_prediction.update_layout(
        title="State of Health Prediction (Next 12 Months)",
        template="plotly_dark",
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_prediction, use_container_width=True)
    
    # Anomaly detection
    st.markdown("#### 🔍 Anomaly Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Simulated anomaly scores
        anomaly_scores = np.random.beta(2, 5, 100) * 100
        anomaly_scores[np.random.choice(100, 5)] = np.random.uniform(80, 100, 5)
        
        fig_anomaly = go.Figure()
        fig_anomaly.add_trace(go.Scatter(
            x=battery_data['timestamp'],
            y=anomaly_scores,
            mode='markers',
            marker=dict(
                size=8,
                color=anomaly_scores,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="Risk Score")
            ),
            name='Anomaly Score'
        ))
        fig_anomaly.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Anomaly Threshold")
        fig_anomaly.update_layout(
            title="Anomaly Detection Score",
            template="plotly_dark",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_anomaly, use_container_width=True)
    
    with col2:
        # Recommendations
        st.markdown("""
        <div class="alert-box alert-info">
            <strong>🎯 Optimization Recommendations</strong><br><br>
            <ul style="margin: 0; padding-left: 1.5rem;">
                <li>Reduce fast charging frequency to extend battery life</li>
                <li>Maintain SOC between 20-80% for optimal longevity</li>
                <li>Monitor cells #23, #45, #67 - slight voltage variance detected</li>
                <li>Schedule maintenance check in 3 months</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-box alert-success" style="margin-top: 1rem;">
            <strong>✓ Best Practices Being Followed</strong><br><br>
            <ul style="margin: 0; padding-left: 1.5rem;">
                <li>Temperature management excellent</li>
                <li>Charging patterns optimal</li>
                <li>No critical alerts detected</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ML Model insights
    st.markdown("#### 🧠 Machine Learning Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Feature importance
        features = ['Temperature', 'Voltage', 'Current', 'Cycles', 'SOC']
        importance = [0.35, 0.25, 0.20, 0.12, 0.08]
        
        fig_importance = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Viridis',
                showscale=False
            )
        ))
        fig_importance.update_layout(
            title="Feature Importance",
            template="plotly_dark",
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        # Model accuracy
        fig_accuracy = go.Figure(go.Indicator(
            mode="gauge+number",
            value=96.5,
            title={'text': "Model Accuracy (%)", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#00ff88"},
                'steps': [
                    {'range': [0, 70], 'color': "#ff6b6b"},
                    {'range': [70, 85], 'color': "#fbbf24"},
                    {'range': [85, 100], 'color': "#4ade80"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        fig_accuracy.update_layout(
            template="plotly_dark",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with col3:
        # Prediction confidence
        fig_confidence = go.Figure(go.Indicator(
            mode="gauge+number",
            value=92.3,
            title={'text': "Prediction Confidence (%)", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [0, 60], 'color': "#ff6b6b"},
                    {'range': [60, 80], 'color': "#fbbf24"},
                    {'range': [80, 100], 'color': "#4ade80"}
                ],
            }
        ))
        fig_confidence.update_layout(
            template="plotly_dark",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_confidence, use_container_width=True)

# =======================
# FOOTER
# =======================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8b92a8; padding: 2rem 0;'>
    <p><strong>EV Battery Digital Twin Dashboard</strong> | Version 2.0</p>
    <p>Last updated: {}</p>
    <p>🔋 Powered by Advanced Battery Analytics</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

# Auto-refresh functionality
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
