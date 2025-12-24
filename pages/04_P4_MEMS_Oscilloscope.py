# -*- coding: utf-8 -*-
"""
P4: MEMS Neural Surrogate - Digital Oscilloscope
================================================
Real-time LSTM inference for electrostatic MEMS mirror dynamics.
Dark mode oscilloscope aesthetic with green traces on black grid.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import time

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="P4: MEMS Oscilloscope",
    page_icon="P4",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - OSCILLOSCOPE DARK MODE
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #0d1117 100%);
    }

    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #FFFFFF !important;
    }

    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    .page-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        text-align: center;
        letter-spacing: -0.02em;
    }

    .page-subtitle {
        font-size: 1.1rem;
        color: #a0aec0 !important;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 400;
    }

    .project-desc {
        color: #c0c8d0 !important;
        font-size: 0.95rem;
        line-height: 1.6;
        text-align: center;
        max-width: 900px;
        margin: 0 auto 1.5rem auto;
    }

    .section-header {
        color: #2ecc71 !important;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1a3a1a;
    }

    .subsection-header {
        color: #FFFFFF !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }

    /* Method cards */
    .method-card {
        background: linear-gradient(145deg, #0d1a0d 0%, #0a120a 100%);
        border: 1px solid #1a3a1a;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
    }

    .method-number {
        display: inline-block;
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white !important;
        font-weight: 700;
        font-size: 0.8rem;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        margin-right: 0.5rem;
    }

    .method-title {
        color: #FFFFFF !important;
        font-weight: 600;
        display: inline;
    }

    .method-desc {
        color: #c0c8d0 !important;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        line-height: 1.6;
    }

    .method-desc strong {
        color: #2ecc71 !important;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #0d1a0d 0%, #0a120a 100%);
        border: 2px solid #1a3a1a;
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
    }

    .metric-card-label {
        color: #888 !important;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .metric-card-value {
        color: #2ecc71 !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 10px rgba(46, 204, 113, 0.5);
    }

    .metric-card-delta {
        font-size: 0.75rem;
        font-weight: 500;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }

    .delta-good {
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71 !important;
    }

    .delta-warning {
        background: rgba(241, 196, 15, 0.2);
        color: #f1c40f !important;
    }

    .delta-neutral {
        background: rgba(100, 100, 100, 0.2);
        color: #888 !important;
    }

    /* Physics equation box */
    .physics-box {
        background: linear-gradient(145deg, #0d1a0d 0%, #0a120a 100%);
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }

    .physics-equation {
        color: #2ecc71 !important;
        font-family: 'Courier New', monospace;
        font-size: 1.3rem;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(46, 204, 113, 0.5);
    }

    /* Channel indicators */
    .channel-ch1 {
        color: #2ecc71 !important;
        font-weight: bold;
    }

    .channel-ch2 {
        color: #f1c40f !important;
        font-weight: bold;
    }

    /* DOE table */
    .doe-table {
        background: #0d1a0d;
        border: 1px solid #1a3a1a;
        border-radius: 8px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.parent
MODEL_PATH = SCRIPT_DIR / "models" / "best_mems_model.pth"
DATA_PATH = SCRIPT_DIR / "Data" / "p4_mems_dataset.npz"
DOE_PATH = SCRIPT_DIR / "models" / "p4_mems_doe_results.json"

# Physics parameters
F0 = 2000.0  # Resonant frequency (Hz)
Q = 50.0     # Quality factor
FS = 100000  # Sampling rate (Hz)

# =============================================================================
# LSTM MODEL
# =============================================================================
class MEMS_LSTM(nn.Module):
    """LSTM model for MEMS dynamics prediction."""

    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.decoder(last_hidden)


# =============================================================================
# MODEL LOADING (CACHED)
# =============================================================================
@st.cache_resource
def load_model():
    """Load trained LSTM model."""
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        config = checkpoint['config']

        model = MEMS_LSTM(
            input_dim=2,
            hidden_dim=config['hidden_dim'],
            num_layers=2,
            dropout=0.1
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load normalization factor
        data = np.load(DATA_PATH, allow_pickle=True)
        norm_factor = float(data['norm_factor'])

        return model, config, norm_factor, "Model Loaded"
    except Exception as e:
        return None, None, None, str(e)


@st.cache_data
def load_doe_data():
    """Load DOE results."""
    try:
        with open(DOE_PATH, 'r') as f:
            return json.load(f)
    except:
        return None


# =============================================================================
# SIGNAL GENERATORS
# =============================================================================
def generate_square_wave(t, frequency, amplitude):
    """Generate square wave signal."""
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))


def generate_sine_sweep(t, f_start, f_end, amplitude):
    """Generate sine sweep (chirp) signal."""
    f_instant = f_start + (f_end - f_start) * t / t[-1]
    phase = 2 * np.pi * np.cumsum(f_instant) * (t[1] - t[0])
    return amplitude * np.sin(phase)


def generate_impulse(t, amplitude, pulse_width=0.0005):
    """Generate impulse signal."""
    signal = np.zeros_like(t)
    pulse_start = int(0.1 * len(t))
    pulse_samples = int(pulse_width * FS)
    signal[pulse_start:pulse_start + pulse_samples] = amplitude
    return signal


# =============================================================================
# LSTM INFERENCE ENGINE
# =============================================================================
def run_inference(model, voltage, seq_length, norm_factor):
    """Run LSTM inference on voltage sequence."""
    n = len(voltage)
    theta = np.zeros(n)

    v_max = max(abs(voltage.max()), abs(voltage.min()), 1.0)
    v_norm = voltage / v_max

    v_mean, v_std = 0.0, 0.5
    th_mean, th_std = 0.0, 0.1

    with torch.no_grad():
        for i in range(seq_length, n):
            v_window = v_norm[i-seq_length:i]
            th_window = theta[i-seq_length:i]

            x = np.stack([v_window, th_window], axis=1)
            x[:, 0] = (x[:, 0] - v_mean) / (v_std + 1e-8)
            x[:, 1] = (x[:, 1] - th_mean) / (th_std + 1e-8)

            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            y_pred = model(x_tensor).numpy()[0, 0]
            theta[i] = y_pred * th_std + th_mean

    return theta * norm_factor


# =============================================================================
# METRICS CALCULATION
# =============================================================================
def calculate_metrics(t, theta, voltage):
    """Calculate settling time and overshoot percentage."""
    v_diff = np.abs(np.diff(voltage))
    if v_diff.max() > 0.1 * abs(voltage).max():
        change_idx = np.argmax(v_diff > 0.1 * v_diff.max())
    else:
        change_idx = 0

    theta_after = theta[change_idx:]
    t_after = t[change_idx:]

    if len(theta_after) < 10:
        return 0.0, 0.0

    steady_state = np.mean(theta_after[-len(theta_after)//10:])

    if abs(steady_state) < 1e-10:
        return 0.0, 0.0

    if steady_state > 0:
        peak_val = theta_after.max()
    else:
        peak_val = theta_after.min()

    overshoot = abs((peak_val - steady_state) / steady_state) * 100

    tolerance = 0.05 * abs(steady_state)
    settled = np.abs(theta_after - steady_state) < tolerance

    settling_idx = len(theta_after) - 1
    for i in range(len(settled) - 1, -1, -1):
        if not settled[i]:
            settling_idx = i
            break

    settling_time = (t_after[min(settling_idx + 1, len(t_after)-1)] - t_after[0]) * 1000

    return settling_time, overshoot


# =============================================================================
# OSCILLOSCOPE PLOT
# =============================================================================
def create_oscilloscope_plot(t, voltage, theta):
    """Create oscilloscope-style dual-axis plot."""
    t_ms = t * 1000

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Channel 1: Voltage (Green)
    fig.add_trace(
        go.Scatter(
            x=t_ms, y=voltage,
            mode='lines',
            name='CH1: Voltage',
            line=dict(color='#2ecc71', width=2),
            hovertemplate='Time: %{x:.2f}ms<br>Voltage: %{y:.1f}V<extra></extra>'
        ),
        secondary_y=False
    )

    # Channel 2: Angle (Yellow)
    fig.add_trace(
        go.Scatter(
            x=t_ms, y=theta * 1000,
            mode='lines',
            name='CH2: Angle',
            line=dict(color='#f1c40f', width=2),
            hovertemplate='Time: %{x:.2f}ms<br>Angle: %{y:.2f}mrad<extra></extra>'
        ),
        secondary_y=True
    )

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(family='Courier New', color='#888'),
        title=dict(
            text='<b>OSCILLOSCOPE TRACE</b>',
            font=dict(size=16, color='#2ecc71'),
            x=0.5
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=11)
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        height=450,
        hovermode='x unified'
    )

    fig.update_xaxes(
        title_text='Time (ms)',
        title_font=dict(size=12, color='#888'),
        gridcolor='#1a3a1a',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='#2a5a2a',
        zerolinewidth=2,
        tickfont=dict(size=10, color='#2ecc71'),
        range=[0, t_ms[-1]]
    )

    fig.update_yaxes(
        title_text='Voltage (V)',
        title_font=dict(size=12, color='#2ecc71'),
        gridcolor='#1a3a1a',
        gridwidth=1,
        zeroline=True,
        zerolinecolor='#2a5a2a',
        tickfont=dict(size=10, color='#2ecc71'),
        secondary_y=False
    )

    fig.update_yaxes(
        title_text='Angle (mrad)',
        title_font=dict(size=12, color='#f1c40f'),
        gridcolor='#3a3a1a',
        showgrid=False,
        tickfont=dict(size=10, color='#f1c40f'),
        secondary_y=True
    )

    return fig


# =============================================================================
# DOE CHART
# =============================================================================
def create_doe_chart(doe_data):
    """Create DOE results bar chart."""
    experiments = doe_data['experiments']

    labels = [f"seq={e['seq_length']}<br>hid={e['hidden_dim']}" for e in experiments]
    val_losses = [e['val_loss'] for e in experiments]
    train_times = [e['train_time_s'] for e in experiments]

    best_idx = np.argmin(val_losses)
    colors = ['#2ecc71' if i == best_idx else '#1a5a3a' for i in range(len(experiments))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=val_losses,
        marker_color=colors,
        text=[f'{v:.4f}' for v in val_losses],
        textposition='outside',
        textfont=dict(color='#2ecc71', size=10),
        hovertemplate='Config: %{x}<br>Val Loss: %{y:.6f}<extra></extra>'
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        title=dict(
            text='<b>DOE Grid Search: Validation Loss by Configuration</b>',
            font=dict(size=14, color='#2ecc71'),
            x=0.5
        ),
        xaxis=dict(
            title='Configuration',
            tickfont=dict(size=10, color='#888'),
            gridcolor='#1a3a1a'
        ),
        yaxis=dict(
            title='Validation Loss (MSE)',
            tickfont=dict(size=10, color='#2ecc71'),
            gridcolor='#1a3a1a'
        ),
        height=400,
        margin=dict(l=60, r=40, t=60, b=60)
    )

    return fig


# =============================================================================
# MAIN PAGE
# =============================================================================

# Header
st.markdown('<h1 class="page-title">P4: MEMS Digital Oscilloscope</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Neural Surrogate for Electrostatic Mirror Dynamics</p>', unsafe_allow_html=True)

st.markdown("""
<p class="project-desc">
    An <strong>LSTM neural surrogate</strong> trained on second-order ODE dynamics of an electrostatic MEMS mirror.
    The model predicts time-domain angular response to arbitrary voltage waveforms with <strong>sub-millisecond</strong>
    temporal resolution, capturing underdamped ringing behavior at f<sub>0</sub> = 2 kHz.
</p>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(["Methodology", "Oscilloscope", "DOE Analysis"])

# =============================================================================
# TAB 1: METHODOLOGY
# =============================================================================
with tab1:
    st.markdown('<h2 class="section-header">System Methodology</h2>', unsafe_allow_html=True)

    # Physics equation
    st.markdown("""
    <div class="physics-box">
        <p class="physics-equation">I &middot; &theta;''(t) + c &middot; &theta;'(t) + k &middot; &theta;(t) = &tau;(V)</p>
        <p style="color: #888; margin-top: 0.5rem;">Second-Order Damped Harmonic Oscillator with Electrostatic Torque</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">OBJ</span>
            <span class="method-title">Objective</span>
            <p class="method-desc">
                Train an <strong>LSTM neural surrogate</strong> to predict the time-domain angular response
                of an electrostatic MEMS mirror to arbitrary voltage inputs, replacing expensive ODE integration.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">1</span>
            <span class="method-title">Physics Parameters</span>
            <p class="method-desc">
                <strong>Resonant Frequency:</strong> f<sub>0</sub> = 2000 Hz<br>
                <strong>Quality Factor:</strong> Q = 50 (underdamped)<br>
                <strong>Damping Ratio:</strong> &zeta; = 0.01<br>
                <strong>Decay Time:</strong> &tau; = 15.9 ms
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">2</span>
            <span class="method-title">Electrostatic Torque Law</span>
            <p class="method-desc">
                <strong>&tau;(V) = k<sub>torque</sub> &middot; V &middot; |V|</strong><br>
                Electrostatic force scales with V<sup>2</sup>, with sign preserved for bidirectional actuation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">3</span>
            <span class="method-title">Training Data Generation</span>
            <p class="method-desc">
                <strong>100 experiments</strong> with mixed signals:<br>
                Step functions, chirp sweeps, band-limited noise, sine bursts.<br>
                <strong>10,000 samples</strong> per experiment at 100 kHz.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">4</span>
            <span class="method-title">LSTM Architecture</span>
            <p class="method-desc">
                <strong>Input:</strong> Sliding window of [V(t), &theta;(t)] pairs<br>
                <strong>Encoder:</strong> 2-layer LSTM with hidden_dim neurons<br>
                <strong>Decoder:</strong> MLP (hidden &rarr; hidden/2 &rarr; 1)<br>
                <strong>Output:</strong> &theta;(t+1) prediction
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">5</span>
            <span class="method-title">Autoregressive Inference</span>
            <p class="method-desc">
                Point-by-point prediction using <strong>sliding window</strong>.<br>
                Each prediction becomes input for the next timestep,
                enabling real-time simulation of arbitrary waveforms.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: OSCILLOSCOPE
# =============================================================================
with tab2:
    st.markdown('<h2 class="section-header">Interactive Oscilloscope</h2>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("<- Back to Home", use_container_width=True):
        st.switch_page("Home.py")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Signal Generator")

    signal_type = st.sidebar.selectbox(
        "Signal Type",
        ["Square Wave", "Sine Sweep", "Impulse"],
        index=0
    )

    if signal_type == "Sine Sweep":
        freq_start = st.sidebar.slider("Start Freq (Hz)", 100, 2000, 200, 50)
        freq_end = st.sidebar.slider("End Freq (Hz)", 500, 5000, 3000, 100)
        frequency = (freq_start, freq_end)
    else:
        frequency = st.sidebar.slider("Frequency (Hz)", 100, 500, 200, 10)

    amplitude = st.sidebar.slider("Amplitude (V)", 10, 100, 50, 5)
    duration_ms = st.sidebar.slider("Duration (ms)", 5, 50, 10, 5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Status")

    # Load model
    model, config, norm_factor, model_status = load_model()

    if model is not None:
        st.sidebar.success("Model Loaded")
        st.sidebar.caption(f"seq_length: {config['seq_length']}, hidden_dim: {config['hidden_dim']}")
        model_active = True
    else:
        st.sidebar.error("Model Not Found")
        st.sidebar.caption(model_status)
        model_active = False

    # Run button
    run_trace = st.button("RUN TRACE", type="primary", use_container_width=True)

    if run_trace or 'trace_data' not in st.session_state:
        if model_active:
            with st.spinner("Generating signal and running LSTM inference..."):
                duration = duration_ms / 1000.0
                n_samples = int(duration * FS)
                t = np.linspace(0, duration, n_samples)

                if signal_type == "Square Wave":
                    voltage = generate_square_wave(t, frequency, amplitude)
                elif signal_type == "Sine Sweep":
                    voltage = generate_sine_sweep(t, frequency[0], frequency[1], amplitude)
                else:
                    voltage = generate_impulse(t, amplitude)

                start_time = time.time()
                theta = run_inference(model, voltage, config['seq_length'], norm_factor)
                inference_time = (time.time() - start_time) * 1000

                settling_time, overshoot = calculate_metrics(t, theta, voltage)

                st.session_state['trace_data'] = {
                    't': t, 'voltage': voltage, 'theta': theta,
                    'settling_time': settling_time, 'overshoot': overshoot,
                    'inference_time': inference_time
                }

    # Display results
    if model_active and 'trace_data' in st.session_state:
        trace = st.session_state['trace_data']

        # Metrics
        st.markdown('<p class="subsection-header">Measurements</p>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Settling Time</p>
                <p class="metric-card-value">{trace['settling_time']:.2f} ms</p>
                <span class="metric-card-delta delta-good">5% Tolerance</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            delta_class = "delta-warning" if trace['overshoot'] > 50 else "delta-good"
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Overshoot</p>
                <p class="metric-card-value">{trace['overshoot']:.1f}%</p>
                <span class="metric-card-delta {delta_class}">Peak vs Steady</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            peak_angle = np.max(np.abs(trace['theta'])) * 1000
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Peak Angle</p>
                <p class="metric-card-value">{peak_angle:.2f} mrad</p>
                <span class="metric-card-delta delta-neutral">Maximum</span>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Inference Time</p>
                <p class="metric-card-value">{trace['inference_time']:.0f} ms</p>
                <span class="metric-card-delta delta-good">LSTM Engine</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Oscilloscope display
        st.markdown('<p class="subsection-header">Oscilloscope Display</p>', unsafe_allow_html=True)

        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            st.markdown('<span class="channel-ch1">&#9679; CH1: Voltage Input (V)</span>', unsafe_allow_html=True)
        with col_ch2:
            st.markdown('<span class="channel-ch2">&#9679; CH2: Mirror Angle (mrad)</span>', unsafe_allow_html=True)

        fig = create_oscilloscope_plot(trace['t'], trace['voltage'], trace['theta'])
        st.plotly_chart(fig, use_container_width=True)

        if trace['overshoot'] > 20:
            st.info(f"**Ringing Detected**: Underdamped oscillations with {trace['overshoot']:.1f}% overshoot (Q={Q}).")

    elif not model_active:
        st.warning("Model not loaded. Please run `python p4_mems_doe_train.py` to train the model.")

# =============================================================================
# TAB 3: DOE ANALYSIS
# =============================================================================
with tab3:
    st.markdown('<h2 class="section-header">Neural Architecture Search: Grid Search Results</h2>', unsafe_allow_html=True)

    doe_data = load_doe_data()

    if doe_data is not None:
        experiments = doe_data['experiments']
        best_config = doe_data['best_config']

        # Optimal configuration
        st.markdown('<p class="subsection-header">Optimal Configuration (Lowest Validation Loss)</p>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #2ecc71;">
                <p class="metric-card-label">Sequence Length</p>
                <p class="metric-card-value">{best_config['seq_length']}</p>
                <span class="metric-card-delta delta-good">Timesteps</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #2ecc71;">
                <p class="metric-card-label">Hidden Dim</p>
                <p class="metric-card-value">{best_config['hidden_dim']}</p>
                <span class="metric-card-delta delta-good">LSTM Units</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #2ecc71;">
                <p class="metric-card-label">Parameters</p>
                <p class="metric-card-value">{best_config['n_parameters']:,}</p>
                <span class="metric-card-delta delta-good">Trainable</span>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            best_loss = min(e['val_loss'] for e in experiments)
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #2ecc71;">
                <p class="metric-card-label">Best Val Loss</p>
                <p class="metric-card-value">{best_loss:.4f}</p>
                <span class="metric-card-delta delta-good">MSE</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # DOE Chart
        st.markdown('<p class="subsection-header">Grid Search Comparison</p>', unsafe_allow_html=True)
        fig = create_doe_chart(doe_data)
        st.plotly_chart(fig, use_container_width=True)

        # Results table
        st.markdown('<p class="subsection-header">All Experiments</p>', unsafe_allow_html=True)

        table_data = {
            'Exp': [e['experiment_id'] for e in experiments],
            'Seq Length': [e['seq_length'] for e in experiments],
            'Hidden Dim': [e['hidden_dim'] for e in experiments],
            'Parameters': [f"{e['n_parameters']:,}" for e in experiments],
            'Train Loss': [f"{e['train_loss']:.6f}" for e in experiments],
            'Val Loss': [f"{e['val_loss']:.6f}" for e in experiments],
            'Time (s)': [f"{e['train_time_s']:.1f}" for e in experiments]
        }

        st.dataframe(table_data, use_container_width=True, hide_index=True)

        # Analysis
        st.markdown('<p class="subsection-header">Analysis</p>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="method-card">
            <span class="method-number">1</span>
            <span class="method-title">Optimal Sequence Length</span>
            <p class="method-desc">
                <strong>seq_length = {best_config['seq_length']}</strong> provides the best balance.
                At 100 kHz sampling, this corresponds to <strong>{best_config['seq_length']/100:.1f} ms</strong> of history,
                approximately matching the resonant period (0.5 ms at 2 kHz).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="method-card">
            <span class="method-number">2</span>
            <span class="method-title">Hidden Dimension Impact</span>
            <p class="method-desc">
                <strong>hidden_dim = {best_config['hidden_dim']}</strong> captures the dynamics adequately.
                Larger capacity allows the LSTM to learn the complex underdamped oscillation patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("DOE results not found. Run `python p4_mems_doe_train.py` to generate.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #444;">
    <p style="font-size: 0.8rem;">
        P4: MEMS Neural Surrogate | LSTM trained on 2nd-order ODE dynamics
    </p>
    <p style="font-size: 0.7rem; color: #333;">
        <a href="/" style="color: #2ecc71;">Back to Dashboard</a>
    </p>
</div>
""", unsafe_allow_html=True)
