# -*- coding: utf-8 -*-
"""
P4: MEMS Neural Surrogate - Digital Oscilloscope
================================================
Real-time LSTM inference for electrostatic MEMS mirror dynamics.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

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
# CUSTOM CSS - MATCHING P1/P2/P3 FORMAT
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #0E1117 100%);
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        color: #667eea !important;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2d2d44;
    }

    .subsection-header {
        color: #FFFFFF !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }

    /* Method cards */
    .method-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 1px solid #2d2d44;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
    }

    .method-number {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        color: #FFFFFF !important;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 2px solid #4B0082;
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
    }

    .metric-card-label {
        color: #a0aec0 !important;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .metric-card-value {
        color: #FFFFFF !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
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
        background: rgba(102, 126, 234, 0.2);
        color: #667eea !important;
    }

    /* Physics equation box */
    .physics-box {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 1px solid #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .physics-equation {
        color: #667eea !important;
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
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
F0 = 2000.0
Q = 50.0
FS = 100000

# =============================================================================
# LSTM MODEL
# =============================================================================
class MEMS_LSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                           num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.decoder(lstm_out[:, -1, :])


# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        model = MEMS_LSTM(input_dim=2, hidden_dim=config['hidden_dim'], num_layers=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        data = np.load(DATA_PATH, allow_pickle=True)
        norm_factor = float(data['norm_factor'])
        return model, config, norm_factor, True
    except Exception as e:
        return None, None, None, False


@st.cache_data
def load_doe_data():
    try:
        with open(DOE_PATH, 'r') as f:
            return json.load(f)
    except:
        return None


# =============================================================================
# SIGNAL GENERATORS
# =============================================================================
def generate_square_wave(t, frequency, amplitude):
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))


def generate_sine_sweep(t, f_start, f_end, amplitude):
    f_instant = f_start + (f_end - f_start) * t / t[-1]
    phase = 2 * np.pi * np.cumsum(f_instant) * (t[1] - t[0])
    return amplitude * np.sin(phase)


def generate_impulse(t, amplitude):
    signal = np.zeros_like(t)
    pulse_start = int(0.1 * len(t))
    pulse_samples = int(0.0005 * FS)
    signal[pulse_start:pulse_start + pulse_samples] = amplitude
    return signal


# =============================================================================
# INFERENCE ENGINE
# =============================================================================
def run_inference(model, voltage, seq_length, norm_factor):
    n = len(voltage)
    theta = np.zeros(n)
    v_max = max(abs(voltage.max()), abs(voltage.min()), 1.0)
    v_norm = voltage / v_max

    with torch.no_grad():
        for i in range(seq_length, n):
            v_window = v_norm[i-seq_length:i]
            th_window = theta[i-seq_length:i]
            x = np.stack([v_window, th_window], axis=1)
            x[:, 0] = x[:, 0] / 0.5
            x[:, 1] = x[:, 1] / 0.1
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            theta[i] = model(x_tensor).numpy()[0, 0] * 0.1

    return theta * norm_factor


def calculate_metrics(t, theta, voltage):
    v_diff = np.abs(np.diff(voltage))
    change_idx = np.argmax(v_diff > 0.1 * v_diff.max()) if v_diff.max() > 0.1 * abs(voltage).max() else 0
    theta_after = theta[change_idx:]
    t_after = t[change_idx:]

    if len(theta_after) < 10:
        return 0.0, 0.0

    steady_state = np.mean(theta_after[-len(theta_after)//10:])
    if abs(steady_state) < 1e-10:
        return 0.0, 0.0

    peak_val = theta_after.max() if steady_state > 0 else theta_after.min()
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
# PLOTS
# =============================================================================
def create_oscilloscope_plot(t, voltage, theta):
    t_ms = t * 1000
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=t_ms, y=voltage, mode='lines', name='Voltage (V)',
        line=dict(color='#667eea', width=2.5)
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=t_ms, y=theta * 1000, mode='lines', name='Angle (mrad)',
        line=dict(color='#2ecc71', width=2.5)
    ), secondary_y=True)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.5)',
        font=dict(family='Inter', color='#a0aec0'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=60, r=60, t=40, b=60),
        height=400,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text='Time (ms)', gridcolor='#2d2d44', zerolinecolor='#2d2d44')
    fig.update_yaxes(title_text='Voltage (V)', title_font=dict(color='#667eea'),
                     gridcolor='#2d2d44', tickfont=dict(color='#667eea'), secondary_y=False)
    fig.update_yaxes(title_text='Angle (mrad)', title_font=dict(color='#2ecc71'),
                     gridcolor='#2d2d44', tickfont=dict(color='#2ecc71'), secondary_y=True)

    return fig


def create_doe_chart(doe_data):
    experiments = doe_data['experiments']
    labels = [f"seq={e['seq_length']}, hid={e['hidden_dim']}" for e in experiments]
    val_losses = [e['val_loss'] for e in experiments]
    best_idx = np.argmin(val_losses)
    colors = ['#2ecc71' if i == best_idx else '#667eea' for i in range(len(experiments))]

    fig = go.Figure(go.Bar(
        x=labels, y=val_losses, marker_color=colors,
        text=[f'{v:.4f}' for v in val_losses], textposition='outside',
        textfont=dict(color='#a0aec0', size=10)
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.5)',
        title=dict(text='Validation Loss by Configuration', font=dict(size=14, color='#667eea'), x=0.5),
        xaxis=dict(title='Configuration', tickfont=dict(size=9)),
        yaxis=dict(title='Validation Loss (MSE)', gridcolor='#2d2d44'),
        height=350, margin=dict(l=60, r=40, t=60, b=80)
    )
    return fig


# =============================================================================
# MAIN
# =============================================================================
model, config, norm_factor, model_loaded = load_model()
doe_data = load_doe_data()

# Header
st.markdown('<h1 class="page-title">P4: MEMS Digital Oscilloscope</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Neural Surrogate for Electrostatic Mirror Dynamics</p>', unsafe_allow_html=True)

st.markdown("""
<p class="project-desc">
    An <strong>LSTM neural surrogate</strong> trained on second-order ODE dynamics of an electrostatic MEMS mirror.
    The model predicts time-domain angular response to arbitrary voltage waveforms, capturing underdamped
    ringing behavior at f<sub>0</sub> = 2 kHz with Q = 50.
</p>
""", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(["Methodology", "Oscilloscope", "DOE Analysis"])

# =============================================================================
# TAB 1: METHODOLOGY
# =============================================================================
with tab1:
    st.markdown('<h2 class="section-header">System Methodology</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="physics-box">
        <p class="physics-equation">I &middot; &theta;''(t) + c &middot; &theta;'(t) + k &middot; &theta;(t) = &tau;(V)</p>
        <p style="color: #a0aec0; margin-top: 0.5rem; text-align: center;">Second-Order Damped Harmonic Oscillator</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">OBJ</span>
            <span class="method-title">Objective</span>
            <p class="method-desc">
                Train an <strong>LSTM neural surrogate</strong> to predict angular response
                of an electrostatic MEMS mirror, replacing ODE integration.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">1</span>
            <span class="method-title">Physics Parameters</span>
            <p class="method-desc">
                <strong>Resonant Frequency:</strong> f₀ = 2000 Hz<br>
                <strong>Quality Factor:</strong> Q = 50<br>
                <strong>Damping Ratio:</strong> ζ = 0.01
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">2</span>
            <span class="method-title">Torque Law</span>
            <p class="method-desc">
                <strong>τ(V) = k · V · |V|</strong><br>
                Electrostatic force ∝ V², sign preserved for bidirectional actuation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">3</span>
            <span class="method-title">Training Data</span>
            <p class="method-desc">
                <strong>100 experiments</strong> with mixed signals<br>
                Step, chirp, noise, sine bursts<br>
                <strong>10,000 samples</strong> @ 100 kHz
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">4</span>
            <span class="method-title">LSTM Architecture</span>
            <p class="method-desc">
                <strong>Input:</strong> Sliding window [V(t), θ(t)]<br>
                <strong>Encoder:</strong> 2-layer LSTM<br>
                <strong>Output:</strong> θ(t+1) prediction
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">5</span>
            <span class="method-title">Inference</span>
            <p class="method-desc">
                <strong>Autoregressive</strong> point-by-point prediction.<br>
                Each output becomes next input for real-time simulation.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: OSCILLOSCOPE
# =============================================================================
with tab2:
    st.markdown('<h2 class="section-header">Real-Time Inference Engine</h2>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("Back to Home", use_container_width=True):
        st.switch_page("Home.py")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Signal Parameters")

    signal_type = st.sidebar.selectbox("Signal Type", ["Square Wave", "Sine Sweep", "Impulse"])

    if signal_type == "Sine Sweep":
        freq_start = st.sidebar.slider("Start Frequency (Hz)", 100, 2000, 200, 50)
        freq_end = st.sidebar.slider("End Frequency (Hz)", 500, 5000, 3000, 100)
    else:
        frequency = st.sidebar.slider("Frequency (Hz)", 100, 500, 200, 10)

    amplitude = st.sidebar.slider("Voltage Amplitude (V)", 10, 100, 50, 5)
    duration_ms = st.sidebar.slider("Duration (ms)", 5, 50, 10, 5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Status")

    if model_loaded:
        st.sidebar.success("Model Loaded")
        st.sidebar.caption(f"seq={config['seq_length']}, hidden={config['hidden_dim']}")
    else:
        st.sidebar.error("Model Not Found")
        st.sidebar.caption("Run p4_mems_doe_train.py")

    # Auto-run inference
    if model_loaded:
        duration = duration_ms / 1000.0
        n_samples = int(duration * FS)
        t = np.linspace(0, duration, n_samples)

        if signal_type == "Square Wave":
            voltage = generate_square_wave(t, frequency, amplitude)
        elif signal_type == "Sine Sweep":
            voltage = generate_sine_sweep(t, freq_start, freq_end, amplitude)
        else:
            voltage = generate_impulse(t, amplitude)

        theta = run_inference(model, voltage, config['seq_length'], norm_factor)
        settling_time, overshoot = calculate_metrics(t, theta, voltage)
        peak_angle = np.max(np.abs(theta)) * 1000

        # Metrics
        st.markdown('<p class="subsection-header">Predicted Metrics</p>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #2ecc71;">
                <p class="metric-card-label">Settling Time</p>
                <p class="metric-card-value" style="color: #2ecc71 !important;">{settling_time:.2f} ms</p>
                <span class="metric-card-delta delta-good">5% Band</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            ov_color = "#f1c40f" if overshoot > 50 else "#2ecc71"
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Overshoot</p>
                <p class="metric-card-value" style="color: {ov_color} !important;">{overshoot:.1f}%</p>
                <span class="metric-card-delta delta-neutral">Peak/Steady</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
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
                <p class="metric-card-label">Resonance</p>
                <p class="metric-card-value">{F0:.0f} Hz</p>
                <span class="metric-card-delta delta-neutral">f₀</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Plot
        st.markdown('<p class="subsection-header">Oscilloscope Trace</p>', unsafe_allow_html=True)
        fig = create_oscilloscope_plot(t, voltage, theta)
        st.plotly_chart(fig, use_container_width=True)

        if overshoot > 20:
            st.info(f"**Ringing Detected**: Underdamped oscillations with {overshoot:.1f}% overshoot (Q={Q}).")

    else:
        st.warning("Model not loaded. Run `python p4_mems_doe_train.py` first.")

# =============================================================================
# TAB 3: DOE ANALYSIS
# =============================================================================
with tab3:
    st.markdown('<h2 class="section-header">Neural Architecture Search</h2>', unsafe_allow_html=True)

    if doe_data:
        experiments = doe_data['experiments']
        best_config = doe_data['best_config']

        st.markdown('<p class="subsection-header">Optimal Configuration</p>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #2ecc71;">
                <p class="metric-card-label">Sequence Length</p>
                <p class="metric-card-value" style="color: #2ecc71 !important;">{best_config['seq_length']}</p>
                <span class="metric-card-delta delta-good">Timesteps</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #2ecc71;">
                <p class="metric-card-label">Hidden Dim</p>
                <p class="metric-card-value" style="color: #2ecc71 !important;">{best_config['hidden_dim']}</p>
                <span class="metric-card-delta delta-good">LSTM Units</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #2ecc71;">
                <p class="metric-card-label">Parameters</p>
                <p class="metric-card-value" style="color: #2ecc71 !important;">{best_config['n_parameters']:,}</p>
                <span class="metric-card-delta delta-good">Trainable</span>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            best_loss = min(e['val_loss'] for e in experiments)
            st.markdown(f"""
            <div class="metric-card" style="border-color: #2ecc71;">
                <p class="metric-card-label">Best Val Loss</p>
                <p class="metric-card-value" style="color: #2ecc71 !important;">{best_loss:.4f}</p>
                <span class="metric-card-delta delta-good">MSE</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Chart
        st.markdown('<p class="subsection-header">Grid Search Comparison</p>', unsafe_allow_html=True)
        fig = create_doe_chart(doe_data)
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.markdown('<p class="subsection-header">All Experiments</p>', unsafe_allow_html=True)
        st.dataframe({
            'Seq Length': [e['seq_length'] for e in experiments],
            'Hidden Dim': [e['hidden_dim'] for e in experiments],
            'Parameters': [f"{e['n_parameters']:,}" for e in experiments],
            'Val Loss': [f"{e['val_loss']:.6f}" for e in experiments],
            'Time (s)': [f"{e['train_time_s']:.1f}" for e in experiments]
        }, use_container_width=True, hide_index=True)

    else:
        st.warning("DOE results not found. Run `python p4_mems_doe_train.py` first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #4a5568; font-size: 0.8rem;">
    P4: MEMS Neural Surrogate | LSTM on 2nd-order ODE dynamics
</div>
""", unsafe_allow_html=True)
