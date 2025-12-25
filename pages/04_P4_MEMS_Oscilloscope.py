# -*- coding: utf-8 -*-
"""
Project 4: MEMS Digital Oscilloscope - Neural Surrogate for Mirror Dynamics
Physical AI Architect Dashboard
"""
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import os

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
# ROBUST PATH RESOLUTION
# =============================================================================
try:
    SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
except:
    SCRIPT_DIR = Path.cwd()

MODEL_PATH = SCRIPT_DIR / "models" / "best_mems_model.pth"
DATA_PATH = SCRIPT_DIR / "Data" / "p4_mems_dataset.npz"
DOE_PATH = SCRIPT_DIR / "models" / "p4_mems_doe_results.json"

# =============================================================================
# CUSTOM CSS - MATCHING P2/P3 FORMAT
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: linear-gradient(180deg, #0a0a0f 0%, #0E1117 100%); }
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div { color: #FFFFFF !important; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #FFFFFF !important; }
    [data-testid="stMetricDelta"] { color: #2ecc71 !important; }

    .page-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.25rem; text-align: center;
    }
    .page-subtitle { font-size: 1.1rem; color: #a0aec0 !important; text-align: center; margin-bottom: 1rem; }
    .project-desc { color: #c0c8d0 !important; font-size: 0.95rem; line-height: 1.6; text-align: center; max-width: 900px; margin: 0 auto 1.5rem auto; }
    .project-desc strong { color: #FFFFFF !important; }

    .section-header { color: #667eea !important; font-size: 1.4rem; font-weight: 600; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #2d2d44; }
    .subsection-header { color: #FFFFFF !important; font-size: 1.1rem; font-weight: 600; margin: 1.5rem 0 1rem 0; }

    .method-card { background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%); border: 1px solid #2d2d44; border-radius: 12px; padding: 1.25rem; margin: 0.75rem 0; }
    .method-number { display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white !important; font-weight: 700; font-size: 0.8rem; padding: 0.25rem 0.6rem; border-radius: 6px; margin-right: 0.5rem; }
    .method-title { color: #FFFFFF !important; font-weight: 600; display: inline; }
    .method-desc { color: #c0c8d0 !important; font-size: 0.9rem; margin-top: 0.5rem; line-height: 1.6; }
    .method-desc strong { color: #FFFFFF !important; }

    .metric-card { background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%); border: 2px solid #667eea; border-radius: 16px; padding: 1.5rem; text-align: center; box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2); min-height: 140px; display: flex; flex-direction: column; justify-content: center; }
    .metric-card-label { color: #667eea !important; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.25rem; }
    .metric-card-value { color: #FFFFFF !important; font-size: 2rem; font-weight: 700; margin: 0.25rem 0; }
    .metric-card-delta { font-size: 0.85rem; padding: 0.3rem 0.6rem; border-radius: 6px; display: inline-block; margin-top: 0.25rem; }
    .delta-good { background: rgba(46, 204, 113, 0.2); color: #2ecc71 !important; border: 1px solid rgba(46, 204, 113, 0.4); }
    .delta-neutral { background: rgba(102, 126, 234, 0.2); color: #667eea !important; }
    .delta-warning { background: rgba(241, 196, 15, 0.2); color: #f1c40f !important; border: 1px solid rgba(241, 196, 15, 0.4); }

    .physics-box { background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%); border: 1px solid #667eea; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; text-align: center; }
    .physics-equation { color: #667eea !important; font-family: 'Courier New', monospace; font-size: 1.2rem; font-weight: bold; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] { background: #1a1a2e; border: 1px solid #2d2d44; border-radius: 8px; padding: 0.75rem 1.5rem; color: #a0aec0 !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-color: transparent; color: white !important; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

    /* SIDEBAR STYLING - CRITICAL FOR READABILITY */
    [data-testid="stSidebar"] { background: #1a1a2e !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stSlider label { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #FFFFFF !important; }

    /* Dropdown/Selectbox styling for dark mode */
    [data-testid="stSidebar"] [data-baseweb="select"] { background-color: #2d2d44 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div { background-color: #2d2d44 !important; color: #FFFFFF !important; }
    [data-baseweb="popover"] { background-color: #1a1a2e !important; }
    [data-baseweb="popover"] li { background-color: #1a1a2e !important; color: #FFFFFF !important; }
    [data-baseweb="popover"] li:hover { background-color: #667eea !important; color: #FFFFFF !important; }
    [data-baseweb="menu"] { background-color: #1a1a2e !important; }
    [data-baseweb="menu"] [role="option"] { color: #FFFFFF !important; background-color: #1a1a2e !important; }
    [data-baseweb="menu"] [role="option"]:hover { background-color: #667eea !important; color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PHYSICS CONSTANTS
# =============================================================================
F0 = 2000.0  # Resonant frequency (Hz)
Q = 50.0     # Quality factor
ZETA = 1.0 / (2.0 * Q)  # Damping ratio = 0.01
FS = 100000  # Sampling rate (Hz)

# =============================================================================
# LSTM MODEL DEFINITION
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
# MODEL & DATA LOADING (CACHED)
# =============================================================================
@st.cache_resource
def load_model():
    """Load pretrained LSTM model."""
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        model = MEMS_LSTM(input_dim=2, hidden_dim=config['hidden_dim'], num_layers=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Load normalization factor
        data = np.load(DATA_PATH, allow_pickle=True)
        norm_factor = float(data['norm_factor'])

        return model, config, norm_factor, True, "Model Loaded"
    except Exception as e:
        return None, None, 1.0, False, str(e)


@st.cache_data
def load_doe_results():
    """Load DOE experiment results."""
    try:
        with open(DOE_PATH, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return None


# =============================================================================
# SIGNAL GENERATORS
# =============================================================================
def generate_square_wave(t, frequency, amplitude):
    """Generate square wave signal."""
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))


def generate_sine_sweep(t, f_start, f_end, amplitude):
    """Generate frequency sweep (chirp) signal."""
    f_instant = f_start + (f_end - f_start) * t / t[-1]
    phase = 2 * np.pi * np.cumsum(f_instant) * (t[1] - t[0])
    return amplitude * np.sin(phase)


def generate_impulse(t, amplitude):
    """Generate impulse signal."""
    signal = np.zeros_like(t)
    pulse_start = int(0.1 * len(t))
    pulse_samples = max(1, int(0.0005 * FS))
    signal[pulse_start:pulse_start + pulse_samples] = amplitude
    return signal


# =============================================================================
# LSTM INFERENCE ENGINE
# =============================================================================
def run_lstm_inference(model, voltage, seq_length, norm_factor):
    """Run autoregressive LSTM inference."""
    n = len(voltage)
    theta = np.zeros(n)

    # Normalize voltage to [-1, 1]
    v_max = max(abs(voltage.max()), abs(voltage.min()), 1.0)
    v_norm = voltage / v_max

    with torch.no_grad():
        for i in range(seq_length, n):
            v_window = v_norm[i-seq_length:i]
            th_window = theta[i-seq_length:i]

            # Stack and normalize features
            x = np.stack([v_window, th_window], axis=1)
            x[:, 0] = x[:, 0] / 0.5  # Voltage normalization
            x[:, 1] = x[:, 1] / 0.1  # Theta normalization

            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            theta[i] = model(x_tensor).numpy()[0, 0] * 0.1

    return theta * norm_factor


def calculate_metrics(t, theta, voltage):
    """Calculate settling time and overshoot."""
    # Find voltage step
    v_diff = np.abs(np.diff(voltage))
    if v_diff.max() > 0.1 * abs(voltage).max():
        change_idx = np.argmax(v_diff > 0.1 * v_diff.max())
    else:
        change_idx = 0

    theta_after = theta[change_idx:]

    if len(theta_after) < 20:
        return 0.0, 0.0

    # Steady state = last 10% average
    steady_state = np.mean(theta_after[-len(theta_after)//10:])

    if abs(steady_state) < 1e-10:
        return 0.0, 0.0

    # Peak and overshoot
    peak_val = theta_after.max() if steady_state > 0 else theta_after.min()
    overshoot = abs((peak_val - steady_state) / steady_state) * 100

    # Settling time (5% band)
    tolerance = 0.05 * abs(steady_state)
    settled = np.abs(theta_after - steady_state) < tolerance

    settling_idx = len(theta_after) - 1
    for i in range(len(settled) - 1, -1, -1):
        if not settled[i]:
            settling_idx = i
            break

    dt = t[1] - t[0]
    settling_time = settling_idx * dt * 1000  # ms

    return settling_time, overshoot


# =============================================================================
# PLOTLY CHARTS
# =============================================================================
def create_oscilloscope_chart(t, voltage, theta):
    """Create dual-axis oscilloscope plot."""
    t_ms = t * 1000

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Voltage trace (purple)
    fig.add_trace(go.Scatter(
        x=t_ms, y=voltage, mode='lines', name='Voltage (V)',
        line=dict(color='#667eea', width=2.5)
    ), secondary_y=False)

    # Angle trace (green)
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
    """Create DOE results bar chart."""
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
        xaxis=dict(title='Configuration', tickfont=dict(size=9), gridcolor='#2d2d44'),
        yaxis=dict(title='Validation Loss (MSE)', gridcolor='#2d2d44'),
        height=350, margin=dict(l=60, r=40, t=60, b=100)
    )

    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Load model and DOE data
model, config, norm_factor, model_loaded, model_status = load_model()
doe_data = load_doe_results()

# Header
st.markdown('<h1 class="page-title">P4: MEMS Digital Oscilloscope</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Neural Surrogate for Electrostatic Mirror Dynamics</p>', unsafe_allow_html=True)

st.markdown("""
<p class="project-desc">
    An <strong>LSTM neural surrogate</strong> trained on the second-order ODE dynamics of an electrostatic MEMS mirror.
    The model predicts time-domain angular response to arbitrary voltage waveforms, capturing <strong>underdamped ringing</strong>
    at f<sub>0</sub> = 2 kHz with Q = 50 (damping ratio &zeta; = 0.01).
</p>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

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
        <p style="color: #a0aec0; margin-top: 0.5rem;">Second-Order Damped Harmonic Oscillator with Electrostatic Torque</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">OBJ</span>
            <span class="method-title">Objective</span>
            <p class="method-desc">
                Train an <strong>LSTM neural surrogate</strong> to predict the angular response
                of an electrostatic MEMS mirror, replacing expensive ODE integration with
                <strong>sub-millisecond inference</strong>.
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
            <span class="method-title">Electrostatic Torque</span>
            <p class="method-desc">
                <strong>&tau;(V) = k &middot; V &middot; |V|</strong><br>
                Electrostatic force scales with V<sup>2</sup>, with sign preserved
                for bidirectional actuation of the mirror.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">3</span>
            <span class="method-title">Training Data</span>
            <p class="method-desc">
                <strong>100 experiments</strong> with mixed signal types:<br>
                Step functions, chirp sweeps, band-limited noise, sine bursts.<br>
                <strong>10,000 samples</strong> per experiment at 100 kHz sampling.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">4</span>
            <span class="method-title">LSTM Architecture</span>
            <p class="method-desc">
                <strong>Input:</strong> Sliding window of [V(t), &theta;(t)] pairs<br>
                <strong>Encoder:</strong> 2-layer LSTM (hidden_dim=64)<br>
                <strong>Decoder:</strong> MLP (64 &rarr; 32 &rarr; 1)<br>
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
                Each prediction &theta;(t+1) becomes input for the next timestep,
                enabling simulation of arbitrary waveforms.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: OSCILLOSCOPE
# =============================================================================
with tab2:
    st.markdown('<h2 class="section-header">Interactive Inference Engine</h2>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("Back to Home", use_container_width=True):
        st.switch_page("Home.py")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Signal Parameters")

    signal_type = st.sidebar.selectbox(
        "Signal Type",
        ["Square Wave", "Sine Sweep", "Impulse"],
        help="Type of input voltage waveform"
    )

    if signal_type == "Sine Sweep":
        freq_start = st.sidebar.slider("Start Frequency (Hz)", 100, 2000, 200, 50)
        freq_end = st.sidebar.slider("End Frequency (Hz)", 500, 5000, 3000, 100)
    else:
        frequency = st.sidebar.slider("Frequency (Hz)", 100, 500, 200, 10,
                                      help="Signal frequency")

    amplitude = st.sidebar.slider("Voltage Amplitude (V)", 10, 100, 50, 5,
                                  help="Peak voltage")
    duration_ms = st.sidebar.slider("Duration (ms)", 5, 50, 10, 5,
                                    help="Simulation window")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Status")

    if model_loaded:
        st.sidebar.success("Model Loaded")
        st.sidebar.caption(f"seq_length={config['seq_length']}, hidden_dim={config['hidden_dim']}")
    else:
        st.sidebar.error("Model Not Found")
        st.sidebar.caption(f"Error: {model_status}")

    # Run inference automatically
    if model_loaded:
        # Generate signal
        duration = duration_ms / 1000.0
        n_samples = int(duration * FS)
        t = np.linspace(0, duration, n_samples)

        if signal_type == "Square Wave":
            voltage = generate_square_wave(t, frequency, amplitude)
        elif signal_type == "Sine Sweep":
            voltage = generate_sine_sweep(t, freq_start, freq_end, amplitude)
        else:
            voltage = generate_impulse(t, amplitude)

        # Run LSTM inference
        theta = run_lstm_inference(model, voltage, config['seq_length'], norm_factor)

        # Calculate metrics
        settling_time, overshoot = calculate_metrics(t, theta, voltage)
        peak_angle = np.max(np.abs(theta)) * 1000  # mrad

        # Display metrics
        st.markdown('<p class="subsection-header">Predicted Response Metrics</p>', unsafe_allow_html=True)

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
            ov_class = "delta-warning" if overshoot > 50 else "delta-good"
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Overshoot</p>
                <p class="metric-card-value">{overshoot:.1f}%</p>
                <span class="metric-card-delta {ov_class}">Peak/Steady</span>
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
                <span class="metric-card-delta delta-neutral">f<sub>0</sub></span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Oscilloscope plot
        st.markdown('<p class="subsection-header">Oscilloscope Trace</p>', unsafe_allow_html=True)

        fig = create_oscilloscope_chart(t, voltage, theta)
        st.plotly_chart(fig, use_container_width=True)

        # Physics insight
        if overshoot > 20:
            st.markdown(f"""
            <div class="method-card">
                <span class="method-number">!</span>
                <span class="method-title">Ringing Detected</span>
                <p class="method-desc">
                    The MEMS mirror exhibits <strong>underdamped oscillations</strong> with {overshoot:.1f}% overshoot.
                    This is expected for Q={Q:.0f} (&zeta;={ZETA:.3f}). The oscillations decay with time constant
                    &tau; = 1/(&zeta;&omega;<sub>n</sub>) = 15.9 ms.
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("Model not loaded. Please ensure `models/best_mems_model.pth` exists.")
        st.info("Run `python p4_mems_doe_train.py` to train the model.")

# =============================================================================
# TAB 3: DOE ANALYSIS
# =============================================================================
with tab3:
    st.markdown('<h2 class="section-header">Neural Architecture Search: DOE Results</h2>', unsafe_allow_html=True)

    if doe_data is not None:
        experiments = doe_data['experiments']
        best_config = doe_data['best_config']

        # Optimal configuration
        st.markdown('<p class="subsection-header">Optimal Configuration (Lowest Validation Loss)</p>', unsafe_allow_html=True)

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
                <p class="metric-card-label">Hidden Dimension</p>
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

        # DOE Chart
        st.markdown('<p class="subsection-header">Grid Search Comparison</p>', unsafe_allow_html=True)

        fig = create_doe_chart(doe_data)
        st.plotly_chart(fig, use_container_width=True)

        # Experiments table
        st.markdown('<p class="subsection-header">All Experiments</p>', unsafe_allow_html=True)

        table_data = {
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
                At 100 kHz sampling, this corresponds to <strong>{best_config['seq_length']/100:.1f} ms</strong> of temporal history,
                which captures approximately one full oscillation period at f<sub>0</sub> = 2 kHz (T = 0.5 ms).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="method-card">
            <span class="method-number">2</span>
            <span class="method-title">Hidden Dimension Impact</span>
            <p class="method-desc">
                <strong>hidden_dim = {best_config['hidden_dim']}</strong> provides sufficient capacity to learn
                the complex underdamped dynamics. The 2-layer LSTM with {best_config['n_parameters']:,} parameters
                can model the phase relationships between voltage input and angular response.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("DOE results not found.")
        st.info("Run `python p4_mems_doe_train.py` to generate DOE results.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #4a5568; font-size: 0.85rem;">
    P4: MEMS Neural Surrogate | LSTM trained on 2nd-order ODE dynamics | f<sub>0</sub> = 2 kHz, Q = 50
</div>
""", unsafe_allow_html=True)
