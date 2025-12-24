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
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="P4 | MEMS Oscilloscope",
    page_icon="📟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# OSCILLOSCOPE DARK MODE CSS
# =============================================================================
st.markdown("""
<style>
    /* Dark oscilloscope base */
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #0d1117 100%);
    }

    /* Oscilloscope frame */
    .scope-frame {
        background: linear-gradient(145deg, #1a1a1a 0%, #0d0d0d 100%);
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 0 30px rgba(46, 204, 113, 0.15),
                    inset 0 0 60px rgba(0, 0, 0, 0.5);
    }

    /* Control panel */
    .control-panel {
        background: linear-gradient(145deg, #1a1a2e 0%, #0d0d15 100%);
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
    }

    /* Metric display (LED style) */
    .led-display {
        background: #000;
        border: 2px solid #333;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        font-family: 'Courier New', monospace;
        text-align: center;
    }

    .led-value {
        color: #2ecc71;
        font-size: 1.8rem;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(46, 204, 113, 0.8);
    }

    .led-label {
        color: #666;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Warning LED */
    .led-warning {
        color: #f1c40f;
        text-shadow: 0 0 10px rgba(241, 196, 15, 0.8);
    }

    /* Title styling */
    .scope-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2ecc71;
        text-align: center;
        text-shadow: 0 0 20px rgba(46, 204, 113, 0.5);
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    .scope-subtitle {
        color: #666;
        text-align: center;
        font-size: 0.9rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }

    /* Channel indicators */
    .channel-ch1 {
        color: #2ecc71;
        font-weight: bold;
    }

    .channel-ch2 {
        color: #f1c40f;
        font-weight: bold;
    }

    /* Knob styling for sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #2ecc71, #27ae60);
    }

    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Status indicator */
    .status-ready {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #2ecc71;
        border-radius: 50%;
        box-shadow: 0 0 10px #2ecc71;
        margin-right: 8px;
    }

    .status-text {
        color: #2ecc71;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Physics info box */
    .physics-box {
        background: rgba(46, 204, 113, 0.1);
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .physics-equation {
        color: #2ecc71;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.parent
MODEL_PATH = SCRIPT_DIR / "models" / "best_mems_model.pth"
DATA_PATH = SCRIPT_DIR / "data" / "p4_mems_dataset.npz"
DOE_PATH = SCRIPT_DIR / "models" / "p4_mems_doe_results.json"

# Physics parameters
F0 = 2000.0  # Resonant frequency (Hz)
Q = 50.0     # Quality factor
FS = 100000  # Sampling rate (Hz)
DURATION = 0.010  # 10ms display window
N_SAMPLES = 1000  # Points to generate

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

        return model, config, norm_factor, True
    except Exception as e:
        return None, None, None, False


# =============================================================================
# SIGNAL GENERATORS
# =============================================================================
def generate_square_wave(t, frequency, amplitude):
    """Generate square wave signal."""
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))


def generate_sine_sweep(t, f_start, f_end, amplitude):
    """Generate sine sweep (chirp) signal."""
    # Linear frequency sweep
    f_instant = f_start + (f_end - f_start) * t / t[-1]
    phase = 2 * np.pi * np.cumsum(f_instant) * (t[1] - t[0])
    return amplitude * np.sin(phase)


def generate_impulse(t, amplitude, pulse_width=0.0005):
    """Generate impulse signal."""
    signal = np.zeros_like(t)
    # Impulse at 10% of the trace
    pulse_start = int(0.1 * len(t))
    pulse_samples = int(pulse_width * FS)
    signal[pulse_start:pulse_start + pulse_samples] = amplitude
    return signal


# =============================================================================
# LSTM INFERENCE ENGINE
# =============================================================================
def run_inference(model, voltage, seq_length, norm_factor):
    """
    Run LSTM inference on voltage sequence.
    Point-by-point autoregressive prediction.
    """
    n = len(voltage)
    theta = np.zeros(n)

    # Normalize voltage to [-1, 1] range
    v_max = max(abs(voltage.max()), abs(voltage.min()), 1.0)
    v_norm = voltage / v_max

    # Normalization stats (approximate)
    v_mean, v_std = 0.0, 0.5
    th_mean, th_std = 0.0, 0.1

    with torch.no_grad():
        for i in range(seq_length, n):
            # Build sliding window
            v_window = v_norm[i-seq_length:i]
            th_window = theta[i-seq_length:i]

            # Stack and normalize
            x = np.stack([v_window, th_window], axis=1)
            x[:, 0] = (x[:, 0] - v_mean) / (v_std + 1e-8)
            x[:, 1] = (x[:, 1] - th_mean) / (th_std + 1e-8)

            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

            # Predict
            y_pred = model(x_tensor).numpy()[0, 0]
            theta[i] = y_pred * th_std + th_mean

    # Denormalize to actual angle
    theta_actual = theta * norm_factor

    return theta_actual


# =============================================================================
# METRICS CALCULATION
# =============================================================================
def calculate_metrics(t, theta, voltage):
    """Calculate settling time and overshoot percentage."""
    # Find step/change point (where voltage first changes significantly)
    v_diff = np.abs(np.diff(voltage))
    if v_diff.max() > 0.1 * abs(voltage).max():
        change_idx = np.argmax(v_diff > 0.1 * v_diff.max())
    else:
        change_idx = 0

    # Subset after change
    theta_after = theta[change_idx:]
    t_after = t[change_idx:]

    if len(theta_after) < 10:
        return 0.0, 0.0

    # Steady state value (last 10% average)
    steady_state = np.mean(theta_after[-len(theta_after)//10:])

    if abs(steady_state) < 1e-10:
        return 0.0, 0.0

    # Peak value
    if steady_state > 0:
        peak_val = theta_after.max()
    else:
        peak_val = theta_after.min()

    # Overshoot percentage
    overshoot = abs((peak_val - steady_state) / steady_state) * 100

    # Settling time (time to stay within 5% of steady state)
    tolerance = 0.05 * abs(steady_state)
    settled = np.abs(theta_after - steady_state) < tolerance

    # Find last time it exceeded tolerance
    settling_idx = len(theta_after) - 1
    for i in range(len(settled) - 1, -1, -1):
        if not settled[i]:
            settling_idx = i
            break

    settling_time = (t_after[min(settling_idx + 1, len(t_after)-1)] - t_after[0]) * 1000  # ms

    return settling_time, overshoot


# =============================================================================
# OSCILLOSCOPE PLOT
# =============================================================================
def create_oscilloscope_plot(t, voltage, theta):
    """Create oscilloscope-style dual-axis plot."""
    t_ms = t * 1000  # Convert to ms

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Channel 1: Voltage (Green)
    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=voltage,
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
            x=t_ms,
            y=theta * 1000,  # Convert to mrad
            mode='lines',
            name='CH2: Angle',
            line=dict(color='#f1c40f', width=2),
            hovertemplate='Time: %{x:.2f}ms<br>Angle: %{y:.2f}mrad<extra></extra>'
        ),
        secondary_y=True
    )

    # Oscilloscope styling
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

    # X-axis (Time)
    fig.update_xaxes(
        title_text='Time (ms)',
        title_font=dict(size=12, color='#888'),
        gridcolor='#1a3a1a',
        gridwidth=1,
        showgrid=True,
        zeroline=True,
        zerolinecolor='#2a5a2a',
        zerolinewidth=2,
        tickfont=dict(size=10, color='#2ecc71'),
        range=[0, t_ms[-1]]
    )

    # Y-axis Left (Voltage - Green)
    fig.update_yaxes(
        title_text='Voltage (V)',
        title_font=dict(size=12, color='#2ecc71'),
        gridcolor='#1a3a1a',
        gridwidth=1,
        showgrid=True,
        zeroline=True,
        zerolinecolor='#2a5a2a',
        zerolinewidth=1,
        tickfont=dict(size=10, color='#2ecc71'),
        secondary_y=False
    )

    # Y-axis Right (Angle - Yellow)
    fig.update_yaxes(
        title_text='Angle (mrad)',
        title_font=dict(size=12, color='#f1c40f'),
        gridcolor='#3a3a1a',
        gridwidth=1,
        showgrid=False,
        tickfont=dict(size=10, color='#f1c40f'),
        secondary_y=True
    )

    return fig


# =============================================================================
# MAIN PAGE
# =============================================================================
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <p class="scope-title">📟 P4: MEMS OSCILLOSCOPE</p>
        <p class="scope-subtitle">Neural Surrogate for Electrostatic Mirror Dynamics</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model, config, norm_factor, model_loaded = load_model()

    if not model_loaded:
        st.error("⚠️ Model not found. Please run `python p4_mems_doe_train.py` first.")
        st.stop()

    # Status indicator
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <span class="status-ready"></span>
        <span class="status-text">Model Loaded • seq_length={} • hidden_dim={}</span>
    </div>
    """.format(config['seq_length'], config['hidden_dim']), unsafe_allow_html=True)

    # Physics info
    with st.expander("📐 Physics Model", expanded=False):
        st.markdown("""
        <div class="physics-box">
            <p class="physics-equation">I·θ''(t) + c·θ'(t) + k·θ(t) = τ(V)</p>
            <p style="color: #888; text-align: center; margin-top: 0.5rem;">
                Electrostatic MEMS Mirror • f₀ = 2000 Hz • Q = 50
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Resonant Freq:** 2000 Hz")
        with col2:
            st.markdown("**Q-Factor:** 50")
        with col3:
            st.markdown("**Damping:** ζ = 0.01")

    st.markdown("---")

    # ==========================================================================
    # CONTROL PANEL
    # ==========================================================================
    st.markdown("### 🎛️ Signal Generator")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

    with col_ctrl1:
        signal_type = st.selectbox(
            "Signal Type",
            ["Square Wave", "Sine Sweep", "Impulse"],
            index=0,
            help="Select the input waveform type"
        )

    with col_ctrl2:
        if signal_type == "Sine Sweep":
            freq_start = st.slider(
                "Start Frequency (Hz)",
                min_value=100,
                max_value=2000,
                value=200,
                step=50
            )
            freq_end = st.slider(
                "End Frequency (Hz)",
                min_value=500,
                max_value=5000,
                value=3000,
                step=100
            )
            frequency = (freq_start, freq_end)
        else:
            frequency = st.slider(
                "Frequency (Hz)",
                min_value=100,
                max_value=500,
                value=200,
                step=10,
                help="Signal frequency"
            )

    with col_ctrl3:
        amplitude = st.slider(
            "Voltage Amplitude (V)",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            help="Peak voltage amplitude"
        )

    # Duration control
    duration_ms = st.slider(
        "Display Window (ms)",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Time window to display"
    )
    duration = duration_ms / 1000.0

    # Generate signal button
    run_button = st.button("▶️ RUN TRACE", type="primary", use_container_width=True)

    st.markdown("---")

    # ==========================================================================
    # SIGNAL GENERATION & INFERENCE
    # ==========================================================================
    if run_button or 'last_trace' not in st.session_state:
        with st.spinner("⚡ Generating signal and running LSTM inference..."):
            # Time array
            n_samples = int(duration * FS)
            t = np.linspace(0, duration, n_samples)

            # Generate signal
            if signal_type == "Square Wave":
                voltage = generate_square_wave(t, frequency, amplitude)
            elif signal_type == "Sine Sweep":
                voltage = generate_sine_sweep(t, frequency[0], frequency[1], amplitude)
            else:  # Impulse
                voltage = generate_impulse(t, amplitude)

            # Run LSTM inference
            start_time = time.time()
            theta = run_inference(model, voltage, config['seq_length'], norm_factor)
            inference_time = (time.time() - start_time) * 1000

            # Calculate metrics
            settling_time, overshoot = calculate_metrics(t, theta, voltage)

            # Store in session state
            st.session_state['last_trace'] = {
                't': t,
                'voltage': voltage,
                'theta': theta,
                'settling_time': settling_time,
                'overshoot': overshoot,
                'inference_time': inference_time,
                'signal_type': signal_type
            }

    # Get trace data
    trace = st.session_state.get('last_trace', None)

    if trace is not None:
        # =======================================================================
        # METRICS DISPLAY (LED Style)
        # =======================================================================
        st.markdown("### 📊 Measurements")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.markdown(f"""
            <div class="led-display">
                <p class="led-label">Settling Time</p>
                <p class="led-value">{trace['settling_time']:.2f}</p>
                <p class="led-label">ms</p>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            overshoot_class = "led-warning" if trace['overshoot'] > 50 else ""
            st.markdown(f"""
            <div class="led-display">
                <p class="led-label">Overshoot</p>
                <p class="led-value {overshoot_class}">{trace['overshoot']:.1f}</p>
                <p class="led-label">%</p>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            peak_angle = np.max(np.abs(trace['theta'])) * 1000  # mrad
            st.markdown(f"""
            <div class="led-display">
                <p class="led-label">Peak Angle</p>
                <p class="led-value">{peak_angle:.2f}</p>
                <p class="led-label">mrad</p>
            </div>
            """, unsafe_allow_html=True)

        with col_m4:
            st.markdown(f"""
            <div class="led-display">
                <p class="led-label">Inference</p>
                <p class="led-value">{trace['inference_time']:.0f}</p>
                <p class="led-label">ms</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # =======================================================================
        # OSCILLOSCOPE DISPLAY
        # =======================================================================
        st.markdown("### 📺 Oscilloscope Display")

        # Channel legend
        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            st.markdown('<span class="channel-ch1">● CH1: Voltage Input</span>', unsafe_allow_html=True)
        with col_ch2:
            st.markdown('<span class="channel-ch2">● CH2: Mirror Angle (Predicted)</span>', unsafe_allow_html=True)

        # Create and display plot
        fig = create_oscilloscope_plot(trace['t'], trace['voltage'], trace['theta'])
        st.plotly_chart(fig, use_container_width=True)

        # Analysis note
        if trace['overshoot'] > 20:
            st.info(f"🔔 **Ringing Detected**: The MEMS mirror exhibits underdamped oscillations with {trace['overshoot']:.1f}% overshoot. "
                   f"This is expected for Q={Q} (low damping).")

    # ==========================================================================
    # DOE RESULTS TAB
    # ==========================================================================
    st.markdown("---")

    with st.expander("🔬 DOE Training Results", expanded=False):
        try:
            with open(DOE_PATH, 'r') as f:
                doe_data = json.load(f)

            st.markdown("#### Model Selection via Grid Search")

            # Create DOE results table
            exp_data = doe_data['experiments']
            doe_df_data = {
                'Seq Length': [e['seq_length'] for e in exp_data],
                'Hidden Dim': [e['hidden_dim'] for e in exp_data],
                'Parameters': [f"{e['n_parameters']:,}" for e in exp_data],
                'Val Loss': [f"{e['val_loss']:.6f}" for e in exp_data],
                'Train Time': [f"{e['train_time_s']:.1f}s" for e in exp_data]
            }

            st.dataframe(doe_df_data, use_container_width=True)

            # Highlight best
            best = doe_data['best_config']
            st.success(f"**Best Model**: seq_length={best['seq_length']}, hidden_dim={best['hidden_dim']} "
                      f"({best['n_parameters']:,} parameters)")

        except FileNotFoundError:
            st.warning("DOE results not found. Run `python p4_mems_doe_train.py` to generate.")

    # ==========================================================================
    # FOOTER
    # ==========================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #444;">
        <p style="font-size: 0.8rem;">
            P4: MEMS Neural Surrogate | LSTM trained on 2nd-order ODE dynamics
        </p>
        <p style="font-size: 0.7rem; color: #333;">
            <a href="/" style="color: #2ecc71;">← Back to Dashboard</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
