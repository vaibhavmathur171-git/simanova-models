# -*- coding: utf-8 -*-
"""
Project 5: LCOS Fringing Designer - Neural Surrogate for LC Director Profile
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
    page_title="P5: LCOS Designer",
    page_icon="P5",
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

MODEL_PATH = SCRIPT_DIR / "models" / "best_lcos_model.pth"
DATA_PATH = SCRIPT_DIR / "data" / "p5_lcos_dataset.npz"
DOE_PATH = SCRIPT_DIR / "models" / "p5_doe_results.json"

# =============================================================================
# CUSTOM CSS - MATCHING P2/P3/P4 FORMAT
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
    .delta-bad { background: rgba(231, 76, 60, 0.2); color: #e74c3c !important; border: 1px solid rgba(231, 76, 60, 0.4); }

    .physics-box { background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%); border: 1px solid #667eea; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; text-align: center; }
    .physics-equation { color: #667eea !important; font-family: 'Courier New', monospace; font-size: 1.2rem; font-weight: bold; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] { background: #1a1a2e; border: 1px solid #2d2d44; border-radius: 8px; padding: 0.75rem 1.5rem; color: #a0aec0 !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-color: transparent; color: white !important; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}

    /* SIDEBAR STYLING */
    [data-testid="stSidebar"] { background: #1a1a2e !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stSlider label { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #FFFFFF !important; }

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
N_PIXELS = 8
GRID_SIZE = 128
POINTS_PER_PIXEL = GRID_SIZE // N_PIXELS
V_MAX = 5.0

# =============================================================================
# 1D U-NET MODEL DEFINITION
# =============================================================================
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock1D(in_channels, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock1D(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.size(2) != skip.size(2):
            diff = skip.size(2) - x.size(2)
            x = nn.functional.pad(x, [diff // 2, diff - diff // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class LCOS_UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=3, base_filters=16):
        super().__init__()
        self.depth = depth
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            self.encoders.append(EncoderBlock1D(in_ch, filters[i]))
            in_ch = filters[i]

        self.bottleneck = ConvBlock1D(filters[depth - 1], filters[depth])

        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.decoders.append(DecoderBlock1D(filters[i + 1], filters[i]))

        self.output = nn.Conv1d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            features, x = encoder(x)
            skips.append(features)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return self.output(x)


# =============================================================================
# MODEL & DATA LOADING (CACHED)
# =============================================================================
@st.cache_resource
def load_model():
    """Load pretrained U-Net model."""
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        v_max = checkpoint.get('v_max', V_MAX)

        model = LCOS_UNet(
            in_channels=1, out_channels=1,
            depth=config['depth'],
            base_filters=config['base_filters']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, config, v_max, True, "Model Loaded"
    except Exception as e:
        return None, None, V_MAX, False, str(e)


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
# PRESET PATTERNS
# =============================================================================
def get_preset_pattern(name):
    """Get predefined voltage patterns."""
    patterns = {
        "1-ON-1-OFF": [5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0],
        "All ON": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "All OFF": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Gradient": [0.0, 0.7, 1.4, 2.1, 2.9, 3.6, 4.3, 5.0],
        "Center Block": [0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.0],
        "Single Pixel (P4)": [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        "Custom": None
    }
    return patterns.get(name)


# =============================================================================
# INFERENCE ENGINE
# =============================================================================
def expand_pixels_to_grid(pixel_voltages):
    """Expand 8 pixel voltages to 128-point grid."""
    grid = np.zeros(GRID_SIZE)
    for i, v in enumerate(pixel_voltages):
        start = i * POINTS_PER_PIXEL
        grid[start:start + POINTS_PER_PIXEL] = v
    return grid


def compute_ideal_response(voltage_grid):
    """Compute ideal response (no fringing)."""
    return voltage_grid / V_MAX


def run_inference(model, pixel_voltages, v_max_norm):
    """Run U-Net inference."""
    voltage_grid = expand_pixels_to_grid(pixel_voltages)

    v_tensor = torch.tensor(voltage_grid / v_max_norm, dtype=torch.float32)
    v_tensor = v_tensor.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        phase_pred = model(v_tensor).numpy()[0, 0]

    return voltage_grid, phase_pred


def compute_metrics(voltage_grid, phase_response, ideal_response):
    """Compute contrast loss and fringing width."""
    metrics = {}

    on_mask = voltage_grid > 0.1
    off_mask = voltage_grid < 0.1

    if np.any(on_mask) and np.any(off_mask):
        ideal_on = np.mean(ideal_response[on_mask])
        ideal_off = np.mean(ideal_response[off_mask])
        ideal_contrast = ideal_on - ideal_off

        actual_on = np.mean(phase_response[on_mask])
        actual_off = np.mean(phase_response[off_mask])
        actual_contrast = actual_on - actual_off

        if ideal_contrast > 0:
            contrast_loss = (1 - actual_contrast / ideal_contrast) * 100
        else:
            contrast_loss = 0.0
        metrics['contrast_loss'] = max(0, contrast_loss)
    else:
        metrics['contrast_loss'] = 0.0

    # Fringing width
    voltage_diff = np.abs(np.diff(voltage_grid))
    edge_positions = np.where(voltage_diff > 0.1)[0]

    if len(edge_positions) > 0:
        total_width = 0
        n_edges = 0
        for edge_pos in edge_positions:
            start = max(0, edge_pos - 10)
            end = min(GRID_SIZE - 1, edge_pos + 10)
            phase_section = phase_response[start:end]
            if len(phase_section) > 2:
                p_min, p_max = phase_section.min(), phase_section.max()
                if p_max - p_min > 0.05:
                    width = end - start
                    total_width += width * 0.5
                    n_edges += 1
        metrics['fringe_width'] = total_width / max(1, n_edges)
        metrics['n_edges'] = len(edge_positions)
    else:
        metrics['fringe_width'] = 0.0
        metrics['n_edges'] = 0

    metrics['peak_phase'] = np.max(phase_response)
    metrics['min_phase'] = np.min(phase_response)

    return metrics


# =============================================================================
# PLOTLY CHARTS
# =============================================================================
def create_lcos_chart(voltage_grid, phase_response, ideal_response):
    """Create dual-axis LCOS visualization."""
    x = np.arange(GRID_SIZE)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Voltage Command (Input)', 'LC Phase Response (Output)'),
        vertical_spacing=0.15,
        row_heights=[0.4, 0.6]
    )

    # Voltage trace (cyan)
    fig.add_trace(go.Scatter(
        x=x, y=voltage_grid, mode='lines', name='Voltage',
        line=dict(color='#00bcd4', width=2, shape='hv'),
        fill='tozeroy', fillcolor='rgba(0, 188, 212, 0.15)'
    ), row=1, col=1)

    # Pixel boundaries
    for i in range(1, N_PIXELS):
        boundary = i * POINTS_PER_PIXEL
        fig.add_vline(x=boundary, line=dict(color='rgba(102, 126, 234, 0.3)', width=1, dash='dot'), row=1, col=1)
        fig.add_vline(x=boundary, line=dict(color='rgba(102, 126, 234, 0.3)', width=1, dash='dot'), row=2, col=1)

    # Ideal response (dotted)
    fig.add_trace(go.Scatter(
        x=x, y=ideal_response, mode='lines', name='Ideal (No Fringing)',
        line=dict(color='rgba(255, 255, 255, 0.4)', width=2, dash='dot')
    ), row=2, col=1)

    # Actual LC response (magenta)
    fig.add_trace(go.Scatter(
        x=x, y=phase_response, mode='lines', name='LC Response',
        line=dict(color='#e91e63', width=3),
        fill='tozeroy', fillcolor='rgba(233, 30, 99, 0.15)'
    ), row=2, col=1)

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.5)',
        font=dict(family='Inter', color='#a0aec0'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=60, r=40, t=60, b=40),
        height=450,
        hovermode='x unified'
    )

    fig.update_xaxes(title_text='Grid Position', gridcolor='#2d2d44', zerolinecolor='#2d2d44', row=2, col=1)
    fig.update_xaxes(gridcolor='#2d2d44', zerolinecolor='#2d2d44', row=1, col=1)
    fig.update_yaxes(title_text='Voltage (V)', title_font=dict(color='#00bcd4'),
                     gridcolor='#2d2d44', range=[-0.5, 5.5], row=1, col=1)
    fig.update_yaxes(title_text='Phase', title_font=dict(color='#e91e63'),
                     gridcolor='#2d2d44', range=[-0.1, 1.2], row=2, col=1)

    return fig


def create_doe_chart(doe_data):
    """Create DOE results bar chart."""
    experiments = doe_data['experiments']

    labels = [f"d={e['depth']}, f={e['base_filters']}" for e in experiments]
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
        xaxis=dict(title='Configuration', tickfont=dict(size=10), gridcolor='#2d2d44'),
        yaxis=dict(title='Validation Loss (MSE)', gridcolor='#2d2d44'),
        height=350, margin=dict(l=60, r=40, t=60, b=80)
    )

    return fig


# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Load model and DOE data
model, config, v_max_norm, model_loaded, model_status = load_model()
doe_data = load_doe_results()

# Header
st.markdown('<h1 class="page-title">P5: LCOS Fringing Designer</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Neural Surrogate for LC Director Profile</p>', unsafe_allow_html=True)

st.markdown("""
<p class="project-desc">
    A <strong>1D U-Net neural surrogate</strong> trained on the LC continuum equations for LCOS displays.
    The model predicts phase retardation profiles accounting for <strong>fringing field effects</strong>
    where elastic coupling between LC molecules causes smooth transitions at pixel boundaries.
</p>
""", unsafe_allow_html=True)

st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(["Methodology", "Design Tool", "DOE Analysis"])

# =============================================================================
# TAB 1: METHODOLOGY
# =============================================================================
with tab1:
    st.markdown('<h2 class="section-header">System Methodology</h2>', unsafe_allow_html=True)

    # Physics equation
    st.markdown("""
    <div class="physics-box">
        <p class="physics-equation">theta[i] = (theta[i-1] + theta[i+1])/2 + C * V^2 * sin(2*theta)</p>
        <p style="color: #a0aec0; margin-top: 0.5rem;">1D Relaxation Equation: Elastic Coupling + Electric Torque</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">OBJ</span>
            <span class="method-title">Objective</span>
            <p class="method-desc">
                Train a <strong>1D U-Net neural surrogate</strong> to predict the LC director profile
                from voltage commands, replacing iterative relaxation solvers with
                <strong>sub-millisecond inference</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">1</span>
            <span class="method-title">LC Physics Parameters</span>
            <p class="method-desc">
                <strong>Grid Size:</strong> 128 points (8 pixels x 16 points)<br>
                <strong>Elastic Constant:</strong> K = 12 pN (splay)<br>
                <strong>Dielectric Anisotropy:</strong> delta-epsilon = 10<br>
                <strong>Cell Gap:</strong> d = 5 um
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">2</span>
            <span class="method-title">Fringing Field Effect</span>
            <p class="method-desc">
                Adjacent pixels share electric field lines. The <strong>fringing field</strong>
                causes LC molecules at boundaries to partially align with neighbors,
                creating smooth transitions instead of sharp edges.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">3</span>
            <span class="method-title">Training Data</span>
            <p class="method-desc">
                <strong>2000 samples</strong> with mixed voltage patterns:<br>
                Random voltages, gradients, alternating, single-pixel ON.<br>
                Physics solver: vectorized 1D relaxation (1000 iterations).
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">4</span>
            <span class="method-title">1D U-Net Architecture</span>
            <p class="method-desc">
                <strong>Input:</strong> (1, 128) voltage pattern<br>
                <strong>Encoder:</strong> 3-level (16->32->64->128 filters)<br>
                <strong>Decoder:</strong> Skip connections + upsampling<br>
                <strong>Output:</strong> (1, 128) phase profile
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">5</span>
            <span class="method-title">Edge-Aware Training</span>
            <p class="method-desc">
                <strong>Combined Loss:</strong> MSE + Gradient Loss + Edge-Weighted MSE<br>
                Gradient loss penalizes slope errors at pixel boundaries,
                ensuring accurate fringing field prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: DESIGN TOOL
# =============================================================================
with tab2:
    st.markdown('<h2 class="section-header">Interactive Design Tool</h2>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("Back to Home", use_container_width=True):
        st.switch_page("Home.py")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Pattern Selection")

    pattern_choice = st.sidebar.selectbox(
        "Preset Pattern",
        ["Custom", "1-ON-1-OFF", "All ON", "All OFF", "Gradient", "Center Block", "Single Pixel (P4)"],
        help="Select a preset pattern or customize"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Pixel Voltages (0-5V)")

    # Initialize session state
    if 'pixel_voltages' not in st.session_state:
        st.session_state.pixel_voltages = [0.0, 5.0, 5.0, 0.0, 2.5, 0.0, 5.0, 0.0]

    # Apply preset if selected
    if pattern_choice != "Custom":
        preset = get_preset_pattern(pattern_choice)
        if preset:
            st.session_state.pixel_voltages = preset

    # Pixel sliders
    pixel_voltages = []
    for i in range(N_PIXELS):
        v = st.sidebar.slider(
            f"P{i}",
            min_value=0.0,
            max_value=5.0,
            value=float(st.session_state.pixel_voltages[i]),
            step=0.1,
            key=f"pixel_{i}"
        )
        pixel_voltages.append(v)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Status")

    if model_loaded:
        st.sidebar.success("Model Loaded")
        st.sidebar.caption(f"depth={config['depth']}, filters={config['base_filters']}")
    else:
        st.sidebar.error("Model Not Found")
        st.sidebar.caption(f"Error: {model_status}")

    # Run inference
    if model_loaded:
        voltage_grid, phase_response = run_inference(model, pixel_voltages, v_max_norm)
        ideal_response = compute_ideal_response(voltage_grid)
        metrics = compute_metrics(voltage_grid, phase_response, ideal_response)

        # Metrics display
        st.markdown('<p class="subsection-header">Performance Metrics</p>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            loss_class = "delta-good" if metrics['contrast_loss'] < 5 else ("delta-warning" if metrics['contrast_loss'] < 15 else "delta-bad")
            st.markdown(f"""
            <div class="metric-card" style="border-color: {'#2ecc71' if metrics['contrast_loss'] < 5 else '#667eea'};">
                <p class="metric-card-label">Contrast Loss</p>
                <p class="metric-card-value">{metrics['contrast_loss']:.1f}%</p>
                <span class="metric-card-delta {loss_class}">{"Excellent" if metrics['contrast_loss'] < 5 else ("Moderate" if metrics['contrast_loss'] < 15 else "High")}</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            fringe_class = "delta-good" if metrics['fringe_width'] < 5 else ("delta-warning" if metrics['fringe_width'] < 10 else "delta-neutral")
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Fringe Width</p>
                <p class="metric-card-value">{metrics['fringe_width']:.1f}</p>
                <span class="metric-card-delta {fringe_class}">Grid Points</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Peak Phase</p>
                <p class="metric-card-value">{metrics['peak_phase']:.3f}</p>
                <span class="metric-card-delta delta-neutral">Maximum</span>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Active Edges</p>
                <p class="metric-card-value">{metrics['n_edges']}</p>
                <span class="metric-card-delta delta-neutral">Transitions</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Visualization
        st.markdown('<p class="subsection-header">Signal Visualization</p>', unsafe_allow_html=True)

        fig = create_lcos_chart(voltage_grid, phase_response, ideal_response)
        st.plotly_chart(fig, use_container_width=True)

        # Physics insight
        if metrics['contrast_loss'] > 10:
            st.markdown(f"""
            <div class="method-card">
                <span class="method-number">!</span>
                <span class="method-title">Fringing Field Impact</span>
                <p class="method-desc">
                    The alternating pattern shows <strong>{metrics['contrast_loss']:.1f}% contrast loss</strong> due to fringing.
                    Adjacent ON/OFF pixels create electric field spreading that smooths the LC director profile.
                    Consider using larger pixel groups or compensation algorithms.
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.error("Model not loaded. Please ensure `models/best_lcos_model.pth` exists.")
        st.info("Run `python p5_retrain.py` to train the model.")

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
                <p class="metric-card-label">Depth</p>
                <p class="metric-card-value" style="color: #2ecc71 !important;">{best_config['depth']}</p>
                <span class="metric-card-delta delta-good">Encoder Levels</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #2ecc71;">
                <p class="metric-card-label">Base Filters</p>
                <p class="metric-card-value" style="color: #2ecc71 !important;">{best_config['base_filters']}</p>
                <span class="metric-card-delta delta-good">First Layer</span>
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
            'Depth': [e['depth'] for e in experiments],
            'Filters': [e['base_filters'] for e in experiments],
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
            <span class="method-title">Optimal Depth</span>
            <p class="method-desc">
                <strong>depth = {best_config['depth']}</strong> provides the best receptive field.
                With 3 encoder levels, the network can capture spatial correlations across
                <strong>{2**best_config['depth'] * 3} grid points</strong>, spanning multiple pixel boundaries.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="method-card">
            <span class="method-number">2</span>
            <span class="method-title">Filter Efficiency</span>
            <p class="method-desc">
                <strong>base_filters = {best_config['base_filters']}</strong> achieves comparable accuracy to larger networks.
                The U-Net with {best_config['n_parameters']:,} parameters learns the voltage-to-phase mapping
                efficiently through skip connections that preserve edge information.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("DOE results not found.")
        st.info("Run `python p5_doe_train.py` to generate DOE results.")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #4a5568; font-size: 0.85rem;">
    P5: LCOS Neural Surrogate | 1D U-Net trained on LC relaxation physics | 8 pixels x 16 points/pixel
</div>
""", unsafe_allow_html=True)
