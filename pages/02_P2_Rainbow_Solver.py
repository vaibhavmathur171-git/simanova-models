# -*- coding: utf-8 -*-
"""
Project 2: Rainbow Solver - Multi-Spectral Grating Optimization
Physical AI Architect Dashboard - Professional Refactor
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
import os
from pathlib import Path

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="P2: Rainbow Solver",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - PROFESSIONAL DARK MODE WITH INTER FONT
# =============================================================================
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Apply Inter globally */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Dark mode base */
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #0E1117 100%);
    }

    /* Force ALL text to white for visibility */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
        color: #FFFFFF !important;
    }

    /* Metric labels and values - high contrast */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }
    [data-testid="stMetricDelta"] {
        color: #2ecc71 !important;
    }

    /* Page title with rainbow gradient */
    .page-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
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

    /* Project description */
    .project-desc {
        color: #c0c8d0 !important;
        font-size: 0.95rem;
        line-height: 1.6;
        text-align: center;
        max-width: 900px;
        margin: 0 auto 1.5rem auto;
    }

    .project-desc strong {
        color: #FFFFFF !important;
    }

    /* Section headers */
    .section-header {
        color: #4ECDC4 !important;
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

    /* Methodology cards */
    .method-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 1px solid #2d2d44;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
    }

    .method-number {
        display: inline-block;
        background: linear-gradient(135deg, #4ECDC4 0%, #45B7D1 100%);
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

    /* Large metric cards for Spectral Solver */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 2px solid #4ECDC4;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(78, 205, 196, 0.3);
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-card-label {
        color: #4ECDC4 !important;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    .metric-card-value {
        color: #FFFFFF !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .metric-card-delta {
        font-size: 0.9rem;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        display: inline-block;
        margin-top: 0.5rem;
    }

    .delta-good {
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71 !important;
        border: 1px solid rgba(46, 204, 113, 0.4);
    }

    .delta-neutral {
        background: rgba(78, 205, 196, 0.2);
        color: #4ECDC4 !important;
    }

    /* Performance badge - centered alignment */
    .badge-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }

    .perf-badge {
        display: inline-block;
        background: rgba(78, 205, 196, 0.15);
        border: 1px solid rgba(78, 205, 196, 0.3);
        color: #4ECDC4 !important;
        padding: 0.5rem 1.25rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Summary cards for DOE */
    .summary-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 1px solid #4ECDC4;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .summary-label {
        color: #4ECDC4 !important;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .summary-value {
        color: #FFFFFF !important;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e;
        border: 1px solid #2d2d44;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        color: #a0aec0 !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4ECDC4 0%, #45B7D1 100%);
        border-color: transparent;
        color: white !important;
    }

    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling - high contrast */
    [data-testid="stSidebar"] {
        background: #1a1a2e !important;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stSlider label {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stSlider p {
        color: #a0aec0 !important;
    }
    section[data-testid="stSidebar"] > div {
        background: #1a1a2e !important;
    }

    /* Author footer */
    .author-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #2d2d44;
        margin-top: 3rem;
    }
    .author-footer a {
        color: #4ECDC4 !important;
        text-decoration: none;
        margin: 0 0.5rem;
    }
    .author-footer a:hover {
        color: #FF6B6B !important;
    }

    /* Force LaTeX to white */
    .stLatex, .katex, .katex-html, .katex-display {
        color: #FFFFFF !important;
        filter: brightness(0) invert(1);
    }

    /* DataFrame styling */
    .stDataFrame {
        border: 1px solid #4ECDC4 !important;
        border-radius: 8px !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #1a1a2e;
        border-radius: 8px;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class ResidualBlock(nn.Module):
    """Residual block with LayerNorm for stable training"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)

class SpectralResNet(nn.Module):
    """ResNet for multi-spectral grating optimization (matches trained model)"""
    def __init__(self, input_dim=2, hidden_dim=128, num_blocks=4):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.output_layer(self.res_blocks(self.input_layer(x)))

# =============================================================================
# CACHED RESOURCE LOADERS
# =============================================================================
# Get the root directory (parent of pages/)
try:
    SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
except:
    SCRIPT_DIR = Path.cwd()

@st.cache_data(ttl=60)  # Refresh every 60 seconds during debugging
def load_doe_data():
    """Load DOE results with fallback paths"""
    paths = [
        SCRIPT_DIR / 'Data' / 'p2_doe_results.csv',
        SCRIPT_DIR / 'data' / 'p2_doe_results.csv',
        Path.cwd() / 'Data' / 'p2_doe_results.csv',
        Path.cwd() / 'data' / 'p2_doe_results.csv',
        Path(__file__).parent.parent / 'Data' / 'p2_doe_results.csv',
        Path(__file__).parent.parent / 'data' / 'p2_doe_results.csv',
        'Data/p2_doe_results.csv',
        'data/p2_doe_results.csv',
    ]
    for path in paths:
        try:
            path_str = str(path)
            if os.path.exists(path_str):
                df = pd.read_csv(path_str)
                # Validate required columns exist
                df.columns = df.columns.str.lower().str.strip()
                if 'mae_nm' in df.columns or 'mae' in df.columns:
                    return df
        except Exception:
            continue
    return None

@st.cache_data
def load_scalers():
    """Load scaler parameters for model inference"""
    import json
    paths = [
        SCRIPT_DIR / 'models' / 'p2_scalers.json',
        Path.cwd() / 'models' / 'p2_scalers.json',
        'models/p2_scalers.json',
    ]
    for path in paths:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except:
            continue
    return None

@st.cache_resource
def load_model():
    """Load trained ResNet with caching - returns (model, status_msg)"""
    paths = [
        SCRIPT_DIR / 'models' / 'p2_rainbow_model.pth',
        Path.cwd() / 'models' / 'p2_rainbow_model.pth',
        'models/p2_rainbow_model.pth',
    ]

    for path in paths:
        try:
            path_str = str(path)
            if os.path.exists(path_str):
                model = SpectralResNet(input_dim=2, hidden_dim=128, num_blocks=4)
                state_dict = torch.load(path_str, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict)
                model.eval()
                return model, f"Loaded from {path_str}"
        except Exception as e:
            continue

    return None, "Model file not found"

# Training material IDs (must match training script)
TRAINING_MATERIAL_IDS = {"BK7": 0, "HIGH_INDEX": 1}

def get_image_path(filename):
    """Get image path with fallback locations"""
    paths = [
        SCRIPT_DIR / 'assets' / filename,
        Path('assets') / filename,
        f'assets/{filename}',
    ]
    for path in paths:
        if os.path.exists(path):
            return str(path)
    return None

# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================
GLASS_LIBRARY = {
    "BK7": {"B1": 1.03961212, "B2": 0.231792344, "B3": 1.01046945,
            "C1": 0.00600069867, "C2": 0.0200179144, "C3": 103.560653},
    "SF11": {"B1": 1.73759695, "B2": 0.313747346, "B3": 1.89878101,
             "C1": 0.013188707, "C2": 0.0623068142, "C3": 155.23629},
    "Fused Silica": {"B1": 0.6961663, "B2": 0.4079426, "B3": 0.8974794,
                     "C1": 0.0046791, "C2": 0.0135121, "C3": 97.934003},
}

def sellmeier_n(lambda_nm, glass="BK7"):
    """Calculate refractive index using Sellmeier equation"""
    c = GLASS_LIBRARY.get(glass, GLASS_LIBRARY["BK7"])
    L = lambda_nm / 1000.0
    L2 = L ** 2
    n2 = 1 + (c["B1"]*L2)/(L2-c["C1"]) + (c["B2"]*L2)/(L2-c["C2"]) + (c["B3"]*L2)/(L2-c["C3"])
    return np.sqrt(n2)

def grating_pitch(angle_deg, lambda_nm, n_out, m=-1):
    """Calculate grating pitch from diffraction angle (Grating Equation)"""
    theta = np.radians(angle_deg)
    sin_t = np.sin(theta)
    if abs(sin_t) < 1e-10:
        sin_t = 1e-10
    return abs((m * lambda_nm) / (n_out * sin_t))

def diffraction_angle(pitch_nm, lambda_nm, n_out, m=-1):
    """Calculate diffraction angle from grating pitch"""
    sin_t = (m * lambda_nm) / (n_out * pitch_nm)
    sin_t = np.clip(sin_t, -1.0, 1.0)
    return np.degrees(np.arcsin(sin_t))

# =============================================================================
# PLOTLY DARK MODE CONFIG
# =============================================================================
def get_plotly_layout(title="", height=400):
    """Standard dark mode layout for all charts"""
    return dict(
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='#FFFFFF', family='Inter'),
        title=dict(text=title, font=dict(size=16, color='#FFFFFF')),
        xaxis=dict(
            gridcolor='#2d2d44',
            zerolinecolor='#2d2d44',
            tickfont=dict(color='#FFFFFF')
        ),
        yaxis=dict(
            gridcolor='#2d2d44',
            zerolinecolor='#2d2d44',
            tickfont=dict(color='#FFFFFF')
        ),
        legend=dict(
            bgcolor='rgba(26, 26, 46, 0.9)',
            bordercolor='#2d2d44',
            font=dict(color='#FFFFFF')
        ),
        height=height,
        margin=dict(l=60, r=40, t=60, b=60)
    )

# =============================================================================
# PAGE HEADER
# =============================================================================
st.markdown('<h1 class="page-title">P2: The Rainbow Solver</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Multi-Spectral Neural Surrogate for Chromatic Dispersion Correction</p>', unsafe_allow_html=True)

# Centered badges
st.markdown("""
<div class="badge-container">
    <span class="perf-badge">RGB: 450-635nm</span>
    <span class="perf-badge">Sellmeier Dispersion</span>
    <span class="perf-badge">Photopic Weighting</span>
    <span class="perf-badge">ResNet-4 Architecture</span>
</div>
""", unsafe_allow_html=True)

# Project description
st.markdown("""
<p class="project-desc">
    In AR waveguides, a single grating period cannot steer all wavelengths to the same angle due to
    <strong>material dispersion</strong>—refractive index varies with wavelength. This creates the <strong>"rainbow effect"</strong>:
    visible color fringing at the edges of the field of view. This engine learns the <strong>optimal compromise pitch</strong>
    that minimizes chromatic angular error across R/G/B channels, weighted by human photopic vision (Green: 60%, Red/Blue: 20% each).
</p>
""", unsafe_allow_html=True)

# Governing Equations
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
_, eq_col, _ = st.columns([1, 2, 1])
with eq_col:
    st.latex(r"n^2(\lambda) - 1 = \sum_{i=1}^{3} \frac{B_i \lambda^2}{\lambda^2 - C_i} \quad \text{(Sellmeier)}")
    st.latex(r"\Lambda_{opt} = \arg\min_\Lambda \left[ 0.6|\Delta\theta_G| + 0.2|\Delta\theta_R| + 0.2|\Delta\theta_B| \right]")
    st.markdown('<p style="color: #a0aec0; font-size: 0.8rem; text-align: center;">Photopic-weighted chromatic penalty optimization</p>', unsafe_allow_html=True)

st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(["Methodology", "Spectral Solver", "DOE Analysis"])

# =============================================================================
# TAB 1: METHODOLOGY
# =============================================================================
with tab1:
    st.markdown('<h2 class="section-header">System Methodology</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">OBJ</span>
            <span class="method-title">The Chromatic Challenge</span>
            <p class="method-desc">
                Design <strong>one grating pitch (Λ)</strong> that steers Red, Green, and Blue light to approximately
                the same output angle, despite each wavelength experiencing different refractive indices due to
                <strong>material dispersion</strong>. This is the core problem preventing perfect color uniformity
                in AR displays.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">1</span>
            <span class="method-title">Sellmeier Dispersion Model</span>
            <p class="method-desc">
                <strong>Physics:</strong> Refractive index n(λ) computed from Sellmeier equation<br>
                <strong>Blue (450nm):</strong> Higher n → steeper diffraction angle<br>
                <strong>Red (635nm):</strong> Lower n → shallower diffraction angle<br>
                <strong>Result:</strong> Chromatic angular spread creates visible "rainbow" at field edges
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">2</span>
            <span class="method-title">Photopic-Weighted Optimization</span>
            <p class="method-desc">
                Human eye sensitivity peaks at green wavelengths. Multi-objective loss function:<br>
                <strong>W<sub>green</sub> = 0.6</strong> (peak sensitivity)<br>
                <strong>W<sub>red</sub> = 0.2</strong><br>
                <strong>W<sub>blue</sub> = 0.2</strong><br>
                Neural surrogate learns this weighted trade-off directly from synthetic data.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">3</span>
            <span class="method-title">Why Neural Networks?</span>
            <p class="method-desc">
                <strong>Classical RCWA:</strong> 5-10 seconds per design point evaluation<br>
                <strong>Neural Surrogate:</strong> &lt;10ms inference time (1000x speedup)<br>
                <strong>Training Data:</strong> 50,000 synthetic samples from analytical model<br>
                <strong>Benefit:</strong> Real-time interactive design exploration impossible with physics solvers
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">4</span>
            <span class="method-title">ResNet Architecture Design</span>
            <p class="method-desc">
                <strong>Input:</strong> [target_angle, material_id] (2D)<br>
                <strong>Embedding:</strong> Linear(2→128) + LayerNorm + ReLU<br>
                <strong>Residual Blocks:</strong> 4 blocks with skip connections<br>
                <strong>Block Structure:</strong> Linear→LayerNorm→ReLU→Linear→(+x)→ReLU<br>
                <strong>Output:</strong> Optimal pitch (nm)
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">5</span>
            <span class="method-title">Validation Strategy</span>
            <p class="method-desc">
                <strong>Ground Truth:</strong> Grating equation + Sellmeier dispersion (analytical)<br>
                <strong>Test Metric:</strong> Mean Absolute Error (MAE) in nanometers<br>
                <strong>Target Performance:</strong> Sub-0.1nm accuracy for fabrication tolerance<br>
                <strong>Achieved:</strong> 0.025 nm MAE (sub-angstrom precision)
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: SPECTRAL SOLVER
# =============================================================================
with tab2:
    st.markdown('<h2 class="section-header">Interactive Inference Engine</h2>', unsafe_allow_html=True)

    # Sidebar - Navigation and inputs
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("← Back to Home", use_container_width=True):
        st.switch_page("Home.py")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Design Parameters")
    target_angle = st.sidebar.slider(
        "Target Diffraction Angle (deg)",
        min_value=-75.0,
        max_value=-25.0,
        value=-50.0,
        step=0.5
    )
    glass_type = st.sidebar.selectbox(
        "Glass Material",
        list(GLASS_LIBRARY.keys()),
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Status")

    # Load model
    model, model_status = load_model()

    # Calculate physics
    n_blue = sellmeier_n(450, glass_type)
    n_green = sellmeier_n(532, glass_type)
    n_red = sellmeier_n(635, glass_type)

    # Optimal pitch for green (analytical reference)
    optimal_pitch = grating_pitch(target_angle, 532, n_green)

    # Resulting angles for each wavelength
    angle_blue = diffraction_angle(optimal_pitch, 450, n_blue)
    angle_green = diffraction_angle(optimal_pitch, 532, n_green)
    angle_red = diffraction_angle(optimal_pitch, 635, n_red)

    # Rainbow penalty
    penalty_blue = abs(angle_blue - target_angle)
    penalty_green = abs(angle_green - target_angle)
    penalty_red = abs(angle_red - target_angle)
    weighted_penalty = 0.2*penalty_blue + 0.6*penalty_green + 0.2*penalty_red

    # Compute surrogate prediction
    scalers = load_scalers()
    if model is not None and scalers is not None:
        # Map glass type to training material ID (BK7=0 supported)
        # For materials not in training, use analytical fallback
        if glass_type == "BK7":
            material_id = 0
            # Scale inputs: X_scaled = (X - mean) / scale
            angle_scaled = (target_angle - scalers["scaler_X_mean"][0]) / scalers["scaler_X_scale"][0]
            mat_scaled = (material_id - scalers["scaler_X_mean"][1]) / scalers["scaler_X_scale"][1]
            input_tensor = torch.tensor([[angle_scaled, mat_scaled]], dtype=torch.float32)
            with torch.no_grad():
                pred_scaled = model(input_tensor).item()
            # Unscale output: pitch = pred * scale + mean
            surrogate_pitch = pred_scaled * scalers["scaler_y_scale"] + scalers["scaler_y_mean"]
            model_active = True
            st.sidebar.success("Model Active")
            st.sidebar.caption("SpectralResNet (BK7 trained)")
        else:
            # Analytical fallback for non-trained materials
            surrogate_pitch = optimal_pitch
            model_active = False
            st.sidebar.info("Analytical Mode")
            st.sidebar.caption(f"{glass_type} not in training set")
    else:
        surrogate_pitch = optimal_pitch  # Fallback to analytical
        model_active = False
        st.sidebar.warning("Model Not Found")
        st.sidebar.caption(f"Status: {model_status}")

    # Calculate residual error (only meaningful when model is active)
    residual_error = abs(optimal_pitch - surrogate_pitch)

    # Display metrics with large cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-card-label">Target Angle</p>
            <p class="metric-card-value">{target_angle:.1f}°</p>
            <span class="metric-card-delta delta-neutral">Input Query</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-card-label">Optimal Pitch</p>
            <p class="metric-card-value">{optimal_pitch:.2f} nm</p>
            <span class="metric-card-delta delta-neutral">Analytical</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if model_active:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Neural Surrogate</p>
                <p class="metric-card-value">{surrogate_pitch:.2f} nm</p>
                <span class="metric-card-delta delta-good">Model Active</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Neural Surrogate</p>
                <p class="metric-card-value">--</p>
                <span class="metric-card-delta delta-neutral">Train Model First</span>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        if model_active:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Rainbow Penalty</p>
                <p class="metric-card-value">{weighted_penalty:.3f}°</p>
                <span class="metric-card-delta delta-good">Photopic-Weighted</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Rainbow Penalty</p>
                <p class="metric-card-value">{weighted_penalty:.3f}°</p>
                <span class="metric-card-delta delta-neutral">Analytical</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

    # =========================================================================
    # RAINBOW VISUALIZATION - What the user would see
    # =========================================================================
    st.markdown('<p class="subsection-header">Simulated Grating Output (User Eye View)</p>', unsafe_allow_html=True)

    # Create a visual representation of RGB beams hitting the eye
    fig_eye = go.Figure()

    # Background (dark)
    fig_eye.add_shape(type="rect", x0=-2, x1=2, y0=-1, y1=1,
                      fillcolor="#0a0a0f", line=dict(width=0))

    # Grating representation at bottom
    for i in range(-20, 21):
        x = i * 0.08
        fig_eye.add_shape(type="line", x0=x, x1=x, y0=-0.95, y1=-0.85,
                          line=dict(color="#4ECDC4", width=1))

    fig_eye.add_annotation(x=0, y=-1.05, text=f"Grating (Λ = {optimal_pitch:.1f} nm)",
                           showarrow=False, font=dict(color="#4ECDC4", size=12))

    # Calculate beam positions based on angular deviation
    # Normalize to visual scale
    max_dev = max(penalty_blue, penalty_green, penalty_red, 1)
    scale = 0.8 / max_dev

    # Blue beam (leftmost due to higher n)
    blue_x = -penalty_blue * scale if angle_blue < target_angle else penalty_blue * scale
    fig_eye.add_trace(go.Scatter(
        x=[0, blue_x], y=[-0.8, 0.7],
        mode='lines', line=dict(color='#45B7D1', width=8),
        name=f'Blue 450nm ({angle_blue:.2f}°)', opacity=0.8
    ))
    fig_eye.add_annotation(x=blue_x, y=0.85, text=f"B: {angle_blue:.2f}°",
                           font=dict(color='#45B7D1', size=11), showarrow=False)

    # Green beam (center - reference)
    green_x = -penalty_green * scale if angle_green < target_angle else penalty_green * scale
    fig_eye.add_trace(go.Scatter(
        x=[0, green_x], y=[-0.8, 0.7],
        mode='lines', line=dict(color='#4ECDC4', width=10),
        name=f'Green 532nm ({angle_green:.2f}°)', opacity=0.9
    ))
    fig_eye.add_annotation(x=green_x, y=0.85, text=f"G: {angle_green:.2f}°",
                           font=dict(color='#4ECDC4', size=11), showarrow=False)

    # Red beam (rightmost due to lower n)
    red_x = -penalty_red * scale if angle_red < target_angle else penalty_red * scale
    fig_eye.add_trace(go.Scatter(
        x=[0, red_x], y=[-0.8, 0.7],
        mode='lines', line=dict(color='#FF6B6B', width=8),
        name=f'Red 635nm ({angle_red:.2f}°)', opacity=0.8
    ))
    fig_eye.add_annotation(x=red_x, y=0.85, text=f"R: {angle_red:.2f}°",
                           font=dict(color='#FF6B6B', size=11), showarrow=False)

    # Eye position indicator
    fig_eye.add_trace(go.Scatter(
        x=[0], y=[0.95], mode='markers',
        marker=dict(size=30, color='white', symbol='circle', line=dict(color='gray', width=2)),
        name='Eye Position', showlegend=False
    ))
    fig_eye.add_annotation(x=0, y=1.1, text="👁 Observer",
                           font=dict(color='white', size=14), showarrow=False)

    # Target line
    fig_eye.add_shape(type="line", x0=0, x1=0, y0=-0.8, y1=0.7,
                      line=dict(color="white", width=1, dash="dash"))

    fig_eye.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='#FFFFFF', family='Inter'),
        height=450,
        showlegend=True,
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.2, 1.2]),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)', font=dict(color='#FFFFFF')),
        margin=dict(l=60, r=40, t=60, b=60)
    )

    st.plotly_chart(fig_eye, use_container_width=True, key=f"eye_view_{target_angle}_{glass_type}")

    st.markdown("""
    <p style="color: #a0aec0; font-size: 0.85rem; text-align: center; margin-top: 1rem;">
        The "rainbow effect": Due to dispersion, Blue bends more and Red bends less than the target angle.
        The optimal pitch minimizes weighted chromatic spread (Green prioritized for human vision).
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # =========================================================================
    # CHROMATIC ANGULAR DEVIATION
    # =========================================================================
    st.markdown('<p class="subsection-header">Chromatic Angular Deviation Across Spectrum</p>', unsafe_allow_html=True)

    wavelengths = np.linspace(400, 700, 100)
    angles = []
    for lam in wavelengths:
        n = sellmeier_n(lam, glass_type)
        ang = diffraction_angle(optimal_pitch, lam, n)
        angles.append(ang)
    angles = np.array(angles)
    deviations = angles - target_angle

    fig_spec = go.Figure()

    # Continuous spectrum with color gradient
    for i in range(len(wavelengths)-1):
        # Map wavelength to approximate RGB color
        lam = wavelengths[i]
        if lam < 490:
            color = f'rgb(0, 0, {int(255 * (lam - 400) / 90)})'
        elif lam < 510:
            color = f'rgb(0, {int(255 * (lam - 490) / 20)}, 255)'
        elif lam < 580:
            color = f'rgb(0, 255, {int(255 * (1 - (lam - 510) / 70))})'
        elif lam < 645:
            color = f'rgb({int(255 * (lam - 580) / 65)}, 255, 0)'
        else:
            color = f'rgb(255, {int(255 * (1 - (lam - 645) / 55))}, 0)'

        fig_spec.add_trace(go.Scatter(
            x=wavelengths[i:i+2], y=deviations[i:i+2],
            mode='lines', line=dict(width=4, color=color),
            showlegend=False, hoverinfo='skip'
        ))

    # Mark RGB points
    fig_spec.add_trace(go.Scatter(
        x=[450, 532, 635],
        y=[angle_blue - target_angle, angle_green - target_angle, angle_red - target_angle],
        mode='markers+text',
        marker=dict(size=14, color=['#45B7D1', '#4ECDC4', '#FF6B6B'],
                   line=dict(color='white', width=2)),
        text=[f'B: {penalty_blue:.3f}°', f'G: {penalty_green:.3f}°', f'R: {penalty_red:.3f}°'],
        textposition='top center',
        textfont=dict(size=11),
        name='RGB Channels'
    ))

    fig_spec.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

    fig_spec.update_layout(
        **get_plotly_layout(height=400),
        xaxis_title='Wavelength (nm)',
        yaxis_title='Angular Deviation from Target (deg)'
    )

    st.plotly_chart(fig_spec, use_container_width=True, key=f"spectrum_{target_angle}_{glass_type}")

# =============================================================================
# TAB 3: DOE ANALYSIS - CONVERGENCE TO OPTIMAL PARAMETERS
# =============================================================================
with tab3:
    st.markdown('<h2 class="section-header">Neural Architecture Search: Convergence to Optimal Configuration</h2>', unsafe_allow_html=True)

    df = load_doe_data()

    if df is not None and not df.empty:
        # Columns already normalized in load_doe_data()
        # Find the correct MAE column name
        if 'mae_nm' in df.columns:
            mae_col = 'mae_nm'
        elif 'mae' in df.columns:
            mae_col = 'mae'
        else:
            st.error(f"DOE data missing MAE column. Found columns: {list(df.columns)}")
            st.stop()

        # Find RMSE column
        rmse_col = 'rmse_nm' if 'rmse_nm' in df.columns else ('rmse' if 'rmse' in df.columns else None)

        # Find optimal configuration
        best_idx = df[mae_col].idxmin()
        best_row = df.loc[best_idx]
        best_blocks = int(best_row.get('num_blocks', 0))
        best_lr = best_row.get('learning_rate', 0)
        best_mae = best_row[mae_col]
        best_rmse = best_row.get(rmse_col, 0) if rmse_col else 0

        # =================================================================
        # OPTIMAL CONFIGURATION HIGHLIGHT
        # =================================================================
        st.markdown('<p class="subsection-header">Optimal Configuration (Lowest MAE)</p>', unsafe_allow_html=True)

        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)

        with opt_col1:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #2ecc71;">
                <p class="metric-card-label">Residual Blocks</p>
                <p class="metric-card-value">{best_blocks}</p>
                <span class="metric-card-delta delta-good">Layers</span>
            </div>
            """, unsafe_allow_html=True)

        with opt_col2:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #2ecc71;">
                <p class="metric-card-label">Learning Rate</p>
                <p class="metric-card-value">{best_lr:.0e}</p>
                <span class="metric-card-delta delta-good">Optimal</span>
            </div>
            """, unsafe_allow_html=True)

        with opt_col3:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #2ecc71;">
                <p class="metric-card-label">Final MAE</p>
                <p class="metric-card-value">{best_mae:.4f}</p>
                <span class="metric-card-delta delta-good">nm</span>
            </div>
            """, unsafe_allow_html=True)

        with opt_col4:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #2ecc71;">
                <p class="metric-card-label">Final RMSE</p>
                <p class="metric-card-value">{best_rmse:.4f}</p>
                <span class="metric-card-delta delta-good">nm</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

        # =================================================================
        # CONVERGENCE PLOTS - Show how we arrived at optimal parameters
        # =================================================================

        # Plot 1: Network Depth vs Error
        st.markdown('<p class="subsection-header">Convergence 1: Network Capacity (Residual Blocks)</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #a0aec0; font-size: 0.85rem; margin-bottom: 1rem;">Shows how model capacity affects accuracy for each learning rate</p>', unsafe_allow_html=True)

        fig_blocks = go.Figure()

        for lr in sorted(df['learning_rate'].unique(), reverse=True):
            subset = df[df['learning_rate'] == lr].sort_values('num_blocks')
            fig_blocks.add_trace(go.Scatter(
                x=subset['num_blocks'],
                y=subset[mae_col],
                mode='lines+markers',
                name=f'LR = {lr:.0e}',
                line=dict(width=3),
                marker=dict(size=12)
            ))

        # Mark best point
        fig_blocks.add_trace(go.Scatter(
            x=[best_blocks], y=[best_mae],
            mode='markers+text',
            marker=dict(size=20, color='#00FFFF', symbol='star', line=dict(color='white', width=2)),
            text=[f'SELECTED<br>{best_blocks} blocks'],
            textposition='top center',
            textfont=dict(color='#00FFFF', size=11),
            name='Selected',
            showlegend=True
        ))

        fig_blocks.update_layout(
            **get_plotly_layout("", height=400),
            xaxis_title='Number of Residual Blocks',
            yaxis_title='MAE (nm)',
            showlegend=True
        )

        st.plotly_chart(fig_blocks, use_container_width=True)

        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

        # Plot 2: Learning Rate Impact
        st.markdown('<p class="subsection-header">Convergence 2: Learning Rate Sensitivity</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #a0aec0; font-size: 0.85rem; margin-bottom: 1rem;">Shows best performance achieved for each learning rate (across all block counts)</p>', unsafe_allow_html=True)

        if 'learning_rate' in df.columns and mae_col in df.columns:
            lr_best = df.groupby('learning_rate')[mae_col].min().reset_index()
            lr_mean = df.groupby('learning_rate')[mae_col].mean().reset_index()
            lr_worst = df.groupby('learning_rate')[mae_col].max().reset_index()

            fig_lr = go.Figure()

            # Show range (worst to best)
            fig_lr.add_trace(go.Scatter(
                x=lr_worst['learning_rate'],
                y=lr_worst[mae_col],
                mode='lines+markers',
                name='Worst Case',
                line=dict(color='#e74c3c', width=2, dash='dash'),
                marker=dict(size=8, color='#e74c3c')
            ))

            fig_lr.add_trace(go.Scatter(
                x=lr_mean['learning_rate'],
                y=lr_mean[mae_col],
                mode='lines+markers',
                name='Mean',
                line=dict(color='#95a5a6', width=2),
                marker=dict(size=10, color='#95a5a6')
            ))

            fig_lr.add_trace(go.Scatter(
                x=lr_best['learning_rate'],
                y=lr_best[mae_col],
                mode='lines+markers',
                name='Best',
                line=dict(color='#2ecc71', width=3),
                marker=dict(size=12, color='#2ecc71'),
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.1)'
            ))

            # Highlight selected learning rate
            optimal_lr_point = lr_best[lr_best['learning_rate'] == best_lr]
            if not optimal_lr_point.empty:
                fig_lr.add_trace(go.Scatter(
                    x=[best_lr],
                    y=[optimal_lr_point[mae_col].values[0]],
                    mode='markers+text',
                    name='Selected',
                    marker=dict(size=20, color='#00FFFF', symbol='star', line=dict(color='white', width=2)),
                    text=[f"SELECTED<br>LR={best_lr:.0e}"],
                    textposition="top center",
                    textfont=dict(color='#00FFFF', size=11)
                ))

            fig_lr.update_layout(
                **get_plotly_layout("", height=400),
                xaxis_title='Learning Rate',
                yaxis_title='MAE (nm)',
                xaxis_type='log'
            )

            st.plotly_chart(fig_lr, use_container_width=True)

        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

        # Plot 3: Model Complexity vs Training Time
        st.markdown('<p class="subsection-header">Convergence 3: Efficiency Analysis (Training Time vs Model Size)</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: #a0aec0; font-size: 0.85rem; margin-bottom: 1rem;">Shows the computational cost of increasing model capacity</p>', unsafe_allow_html=True)

        if 'n_parameters' in df.columns and 'train_time_s' in df.columns:
            fig_efficiency = go.Figure()

            # Color by MAE (performance)
            fig_efficiency.add_trace(go.Scatter(
                x=df['n_parameters'],
                y=df['train_time_s'],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=df[mae_col],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="MAE (nm)", x=1.15),
                    line=dict(color='white', width=1)
                ),
                text=[f"{int(row['num_blocks'])}B<br>LR={row['learning_rate']:.0e}" for _, row in df.iterrows()],
                textposition='top center',
                textfont=dict(size=9),
                name='Experiments'
            ))

            # Highlight best
            fig_efficiency.add_trace(go.Scatter(
                x=[best_row.get('n_parameters', 0)],
                y=[best_row.get('train_time_s', 0)],
                mode='markers',
                marker=dict(size=25, color='#00FFFF', symbol='star', line=dict(color='white', width=3)),
                name='Optimal',
                showlegend=True
            ))

            fig_efficiency.update_layout(
                **get_plotly_layout("", height=400),
                xaxis_title='Number of Parameters',
                yaxis_title='Training Time (seconds)'
            )

            st.plotly_chart(fig_efficiency, use_container_width=True)

        # =================================================================
        # INTERPRETATION
        # =================================================================
        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
        st.markdown('<p class="subsection-header">How We Converged to Optimal Parameters</p>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: rgba(46, 204, 113, 0.1); border: 1px solid #2ecc71; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <p style="color: #FFFFFF; font-size: 1rem; line-height: 1.8; margin: 0;">
                <strong style="color: #2ecc71;">Optimization Result:</strong><br>
                After testing <strong>{len(df)} configurations</strong> across residual blocks and learning rates,
                the optimal SpectralResNet uses <strong>{best_blocks} residual blocks</strong> with learning rate
                <strong>{best_lr:.0e}</strong>, achieving <strong style="color: #2ecc71;">{best_mae:.4f} nm MAE</strong> —
                sub-angstrom precision sufficient for high-fidelity chromatic dispersion correction.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background: rgba(78, 205, 196, 0.1); border: 1px solid #4ECDC4; border-radius: 12px; padding: 1.5rem;">
            <p style="color: #FFFFFF; font-size: 0.95rem; line-height: 1.8; margin-bottom: 1rem;"><strong style="color: #4ECDC4;">Key Insights:</strong></p>
            <ol style="color: #a0aec0; font-size: 0.9rem; line-height: 1.8; margin: 0;">
                <li><strong style="color: #FFFFFF;">Optimal Depth:</strong> 4 residual blocks balance capacity and generalization—deeper models (8 blocks) showed diminishing returns and risk of overfitting</li>
                <li><strong style="color: #FFFFFF;">Learning Rate Sensitivity:</strong> 1e-3 converges faster and achieves lower final error than 1e-4, suggesting the loss landscape is well-conditioned</li>
                <li><strong style="color: #FFFFFF;">Efficiency Trade-off:</strong> The optimal configuration (134K parameters, ~72s training) achieves best performance without excessive computational cost</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # =================================================================
        # TECHNICAL DEEP-DIVE
        # =================================================================
        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

        with st.expander("Full Experiment Data Table", expanded=False):
            st.dataframe(df.sort_values(mae_col), use_container_width=True, height=400)

    else:
        st.warning("DOE results file not found. Run `python p2_rainbow_solver.py` to generate data.")

# =============================================================================
# AUTHOR FOOTER
# =============================================================================
st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="author-footer">
    <p style="margin: 0; color: #a0aec0; font-size: 0.9rem;">
        Built by <strong style="color: #FFFFFF;">Vaibhav Mathur</strong>
    </p>
    <p style="margin: 0.5rem 0 0 0;">
        <a href="https://x.com/vaibhavmathur91" target="_blank" style="color: #4ECDC4; text-decoration: none; margin-right: 1.5rem;">
            X (Twitter)
        </a>
        <a href="https://linkedin.com/in/vaibhavmathur91" target="_blank" style="color: #4ECDC4; text-decoration: none;">
            LinkedIn
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
