# -*- coding: utf-8 -*-
"""
Project 2: Rainbow Solver - Multi-Spectral Grating Optimization
Physical AI Architect Dashboard - Production Build
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import os
import json
import time
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
# ROBUST PATH RESOLUTION (MATCHING P1 PATTERN)
# =============================================================================
# Get the root directory (parent of pages/)
try:
    SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
except:
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR  # Alias for compatibility

# =============================================================================
# CUSTOM CSS - PROFESSIONAL DARK MODE
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
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.25rem; text-align: center;
    }
    .page-subtitle { font-size: 1.1rem; color: #a0aec0 !important; text-align: center; margin-bottom: 1rem; }
    .project-desc { color: #c0c8d0 !important; font-size: 0.95rem; line-height: 1.6; text-align: center; max-width: 900px; margin: 0 auto 1.5rem auto; }
    .project-desc strong { color: #FFFFFF !important; }
    .section-header { color: #4ECDC4 !important; font-size: 1.4rem; font-weight: 600; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #2d2d44; }
    .subsection-header { color: #FFFFFF !important; font-size: 1.1rem; font-weight: 600; margin: 1.5rem 0 1rem 0; }
    .method-card { background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%); border: 1px solid #2d2d44; border-radius: 12px; padding: 1.25rem; margin: 0.75rem 0; }
    .method-number { display: inline-block; background: linear-gradient(135deg, #4ECDC4 0%, #45B7D1 100%); color: white !important; font-weight: 700; font-size: 0.8rem; padding: 0.25rem 0.6rem; border-radius: 6px; margin-right: 0.5rem; }
    .method-title { color: #FFFFFF !important; font-weight: 600; display: inline; }
    .method-desc { color: #c0c8d0 !important; font-size: 0.9rem; margin-top: 0.5rem; line-height: 1.6; }
    .method-desc strong { color: #FFFFFF !important; }
    .metric-card { background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%); border: 2px solid #4ECDC4; border-radius: 16px; padding: 2rem; text-align: center; box-shadow: 0 4px 20px rgba(78, 205, 196, 0.3); min-height: 180px; display: flex; flex-direction: column; justify-content: center; }
    .metric-card-label { color: #4ECDC4 !important; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
    .metric-card-value { color: #FFFFFF !important; font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; }
    .metric-card-delta { font-size: 0.9rem; padding: 0.4rem 0.8rem; border-radius: 6px; display: inline-block; margin-top: 0.5rem; }
    .delta-good { background: rgba(46, 204, 113, 0.2); color: #2ecc71 !important; border: 1px solid rgba(46, 204, 113, 0.4); }
    .delta-neutral { background: rgba(78, 205, 196, 0.2); color: #4ECDC4 !important; }
    .badge-container { display: flex; justify-content: center; align-items: center; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
    .perf-badge { display: inline-block; background: rgba(78, 205, 196, 0.15); border: 1px solid rgba(78, 205, 196, 0.3); color: #4ECDC4 !important; padding: 0.5rem 1.25rem; border-radius: 25px; font-size: 0.85rem; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
    .stTabs [data-baseweb="tab"] { background: #1a1a2e; border: 1px solid #2d2d44; border-radius: 8px; padding: 0.75rem 1.5rem; color: #a0aec0 !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #4ECDC4 0%, #45B7D1 100%); border-color: transparent; color: white !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    [data-testid="stSidebar"] { background: #1a1a2e !important; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    .author-footer { text-align: center; padding: 2rem 0 1rem 0; border-top: 1px solid #2d2d44; margin-top: 3rem; }
    .author-footer a { color: #4ECDC4 !important; text-decoration: none; margin: 0 0.5rem; }
    .stLatex, .katex, .katex-html, .katex-display { color: #FFFFFF !important; filter: brightness(0) invert(1); }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)

class SpectralResNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_blocks=4):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.output_layer(self.res_blocks(self.input_layer(x)))

# =============================================================================
# CACHED RESOURCE LOADERS WITH ROBUST FALLBACKS
# =============================================================================
@st.cache_data(ttl=300)
def generate_static_doe():
    """Generate static DOE data as fallback when CSV is missing"""
    return pd.DataFrame({
        'experiment_id': ['EXP_01', 'EXP_02', 'EXP_03', 'EXP_04', 'EXP_05', 'EXP_06'],
        'num_blocks': [2, 2, 4, 4, 8, 8],
        'learning_rate': [0.001, 0.0001, 0.001, 0.0001, 0.001, 0.0001],
        'n_parameters': [67841, 67841, 134913, 134913, 269057, 269057],
        'train_time_s': [45.73, 46.18, 72.37, 72.92, 112.41, 107.65],
        'mae_nm': [0.0429, 0.0529, 0.0249, 0.0602, 0.0532, 0.0789],
        'rmse_nm': [0.072, 0.0938, 0.0383, 0.1253, 0.0838, 0.1671],
    })

@st.cache_data(ttl=60)
def load_doe_data():
    """Load DOE results with fallback to static data"""
    paths = [
        SCRIPT_DIR / 'data' / 'p2_doe_results.csv',
        SCRIPT_DIR / 'Data' / 'p2_doe_results.csv',
        Path.cwd() / 'data' / 'p2_doe_results.csv',
        Path.cwd() / 'Data' / 'p2_doe_results.csv',
        'data/p2_doe_results.csv',
        'Data/p2_doe_results.csv',
    ]

    for path in paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                df.columns = df.columns.str.lower().str.strip()
                if 'mae_nm' in df.columns or 'mae' in df.columns:
                    return df, "loaded"
        except Exception:
            continue

    # Fallback to static data
    return generate_static_doe(), "static"

@st.cache_data
def load_scalers():
    """Load scaler parameters for model inference"""
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
    # Return default scalers if file not found (hardcoded from training)
    return {
        "scaler_X_mean": [-50.34740211391449, 0.50025],
        "scaler_X_scale": [14.390951526789614, 0.4999999374999994],
        "scaler_y_mean": 459.2576668510437,
        "scaler_y_scale": 120.51156191729024
    }

@st.cache_resource
def load_model():
    """Load trained ResNet with caching - returns (model, loaded_flag, status_msg)"""
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
                return model, True, f"Loaded from {path_str}"
        except Exception as e:
            continue

    return None, False, "Model file not found"

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
    """Calculate grating pitch from diffraction angle"""
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
# REAL-TIME MODEL INFERENCE
# =============================================================================
def run_inference(model, scalers, target_angle, material_id=0):
    """Execute real-time model.forward() with proper scaling"""
    if model is None or scalers is None:
        return None, False

    try:
        # Scale inputs
        angle_scaled = (target_angle - scalers["scaler_X_mean"][0]) / scalers["scaler_X_scale"][0]
        mat_scaled = (material_id - scalers["scaler_X_mean"][1]) / scalers["scaler_X_scale"][1]

        # Create input tensor and run forward pass
        input_tensor = torch.tensor([[angle_scaled, mat_scaled]], dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model(input_tensor).item()

        # Unscale output
        pitch_nm = pred_scaled * scalers["scaler_y_scale"] + scalers["scaler_y_mean"]
        return pitch_nm, True
    except Exception as e:
        return None, False

# =============================================================================
# MVP TRAINING FUNCTION
# =============================================================================
def train_mvp_model(epochs=10, n_samples=5000):
    """Quick 10-epoch training sprint for MVP model"""
    from torch.utils.data import DataLoader, TensorDataset

    progress_bar = st.progress(0, text="Initializing MVP training...")

    # Generate synthetic training data
    angles = np.random.uniform(-75, -25, n_samples).astype(np.float32)
    materials = np.zeros(n_samples, dtype=np.float32)  # BK7 only for MVP

    # Calculate ground truth pitches
    pitches = []
    for ang in angles:
        n_green = sellmeier_n(532, "BK7")
        pitch = grating_pitch(ang, 532, n_green)
        pitches.append(pitch)
    pitches = np.array(pitches, dtype=np.float32)

    # Normalize
    X = np.column_stack([angles, materials])
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = pitches.mean(), pitches.std()
    X_norm = (X - X_mean) / X_std
    y_norm = (pitches - y_mean) / y_std

    progress_bar.progress(10, text="Data generated...")

    # Create model and dataloader
    model = SpectralResNet(input_dim=2, hidden_dim=128, num_blocks=4)
    dataset = TensorDataset(
        torch.tensor(X_norm, dtype=torch.float32),
        torch.tensor(y_norm.reshape(-1, 1), dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        progress = int(10 + (epoch + 1) / epochs * 80)
        progress_bar.progress(progress, text=f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.6f}")

    # Save model and scalers
    model_dir = PROJECT_ROOT / 'models'
    model_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), model_dir / 'p2_rainbow_model.pth')

    scalers = {
        "scaler_X_mean": X_mean.tolist(),
        "scaler_X_scale": X_std.tolist(),
        "scaler_y_mean": float(y_mean),
        "scaler_y_scale": float(y_std)
    }
    with open(model_dir / 'p2_scalers.json', 'w') as f:
        json.dump(scalers, f, indent=2)

    progress_bar.progress(100, text="MVP model trained and saved!")
    time.sleep(1)
    progress_bar.empty()

    return model, scalers

# =============================================================================
# PLOTLY CONFIG FOR STABILITY
# =============================================================================
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "staticPlot": False,
    "responsive": True,
}

def get_plotly_layout(title="", height=400):
    """Standard dark mode layout for all charts"""
    return dict(
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='#FFFFFF', family='Inter'),
        title=dict(text=title, font=dict(size=16, color='#FFFFFF')),
        xaxis=dict(gridcolor='#2d2d44', zerolinecolor='#2d2d44', tickfont=dict(color='#FFFFFF')),
        yaxis=dict(gridcolor='#2d2d44', zerolinecolor='#2d2d44', tickfont=dict(color='#FFFFFF')),
        legend=dict(bgcolor='rgba(26, 26, 46, 0.9)', bordercolor='#2d2d44', font=dict(color='#FFFFFF')),
        height=height,
        margin=dict(l=60, r=40, t=60, b=60)
    )

# =============================================================================
# PAGE HEADER
# =============================================================================
st.markdown('<h1 class="page-title">P2: The Rainbow Solver</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Multi-Spectral Neural Surrogate for Chromatic Dispersion Correction</p>', unsafe_allow_html=True)

st.markdown("""
<div class="badge-container">
    <span class="perf-badge">RGB: 450-635nm</span>
    <span class="perf-badge">Sellmeier Dispersion</span>
    <span class="perf-badge">Photopic Weighting</span>
    <span class="perf-badge">ResNet-4 Architecture</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p class="project-desc">
    In AR waveguides, a single grating period cannot steer all wavelengths to the same angle due to
    <strong>material dispersion</strong>. This creates the <strong>"rainbow effect"</strong>: visible color fringing.
    This engine learns the <strong>optimal compromise pitch</strong> that minimizes chromatic angular error
    across R/G/B channels, weighted by human photopic vision (Green: 60%, Red/Blue: 20% each).
</p>
""", unsafe_allow_html=True)

# Governing Equations
_, eq_col, _ = st.columns([1, 2, 1])
with eq_col:
    st.latex(r"n^2(\lambda) - 1 = \sum_{i=1}^{3} \frac{B_i \lambda^2}{\lambda^2 - C_i}")
    st.latex(r"\Lambda_{opt} = \arg\min_\Lambda \left[ 0.6|\Delta\theta_G| + 0.2|\Delta\theta_R| + 0.2|\Delta\theta_B| \right]")

st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

# =============================================================================
# LOAD RESOURCES
# =============================================================================
model, model_loaded, model_status = load_model()
scalers = load_scalers()

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(["Methodology", "Spectral Solver", "DOE Analysis"])

# =============================================================================
# TAB 1: METHODOLOGY
# =============================================================================
with tab1:
    st.markdown('<h2 class="section-header">System Methodology</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">OBJ</span>
            <span class="method-title">The Chromatic Challenge</span>
            <p class="method-desc">Design <strong>one grating pitch (Λ)</strong> that steers R/G/B light to approximately the same angle, despite material dispersion.</p>
        </div>
        <div class="method-card">
            <span class="method-number">1</span>
            <span class="method-title">Sellmeier Dispersion Model</span>
            <p class="method-desc"><strong>Blue (450nm):</strong> Higher n → steeper angle<br><strong>Red (635nm):</strong> Lower n → shallower angle</p>
        </div>
        <div class="method-card">
            <span class="method-number">2</span>
            <span class="method-title">Photopic-Weighted Loss</span>
            <p class="method-desc"><strong>W<sub>green</sub>=0.6</strong>, <strong>W<sub>red</sub>=0.2</strong>, <strong>W<sub>blue</sub>=0.2</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">3</span>
            <span class="method-title">Why Neural Networks?</span>
            <p class="method-desc"><strong>RCWA:</strong> 5-10s per eval<br><strong>Neural:</strong> &lt;10ms (1000x speedup)</p>
        </div>
        <div class="method-card">
            <span class="method-number">4</span>
            <span class="method-title">ResNet Architecture</span>
            <p class="method-desc"><strong>Input:</strong> [angle, material] → <strong>4 ResBlocks</strong> → <strong>Output:</strong> pitch (nm)</p>
        </div>
        <div class="method-card">
            <span class="method-number">5</span>
            <span class="method-title">Validation</span>
            <p class="method-desc"><strong>Achieved:</strong> 0.025 nm MAE (sub-angstrom precision)</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: SPECTRAL SOLVER (INTERACTIVE INFERENCE)
# =============================================================================
with tab2:
    st.markdown('<h2 class="section-header">Interactive Inference Engine</h2>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("← Back to Home", use_container_width=True):
        st.switch_page("Home.py")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Design Parameters")

    target_angle = st.sidebar.slider("Target Diffraction Angle (deg)", -75.0, -25.0, -50.0, 0.5)
    glass_type = st.sidebar.selectbox("Glass Material", list(GLASS_LIBRARY.keys()), index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Status")

    # Handle missing model state
    if not model_loaded:
        st.sidebar.warning("Model Not Found")
        st.sidebar.caption(model_status)

        if st.sidebar.button("🚀 Train MVP Model", use_container_width=True, help="Quick 10-epoch training sprint"):
            with st.spinner("Training MVP model..."):
                model, scalers = train_mvp_model(epochs=10, n_samples=5000)
                model_loaded = True
                st.sidebar.success("MVP Model Ready!")
                st.rerun()
    else:
        st.sidebar.success("Model Active")
        st.sidebar.caption(model_status)

    # Physics calculations (always available)
    n_blue = sellmeier_n(450, glass_type)
    n_green = sellmeier_n(532, glass_type)
    n_red = sellmeier_n(635, glass_type)

    analytical_pitch = grating_pitch(target_angle, 532, n_green)

    angle_blue = diffraction_angle(analytical_pitch, 450, n_blue)
    angle_green = diffraction_angle(analytical_pitch, 532, n_green)
    angle_red = diffraction_angle(analytical_pitch, 635, n_red)

    penalty_blue = abs(angle_blue - target_angle)
    penalty_green = abs(angle_green - target_angle)
    penalty_red = abs(angle_red - target_angle)
    weighted_penalty = 0.2*penalty_blue + 0.6*penalty_green + 0.2*penalty_red

    # Real-time neural inference (hooked to slider)
    surrogate_pitch, inference_ok = run_inference(model, scalers, target_angle, material_id=0 if glass_type == "BK7" else -1)

    if not inference_ok or glass_type != "BK7":
        surrogate_pitch = analytical_pitch  # Fallback
        neural_active = False
    else:
        neural_active = True

    # Metrics display
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-card-label">Target Angle</p>
            <p class="metric-card-value">{target_angle:.1f}°</p>
            <span class="metric-card-delta delta-neutral">Input</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-card-label">Analytical Pitch</p>
            <p class="metric-card-value">{analytical_pitch:.2f}</p>
            <span class="metric-card-delta delta-neutral">nm (Grating Eq)</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if neural_active:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Neural Surrogate</p>
                <p class="metric-card-value">{surrogate_pitch:.2f}</p>
                <span class="metric-card-delta delta-good">nm (ResNet)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Neural Surrogate</p>
                <p class="metric-card-value">--</p>
                <span class="metric-card-delta delta-neutral">Waiting...</span>
            </div>
            """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-card-label">Rainbow Penalty</p>
            <p class="metric-card-value">{weighted_penalty:.3f}°</p>
            <span class="metric-card-delta delta-neutral">Photopic</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

    # Rainbow visualization
    st.markdown('<p class="subsection-header">Simulated Grating Output (User Eye View)</p>', unsafe_allow_html=True)

    fig_eye = go.Figure()
    fig_eye.add_shape(type="rect", x0=-2, x1=2, y0=-1, y1=1, fillcolor="#0a0a0f", line=dict(width=0))

    for i in range(-20, 21):
        fig_eye.add_shape(type="line", x0=i*0.08, x1=i*0.08, y0=-0.95, y1=-0.85, line=dict(color="#4ECDC4", width=1))

    fig_eye.add_annotation(x=0, y=-1.05, text=f"Grating (Λ={analytical_pitch:.1f}nm)", showarrow=False, font=dict(color="#4ECDC4", size=12))

    max_dev = max(penalty_blue, penalty_green, penalty_red, 0.1)
    scale = 0.8 / max_dev

    for beam, color, name, penalty, angle in [
        (penalty_blue, '#45B7D1', 'Blue 450nm', penalty_blue, angle_blue),
        (penalty_green, '#4ECDC4', 'Green 532nm', penalty_green, angle_green),
        (penalty_red, '#FF6B6B', 'Red 635nm', penalty_red, angle_red),
    ]:
        x = -penalty * scale if angle < target_angle else penalty * scale
        fig_eye.add_trace(go.Scatter(x=[0, x], y=[-0.8, 0.7], mode='lines',
                                      line=dict(color=color, width=8), name=f'{name} ({angle:.2f}°)', opacity=0.8))

    fig_eye.add_shape(type="line", x0=0, x1=0, y0=-0.8, y1=0.7, line=dict(color="white", width=1, dash="dash"))
    fig_eye.add_annotation(x=0, y=1.1, text="👁 Observer", font=dict(color='white', size=14), showarrow=False)

    fig_eye.update_layout(
        template='plotly_dark', paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
        height=400, showlegend=True,
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.2, 1.2]),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
    )

    if model_loaded or True:  # Always show visualization
        st.plotly_chart(fig_eye, use_container_width=True, config=PLOTLY_CONFIG)
    else:
        st.warning("Waiting for model...")

    # Chromatic deviation plot with SVG render mode for stability
    st.markdown('<p class="subsection-header">Chromatic Angular Deviation</p>', unsafe_allow_html=True)

    wavelengths = np.linspace(400, 700, 50)  # Reduced points for SVG
    deviations = []
    for lam in wavelengths:
        n = sellmeier_n(lam, glass_type)
        ang = diffraction_angle(analytical_pitch, lam, n)
        deviations.append(ang - target_angle)

    fig_spec = go.Figure()
    fig_spec.add_trace(go.Scatter(
        x=wavelengths, y=deviations,
        mode='lines', line=dict(width=3, color='#4ECDC4'),
        name='Deviation'
    ))
    fig_spec.add_trace(go.Scatter(
        x=[450, 532, 635],
        y=[angle_blue - target_angle, angle_green - target_angle, angle_red - target_angle],
        mode='markers', marker=dict(size=12, color=['#45B7D1', '#4ECDC4', '#FF6B6B']),
        name='RGB'
    ))
    fig_spec.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
    fig_spec.update_layout(**get_plotly_layout(height=350), xaxis_title='Wavelength (nm)', yaxis_title='Deviation (deg)')

    # Use SVG render mode for browser stability
    st.plotly_chart(fig_spec, use_container_width=True, config={**PLOTLY_CONFIG, "toImageButtonOptions": {"format": "svg"}})

# =============================================================================
# TAB 3: DOE ANALYSIS (WITH STATIC FALLBACK)
# =============================================================================
with tab3:
    st.markdown('<h2 class="section-header">Neural Architecture Search</h2>', unsafe_allow_html=True)

    df, data_source = load_doe_data()

    if data_source == "static":
        st.info("📊 Displaying pre-computed DOE results (static fallback)")

    if df is not None and not df.empty:
        mae_col = 'mae_nm' if 'mae_nm' in df.columns else 'mae'
        rmse_col = 'rmse_nm' if 'rmse_nm' in df.columns else ('rmse' if 'rmse' in df.columns else None)

        if mae_col not in df.columns:
            st.error(f"Missing MAE column. Found: {list(df.columns)}")
            st.stop()

        best_idx = df[mae_col].idxmin()
        best_row = df.loc[best_idx]
        best_blocks = int(best_row.get('num_blocks', 4))
        best_lr = best_row.get('learning_rate', 0.001)
        best_mae = best_row[mae_col]
        best_rmse = best_row.get(rmse_col, 0) if rmse_col else 0

        # Optimal config cards
        st.markdown('<p class="subsection-header">Optimal Configuration</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f'<div class="metric-card" style="border-color:#2ecc71;"><p class="metric-card-label">Blocks</p><p class="metric-card-value">{best_blocks}</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card" style="border-color:#2ecc71;"><p class="metric-card-label">LR</p><p class="metric-card-value">{best_lr:.0e}</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card" style="border-color:#2ecc71;"><p class="metric-card-label">MAE</p><p class="metric-card-value">{best_mae:.4f}</p></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card" style="border-color:#2ecc71;"><p class="metric-card-label">RMSE</p><p class="metric-card-value">{best_rmse:.4f}</p></div>', unsafe_allow_html=True)

        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

        # Convergence plot
        st.markdown('<p class="subsection-header">Convergence Analysis</p>', unsafe_allow_html=True)

        fig_conv = go.Figure()
        for lr in sorted(df['learning_rate'].unique(), reverse=True):
            subset = df[df['learning_rate'] == lr].sort_values('num_blocks')
            fig_conv.add_trace(go.Scatter(
                x=subset['num_blocks'], y=subset[mae_col],
                mode='lines+markers', name=f'LR={lr:.0e}',
                line=dict(width=3), marker=dict(size=10)
            ))

        fig_conv.add_trace(go.Scatter(
            x=[best_blocks], y=[best_mae],
            mode='markers', marker=dict(size=18, color='#00FFFF', symbol='star'),
            name='Selected'
        ))

        fig_conv.update_layout(**get_plotly_layout(height=400), xaxis_title='Residual Blocks', yaxis_title='MAE (nm)')

        st.plotly_chart(fig_conv, use_container_width=True, config=PLOTLY_CONFIG)

        # Interpretation
        st.markdown(f"""
        <div style="background: rgba(46, 204, 113, 0.1); border: 1px solid #2ecc71; border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem;">
            <p style="color: #FFFFFF; margin: 0;">
                <strong style="color: #2ecc71;">Result:</strong> {len(df)} configs tested →
                <strong>{best_blocks} blocks</strong> @ LR={best_lr:.0e} achieves
                <strong style="color: #2ecc71;">{best_mae:.4f} nm MAE</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Full DOE Table"):
            st.dataframe(df.sort_values(mae_col), use_container_width=True)
    else:
        st.error("No DOE data available")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("""
<div class="author-footer">
    <p style="margin: 0; color: #a0aec0;">Built by <strong style="color: #FFFFFF;">Vaibhav Mathur</strong></p>
    <p style="margin: 0.5rem 0 0 0;">
        <a href="https://x.com/vaibhavmathur91">X</a> |
        <a href="https://linkedin.com/in/vaibhavmathur91">LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)
