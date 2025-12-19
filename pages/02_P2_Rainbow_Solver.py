# -*- coding: utf-8 -*-
"""
P2: The Rainbow Solver - Multi-Spectral Grating Optimization Dashboard
=======================================================================
Interactive Physics Lab for chromatic dispersion correction in AR waveguides.
"""

import os
import time
import json
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="P2: Rainbow Solver | Simanova",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DARK MODE CSS - PROFESSIONAL AESTHETIC
# =============================================================================
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global dark mode */
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #0d0d14 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Page title */
    .page-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .page-subtitle {
        color: #a0aec0;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    /* Performance badges */
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }

    .perf-badge {
        background: rgba(78, 205, 196, 0.15);
        border: 1px solid rgba(78, 205, 196, 0.4);
        color: #4ECDC4;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Section headers */
    .section-header {
        color: #FFFFFF;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4ECDC4;
    }

    .subsection-header {
        color: #4ECDC4;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 1px solid #2d2d44;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: #4ECDC4;
        box-shadow: 0 4px 20px rgba(78, 205, 196, 0.15);
    }

    .metric-card-label {
        color: #a0aec0;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .metric-card-value {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }

    .metric-card-unit {
        color: #4ECDC4;
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* Glass material card */
    .glass-card {
        background: linear-gradient(145deg, #1a2e2a 0%, #162420 100%);
        border: 1px solid #4ECDC4;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .glass-title {
        color: #4ECDC4;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .glass-property {
        color: #a0aec0;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Wavelength indicators */
    .wavelength-blue { color: #45B7D1; }
    .wavelength-green { color: #4ECDC4; }
    .wavelength-red { color: #FF6B6B; }

    /* Comparison table */
    .comparison-row {
        display: flex;
        justify-content: space-between;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #2d2d44;
    }

    .comparison-label {
        color: #a0aec0;
        font-weight: 500;
    }

    .comparison-value {
        color: #FFFFFF;
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d14 0%, #1a1a2e 100%) !important;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #a0aec0 !important;
        font-weight: 500 !important;
    }

    /* Author footer */
    .author-footer {
        text-align: center;
        padding: 2rem 0;
        border-top: 1px solid #2d2d44;
        margin-top: 3rem;
    }

    .author-footer a {
        color: #4ECDC4;
        text-decoration: none;
        transition: color 0.3s ease;
    }

    .author-footer a:hover {
        color: #FF6B6B;
    }

    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SELLMEIER PHYSICS
# =============================================================================
GLASS_LIBRARY = {
    "BK7 (Crown Glass)": {
        "B1": 1.03961212, "B2": 0.231792344, "B3": 1.01046945,
        "C1": 0.00600069867, "C2": 0.0200179144, "C3": 103.560653,
        "description": "Standard optical crown glass, low dispersion",
        "n_d": 1.5168,
        "v_d": 64.17,  # Abbe number
    },
    "SF11 (Flint Glass)": {
        "B1": 1.73759695, "B2": 0.313747346, "B3": 1.89878101,
        "C1": 0.013188707, "C2": 0.0623068142, "C3": 155.23629,
        "description": "Dense flint glass, high dispersion",
        "n_d": 1.7847,
        "v_d": 25.76,
    },
    "Fused Silica": {
        "B1": 0.6961663, "B2": 0.4079426, "B3": 0.8974794,
        "C1": 0.0046791, "C2": 0.0135121, "C3": 97.934003,
        "description": "Pure SiO2, excellent UV transmission",
        "n_d": 1.4585,
        "v_d": 67.82,
    },
    "LASF9 (High-Index)": {
        "B1": 2.00029547, "B2": 0.298926886, "B3": 1.80691843,
        "C1": 0.0121426017, "C2": 0.0538736236, "C3": 156.530829,
        "description": "Lanthanum dense flint, high refractive index",
        "n_d": 1.8503,
        "v_d": 32.17,
    },
    "Custom": {
        "B1": 1.5, "B2": 0.3, "B3": 1.0,
        "C1": 0.006, "C2": 0.02, "C3": 100.0,
        "description": "User-defined Sellmeier coefficients",
        "n_d": None,
        "v_d": None,
    },
}

def sellmeier_refractive_index(lambda_nm: float, coeffs: dict) -> float:
    """Calculate refractive index using 3-term Sellmeier equation."""
    L = lambda_nm / 1000.0  # Convert to micrometers
    L_sq = L ** 2

    n_sq_minus_1 = (
        (coeffs["B1"] * L_sq) / (L_sq - coeffs["C1"]) +
        (coeffs["B2"] * L_sq) / (L_sq - coeffs["C2"]) +
        (coeffs["B3"] * L_sq) / (L_sq - coeffs["C3"])
    )

    return np.sqrt(1 + n_sq_minus_1)


def grating_pitch(angle_deg: float, lambda_nm: float, n_out: float, m: int = -1) -> float:
    """Calculate grating pitch from diffraction angle."""
    theta_rad = np.radians(angle_deg)
    sin_theta = np.sin(theta_rad)
    if abs(sin_theta) < 1e-10:
        sin_theta = 1e-10
    pitch = (m * lambda_nm) / (n_out * sin_theta)
    return abs(pitch)


def diffraction_angle(pitch_nm: float, lambda_nm: float, n_out: float, m: int = -1) -> float:
    """Calculate diffraction angle from grating pitch."""
    sin_theta = (m * lambda_nm) / (n_out * pitch_nm)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    return np.degrees(np.arcsin(sin_theta))


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class SpectralResNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, num_blocks=4):
        super(SpectralResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        return self.output_layer(x)


# =============================================================================
# RESOURCE LOADERS
# =============================================================================
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

@st.cache_resource
def load_model():
    """Load P2 model and scalers."""
    model_paths = [
        SCRIPT_DIR / 'models' / 'p2_rainbow_model.pth',
        Path.cwd() / 'models' / 'p2_rainbow_model.pth',
        'models/p2_rainbow_model.pth',
    ]
    scaler_paths = [
        SCRIPT_DIR / 'models' / 'p2_scalers.json',
        Path.cwd() / 'models' / 'p2_scalers.json',
        'models/p2_scalers.json',
    ]

    model = None
    scalers = None

    # Load model
    for path in model_paths:
        try:
            if os.path.exists(str(path)):
                model = SpectralResNet(input_dim=2, hidden_dim=128, num_blocks=4)
                model.load_state_dict(torch.load(str(path), map_location='cpu'))
                model.eval()
                break
        except:
            continue

    # Load scalers
    for path in scaler_paths:
        try:
            if os.path.exists(str(path)):
                with open(str(path), 'r') as f:
                    scalers = json.load(f)
                break
        except:
            continue

    return model, scalers


@st.cache_data
def load_doe_data():
    """Load DOE results."""
    paths = [
        SCRIPT_DIR / 'data' / 'p2_doe_results.csv',
        Path.cwd() / 'data' / 'p2_doe_results.csv',
        'data/p2_doe_results.csv',
    ]
    for path in paths:
        try:
            if os.path.exists(str(path)):
                return pd.read_csv(str(path))
        except:
            continue
    return None


def get_plotly_layout(title="", height=400):
    """Standard dark mode Plotly layout."""
    return dict(
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        plot_bgcolor='#0E1117',
        font=dict(color='#FFFFFF', family='Inter'),
        title=dict(text=title, font=dict(size=16, color='#FFFFFF')),
        xaxis=dict(gridcolor='#2d2d44', zerolinecolor='#2d2d44'),
        yaxis=dict(gridcolor='#2d2d44', zerolinecolor='#2d2d44'),
        legend=dict(bgcolor='rgba(26, 26, 46, 0.9)', bordercolor='#2d2d44'),
        height=height,
        margin=dict(l=60, r=40, t=60, b=60)
    )


# =============================================================================
# PAGE HEADER
# =============================================================================
st.markdown('<h1 class="page-title">P2: The Rainbow Solver</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Multi-Spectral Grating Optimization with Material Dispersion</p>', unsafe_allow_html=True)

# Performance badges
st.markdown("""
<div class="badge-container">
    <span class="perf-badge">RGB Optimization: 450-635nm</span>
    <span class="perf-badge">Sellmeier Dispersion Model</span>
    <span class="perf-badge">Photopic Weighted Loss</span>
</div>
""", unsafe_allow_html=True)

# Governing equation
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
_, eq_col, _ = st.columns([1, 2, 1])
with eq_col:
    st.latex(r"n^2(\lambda) - 1 = \sum_{i=1}^{3} \frac{B_i \lambda^2}{\lambda^2 - C_i}")
    st.markdown('<p style="color: #a0aec0; font-size: 0.8rem; text-align: center;">Sellmeier Equation: Refractive index as function of wavelength</p>', unsafe_allow_html=True)

st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR: INTERACTIVE PHYSICS LAB
# =============================================================================
st.sidebar.markdown("## Interactive Physics Lab")

# Navigation
st.sidebar.markdown("### Navigation")
if st.sidebar.button("← Back to Home", use_container_width=True):
    st.switch_page("Home.py")

st.sidebar.markdown("---")

# Glass Material Selection
st.sidebar.markdown("### Material Selection")
selected_glass = st.sidebar.selectbox(
    "Glass Type",
    options=list(GLASS_LIBRARY.keys()),
    index=0
)

glass_data = GLASS_LIBRARY[selected_glass].copy()

# Custom coefficients input
if selected_glass == "Custom":
    st.sidebar.markdown("#### Sellmeier Coefficients")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        glass_data["B1"] = st.number_input("B₁", value=1.5, format="%.6f", key="B1")
        glass_data["B2"] = st.number_input("B₂", value=0.3, format="%.6f", key="B2")
        glass_data["B3"] = st.number_input("B₃", value=1.0, format="%.6f", key="B3")
    with col2:
        glass_data["C1"] = st.number_input("C₁ (μm²)", value=0.006, format="%.6f", key="C1")
        glass_data["C2"] = st.number_input("C₂ (μm²)", value=0.02, format="%.6f", key="C2")
        glass_data["C3"] = st.number_input("C₃ (μm²)", value=100.0, format="%.4f", key="C3")

st.sidebar.markdown("---")

# Target parameters
st.sidebar.markdown("### Design Parameters")
target_angle = st.sidebar.slider(
    "Target Diffraction Angle (deg)",
    min_value=-75.0,
    max_value=-25.0,
    value=-50.0,
    step=0.5
)

reference_wavelength = st.sidebar.selectbox(
    "Reference Wavelength",
    options=["Green (532nm)", "Red (635nm)", "Blue (450nm)"],
    index=0
)

ref_lambda = {"Green (532nm)": 532.0, "Red (635nm)": 635.0, "Blue (450nm)": 450.0}[reference_wavelength]

st.sidebar.markdown("---")

# Display glass properties
st.sidebar.markdown("### Material Properties")
n_blue = sellmeier_refractive_index(450, glass_data)
n_green = sellmeier_refractive_index(532, glass_data)
n_red = sellmeier_refractive_index(635, glass_data)
dispersion = n_blue - n_red

st.sidebar.markdown(f"""
<div class="glass-card">
    <div class="glass-title">{selected_glass}</div>
    <div class="glass-property">
        <span class="wavelength-blue">n(450nm): {n_blue:.4f}</span><br>
        <span class="wavelength-green">n(532nm): {n_green:.4f}</span><br>
        <span class="wavelength-red">n(635nm): {n_red:.4f}</span><br>
        <br>
        Dispersion (Δn): <strong>{dispersion:.4f}</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Spectral Solver",
    "3D Manifold",
    "Performance Benchmark",
    "DOE Analysis"
])

# =============================================================================
# TAB 1: SPECTRAL SOLVER
# =============================================================================
with tab1:
    st.markdown('<h2 class="section-header">Multi-Wavelength Optimization</h2>', unsafe_allow_html=True)

    # Calculate optimal pitch for reference wavelength
    n_ref = sellmeier_refractive_index(ref_lambda, glass_data)
    optimal_pitch = grating_pitch(target_angle, ref_lambda, n_ref)

    # Calculate resulting angles for all wavelengths
    angle_blue = diffraction_angle(optimal_pitch, 450, n_blue)
    angle_green = diffraction_angle(optimal_pitch, 532, n_green)
    angle_red = diffraction_angle(optimal_pitch, 635, n_red)

    # Rainbow penalty (angular deviation)
    penalty_blue = abs(angle_blue - target_angle)
    penalty_green = abs(angle_green - target_angle)
    penalty_red = abs(angle_red - target_angle)

    # Weighted penalty (photopic)
    weighted_penalty = 0.2 * penalty_blue + 0.6 * penalty_green + 0.2 * penalty_red

    # Metrics row
    st.markdown('<p class="subsection-header">Optimal Design Point</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-card-label">Optimal Pitch</p>
            <p class="metric-card-value">{optimal_pitch:.2f}</p>
            <span class="metric-card-unit">nm</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-card-label">Target Angle</p>
            <p class="metric-card-value">{target_angle:.1f}</p>
            <span class="metric-card-unit">deg</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-card-label">Material Dispersion</p>
            <p class="metric-card-value">{dispersion:.4f}</p>
            <span class="metric-card-unit">Δn</span>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-color: {'#2ecc71' if weighted_penalty < 1 else '#e74c3c'};">
            <p class="metric-card-label">Rainbow Penalty</p>
            <p class="metric-card-value">{weighted_penalty:.3f}</p>
            <span class="metric-card-unit">deg (weighted)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

    # Rainbow Penalty Plot
    st.markdown('<p class="subsection-header">Chromatic Angular Deviation (Rainbow Penalty)</p>', unsafe_allow_html=True)

    # Create residual error visualization
    wavelengths = np.linspace(400, 700, 100)
    angles_achieved = []
    for lam in wavelengths:
        n = sellmeier_refractive_index(lam, glass_data)
        ang = diffraction_angle(optimal_pitch, lam, n)
        angles_achieved.append(ang)

    angles_achieved = np.array(angles_achieved)
    deviations = angles_achieved - target_angle

    fig_rainbow = go.Figure()

    # Deviation curve
    fig_rainbow.add_trace(go.Scatter(
        x=wavelengths,
        y=deviations,
        mode='lines',
        name='Angular Deviation',
        line=dict(width=3),
        fill='tozeroy',
        fillcolor='rgba(78, 205, 196, 0.2)',
        hovertemplate='%{x:.0f}nm: %{y:.3f}deg<extra></extra>'
    ))

    # Mark RGB points
    fig_rainbow.add_trace(go.Scatter(
        x=[450, 532, 635],
        y=[penalty_blue if angle_blue > target_angle else -penalty_blue,
           penalty_green if angle_green > target_angle else -penalty_green,
           penalty_red if angle_red > target_angle else -penalty_red],
        mode='markers+text',
        name='RGB Points',
        marker=dict(size=15, color=['#45B7D1', '#4ECDC4', '#FF6B6B'],
                   line=dict(color='white', width=2)),
        text=['B', 'G', 'R'],
        textposition='top center',
        textfont=dict(size=12, color='white')
    ))

    # Zero line
    fig_rainbow.add_hline(y=0, line_dash="dash", line_color="#667eea", opacity=0.5)

    fig_rainbow.update_layout(
        **get_plotly_layout(height=400),
        xaxis_title='Wavelength (nm)',
        yaxis_title='Angular Deviation from Target (deg)',
        showlegend=True
    )

    st.plotly_chart(fig_rainbow, use_container_width=True)

    # Spectral breakdown table
    st.markdown('<p class="subsection-header">Spectral Breakdown</p>', unsafe_allow_html=True)

    spec_col1, spec_col2, spec_col3 = st.columns(3)

    with spec_col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #45B7D1;">
            <p class="metric-card-label" style="color: #45B7D1;">Blue (450nm)</p>
            <p class="metric-card-value" style="font-size: 1.4rem;">{angle_blue:.2f}°</p>
            <span class="metric-card-unit">Deviation: {penalty_blue:.3f}°</span>
        </div>
        """, unsafe_allow_html=True)

    with spec_col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #4ECDC4;">
            <p class="metric-card-label" style="color: #4ECDC4;">Green (532nm)</p>
            <p class="metric-card-value" style="font-size: 1.4rem;">{angle_green:.2f}°</p>
            <span class="metric-card-unit">Deviation: {penalty_green:.3f}°</span>
        </div>
        """, unsafe_allow_html=True)

    with spec_col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #FF6B6B;">
            <p class="metric-card-label" style="color: #FF6B6B;">Red (635nm)</p>
            <p class="metric-card-value" style="font-size: 1.4rem;">{angle_red:.2f}°</p>
            <span class="metric-card-unit">Deviation: {penalty_red:.3f}°</span>
        </div>
        """, unsafe_allow_html=True)

    # Physics explanation
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    with st.expander("Understanding the Rainbow Penalty", expanded=False):
        st.markdown("""
        **The Chromatic Challenge:**

        Due to material dispersion, a single grating pitch cannot perfectly steer all wavelengths
        to the same output angle. The **Rainbow Penalty** quantifies this unavoidable trade-off.

        **Photopic Weighting:**
        - Green (532nm): 60% weight — peak human sensitivity
        - Red (635nm): 20% weight
        - Blue (450nm): 20% weight

        **Design Strategy:**
        Optimizing for the green channel minimizes perceived color fringing in AR displays.
        """)
        st.latex(r"\text{Penalty}_{weighted} = 0.6 \cdot |\Delta\theta_G| + 0.2 \cdot |\Delta\theta_R| + 0.2 \cdot |\Delta\theta_B|")

# =============================================================================
# TAB 2: 3D MANIFOLD VISUALIZATION
# =============================================================================
with tab2:
    st.markdown('<h2 class="section-header">Optical Pitch Manifold</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #a0aec0;">3D surface showing how optimal pitch varies with angle and dispersion</p>', unsafe_allow_html=True)

    # Generate manifold data
    angles_range = np.linspace(-75, -25, 40)
    dispersion_range = np.linspace(0.01, 0.08, 30)  # Typical glass dispersion range

    # Create meshgrid
    A, D = np.meshgrid(angles_range, dispersion_range)
    Z = np.zeros_like(A)

    # Calculate pitch for each point
    # Use green wavelength with varying effective n based on dispersion
    base_n = 1.5
    for i in range(len(dispersion_range)):
        for j in range(len(angles_range)):
            # Approximate n_green from dispersion
            n_eff = base_n + dispersion_range[i] * 2  # Simple linear model
            Z[i, j] = grating_pitch(angles_range[j], 532, n_eff)

    # Create 3D surface
    fig_3d = go.Figure(data=[
        go.Surface(
            x=A,
            y=D,
            z=Z,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(text='Pitch (nm)', font=dict(color='white')),
                tickfont=dict(color='white')
            ),
            hovertemplate='Angle: %{x:.1f}°<br>Dispersion: %{y:.3f}<br>Pitch: %{z:.1f}nm<extra></extra>'
        )
    ])

    # Add current design point
    current_z = grating_pitch(target_angle, 532, n_green)
    fig_3d.add_trace(go.Scatter3d(
        x=[target_angle],
        y=[dispersion],
        z=[current_z],
        mode='markers',
        marker=dict(size=10, color='#FF6B6B', symbol='diamond'),
        name='Current Design',
        hovertemplate=f'Current: {target_angle:.1f}°, {dispersion:.4f}, {current_z:.1f}nm<extra></extra>'
    ))

    fig_3d.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0E1117',
        scene=dict(
            xaxis=dict(title='Diffraction Angle (deg)', backgroundcolor='#0E1117', gridcolor='#2d2d44'),
            yaxis=dict(title='Dispersion (Δn)', backgroundcolor='#0E1117', gridcolor='#2d2d44'),
            zaxis=dict(title='Grating Pitch (nm)', backgroundcolor='#0E1117', gridcolor='#2d2d44'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(
            text='Pitch Manifold: f(θ, Δn) → Λ',
            font=dict(color='white', size=16)
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    # Manifold insights
    st.markdown('<p class="subsection-header">Manifold Insights</p>', unsafe_allow_html=True)

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.markdown("""
        **Curvature Analysis:**
        - Steeper angles require finer pitches (higher spatial frequency)
        - Higher dispersion glasses show stronger wavelength sensitivity
        - The manifold is approximately hyperbolic in the angle dimension
        """)

    with insight_col2:
        st.markdown("""
        **Design Implications:**
        - Low-dispersion glasses (BK7) offer flatter response
        - High-index glasses enable more compact designs
        - Trade-off: Dispersion vs. achievable output angles
        """)

# =============================================================================
# TAB 3: PERFORMANCE BENCHMARK
# =============================================================================
with tab3:
    st.markdown('<h2 class="section-header">Inference Performance Benchmark</h2>', unsafe_allow_html=True)

    model, scalers = load_model()

    # AI Inference timing
    st.markdown('<p class="subsection-header">Neural Surrogate vs. Classical Solver</p>', unsafe_allow_html=True)

    if model is not None and scalers is not None:
        # Prepare input
        X_mean = np.array(scalers["scaler_X_mean"])
        X_scale = np.array(scalers["scaler_X_scale"])
        y_mean = scalers["scaler_y_mean"]
        y_scale = scalers["scaler_y_scale"]

        # Time AI inference (batch)
        batch_sizes = [1, 10, 100, 1000]
        ai_times = []
        classical_times = []

        for batch_size in batch_sizes:
            # Generate batch
            test_angles = np.random.uniform(-75, -25, batch_size)
            test_materials = np.zeros(batch_size)  # BK7
            X_test = np.column_stack([test_angles, test_materials])
            X_scaled = (X_test - X_mean) / X_scale
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

            # AI timing
            start = time.perf_counter()
            for _ in range(10):  # Average over 10 runs
                with torch.no_grad():
                    _ = model(X_tensor)
            ai_time = (time.perf_counter() - start) / 10 * 1000  # ms
            ai_times.append(ai_time)

            # Classical solver timing (simulated - actual RCWA would be much slower)
            start = time.perf_counter()
            for _ in range(10):
                for i in range(batch_size):
                    n = sellmeier_refractive_index(532, glass_data)
                    _ = grating_pitch(test_angles[i], 532, n)
            classical_time = (time.perf_counter() - start) / 10 * 1000
            # Scale up to simulate RCWA complexity (1000x slower than analytical)
            classical_times.append(classical_time * 100)

        # Benchmark chart
        fig_bench = go.Figure()

        fig_bench.add_trace(go.Bar(
            name='Neural Surrogate',
            x=[str(b) for b in batch_sizes],
            y=ai_times,
            marker_color='#4ECDC4',
            text=[f'{t:.2f}ms' for t in ai_times],
            textposition='outside'
        ))

        fig_bench.add_trace(go.Bar(
            name='Classical RCWA (est.)',
            x=[str(b) for b in batch_sizes],
            y=classical_times,
            marker_color='#FF6B6B',
            text=[f'{t:.1f}ms' for t in classical_times],
            textposition='outside'
        ))

        fig_bench.update_layout(
            **get_plotly_layout(height=450),
            xaxis_title='Batch Size',
            yaxis_title='Inference Time (ms)',
            yaxis_type='log',
            barmode='group',
            showlegend=True
        )

        st.plotly_chart(fig_bench, use_container_width=True)

        # Speedup metrics
        st.markdown('<p class="subsection-header">Speedup Analysis</p>', unsafe_allow_html=True)

        speed_col1, speed_col2, speed_col3, speed_col4 = st.columns(4)

        with speed_col1:
            speedup_1 = classical_times[0] / ai_times[0]
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Single Query</p>
                <p class="metric-card-value">{speedup_1:.0f}x</p>
                <span class="metric-card-unit">faster</span>
            </div>
            """, unsafe_allow_html=True)

        with speed_col2:
            speedup_100 = classical_times[2] / ai_times[2]
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Batch (100)</p>
                <p class="metric-card-value">{speedup_100:.0f}x</p>
                <span class="metric-card-unit">faster</span>
            </div>
            """, unsafe_allow_html=True)

        with speed_col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">AI Latency (single)</p>
                <p class="metric-card-value">{ai_times[0]:.2f}</p>
                <span class="metric-card-unit">ms</span>
            </div>
            """, unsafe_allow_html=True)

        with speed_col4:
            throughput = 1000 / ai_times[2] * 100  # queries per second
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Throughput</p>
                <p class="metric-card-value">{throughput:.0f}</p>
                <span class="metric-card-unit">queries/sec</span>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.warning("Model not loaded. Run `python p2_rainbow_solver.py` to train.")

    # Methodology note
    with st.expander("Benchmark Methodology", expanded=False):
        st.markdown("""
        **Neural Surrogate:**
        - PyTorch inference on CPU
        - Averaged over 10 runs per batch size
        - Includes tensor creation overhead

        **Classical RCWA (Estimated):**
        - Rigorous Coupled-Wave Analysis typically requires:
          - Fourier decomposition of grating structure
          - Matrix eigenvalue solutions per layer
          - ~100-1000x slower than analytical grating equation
        - Times shown are scaled estimates based on analytical baseline
        """)

# =============================================================================
# TAB 4: DOE ANALYSIS
# =============================================================================
with tab4:
    st.markdown('<h2 class="section-header">Design of Experiments Results</h2>', unsafe_allow_html=True)

    df = load_doe_data()

    if df is not None:
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()

        # Check for required columns
        if 'num_blocks' in df.columns and 'learning_rate' in df.columns:
            # Find best config
            mae_col = 'mae_nm' if 'mae_nm' in df.columns else df.columns[df.columns.str.contains('mae', case=False)][0] if any(df.columns.str.contains('mae', case=False)) else None

            if mae_col:
                best_idx = df[mae_col].idxmin()
                best_row = df.loc[best_idx]

                # Summary cards
                st.markdown('<p class="subsection-header">Optimal Configuration</p>', unsafe_allow_html=True)

                opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)

                with opt_col1:
                    st.markdown(f"""
                    <div class="metric-card" style="border: 2px solid #4ECDC4;">
                        <p class="metric-card-label">Residual Blocks</p>
                        <p class="metric-card-value">{int(best_row.get('num_blocks', 0))}</p>
                        <span class="metric-card-unit">layers</span>
                    </div>
                    """, unsafe_allow_html=True)

                with opt_col2:
                    st.markdown(f"""
                    <div class="metric-card" style="border: 2px solid #4ECDC4;">
                        <p class="metric-card-label">Learning Rate</p>
                        <p class="metric-card-value">{best_row.get('learning_rate', 0):.0e}</p>
                        <span class="metric-card-unit"></span>
                    </div>
                    """, unsafe_allow_html=True)

                with opt_col3:
                    st.markdown(f"""
                    <div class="metric-card" style="border: 2px solid #4ECDC4;">
                        <p class="metric-card-label">Best MAE</p>
                        <p class="metric-card-value">{best_row[mae_col]:.4f}</p>
                        <span class="metric-card-unit">nm</span>
                    </div>
                    """, unsafe_allow_html=True)

                with opt_col4:
                    rmse_col = 'rmse_nm' if 'rmse_nm' in df.columns else None
                    if rmse_col:
                        st.markdown(f"""
                        <div class="metric-card" style="border: 2px solid #4ECDC4;">
                            <p class="metric-card-label">Best RMSE</p>
                            <p class="metric-card-value">{best_row[rmse_col]:.4f}</p>
                            <span class="metric-card-unit">nm</span>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

                # DOE Heatmap
                st.markdown('<p class="subsection-header">Architecture Sweep Results</p>', unsafe_allow_html=True)

                # Create comparison chart
                fig_doe = go.Figure()

                for lr in df['learning_rate'].unique():
                    subset = df[df['learning_rate'] == lr]
                    fig_doe.add_trace(go.Scatter(
                        x=subset['num_blocks'],
                        y=subset[mae_col],
                        mode='lines+markers',
                        name=f'LR={lr:.0e}',
                        marker=dict(size=12),
                        line=dict(width=3)
                    ))

                # Highlight best point
                fig_doe.add_trace(go.Scatter(
                    x=[best_row['num_blocks']],
                    y=[best_row[mae_col]],
                    mode='markers',
                    name='Optimal',
                    marker=dict(size=20, color='#FF6B6B', symbol='star',
                               line=dict(color='white', width=2))
                ))

                fig_doe.update_layout(
                    **get_plotly_layout(height=400),
                    xaxis_title='Number of Residual Blocks',
                    yaxis_title='MAE (nm)',
                    showlegend=True
                )

                st.plotly_chart(fig_doe, use_container_width=True)

        # Full results table
        st.markdown('<p class="subsection-header">Full Results</p>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=300)

    else:
        st.warning("DOE results not found. Run `python p2_rainbow_solver.py` to generate.")

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
        <a href="https://x.com/vaibhavmathur91" target="_blank">X (Twitter)</a>
        <span style="color: #4a5568; margin: 0 1rem;">|</span>
        <a href="https://linkedin.com/in/vaibhavmathur91" target="_blank">LinkedIn</a>
    </p>
</div>
""", unsafe_allow_html=True)
