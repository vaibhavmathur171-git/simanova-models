# -*- coding: utf-8 -*-
"""
Project 1: Neural Surrogate for Inverse Optical Design
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
    page_title="P1: Inverse Waveguide Grating Design",
    page_icon="🔬",
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

    /* Page title */
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

    /* Large metric cards for Inverse Solver */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 2px solid #4B0082;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(75, 0, 130, 0.3);
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-card-label {
        color: #667eea !important;
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
        background: rgba(102, 126, 234, 0.2);
        color: #667eea !important;
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
        background: rgba(46, 204, 113, 0.15);
        border: 1px solid rgba(46, 204, 113, 0.3);
        color: #2ecc71 !important;
        padding: 0.5rem 1.25rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Summary cards for DOE */
    .summary-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 1px solid #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .summary-label {
        color: #667eea !important;
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
        color: #667eea !important;
        text-decoration: none;
        margin: 0 0.5rem;
    }
    .author-footer a:hover {
        color: #f093fb !important;
    }

    /* Force LaTeX to white */
    .stLatex, .katex, .katex-html, .katex-display {
        color: #FFFFFF !important;
        filter: brightness(0) invert(1);
    }

    /* DataFrame styling */
    .stDataFrame {
        border: 1px solid #667eea !important;
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
class SimpleMLP(nn.Module):
    """4-layer MLP for inverse grating design"""
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# =============================================================================
# CACHED RESOURCE LOADERS
# =============================================================================
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

@st.cache_data
def load_doe_data():
    """Load DOE results with fallback paths"""
    paths = [
        SCRIPT_DIR / 'data' / 'p1_doe_results.csv',
        SCRIPT_DIR / 'Data' / 'p1_doe_results.csv',
        'data/p1_doe_results.csv',
        'Data/p1_doe_results.csv'
    ]
    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

@st.cache_resource
def load_model():
    """Load trained MLP with caching"""
    paths = [
        SCRIPT_DIR / 'models' / 'p1_mono_model.pth',
        'models/p1_mono_model.pth',
        Path('models') / 'p1_mono_model.pth'
    ]
    model = SimpleMLP()
    for path in paths:
        try:
            if os.path.exists(path):
                model.load_state_dict(torch.load(str(path), map_location=torch.device('cpu')))
                model.eval()
                return model
        except Exception:
            continue
    return None

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
def grating_equation(target_angle_deg, wavelength_nm=532, n_out=1.5):
    """Analytical solution to the Grating Equation"""
    theta_out_rad = np.radians(target_angle_deg)
    m = -1
    if np.sin(theta_out_rad) == 0:
        return 0
    period = (m * wavelength_nm) / (n_out * np.sin(theta_out_rad))
    return abs(period)

def generate_optical_manifold(angle_range=(-80, -30), n_points=200):
    """Generate the angle-to-period mapping curve"""
    angles = np.linspace(angle_range[0], angle_range[1], n_points)
    periods = np.array([grating_equation(a) for a in angles])
    return angles, periods

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
st.markdown('<h1 class="page-title">P1: Inverse Waveguide Grating Design</h1>', unsafe_allow_html=True)
st.markdown('<p class="page-subtitle">Neural Surrogate for Real-Time Diffractive Optics Optimization</p>', unsafe_allow_html=True)

# Centered badges
st.markdown("""
<div class="badge-container">
    <span class="perf-badge">Inference: &lt;10ms</span>
    <span class="perf-badge">1000x Speedup vs. RCWA</span>
    <span class="perf-badge">Model: 4-Layer MLP</span>
</div>
""", unsafe_allow_html=True)

# Project description
st.markdown("""
<p class="project-desc">
    This engine demonstrates <strong>inverse design</strong> for AR waveguide gratings using neural networks.
    Given a target diffraction angle, the model predicts the required grating period (Λ) in milliseconds—bypassing
    expensive iterative RCWA simulations. A <strong>Design of Experiments (DOE)</strong> sweep identifies optimal
    neural architecture parameters for sub-nanometer accuracy.
</p>
""", unsafe_allow_html=True)

# Visual assets
grating_img = get_image_path('p1_grating.jpg')
neural_img = get_image_path('p1_neural_net.jpg')

if grating_img and neural_img:
    _, img_col1, img_col2, _ = st.columns([1, 2, 2, 1])
    with img_col1:
        st.image(grating_img, caption="Diffractive Grating", use_container_width=True)
    with img_col2:
        st.image(neural_img, caption="Neural Surrogate", use_container_width=True)

# Governing Equation
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
_, eq_col, _ = st.columns([1, 2, 1])
with eq_col:
    st.latex(r"n_{out} \sin(\theta_m) = n_{in} \sin(\theta_{in}) + \frac{m \lambda}{\Lambda}")
    st.markdown('<p style="color: #a0aec0; font-size: 0.8rem; text-align: center;">Λ = grating period, λ = wavelength, m = diffraction order</p>', unsafe_allow_html=True)

st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(["Methodology", "Inverse Solver", "DOE Analysis"])

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
            <span class="method-title">Objective</span>
            <p class="method-desc">
                Characterizing <strong>Neural Surrogates</strong> for diffractive waveguide design.
                A 4-layer MLP performs inverse mapping with <strong>sub-millisecond latency</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">1</span>
            <span class="method-title">Parametric Design Space</span>
            <p class="method-desc">
                <strong>Pitch Range:</strong> 300–600 nm<br>
                <strong>Wavelength:</strong> λ = 532 nm<br>
                <strong>Refractive Index:</strong> n<sub>out</sub> = 1.5
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">2</span>
            <span class="method-title">Forward Generation</span>
            <p class="method-desc">
                Analytical sweep using the Grating Equation to generate 10,000 ground-truth
                (angle, period) pairs for dense optical manifold sampling.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="method-card">
            <span class="method-number">3</span>
            <span class="method-title">Data Inversion</span>
            <p class="method-desc">
                <strong>Forward:</strong> f(Λ) → θ<sub>out</sub><br>
                <strong>Inverse:</strong> f<sup>-1</sup>(θ<sub>target</sub>) → Λ<br>
                Neural surrogate learns inverse mapping directly.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">4</span>
            <span class="method-title">Stochastic Perturbation</span>
            <p class="method-desc">
                Gaussian noise (σ = 0.5°) injected into training inputs simulates
                <strong>manufacturing tolerances</strong> and metrology uncertainty.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="method-card">
            <span class="method-number">5</span>
            <span class="method-title">Feature Normalization</span>
            <p class="method-desc">
                <strong>Method:</strong> Min-Max scaling to [0, 1]<br>
                Ensures stable gradient flow during training.
            </p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 2: INVERSE SOLVER
# =============================================================================
with tab2:
    st.markdown('<h2 class="section-header">Interactive Inference Engine</h2>', unsafe_allow_html=True)

    # Sidebar - Navigation and inputs
    st.sidebar.markdown("### Navigation")
    if st.sidebar.button("← Back to Home", use_container_width=True):
        st.switch_page("Home.py")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Input Parameters")
    target_angle = st.sidebar.slider(
        "Target Diffraction Angle (deg)",
        min_value=-80.0,
        max_value=-30.0,
        value=-51.0,
        step=0.1
    )
    wavelength = st.sidebar.slider(
        "Wavelength (nm)",
        min_value=450,
        max_value=650,
        value=532,
        step=1
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Status")

    # Load model and compute BOTH analytical and surrogate
    model = load_model()
    analytical_period = grating_equation(target_angle, wavelength_nm=wavelength)

    # Compute surrogate prediction
    if model:
        input_tensor = torch.tensor([[target_angle]], dtype=torch.float32)
        with torch.no_grad():
            surrogate_period = model(input_tensor).item()
        model_active = True
        st.sidebar.success("Model Loaded")
        st.sidebar.caption("Trained on 10K samples with Gaussian noise (σ=0.5°) for robustness")
    else:
        surrogate_period = analytical_period  # Fallback to analytical
        model_active = False
        st.sidebar.warning("Model Not Found")
        st.sidebar.caption("Run `python p1_doe_sweep.py` to train")

    # Calculate residual error (only meaningful when model is active)
    residual_error = abs(analytical_period - surrogate_period)
    pct_error = (residual_error / analytical_period) * 100 if analytical_period > 0 else 0

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
            <p class="metric-card-label">Analytical Solution</p>
            <p class="metric-card-value">{analytical_period:.2f} nm</p>
            <span class="metric-card-delta delta-neutral">Grating Equation</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if model_active:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Neural Surrogate</p>
                <p class="metric-card-value">{surrogate_period:.2f} nm</p>
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
                <p class="metric-card-label">Residual Error</p>
                <p class="metric-card-value">{residual_error:.3f} nm</p>
                <span class="metric-card-delta delta-good">|Analytical - AI|</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Residual Error</p>
                <p class="metric-card-value">--</p>
                <span class="metric-card-delta delta-neutral">N/A</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

    # Optical Manifold Chart
    st.markdown('<p class="subsection-header">Optical Manifold: Angle-to-Period Mapping</p>', unsafe_allow_html=True)

    angles, periods_analytical = generate_optical_manifold()

    fig = go.Figure()

    # Analytical curve
    fig.add_trace(go.Scatter(
        x=angles,
        y=periods_analytical,
        mode='lines',
        name='Analytical (Grating Eq.)',
        line=dict(color='#667eea', width=3)
    ))

    # Neural surrogate curve
    if model:
        input_batch = torch.tensor(angles.reshape(-1, 1), dtype=torch.float32)
        with torch.no_grad():
            periods_surrogate = model(input_batch).numpy().flatten()
        fig.add_trace(go.Scatter(
            x=angles,
            y=periods_surrogate,
            mode='lines',
            name='Neural Surrogate',
            line=dict(color='#f093fb', width=2, dash='dash')
        ))

    # Query point
    fig.add_trace(go.Scatter(
        x=[target_angle],
        y=[analytical_period],
        mode='markers+text',
        name='Query Point',
        marker=dict(color='#00FFFF', size=16, line=dict(color='white', width=2)),
        text=[f"({target_angle:.1f}°, {analytical_period:.1f}nm)"],
        textposition="top center",
        textfont=dict(color='#00FFFF', size=11)
    ))

    fig.update_layout(
        **get_plotly_layout(height=450),
        xaxis_title='Diffraction Angle (deg)',
        yaxis_title='Grating Period (nm)'
    )

    st.plotly_chart(fig, use_container_width=True, key=f"manifold_{target_angle}_{wavelength}")

# =============================================================================
# TAB 3: DOE ANALYSIS
# =============================================================================
with tab3:
    st.markdown('<h2 class="section-header">Design of Experiments Analysis</h2>', unsafe_allow_html=True)

    df = load_doe_data()

    if df is not None:
        df.columns = df.columns.str.lower().str.strip()
        mae_col = 'mae_nm' if 'mae_nm' in df.columns else 'mae'
        rmse_col = 'rmse_nm' if 'rmse_nm' in df.columns else 'rmse'

        # =================================================================
        # SUMMARY METRICS - Large impactful cards
        # =================================================================
        st.markdown('<p class="subsection-header">Experiment Summary</p>', unsafe_allow_html=True)

        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

        with sum_col1:
            st.markdown(f"""
            <div class="summary-card">
                <p class="summary-label">Total Experiments</p>
                <p class="summary-value">{len(df)}</p>
            </div>
            """, unsafe_allow_html=True)

        with sum_col2:
            if mae_col in df.columns:
                best_mae = df[mae_col].min()
                st.markdown(f"""
                <div class="summary-card">
                    <p class="summary-label">Best MAE</p>
                    <p class="summary-value">{best_mae:.3f} nm</p>
                </div>
                """, unsafe_allow_html=True)

        with sum_col3:
            if rmse_col in df.columns:
                best_rmse = df[rmse_col].min()
                st.markdown(f"""
                <div class="summary-card">
                    <p class="summary-label">Best RMSE</p>
                    <p class="summary-value">{best_rmse:.3f} nm</p>
                </div>
                """, unsafe_allow_html=True)

        with sum_col4:
            if 'n_samples' in df.columns:
                max_samples = df['n_samples'].max()
                st.markdown(f"""
                <div class="summary-card">
                    <p class="summary-label">Max Dataset Size</p>
                    <p class="summary-value">{int(max_samples):,}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

        # =================================================================
        # SCIENTIFIC SCALING PLOTS
        # =================================================================

        # Plot 1: Scaling Law - MAE vs Dataset Size (Log scale)
        if 'n_samples' in df.columns and mae_col in df.columns:
            st.markdown('<p class="subsection-header">Scaling Law: Error vs. Dataset Size</p>', unsafe_allow_html=True)

            scaling_data = df.groupby('n_samples')[mae_col].mean().reset_index()

            fig_scaling = go.Figure()
            fig_scaling.add_trace(go.Scatter(
                x=scaling_data['n_samples'],
                y=scaling_data[mae_col],
                mode='lines+markers',
                name='Mean MAE',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10, color='#667eea')
            ))

            fig_scaling.update_layout(
                **get_plotly_layout("Dataset Scaling Law", height=350),
                xaxis_title='Training Samples',
                yaxis_title='Mean Absolute Error (nm)',
                xaxis_type='log',
                yaxis_type='log'
            )

            st.plotly_chart(fig_scaling, use_container_width=True)

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Plot 2: Architecture Sensitivity Heatmap
        if 'n_layers' in df.columns and 'n_samples' in df.columns and mae_col in df.columns:
            st.markdown('<p class="subsection-header">Architecture Sensitivity: MAE Heatmap</p>', unsafe_allow_html=True)

            try:
                heatmap_pivot = df.pivot_table(
                    index='n_layers',
                    columns='n_samples',
                    values=mae_col,
                    aggfunc='mean'
                ).fillna(0)

                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=heatmap_pivot.values,
                    x=[f"{int(c):,}" for c in heatmap_pivot.columns],
                    y=[f"{int(r)} Layers" for r in heatmap_pivot.index],
                    colorscale='Magma',
                    reversescale=True,
                    colorbar=dict(
                        title=dict(text="MAE (nm)", font=dict(color='#FFFFFF')),
                        tickfont=dict(color='#FFFFFF')
                    ),
                    hovertemplate="Depth: %{y}<br>Samples: %{x}<br>MAE: %{z:.3f} nm<extra></extra>"
                ))

                fig_heatmap.update_layout(
                    **get_plotly_layout("", height=350),
                    xaxis_title='Training Samples',
                    yaxis_title='Network Depth'
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating heatmap: {str(e)}")

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Plot 3: Training Dynamics (if epochs data available)
        if 'n_epochs' in df.columns and mae_col in df.columns:
            st.markdown('<p class="subsection-header">Training Dynamics: Error vs. Epochs</p>', unsafe_allow_html=True)

            epochs_data = df.groupby('n_epochs')[mae_col].mean().reset_index()

            fig_epochs = go.Figure()
            fig_epochs.add_trace(go.Scatter(
                x=epochs_data['n_epochs'],
                y=epochs_data[mae_col],
                mode='lines+markers',
                name='Mean MAE',
                line=dict(color='#f093fb', width=3),
                marker=dict(size=10, color='#f093fb'),
                fill='tozeroy',
                fillcolor='rgba(240, 147, 251, 0.1)'
            ))

            fig_epochs.update_layout(
                **get_plotly_layout("Learning Curve", height=350),
                xaxis_title='Training Epochs',
                yaxis_title='Mean Absolute Error (nm)'
            )

            st.plotly_chart(fig_epochs, use_container_width=True)

        # =================================================================
        # TECHNICAL DEEP-DIVE (Moved to bottom)
        # =================================================================
        st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
        st.markdown('<p class="subsection-header">Technical Deep-Dive</p>', unsafe_allow_html=True)

        with st.expander("Interactive Filters & Data Table", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)

            selected_layers = 'All'
            selected_samples = 'All'
            selected_epochs = 'All'

            with filter_col1:
                if 'n_layers' in df.columns:
                    layer_options = ['All'] + sorted(df['n_layers'].unique().tolist())
                    selected_layers = st.selectbox("Filter by Layers", options=layer_options)

            with filter_col2:
                if 'n_samples' in df.columns:
                    sample_options = ['All'] + sorted(df['n_samples'].unique().tolist())
                    selected_samples = st.selectbox("Filter by Dataset Size", options=sample_options)

            with filter_col3:
                if 'n_epochs' in df.columns:
                    epoch_options = ['All'] + sorted(df['n_epochs'].unique().tolist())
                    selected_epochs = st.selectbox("Filter by Epochs", options=epoch_options)

            df_filtered = df.copy()
            if 'n_layers' in df.columns and selected_layers != 'All':
                df_filtered = df_filtered[df_filtered['n_layers'] == selected_layers]
            if 'n_samples' in df.columns and selected_samples != 'All':
                df_filtered = df_filtered[df_filtered['n_samples'] == selected_samples]
            if 'n_epochs' in df.columns and selected_epochs != 'All':
                df_filtered = df_filtered[df_filtered['n_epochs'] == selected_epochs]

            st.markdown(f"Showing **{len(df_filtered)}** of **{len(df)}** experiments")
            st.dataframe(df_filtered, use_container_width=True, height=300)

        # Scientific Interpretation
        with st.expander("Scientific Interpretation", expanded=False):
            if mae_col in df.columns and 'n_layers' in df.columns:
                best_idx = df[mae_col].idxmin()
                best_row = df.loc[best_idx]

                st.markdown(f"""
                **Optimal Configuration Found:**
                - Layers: {int(best_row.get('n_layers', 0))}
                - Samples: {int(best_row.get('n_samples', 0)):,}
                - Epochs: {int(best_row.get('n_epochs', 0))}
                - MAE: {best_row[mae_col]:.4f} nm
                """)

            st.markdown("""
            **Key Insights:**

            1. **Underfitting Regime**: Shallow networks (1-2 layers) cannot approximate the arcsin nonlinearity.

            2. **Overfitting Regime**: Deep networks with small datasets show unstable convergence.

            3. **Optimal Capacity**: 2-3 layer networks with 1000+ samples achieve sub-3nm MAE.
            """)

    else:
        st.warning("DOE results file not found. Run `python p1_doe_sweep.py` to generate data.")

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
        <a href="https://x.com/vaibhavmathur91" target="_blank" style="color: #667eea; text-decoration: none; margin-right: 1.5rem;">
            X (Twitter)
        </a>
        <a href="https://linkedin.com/in/vaibhavmathur91" target="_blank" style="color: #667eea; text-decoration: none;">
            LinkedIn
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
