# -*- coding: utf-8 -*-
"""
Project 3: Virtual Wind Tunnel - Neural Surrogate for Airfoil Aerodynamics
Real-time pressure distribution prediction using 1D CNN
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
    page_title="P3: Virtual Wind Tunnel",
    page_icon="P3",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - SIMANOVA DARK MODE (matching P1/P2)
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
    [data-testid="stMetricDelta"] {
        color: #2ecc71 !important;
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

    .project-desc strong {
        color: #FFFFFF !important;
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
        margin: 1rem 0 0.5rem 0;
    }

    /* Method cards */
    .method-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 1px solid #2d2d44;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s ease;
    }

    .method-card:hover {
        border-color: #667eea;
    }

    .method-number {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }

    .method-title {
        color: #FFFFFF !important;
        font-weight: 600;
        font-size: 1rem;
    }

    .method-desc {
        color: #a0aec0 !important;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        line-height: 1.5;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 1px solid #2d2d44;
        border-radius: 12px;
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
        margin: 0.25rem 0;
    }

    .metric-card-delta {
        font-size: 0.75rem;
        font-weight: 600;
    }

    .delta-good {
        color: #2ecc71 !important;
    }

    /* Badge container */
    .badge-container {
        display: flex;
        justify-content: center;
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

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #1a1a2e !important;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Physics box */
    .physics-box {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Author footer */
    .author-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #2d2d44;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_plotly_layout(title="", height=400):
    """Standard Plotly layout for dark theme."""
    return dict(
        title=dict(text=title, font=dict(size=16, color='#FFFFFF'), x=0.5) if title else None,
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(family='Inter', color='#FFFFFF'),
        xaxis=dict(
            gridcolor='#2d2d44',
            zerolinecolor='#4a5568',
            tickfont=dict(color='#a0aec0'),
            title_font=dict(color='#FFFFFF')
        ),
        yaxis=dict(
            gridcolor='#2d2d44',
            zerolinecolor='#4a5568',
            tickfont=dict(color='#a0aec0'),
            title_font=dict(color='#FFFFFF')
        ),
        height=height,
        margin=dict(l=60, r=40, t=60, b=60),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(14, 17, 23, 0.8)',
            bordercolor='#2d2d44',
            font=dict(color='#FFFFFF')
        )
    )


# =============================================================================
# NACA GEOMETRY GENERATOR
# =============================================================================
def naca_4digit(m: float, p: float, t: float, n_points: int = 50):
    """Generate NACA 4-digit airfoil coordinates."""
    beta = np.linspace(0, np.pi, n_points)
    xc = 0.5 * (1 - np.cos(beta))

    yt = 5 * t * (
        0.2969 * np.sqrt(xc + 1e-10)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1015 * xc**4
    )

    if p < 0.01 or m < 0.001:
        yc = np.zeros_like(xc)
        dyc_dx = np.zeros_like(xc)
    else:
        yc = np.where(
            xc < p,
            m / p**2 * (2 * p * xc - xc**2),
            m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * xc - xc**2)
        )
        dyc_dx = np.where(
            xc < p,
            2 * m / p**2 * (p - xc),
            2 * m / (1 - p)**2 * (p - xc)
        )

    theta = np.arctan(dyc_dx)
    xu = xc - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = xc + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    x_full = np.concatenate([xu[::-1], xl[1:]])
    y_full = np.concatenate([yu[::-1], yl[1:]])

    return x_full, y_full


def resample_airfoil(x, y, n_target=100):
    """Resample airfoil to fixed number of points using arc-length."""
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    s_norm = s / s[-1]

    s_new = np.linspace(0, 1, n_target)
    x_new = np.interp(s_new, s_norm, x)
    y_new = np.interp(s_new, s_norm, y)
    return x_new, y_new


# =============================================================================
# AEROCNN MODEL
# =============================================================================
class AeroCNN(nn.Module):
    """1D CNN for Airfoil Pressure Prediction."""

    def __init__(self, kernel_size=5, num_filters=32, num_layers=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers

        in_channels = 2
        seq_len = 100

        encoder_layers = []
        current_channels = in_channels
        current_len = seq_len

        for i in range(num_layers):
            out_channels = num_filters * (2 ** i)
            padding = kernel_size // 2

            encoder_layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ])

            if i < num_layers - 1 and i % 2 == 0:
                encoder_layers.append(nn.MaxPool1d(2))
                current_len = current_len // 2

            current_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)
        self.flat_size = current_channels * current_len

        hidden_dim = 256
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 100),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


# =============================================================================
# MODEL LOADING
# =============================================================================
try:
    SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
except:
    SCRIPT_DIR = Path.cwd()


@st.cache_resource
def load_model():
    """Load the trained AeroCNN model."""
    paths = [
        SCRIPT_DIR / "models" / "best_aero_model.pth",
        Path("models/best_aero_model.pth"),
        Path("../models/best_aero_model.pth"),
    ]

    for model_path in paths:
        if model_path.exists():
            try:
                model = AeroCNN(kernel_size=3, num_filters=16, num_layers=2)
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                return model, checkpoint, None
            except Exception as e:
                return None, None, str(e)

    return None, None, "Model file not found"


@st.cache_data
def load_doe_results():
    """Load DOE results for display."""
    paths = [
        SCRIPT_DIR / "models" / "p3_doe_results.json",
        Path("models/p3_doe_results.json"),
    ]

    for doe_path in paths:
        if doe_path.exists():
            try:
                with open(doe_path, 'r') as f:
                    return json.load(f)
            except:
                pass
    return None


# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict_cp(model, x_coords, y_coords):
    """Run inference to predict Cp distribution."""
    X = np.stack([x_coords, y_coords], axis=0).astype(np.float32)
    X_tensor = torch.FloatTensor(X).unsqueeze(0)

    with torch.no_grad():
        Cp_pred = model(X_tensor).numpy().squeeze()

    return Cp_pred


# =============================================================================
# PLOTTING FUNCTIONS (Fixed for Plotly compatibility)
# =============================================================================
def create_airfoil_plot(x, y, naca_code):
    """Create interactive airfoil shape plot."""
    fig = go.Figure()

    # Fill
    fig.add_trace(go.Scatter(
        x=x, y=y,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=3),
        name='Airfoil',
        hovertemplate='x/c: %{x:.3f}<br>y/c: %{y:.4f}<extra></extra>'
    ))

    # Chord line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0],
        mode='lines',
        line=dict(color='#4a5568', width=1, dash='dash'),
        name='Chord Line',
        hoverinfo='skip'
    ))

    layout = get_plotly_layout(f'NACA {naca_code} Airfoil Shape', height=350)
    layout['xaxis']['title'] = 'x/c (Chord Position)'
    layout['xaxis']['range'] = [-0.05, 1.05]
    layout['yaxis']['title'] = 'y/c (Thickness)'
    layout['yaxis']['range'] = [-0.2, 0.2]
    layout['yaxis']['scaleanchor'] = 'x'
    layout['yaxis']['scaleratio'] = 1
    layout['showlegend'] = False

    fig.update_layout(**layout)

    return fig


def create_cp_plot(x, Cp, naca_code):
    """Create interactive Cp distribution plot."""
    fig = go.Figure()

    # Cp curve
    fig.add_trace(go.Scatter(
        x=x, y=Cp,
        mode='lines',
        line=dict(color='#2ecc71', width=3),
        name='Cp (AI Prediction)',
        fill='tozeroy',
        fillcolor='rgba(46, 204, 113, 0.15)',
        hovertemplate='x/c: %{x:.3f}<br>Cp: %{y:.3f}<extra></extra>'
    ))

    # Zero line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0],
        mode='lines',
        line=dict(color='#4a5568', width=1, dash='dash'),
        name='Cp = 0',
        hoverinfo='skip'
    ))

    # Find and mark suction peak
    min_idx = np.argmin(Cp)
    fig.add_trace(go.Scatter(
        x=[x[min_idx]], y=[Cp[min_idx]],
        mode='markers',
        marker=dict(size=12, color='#e74c3c', symbol='diamond'),
        name=f'Suction Peak: {Cp[min_idx]:.2f}',
        hovertemplate='Suction Peak<br>x/c: %{x:.3f}<br>Cp: %{y:.3f}<extra></extra>'
    ))

    layout = get_plotly_layout(f'Pressure Distribution - NACA {naca_code}', height=400)
    layout['xaxis']['title'] = 'x/c (Chord Position)'
    layout['xaxis']['range'] = [-0.05, 1.05]
    layout['yaxis']['title'] = 'Pressure Coefficient (Cp)'
    layout['yaxis']['autorange'] = 'reversed'  # Invert Y-axis for aerodynamics

    fig.update_layout(**layout)

    return fig


def create_doe_chart(doe_results):
    """Create DOE performance comparison chart."""
    if not doe_results:
        return None

    experiments = doe_results.get('experiments', [])
    if not experiments:
        return None

    # Sort by validation loss
    sorted_exp = sorted(experiments, key=lambda x: x['best_val_loss'])

    labels = [f"K{e['kernel_size']}_F{e['num_filters']}_L{e['num_layers']}" for e in sorted_exp]
    val_losses = [e['best_val_loss'] for e in sorted_exp]
    params = [e['params'] / 1000 for e in sorted_exp]

    colors = ['#2ecc71' if i == 0 else '#3498db' if v < 0.085 else '#f39c12' if v < 0.09 else '#e74c3c'
              for i, v in enumerate(val_losses)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Validation Loss by Configuration', 'Accuracy vs Model Complexity'),
        horizontal_spacing=0.12
    )

    # Bar chart
    fig.add_trace(
        go.Bar(
            x=labels, y=val_losses,
            marker_color=colors,
            text=[f'{v:.4f}' for v in val_losses],
            textposition='outside',
            textfont=dict(color='#FFFFFF', size=10),
            hovertemplate='Config: %{x}<br>Val Loss: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Scatter plot
    fig.add_trace(
        go.Scatter(
            x=params, y=val_losses,
            mode='markers+text',
            marker=dict(size=15, color=colors, line=dict(width=2, color='#FFFFFF')),
            text=labels,
            textposition='top center',
            textfont=dict(size=9, color='#a0aec0'),
            hovertemplate='Config: %{text}<br>Params: %{x:.0f}K<br>Val Loss: %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.update_xaxes(title_text='Configuration', row=1, col=1, tickangle=45,
                     gridcolor='#2d2d44', tickfont=dict(color='#a0aec0'))
    fig.update_xaxes(title_text='Parameters (K)', row=1, col=2,
                     gridcolor='#2d2d44', tickfont=dict(color='#a0aec0'))
    fig.update_yaxes(title_text='Validation Loss (MSE)', row=1, col=1,
                     gridcolor='#2d2d44', tickfont=dict(color='#a0aec0'))
    fig.update_yaxes(title_text='Validation Loss (MSE)', row=1, col=2,
                     gridcolor='#2d2d44', tickfont=dict(color='#a0aec0'))

    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#FFFFFF'),
        showlegend=False,
        height=400,
        margin=dict(l=60, r=40, t=80, b=80)
    )

    fig.update_annotations(font=dict(size=14, color='#667eea'))

    return fig


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Load model and DOE results
    model, checkpoint, error = load_model()
    doe_results = load_doe_results()

    # Header
    st.markdown('<h1 class="page-title">P3: Virtual Wind Tunnel</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Neural Surrogate for Real-Time Airfoil Aerodynamics</p>', unsafe_allow_html=True)

    # Performance badges
    st.markdown("""
    <div class="badge-container">
        <span class="perf-badge">Inference: &lt;5ms</span>
        <span class="perf-badge">2000 Training Airfoils</span>
        <span class="perf-badge">Model: 1D CNN (AeroCNN)</span>
    </div>
    """, unsafe_allow_html=True)

    # Project description
    st.markdown("""
    <p class="project-desc">
        This engine demonstrates <strong>neural surrogate modeling</strong> for aerodynamic analysis.
        Given a NACA 4-digit airfoil geometry, the AeroCNN predicts the <strong>pressure coefficient (Cp)</strong>
        distribution in milliseconds - bypassing expensive CFD simulations. A <strong>Design of Experiments (DOE)</strong>
        sweep identifies optimal CNN hyperparameters for minimal prediction error.
    </p>
    """, unsafe_allow_html=True)

    # Governing Equation
    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
    _, eq_col, _ = st.columns([1, 2, 1])
    with eq_col:
        st.latex(r"C_p = 1 - \left(\frac{V}{V_\infty}\right)^2 = \frac{P - P_\infty}{\frac{1}{2}\rho V_\infty^2}")
        st.markdown('<p style="color: #a0aec0; font-size: 0.8rem; text-align: center;">Cp = pressure coefficient, V = local velocity, P = local pressure</p>', unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # Check model status
    if model is None:
        st.error(f"Model loading failed: {error}")
        st.info("Please ensure `models/best_aero_model.pth` exists. Run `p3_doe_train.py` first.")
        return

    # =============================================================================
    # TABS (matching P1/P2 structure)
    # =============================================================================
    tab1, tab2, tab3 = st.tabs(["Methodology", "Design Tool", "DOE Analysis"])

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
                    Build a <strong>Neural Surrogate</strong> for airfoil pressure prediction.
                    A 1D CNN (AeroCNN) maps geometry to Cp distribution with <strong>sub-5ms latency</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="method-card">
                <span class="method-number">1</span>
                <span class="method-title">Parametric Design Space</span>
                <p class="method-desc">
                    <strong>Airfoil Family:</strong> NACA 4-Digit Series<br>
                    <strong>Camber:</strong> 0-9% of chord<br>
                    <strong>Thickness:</strong> 6-24% of chord<br>
                    <strong>Resolution:</strong> 100 surface points
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="method-card">
                <span class="method-number">2</span>
                <span class="method-title">Physics Engine</span>
                <p class="method-desc">
                    <strong>Vortex Panel Method</strong> solver computes ground-truth Cp.
                    Potential flow with <strong>Kutta condition</strong> at trailing edge.
                    2D inviscid, incompressible flow assumption.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="method-card">
                <span class="method-number">3</span>
                <span class="method-title">Dataset Generation</span>
                <p class="method-desc">
                    <strong>2,000 random NACA airfoils</strong> generated with Latin Hypercube sampling.
                    Each sample: (x, y) coordinates + Cp distribution.
                    Split: 80% train / 10% val / 10% test.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="method-card">
                <span class="method-number">4</span>
                <span class="method-title">Neural Architecture</span>
                <p class="method-desc">
                    <strong>AeroCNN:</strong> 1D Convolutional Encoder + MLP Decoder<br>
                    Input: (2, 100) - x,y coordinates<br>
                    Output: (100,) - Cp at each point<br>
                    Activation: ReLU + BatchNorm + Dropout
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="method-card">
                <span class="method-number">5</span>
                <span class="method-title">DOE Optimization</span>
                <p class="method-desc">
                    <strong>Grid Search:</strong> kernel_size [3,7], filters [16,32], layers [2,3]<br>
                    <strong>8 experiments</strong> x 15 epochs each<br>
                    Best model retrained for 30 epochs.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # =============================================================================
    # TAB 2: DESIGN TOOL (Interactive Inference)
    # =============================================================================
    with tab2:
        st.markdown('<h2 class="section-header">Interactive Airfoil Designer</h2>', unsafe_allow_html=True)

        # Sidebar inputs
        st.sidebar.markdown("### Navigation")
        if st.sidebar.button("<- Back to Home", use_container_width=True):
            st.switch_page("Home.py")

        st.sidebar.markdown("---")
        st.sidebar.markdown("### NACA Parameters")

        camber_pct = st.sidebar.slider(
            "Camber (%)",
            min_value=0, max_value=9, value=2, step=1,
            help="Maximum camber as % of chord. 0 = symmetric airfoil."
        )
        m = camber_pct / 100.0

        position_pct = st.sidebar.slider(
            "Camber Position (x10%)",
            min_value=1, max_value=9, value=4, step=1,
            help="Location of max camber in tenths of chord."
        )
        p = position_pct / 10.0

        thickness_pct = st.sidebar.slider(
            "Thickness (%)",
            min_value=6, max_value=24, value=12, step=1,
            help="Maximum thickness as % of chord."
        )
        t = thickness_pct / 100.0

        naca_code = f"{camber_pct}{position_pct}{thickness_pct:02d}"

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"""
        <div style="background: rgba(102, 126, 234, 0.1); border: 1px solid rgba(102, 126, 234, 0.3); border-radius: 8px; padding: 1rem; text-align: center;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #667eea;">NACA {naca_code}</div>
            <div style="font-size: 0.8rem; color: #a0aec0; margin-top: 0.5rem;">
                {camber_pct}% camber at {position_pct*10}% chord<br>
                {thickness_pct}% max thickness
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Generate and predict
        x_raw, y_raw = naca_4digit(m, p, t, n_points=60)
        x_100, y_100 = resample_airfoil(x_raw, y_raw, n_target=100)
        Cp_pred = predict_cp(model, x_100, y_100)

        # Metrics
        min_cp = float(np.min(Cp_pred))
        max_cp = float(np.max(Cp_pred))
        cp_range = max_cp - min_cp

        st.markdown('<p class="subsection-header">Aerodynamic Metrics</p>', unsafe_allow_html=True)

        met_col1, met_col2, met_col3, met_col4 = st.columns(4)

        with met_col1:
            st.markdown(f"""
            <div class="metric-card" style="border: 2px solid #2ecc71;">
                <p class="metric-card-label">Max Suction</p>
                <p class="metric-card-value" style="color: #2ecc71 !important;">{min_cp:.2f}</p>
                <span class="metric-card-delta delta-good">Min Cp</span>
            </div>
            """, unsafe_allow_html=True)

        with met_col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Max Pressure</p>
                <p class="metric-card-value">{max_cp:.2f}</p>
                <span class="metric-card-delta delta-good">Max Cp</span>
            </div>
            """, unsafe_allow_html=True)

        with met_col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Cp Range</p>
                <p class="metric-card-value">{cp_range:.2f}</p>
                <span class="metric-card-delta delta-good">Delta</span>
            </div>
            """, unsafe_allow_html=True)

        with met_col4:
            lift_indicator = abs(min_cp) * (1 + m * 10)
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-card-label">Lift Indicator</p>
                <p class="metric-card-value">{lift_indicator:.2f}</p>
                <span class="metric-card-delta delta-good">Relative</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Plots
        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            airfoil_fig = create_airfoil_plot(x_100, y_100, naca_code)
            st.plotly_chart(airfoil_fig, use_container_width=True, key=f"airfoil_{naca_code}")

        with plot_col2:
            cp_fig = create_cp_plot(x_100, Cp_pred, naca_code)
            st.plotly_chart(cp_fig, use_container_width=True, key=f"cp_{naca_code}")

        # Physics explanation
        st.markdown('<p class="subsection-header">Understanding the Results</p>', unsafe_allow_html=True)

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("""
            <div class="physics-box">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">Pressure Coefficient (Cp)</h4>
                <p style="color: #c0c8d0; font-size: 0.9rem; line-height: 1.6;">
                    <strong>Negative Cp</strong>: Suction (low pressure) - creates lift<br>
                    <strong>Positive Cp</strong>: Compression (high pressure)<br>
                    <strong>Cp = 1</strong>: Stagnation point (flow stops)
                </p>
            </div>
            """, unsafe_allow_html=True)

        with exp_col2:
            st.markdown("""
            <div class="physics-box">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">Suction Peak</h4>
                <p style="color: #c0c8d0; font-size: 0.9rem; line-height: 1.6;">
                    The <strong>suction peak</strong> (minimum Cp) indicates where flow accelerates most.<br>
                    Stronger peak = more lift potential.<br>
                    Sharp peaks may indicate flow separation risk.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # =============================================================================
    # TAB 3: DOE ANALYSIS
    # =============================================================================
    with tab3:
        st.markdown('<h2 class="section-header">Neural Architecture Search: DOE Results</h2>', unsafe_allow_html=True)

        if doe_results:
            best = doe_results.get('best_config', {})
            experiments = doe_results.get('experiments', [])

            # Optimal Configuration
            st.markdown('<p class="subsection-header">Optimal Configuration (Lowest Validation Loss)</p>', unsafe_allow_html=True)

            opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)

            with opt_col1:
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid #2ecc71;">
                    <p class="metric-card-label">Kernel Size</p>
                    <p class="metric-card-value">{best.get('kernel_size', 3)}</p>
                    <span class="metric-card-delta delta-good">Conv1D</span>
                </div>
                """, unsafe_allow_html=True)

            with opt_col2:
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid #2ecc71;">
                    <p class="metric-card-label">Num Filters</p>
                    <p class="metric-card-value">{best.get('num_filters', 16)}</p>
                    <span class="metric-card-delta delta-good">Channels</span>
                </div>
                """, unsafe_allow_html=True)

            with opt_col3:
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid #2ecc71;">
                    <p class="metric-card-label">Num Layers</p>
                    <p class="metric-card-value">{best.get('num_layers', 2)}</p>
                    <span class="metric-card-delta delta-good">Depth</span>
                </div>
                """, unsafe_allow_html=True)

            with opt_col4:
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid #2ecc71;">
                    <p class="metric-card-label">Val MSE</p>
                    <p class="metric-card-value">{best.get('val_loss', 0):.4f}</p>
                    <span class="metric-card-delta delta-good">Loss</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

            # DOE Chart
            st.markdown('<p class="subsection-header">DOE Performance Comparison</p>', unsafe_allow_html=True)

            doe_fig = create_doe_chart(doe_results)
            if doe_fig:
                st.plotly_chart(doe_fig, use_container_width=True, key="doe_chart")

            # Experiments Table
            st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
            st.markdown('<p class="subsection-header">All Experiments</p>', unsafe_allow_html=True)

            if experiments:
                import pandas as pd
                df = pd.DataFrame(experiments)
                df = df[['kernel_size', 'num_filters', 'num_layers', 'params', 'best_val_loss', 'train_time']]
                df.columns = ['Kernel', 'Filters', 'Layers', 'Parameters', 'Val Loss', 'Time (s)']
                df = df.sort_values('Val Loss')
                df['Parameters'] = df['Parameters'].apply(lambda x: f"{x:,}")
                df['Val Loss'] = df['Val Loss'].apply(lambda x: f"{x:.4f}")
                df['Time (s)'] = df['Time (s)'].apply(lambda x: f"{x:.1f}")
                st.dataframe(df, use_container_width=True, hide_index=True)

            # Training Info
            st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="physics-box">
                <h4 style="color: #667eea; margin-bottom: 0.5rem;">Training Details</h4>
                <p style="color: #c0c8d0; font-size: 0.9rem; line-height: 1.6;">
                    <strong>Hardware:</strong> CPU (local training)<br>
                    <strong>Dataset:</strong> 2000 airfoils (1600 train / 200 val / 200 test)<br>
                    <strong>DOE:</strong> 8 experiments x 15 epochs = 120 total training runs<br>
                    <strong>Final Model:</strong> Best config retrained for 30 epochs
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("DOE results not found. Run `p3_doe_train.py` to generate model performance analysis.")

    # =============================================================================
    # FOOTER
    # =============================================================================
    st.markdown("""
    <div class="author-footer">
        <p style="color: #a0aec0; font-size: 0.9rem; margin-bottom: 0.5rem;">
            Built by <strong style="color: #FFFFFF;">Vaibhav Mathur</strong>
        </p>
        <p style="color: #4a5568; font-size: 0.85rem;">
            <a href="https://x.com/vaibhavmathur91" target="_blank">X (Twitter)</a>
            <a href="https://linkedin.com/in/vaibhavmathur91" target="_blank">LinkedIn</a>
            <a href="https://github.com/vaibhavmathur171-git/simanova-models" target="_blank">GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
