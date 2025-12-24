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
# CUSTOM CSS - SIMANOVA DARK MODE
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

    .section-header {
        color: #667eea !important;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2d2d44;
    }

    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 1px solid #2d2d44;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea !important;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .suction-metric {
        background: linear-gradient(145deg, #1a2e1a 0%, #162016 100%);
        border: 1px solid #2ecc71;
    }

    .suction-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2ecc71 !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d12 0%, #0a0a0f 100%);
        border-right: 1px solid #2d2d44;
    }

    [data-testid="stSidebar"] .stSlider label {
        color: #FFFFFF !important;
        font-weight: 500;
    }

    /* Slider track */
    [data-testid="stSidebar"] .stSlider > div > div {
        background-color: #2d2d44 !important;
    }

    /* Info boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .physics-note {
        background: rgba(46, 204, 113, 0.1);
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.85rem;
        color: #a0aec0 !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


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
@st.cache_resource
def load_model():
    """Load the trained AeroCNN model."""
    script_dir = Path(__file__).parent.parent
    model_path = script_dir / "models" / "best_aero_model.pth"

    if not model_path.exists():
        return None, "Model not found"

    try:
        model = AeroCNN(kernel_size=3, num_filters=16, num_layers=2)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_doe_results():
    """Load DOE results for display."""
    script_dir = Path(__file__).parent.parent
    doe_path = script_dir / "models" / "p3_doe_results.json"

    if not doe_path.exists():
        return None

    try:
        with open(doe_path, 'r') as f:
            return json.load(f)
    except:
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
# PLOTTING FUNCTIONS
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

    fig.update_layout(
        title=dict(
            text=f'NACA {naca_code} Airfoil Shape',
            font=dict(size=18, color='#FFFFFF'),
            x=0.5
        ),
        xaxis=dict(
            title='x/c (Chord Position)',
            range=[-0.05, 1.05],
            gridcolor='#2d2d44',
            zerolinecolor='#4a5568',
            tickfont=dict(color='#a0aec0'),
            titlefont=dict(color='#FFFFFF')
        ),
        yaxis=dict(
            title='y/c (Thickness)',
            range=[-0.2, 0.2],
            scaleanchor='x',
            scaleratio=1,
            gridcolor='#2d2d44',
            zerolinecolor='#4a5568',
            tickfont=dict(color='#a0aec0'),
            titlefont=dict(color='#FFFFFF')
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#FFFFFF'),
        showlegend=False,
        height=350,
        margin=dict(l=60, r=40, t=60, b=60)
    )

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

    fig.update_layout(
        title=dict(
            text=f'Pressure Distribution - NACA {naca_code}',
            font=dict(size=18, color='#FFFFFF'),
            x=0.5
        ),
        xaxis=dict(
            title='x/c (Chord Position)',
            range=[-0.05, 1.05],
            gridcolor='#2d2d44',
            zerolinecolor='#4a5568',
            tickfont=dict(color='#a0aec0'),
            titlefont=dict(color='#FFFFFF')
        ),
        yaxis=dict(
            title='Pressure Coefficient (Cp)',
            autorange='reversed',  # Invert Y-axis for aerodynamics convention
            gridcolor='#2d2d44',
            zerolinecolor='#4a5568',
            tickfont=dict(color='#a0aec0'),
            titlefont=dict(color='#FFFFFF')
        ),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#FFFFFF'),
        showlegend=True,
        legend=dict(
            x=0.98, y=0.02,
            xanchor='right', yanchor='bottom',
            bgcolor='rgba(14, 17, 23, 0.8)',
            bordercolor='#2d2d44',
            font=dict(color='#FFFFFF')
        ),
        height=400,
        margin=dict(l=60, r=40, t=60, b=60)
    )

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
    params = [e['params'] / 1000 for e in sorted_exp]  # Convert to K

    # Color scale: green (best) to red (worst)
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

    # Update subplot titles
    fig.update_annotations(font=dict(size=14, color='#667eea'))

    return fig


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Load model
    model, error = load_model()
    doe_results = load_doe_results()

    # Header
    st.markdown('<h1 class="page-title">P3: Virtual Wind Tunnel</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Neural Surrogate for Airfoil Aerodynamics</p>', unsafe_allow_html=True)

    st.markdown("""
    <p class="project-desc">
        Real-time pressure distribution prediction using a <strong>1D Convolutional Neural Network</strong>.
        Adjust the NACA 4-digit parameters to explore how airfoil geometry affects the
        <strong>pressure coefficient (Cp)</strong> distribution. The suction peak indicates lift generation.
    </p>
    """, unsafe_allow_html=True)

    # Check model status
    if model is None:
        st.error(f"Model loading failed: {error}")
        st.info("Please ensure `models/best_aero_model.pth` exists. Run `p3_doe_train.py` first.")
        return

    # -------------------------------------------------------------------------
    # SIDEBAR - AIRFOIL PARAMETERS
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### Airfoil Design Parameters")
        st.markdown('<p class="physics-note">NACA 4-Digit Series</p>', unsafe_allow_html=True)

        st.markdown("---")

        # Camber (first digit)
        camber_pct = st.slider(
            "Camber (Curvature)",
            min_value=0, max_value=9, value=2, step=1,
            help="Maximum camber as % of chord. 0 = symmetric airfoil."
        )
        m = camber_pct / 100.0

        # Position (second digit)
        position_pct = st.slider(
            "Camber Position",
            min_value=1, max_value=9, value=4, step=1,
            help="Location of max camber in tenths of chord (10-90%)."
        )
        p = position_pct / 10.0

        # Thickness (last two digits)
        thickness_pct = st.slider(
            "Thickness",
            min_value=6, max_value=24, value=12, step=1,
            help="Maximum thickness as % of chord."
        )
        t = thickness_pct / 100.0

        # Generate NACA code
        naca_code = f"{camber_pct}{position_pct}{thickness_pct:02d}"

        st.markdown("---")

        st.markdown(f"""
        <div class="info-box">
            <div style="font-size: 1.5rem; font-weight: 700; color: #667eea; text-align: center;">
                NACA {naca_code}
            </div>
            <div style="font-size: 0.8rem; color: #a0aec0; text-align: center; margin-top: 0.5rem;">
                {camber_pct}% camber at {position_pct*10}% chord<br>
                {thickness_pct}% max thickness
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("""
        <div class="physics-note">
            <strong>Physics Guide:</strong><br>
            - More camber = more lift (cambered wings)<br>
            - Forward camber = earlier stall<br>
            - Thicker airfoils = more drag, stronger structure
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # GENERATE AIRFOIL AND PREDICT
    # -------------------------------------------------------------------------
    # Generate airfoil
    x_raw, y_raw = naca_4digit(m, p, t, n_points=60)
    x_100, y_100 = resample_airfoil(x_raw, y_raw, n_target=100)

    # Run AI prediction
    Cp_pred = predict_cp(model, x_100, y_100)

    # Calculate metrics
    min_cp = float(np.min(Cp_pred))
    max_cp = float(np.max(Cp_pred))
    cp_range = max_cp - min_cp

    # -------------------------------------------------------------------------
    # MAIN CONTENT - METRICS
    # -------------------------------------------------------------------------
    st.markdown('<p class="section-header">Real-Time Aerodynamic Analysis</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card suction-metric">
            <div class="suction-value">{min_cp:.2f}</div>
            <div class="metric-label">Max Suction (Min Cp)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{max_cp:.2f}</div>
            <div class="metric-label">Max Pressure (Max Cp)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cp_range:.2f}</div>
            <div class="metric-label">Cp Range</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        # Estimate relative lift (simplified)
        lift_indicator = abs(min_cp) * (1 + m * 10)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{lift_indicator:.2f}</div>
            <div class="metric-label">Lift Indicator</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        airfoil_fig = create_airfoil_plot(x_100, y_100, naca_code)
        st.plotly_chart(airfoil_fig, use_container_width=True)

    with col_right:
        cp_fig = create_cp_plot(x_100, Cp_pred, naca_code)
        st.plotly_chart(cp_fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # PHYSICS EXPLANATION
    # -------------------------------------------------------------------------
    st.markdown('<p class="section-header">Understanding the Results</p>', unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #667eea; margin-bottom: 0.5rem;">Pressure Coefficient (Cp)</h4>
            <p style="color: #c0c8d0; font-size: 0.9rem; line-height: 1.6;">
                <strong>Cp = (P - P_inf) / (0.5 * rho * V^2)</strong><br><br>
                - <strong>Negative Cp</strong>: Suction (low pressure) - creates lift<br>
                - <strong>Positive Cp</strong>: Compression (high pressure)<br>
                - <strong>Cp = 1</strong>: Stagnation point (flow stops)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with exp_col2:
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #667eea; margin-bottom: 0.5rem;">Suction Peak</h4>
            <p style="color: #c0c8d0; font-size: 0.9rem; line-height: 1.6;">
                The <strong>suction peak</strong> (minimum Cp) near the leading edge indicates
                where flow accelerates most rapidly around the airfoil.<br><br>
                - Stronger suction peak = more lift potential<br>
                - Peak location affects stall behavior<br>
                - Sharp peaks may indicate flow separation risk
            </p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # DOE PERFORMANCE CHART
    # -------------------------------------------------------------------------
    st.markdown('<p class="section-header">Model Accuracy Analysis (DOE Results)</p>', unsafe_allow_html=True)

    if doe_results:
        doe_fig = create_doe_chart(doe_results)
        if doe_fig:
            st.plotly_chart(doe_fig, use_container_width=True)

        # Best model info
        best = doe_results.get('best_config', {})
        st.markdown(f"""
        <div class="info-box" style="text-align: center;">
            <h4 style="color: #2ecc71; margin-bottom: 0.5rem;">Best Model Configuration</h4>
            <p style="color: #c0c8d0; font-size: 0.95rem;">
                <strong>Kernel Size:</strong> {best.get('kernel_size', 3)} |
                <strong>Filters:</strong> {best.get('num_filters', 16)} |
                <strong>Layers:</strong> {best.get('num_layers', 2)} |
                <strong>Test MSE:</strong> {best.get('test_loss', 0):.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("DOE results not found. Run `p3_doe_train.py` to generate model performance analysis.")

    # -------------------------------------------------------------------------
    # FOOTER
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #4a5568; font-size: 0.85rem; padding: 1rem 0;">
        <strong>P3: Virtual Wind Tunnel</strong> | Neural Surrogate for NACA Airfoil Aerodynamics<br>
        Model: AeroCNN (1D Conv) | Physics: Vortex Panel Method | Training: 2000 Airfoils
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
