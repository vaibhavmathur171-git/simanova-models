# -*- coding: utf-8 -*-
"""
Project 4: Thermal Management - PINN Dashboard
Interactive dashboard for Physics-Informed Neural Network thermal simulation.

Features:
- Tab 1: Thermal Simulator (Real-time PINN inference)
- Tab 2: AI Research Lab (DOE architecture comparison)

Key Innovation: Zero-Data Training - Model trained using ONLY the Heat Equation!
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- 1. Page Config ---
st.set_page_config(
    page_title="P4: Thermal PINN",
    page_icon="🌡️",
    layout="wide"
)

# =============================================================================
# CONSTANTS
# =============================================================================
# Domain bounds (matching training)
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 0.2
T_MIN, T_MAX = 0.0, 1.0

# Grid resolution for visualization
NX = 100
NY = 20

# Physics
ALPHA = 0.01  # Thermal diffusivity

# Paths (handle case sensitivity)
MODEL_PATH = "models/p4_pinn_model.pth"
DOE_RESULTS_PATH = "Data/p4_doe_results.csv" if os.path.exists("Data/p4_doe_results.csv") else "data/p4_doe_results.csv"


# =============================================================================
# MODEL DEFINITION: ThermalPINN (4-layer Standard - Best from DOE)
# =============================================================================
class ThermalPINN(nn.Module):
    """
    Physics-Informed Neural Network for 2D Heat Equation.

    Architecture (Model B - Standard, 4 layers):
    - Input: (x, y, t) -> 3 neurons
    - Hidden: 4 layers x 64 neurons with Tanh activation
    - Output: u (temperature) -> 1 neuron

    Tanh activation is crucial for PINNs:
    - Infinitely differentiable (C∞)
    - Smooth gradients for autograd
    - Bounded output for stability
    """

    def __init__(self, num_layers: int = 4, neurons_per_layer: int = 64):
        super(ThermalPINN, self).__init__()

        self.num_layers = num_layers
        self.neurons = neurons_per_layer

        # Build network
        layers = []

        # Input layer: (x, y, t) -> neurons
        layers.append(nn.Linear(3, neurons_per_layer))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())

        # Output layer: neurons -> u (temperature)
        layers.append(nn.Linear(neurons_per_layer, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass: (x, y, t) -> u"""
        inputs = torch.cat([x, y, t], dim=1)
        return self.network(inputs)


# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================
def heat_source(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gaussian heat source (hotspot) at (0.2, 0.1).
    Q(x,y) = 5.0 * exp(-((x-0.2)^2 + (y-0.1)^2) / 0.01)
    """
    x0, y0 = 0.2, 0.1
    sigma2 = 0.01
    amplitude = 5.0
    r2 = (x - x0)**2 + (y - y0)**2
    return amplitude * np.exp(-r2 / sigma2)


# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_pinn_model():
    """Load the trained PINN model with error handling."""
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        model = ThermalPINN(num_layers=4, neurons_per_layer=64)
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_doe_results():
    """Load DOE results CSV with fallback paths."""
    paths_to_try = [
        "Data/p4_doe_results.csv",
        "data/p4_doe_results.csv",
        "./Data/p4_doe_results.csv",
        "./data/p4_doe_results.csv",
    ]
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                # Verify expected columns exist
                if 'Loss_Total' in df.columns:
                    return df
            except Exception:
                continue
    return None


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================
def predict_temperature_field(model: ThermalPINN, t_value: float) -> np.ndarray:
    """
    Predict temperature field u(x, y) at a given time t.

    Returns a 2D numpy array of shape (NY, NX).
    """
    # Create grid
    x_vals = np.linspace(X_MIN, X_MAX, NX)
    y_vals = np.linspace(Y_MIN, Y_MAX, NY)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Flatten for model input
    x_flat = X.flatten()
    y_flat = Y.flatten()
    t_flat = np.full_like(x_flat, t_value)

    # Convert to tensors
    x_tensor = torch.tensor(x_flat, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_flat, dtype=torch.float32).unsqueeze(1)
    t_tensor = torch.tensor(t_flat, dtype=torch.float32).unsqueeze(1)

    # Predict
    with torch.no_grad():
        u_pred = model(x_tensor, y_tensor, t_tensor)

    # Reshape to grid
    u_field = u_pred.numpy().reshape(NY, NX)

    return u_field, x_vals, y_vals


# =============================================================================
# MAIN APP
# =============================================================================
st.title("🌡️ P4: Thermal Management with PINN")

# --- EXECUTIVE SUMMARY ---
with st.container():
    st.markdown("### Project Executive Summary")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
            "**1. The Engineering Goal**\n\n"
            "Predict **Thermal Throttling** in AR glasses frames. "
            "Simulate heat diffusion from processor hotspots to prevent "
            "overheating and performance degradation."
        )

    with c2:
        st.info(
            "**2. The Physics**\n\n"
            "**2D Heat Equation**: `du/dt = α(d²u/dx² + d²u/dy²) + Q`\n\n"
            "Where α=0.01 (diffusivity) and Q is a Gaussian heat source "
            "at position (0.2, 0.1) representing the processor."
        )

    with c3:
        st.success(
            "**3. The AI Innovation**\n\n"
            "**Zero-Data Training!** This PINN was trained using ONLY "
            "the Heat Equation - no simulation data required. The network "
            "learns physics directly from the PDE residual."
        )

st.divider()

# --- TABS ---
tab1, tab2 = st.tabs(["🌡️ Thermal Simulator", "🔬 AI Research Lab (DOE)"])

# =============================================================================
# TAB 1: THERMAL SIMULATOR
# =============================================================================
with tab1:
    # Load model
    model = load_pinn_model()

    if model is None:
        st.warning(
            "**Model not found!** The file `models/p4_pinn_model.pth` is missing. "
            "Run `python p4_doe_train.py` to train the model first."
        )
    else:
        # Sidebar controls
        st.sidebar.header("Simulation Controls")
        st.sidebar.markdown("Adjust time to see heat evolution.")

        t_value = st.sidebar.slider(
            "Time (t)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.01,
            help="Simulation time from t=0 (initial) to t=1 (steady state)"
        )

        st.sidebar.divider()
        st.sidebar.markdown("**Domain Info**")
        st.sidebar.caption(f"x: [{X_MIN}, {X_MAX}] (frame length)")
        st.sidebar.caption(f"y: [{Y_MIN}, {Y_MAX}] (frame thickness)")
        st.sidebar.caption(f"Heat source at (0.2, 0.1)")

        st.sidebar.divider()
        st.sidebar.caption("P4: Thermal PINN | SimaNova")

        # Predict temperature field
        u_field, x_vals, y_vals = predict_temperature_field(model, t_value)

        # --- VISUALIZATION ---
        st.subheader(f"Temperature Distribution at t = {t_value:.2f}")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=u_field,
                x=x_vals,
                y=y_vals,
                colorscale='Inferno',
                colorbar=dict(title="Temperature (u)"),
                zmin=0,
                zmax=max(0.5, u_field.max())
            ))

            # Add heat source marker
            fig_heatmap.add_trace(go.Scatter(
                x=[0.2], y=[0.1],
                mode='markers+text',
                marker=dict(size=15, color='cyan', symbol='x'),
                text=['Heat Source'],
                textposition='top center',
                textfont=dict(color='cyan', size=12),
                showlegend=False
            ))

            fig_heatmap.update_layout(
                title=dict(text="2D Temperature Field u(x, y)", x=0.5),
                xaxis_title="X (Frame Length)",
                yaxis_title="Y (Frame Thickness)",
                yaxis=dict(scaleanchor="x", scaleratio=5),  # Aspect ratio
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )

            st.plotly_chart(fig_heatmap, use_container_width=True)

        with col2:
            # Temperature statistics
            st.markdown("**Field Statistics**")
            st.metric("Max Temperature", f"{u_field.max():.4f}")
            st.metric("Mean Temperature", f"{u_field.mean():.4f}")
            st.metric("Min Temperature", f"{u_field.min():.4f}")

            # Heat source info
            st.markdown("**Heat Source**")
            Q_max = heat_source(np.array([0.2]), np.array([0.1]))[0]
            st.caption(f"Location: (0.2, 0.1)")
            st.caption(f"Intensity: {Q_max:.1f} W")

        # Cross-section plot
        st.subheader("Temperature Cross-Section at y = 0.1 (Through Heat Source)")

        # Find closest y index to 0.1
        y_idx = np.argmin(np.abs(y_vals - 0.1))
        temp_cross = u_field[y_idx, :]

        fig_cross = go.Figure()

        fig_cross.add_trace(go.Scatter(
            x=x_vals,
            y=temp_cross,
            mode='lines',
            name='Temperature',
            line=dict(color='#FF6B6B', width=3)
        ))

        # Mark heat source location
        fig_cross.add_vline(x=0.2, line_dash="dash", line_color="cyan",
                           annotation_text="Heat Source", annotation_position="top")

        # Mark boundary condition
        fig_cross.add_vline(x=0.0, line_dash="dot", line_color="gray",
                           annotation_text="BC: u=0", annotation_position="bottom left")

        fig_cross.update_layout(
            title=dict(text=f"Temperature Profile at y = {y_vals[y_idx]:.2f}", x=0.5),
            xaxis_title="X Position",
            yaxis_title="Temperature (u)",
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig_cross, use_container_width=True)

        # Time evolution explanation
        with st.expander("Understanding the Simulation"):
            st.markdown("""
            ### How the PINN Simulation Works

            **Initial Condition (t=0):** The entire domain starts at u=0 (cold).

            **Heat Source:** A Gaussian heat injection at (x=0.2, y=0.1) continuously adds energy.

            **Boundary Condition:** The left edge (x=0) is held at u=0 (Dirichlet BC), acting as a heat sink.

            **Physics:** Heat diffuses from the source according to:
            ```
            du/dt = 0.01 * (d²u/dx² + d²u/dy²) + Q(x,y)
            ```

            **What to Observe:**
            - At **t=0**: Temperature is near zero everywhere
            - As **t increases**: Heat builds up near the source (0.2, 0.1)
            - The heat diffuses outward but is absorbed at the left boundary
            - **Steady state** is approached as t → 1
            """)


# =============================================================================
# TAB 2: AI RESEARCH LAB (DOE)
# =============================================================================
with tab2:
    st.subheader("Design of Experiments: PINN Architecture Search")

    doe_df = load_doe_results()

    if doe_df is not None:
        # --- EXPERIMENT OVERVIEW ---
        st.markdown("### 1. The Zero-Data Training Paradigm")

        st.info(
            "**Revolutionary Approach:** Unlike traditional ML, PINNs require NO training data!\n\n"
            "Instead of learning from simulation outputs, the network learns by minimizing "
            "the PDE residual directly. The loss function encodes physics:\n\n"
            "```\n"
            "Loss = MSE(PDE Residual) + MSE(Boundary Conditions) + MSE(Initial Conditions)\n"
            "```\n\n"
            "This means we can train a thermal simulator without ever running a single FEM simulation!"
        )

        st.divider()

        # --- EXPERIMENT DESIGN ---
        st.markdown("### 2. Architecture Comparison")

        col_exp1, col_exp2, col_exp3 = st.columns(3)

        with col_exp1:
            st.error(
                "**Model A: Shallow**\n\n"
                "Layers: **2**\n\n"
                "Neurons: 64\n\n"
                "Parameters: 4,481\n\n"
                "Hypothesis: May underfit"
            )

        with col_exp2:
            st.success(
                "**Model B: Standard** ✓\n\n"
                "Layers: **4**\n\n"
                "Neurons: 64\n\n"
                "Parameters: 12,801\n\n"
                "Hypothesis: Balanced"
            )

        with col_exp3:
            st.warning(
                "**Model C: Deep**\n\n"
                "Layers: **8**\n\n"
                "Neurons: 64\n\n"
                "Parameters: 29,441\n\n"
                "Hypothesis: May overfit"
            )

        st.divider()

        # --- LOSS CURVES ---
        st.markdown("### 3. Training Loss Curves")

        # Create subplot with PDE loss and Total loss
        fig_loss = make_subplots(rows=1, cols=2,
                                  subplot_titles=("Total Loss vs Epoch", "PDE Loss vs Epoch"))

        colors = {'Model_A_Shallow': '#EF553B', 'Model_B_Standard': '#00CC96', 'Model_C_Deep': '#636EFA'}

        for model_name in doe_df['Model_Name'].unique():
            model_data = doe_df[doe_df['Model_Name'] == model_name]
            color = colors.get(model_name, 'gray')

            # Total Loss
            fig_loss.add_trace(go.Scatter(
                x=model_data['Epoch'],
                y=model_data['Loss_Total'],
                mode='lines+markers',
                name=model_name,
                line=dict(color=color),
                legendgroup=model_name
            ), row=1, col=1)

            # PDE Loss
            fig_loss.add_trace(go.Scatter(
                x=model_data['Epoch'],
                y=model_data['Loss_PDE'],
                mode='lines+markers',
                name=model_name,
                line=dict(color=color),
                legendgroup=model_name,
                showlegend=False
            ), row=1, col=2)

        fig_loss.update_yaxes(type="log", title="Loss (log scale)", row=1, col=1)
        fig_loss.update_yaxes(type="log", title="Loss (log scale)", row=1, col=2)
        fig_loss.update_xaxes(title="Epoch", row=1, col=1)
        fig_loss.update_xaxes(title="Epoch", row=1, col=2)

        fig_loss.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))

        st.plotly_chart(fig_loss, use_container_width=True)

        # --- FINAL METRICS ---
        st.markdown("### 4. Final Performance Metrics")

        # Get final losses
        model_a_loss = doe_df[doe_df['Model_Name'] == 'Model_A_Shallow']['Loss_Total'].iloc[-1]
        model_b_loss = doe_df[doe_df['Model_Name'] == 'Model_B_Standard']['Loss_Total'].iloc[-1]
        model_c_loss = doe_df[doe_df['Model_Name'] == 'Model_C_Deep']['Loss_Total'].iloc[-1]

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.metric(
                "Model A (2 layers)",
                f"{model_a_loss:.6f}",
                delta=f"{(model_a_loss/model_b_loss):.0f}x worse",
                delta_color="inverse"
            )

        with col_r2:
            st.metric(
                "Model B (4 layers) ✓",
                f"{model_b_loss:.6f}",
                delta="BEST",
                delta_color="normal"
            )

        with col_r3:
            st.metric(
                "Model C (8 layers)",
                f"{model_c_loss:.6f}",
                delta=f"{(model_c_loss/model_b_loss):.1f}x worse",
                delta_color="inverse"
            )

        # Bar chart
        final_results = pd.DataFrame({
            'Model': ['Model A\n(2 layers)', 'Model B\n(4 layers)', 'Model C\n(8 layers)'],
            'Final Loss': [model_a_loss, model_b_loss, model_c_loss],
            'Color': ['Shallow', 'Standard', 'Deep']
        })

        fig_bar = px.bar(
            final_results,
            x='Model',
            y='Final Loss',
            color='Color',
            color_discrete_map={'Shallow': '#EF553B', 'Standard': '#00CC96', 'Deep': '#636EFA'},
            title="Final Total Loss Comparison",
            log_y=True,
            text_auto='.6f'
        )
        fig_bar.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # --- KEY INSIGHTS ---
        st.markdown("### 5. Key Insights")

        st.error(
            "**Why Model A Fails (2 Layers)**\n\n"
            "Shallow networks cannot capture the complex gradient landscapes of PDEs:\n\n"
            "- Heat diffusion involves second-order spatial derivatives (Laplacian)\n"
            "- The network must learn `d²u/dx²` and `d²u/dy²` implicitly\n"
            "- 2 layers lack the representational capacity for smooth, physics-consistent solutions\n\n"
            "**Result:** 77x higher loss than Model B"
        )

        st.success(
            "**Why Model B Wins (4 Layers)**\n\n"
            "The 4-layer architecture provides optimal balance:\n\n"
            "- Sufficient depth to capture second-order derivative relationships\n"
            "- Not so deep as to suffer from vanishing gradients\n"
            "- Tanh activations remain effective at this depth\n\n"
            "**Result:** Lowest loss, fastest convergence after epoch 800"
        )

        st.warning(
            "**Why Model C Underperforms (8 Layers)**\n\n"
            "Deeper is not always better for PINNs:\n\n"
            "- Vanishing gradients through 8 Tanh layers\n"
            "- Harder optimization landscape\n"
            "- More parameters but diminishing returns\n\n"
            "**Result:** 3.4x higher loss than Model B despite 2.3x more parameters"
        )

        # --- RAW DATA ---
        with st.expander("View Raw DOE Data"):
            st.dataframe(doe_df, use_container_width=True)

    else:
        st.warning(
            "**DOE results not found!** The file `data/p4_doe_results.csv` is missing. "
            "Run `python p4_doe_train.py` to generate the results."
        )
