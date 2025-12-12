# -*- coding: utf-8 -*-
"""
Project 3: Holographic Waveguide Uniformity
Interactive dashboard for SRG waveguide design and AI prediction.

Features:
- Tab 1: Waveguide Designer (Engineering tool with real units)
- Tab 2: AI Research Lab (DOE results and model comparison)
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
    page_title="P3: Waveguide Uniformity",
    page_icon="🔦",
    layout="wide"
)

# =============================================================================
# CONSTANTS (Engineering Units)
# =============================================================================
GRID_SIZE = 64
ABSORPTION_FACTOR = 0.99

# Physics Constants
REFRACTIVE_INDEX = 1.7
MAX_DEPTH = 400.0  # nm
KOGELNIK_PERIOD = 800.0  # nm
MAX_EFFICIENCY = 0.8

# Depth Range
DEPTH_MIN = 50.0   # nm
DEPTH_MAX = 350.0  # nm

# Model Paths
MODEL_C_PATH = "models/p3_model_C.pth"
DOE_RESULTS_PATH = "data/p3_doe_results.csv"


# =============================================================================
# MODEL DEFINITION: Model C (Deep Receptive - Best)
# =============================================================================
class ModelC_DeepReceptive(nn.Module):
    """
    Model C: "The Deep Receptive" (5-layer CNN)
    Best performing model from DOE with 11x11 receptive field.
    """
    def __init__(self):
        super(ModelC_DeepReceptive, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# PHYSICS ENGINE (Kogelnik + Leaky Bucket)
# =============================================================================
def depth_to_efficiency(depth_nm: np.ndarray) -> np.ndarray:
    """
    Convert etch depth to grating efficiency using Kogelnik approximation.

    Formula: Efficiency = 0.8 * sin^2(PI * depth / 800)

    Peak efficiency (~0.8) occurs at ~200nm depth.
    """
    phase = np.pi * depth_nm / KOGELNIK_PERIOD
    efficiency = MAX_EFFICIENCY * np.sin(phase) ** 2
    return efficiency.astype(np.float32)


def solve_light_propagation(efficiency_map: np.ndarray) -> np.ndarray:
    """
    Solve light propagation using the Leaky Bucket Energy Transport model.

    Physics:
    - Light propagates Left (x=0) to Right (x=63)
    - Each pixel extracts: Extracted = Energy * Efficiency
    - Remaining energy: Remaining = (Energy - Extracted) * 0.99
    """
    grid_size = efficiency_map.shape[0]
    extracted_light = np.zeros((grid_size, grid_size), dtype=np.float32)
    current_energy = np.ones(grid_size, dtype=np.float32)

    for x in range(grid_size):
        extracted_light[:, x] = current_energy * efficiency_map[:, x]
        remaining = current_energy - extracted_light[:, x]
        current_energy = remaining * ABSORPTION_FACTOR

    return extracted_light


def generate_depth_map_from_params(mean_depth: float, gradient_correction: float,
                                    process_noise: float) -> np.ndarray:
    """
    Generate an etch depth map based on engineering parameters.

    Parameters:
        mean_depth: Average etch depth in nm (50-350)
        gradient_correction: Linear ramp coefficient (-1 to 1)
                            Positive = deeper on right (compensates depletion)
        process_noise: Manufacturing jitter in nm (0-20)

    Returns:
        depth_map: 2D array of etch depths in nanometers
    """
    y, x = np.meshgrid(
        np.linspace(0, 1, GRID_SIZE),
        np.linspace(0, 1, GRID_SIZE),
        indexing='ij'
    )

    # Base depth with gradient correction
    # gradient_correction of 1.0 means +100nm increase from left to right
    depth_map = mean_depth + gradient_correction * 100.0 * (x - 0.5)

    # Add manufacturing process noise (Gaussian)
    if process_noise > 0:
        noise = np.random.normal(0, process_noise, (GRID_SIZE, GRID_SIZE))
        depth_map += noise

    # Clip to valid manufacturing range
    depth_map = np.clip(depth_map, DEPTH_MIN, DEPTH_MAX)

    return depth_map.astype(np.float32)


# =============================================================================
# MODEL LOADING
# =============================================================================
@st.cache_resource
def load_model_c():
    """Load the best model (Model C) with error handling."""
    if not os.path.exists(MODEL_C_PATH):
        return None

    try:
        model = ModelC_DeepReceptive()
        state_dict = torch.load(MODEL_C_PATH, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_doe_results():
    """Load DOE results CSV."""
    if os.path.exists(DOE_RESULTS_PATH):
        return pd.read_csv(DOE_RESULTS_PATH)
    return None


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================
def create_plotly_heatmap(data: np.ndarray, title: str, colorscale: str = 'Viridis',
                          zmin: float = None, zmax: float = None,
                          colorbar_title: str = None) -> go.Figure:
    """Create a Plotly heatmap with consistent styling."""
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title=colorbar_title) if colorbar_title else None
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis_title="X (Propagation Direction)",
        yaxis_title="Y",
        yaxis=dict(scaleanchor="x", scaleratio=1, autorange="reversed"),
        margin=dict(l=50, r=50, t=50, b=50),
        height=400
    )

    return fig


def compute_uniformity_score(light_field: np.ndarray) -> float:
    """Compute uniformity score (0-100%). Higher is better."""
    mean_val = np.mean(light_field)
    std_val = np.std(light_field)
    if mean_val < 1e-8:
        return 0.0
    uniformity = max(0, 1.0 - (std_val / mean_val))
    return uniformity * 100


# =============================================================================
# MAIN APP
# =============================================================================
st.title("🔦 P3: Holographic Waveguide Uniformity")

# --- EXECUTIVE SUMMARY ---
with st.container():
    st.markdown("### Project Executive Summary")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
            "**1. The Engineering Goal**\n\n"
            "Achieve **Uniform Light Output** across an AR waveguide. "
            "Design Surface Relief Grating (SRG) etch depths that extract "
            "light evenly, preventing bright/dark spots."
        )

    with c2:
        st.info(
            "**2. The Physics**\n\n"
            "**Kogelnik + Leaky Bucket:** Etch depth (nm) maps to diffraction "
            "efficiency via sin^2 response. Light depletes as it propagates "
            "left-to-right with 1% absorption per column."
        )

    with c3:
        st.info(
            "**3. The AI Strategy**\n\n"
            "**Deep CNN (Model C):** 5-layer network with 11x11 receptive field "
            "captures long-range energy depletion effects. Enables instant "
            "design iteration vs. physics simulation."
        )

st.divider()

# --- TABS ---
tab1, tab2 = st.tabs(["🛠️ Waveguide Designer", "🔬 AI Research Lab (DOE)"])

# =============================================================================
# TAB 1: WAVEGUIDE DESIGNER
# =============================================================================
with tab1:
    # Sidebar controls
    st.sidebar.header("Etch Depth Designer")
    st.sidebar.markdown("Design your SRG waveguide grating.")

    st.sidebar.subheader("Depth Parameters")
    mean_depth = st.sidebar.slider(
        "Mean Etch Depth (nm)",
        min_value=50, max_value=350, value=200, step=10,
        help="Average etch depth across the grating. ~200nm gives peak efficiency."
    )

    gradient_correction = st.sidebar.slider(
        "Gradient Correction",
        min_value=-1.0, max_value=1.0, value=0.5, step=0.1,
        help="Linear ramp to compensate for energy depletion. Positive = deeper on right."
    )

    process_noise = st.sidebar.slider(
        "Process Noise (nm)",
        min_value=0, max_value=20, value=5, step=1,
        help="Simulate manufacturing jitter/variation."
    )

    # Generate button for noise regeneration
    if st.sidebar.button("🎲 Regenerate Noise"):
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("P3: Waveguide Uniformity | SimaNova")

    # Load model
    model = load_model_c()

    if model is None:
        st.warning(
            "**Model not found!** The file `models/p3_model_C.pth` is missing. "
            "Run `python p3_doe_train.py` to train the models first."
        )

    # Generate depth map
    depth_map = generate_depth_map_from_params(mean_depth, gradient_correction, process_noise)

    # Convert to efficiency (Kogelnik)
    efficiency_map = depth_to_efficiency(depth_map)

    # Run physics simulation
    physics_output = solve_light_propagation(efficiency_map)

    # Run AI prediction
    if model is not None:
        # Normalize depth for model input (same as training)
        depth_normalized = depth_map / MAX_DEPTH
        input_tensor = torch.from_numpy(depth_normalized).float()
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)

        with torch.no_grad():
            ai_output_tensor = model(input_tensor)

        ai_output = ai_output_tensor.squeeze().numpy()
    else:
        ai_output = None

    # --- VISUALIZATION ---
    st.subheader("Design Visualization")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**1. Etch Depth Profile**")
        fig1 = create_plotly_heatmap(
            depth_map,
            "Etch Depth (nm)",
            colorscale='Plasma',
            zmin=DEPTH_MIN,
            zmax=DEPTH_MAX,
            colorbar_title="nm"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Depth stats
        st.caption(f"Min: {depth_map.min():.0f} nm | Max: {depth_map.max():.0f} nm | Mean: {depth_map.mean():.0f} nm")

    with col2:
        st.markdown("**2. AI Prediction (Model C)**")
        if ai_output is not None:
            fig2 = create_plotly_heatmap(
                ai_output,
                "Predicted Light Field",
                colorscale='Viridis',
                zmin=0,
                zmax=0.5,
                colorbar_title="Intensity"
            )
            st.plotly_chart(fig2, use_container_width=True)

            uniformity_ai = compute_uniformity_score(ai_output)
            st.caption(f"Mean: {ai_output.mean():.4f} | Uniformity: {uniformity_ai:.1f}%")
        else:
            st.info("Model not available.")

    with col3:
        st.markdown("**3. Physics Ground Truth**")
        fig3 = create_plotly_heatmap(
            physics_output,
            "Simulated Light Field",
            colorscale='Viridis',
            zmin=0,
            zmax=0.5,
            colorbar_title="Intensity"
        )
        st.plotly_chart(fig3, use_container_width=True)

        uniformity_physics = compute_uniformity_score(physics_output)
        st.caption(f"Mean: {physics_output.mean():.4f} | Uniformity: {uniformity_physics:.1f}%")

    # --- MODEL ACCURACY ---
    if ai_output is not None:
        st.divider()
        st.subheader("Model Accuracy")

        mse = np.mean((ai_output - physics_output) ** 2)
        mae = np.mean(np.abs(ai_output - physics_output))
        max_error = np.max(np.abs(ai_output - physics_output))
        correlation = np.corrcoef(ai_output.flatten(), physics_output.flatten())[0, 1]

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("MSE", f"{mse:.6f}")
        with col_m2:
            st.metric("MAE", f"{mae:.6f}")
        with col_m3:
            st.metric("Max Error", f"{max_error:.4f}")
        with col_m4:
            st.metric("Correlation", f"{correlation:.4f}")

        # Error heatmap
        error_map = ai_output - physics_output
        fig_err = create_plotly_heatmap(
            error_map,
            "Prediction Error (AI - Physics)",
            colorscale='RdBu_r',
            zmin=-0.05,
            zmax=0.05,
            colorbar_title="Error"
        )
        fig_err.update_layout(height=300)
        st.plotly_chart(fig_err, use_container_width=True)

    # --- PHYSICS EXPLAINER ---
    with st.expander("Physics Deep Dive: Kogelnik + Leaky Bucket"):
        st.markdown("""
        ### Surface Relief Grating (SRG) Physics

        **1. Kogelnik Approximation (Depth → Efficiency)**

        The diffraction efficiency of an SRG depends on etch depth via:

        $$\\eta(d) = 0.8 \\cdot \\sin^2\\left(\\frac{\\pi \\cdot d}{800}\\right)$$

        Where:
        - $d$ = etch depth in nanometers
        - Peak efficiency (~80%) at $d \\approx 200$ nm
        - This creates a **non-linear** response curve

        **2. Leaky Bucket Energy Transport**

        Light propagates column-by-column from left to right:
        1. `Extracted[y,x] = Energy[y] * Efficiency[y,x]`
        2. `Energy[y] = (Energy[y] - Extracted[y,x]) * 0.99`

        The 0.99 factor represents 1% absorption loss per column.

        **3. Design Strategy**

        To achieve uniform light output:
        - Start with **lower efficiency** on the left (shallow etch)
        - Increase efficiency toward the right (deeper etch)
        - This compensates for energy depletion

        The **Gradient Correction** slider implements this strategy!
        """)


# =============================================================================
# TAB 2: AI RESEARCH LAB (DOE)
# =============================================================================
with tab2:
    st.subheader("Design of Experiments: CNN Architecture Comparison")

    doe_df = load_doe_results()

    if doe_df is not None:
        # --- EXPERIMENT SCOPE ---
        st.markdown("### 1. Experimental Design")

        col_exp1, col_exp2, col_exp3 = st.columns(3)

        with col_exp1:
            st.info(
                "**Model A: Pixel-Wise**\n\n"
                "Architecture: 2x Conv(1x1)\n\n"
                "Receptive Field: **1x1**\n\n"
                "Parameters: 49"
            )

        with col_exp2:
            st.info(
                "**Model B: Standard**\n\n"
                "Architecture: 3x Conv(3x3)\n\n"
                "Receptive Field: **7x7**\n\n"
                "Parameters: 5,089"
            )

        with col_exp3:
            st.success(
                "**Model C: Deep Receptive** ✓\n\n"
                "Architecture: 5x Conv(3x3)\n\n"
                "Receptive Field: **11x11**\n\n"
                "Parameters: 42,049"
            )

        st.divider()

        # --- LOSS CURVES ---
        st.markdown("### 2. Training Loss Curves")

        # Create line chart
        fig_loss = px.line(
            doe_df,
            x='Epoch',
            y='Test_Loss',
            color='Model_Name',
            markers=True,
            log_y=True,
            title="Test Loss vs Epoch (Log Scale)",
            labels={'Test_Loss': 'Test Loss (MSE)', 'Model_Name': 'Model'}
        )

        fig_loss.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Test Loss (MSE, log scale)",
            legend_title="Model",
            height=400
        )

        st.plotly_chart(fig_loss, use_container_width=True)

        # --- FINAL METRICS ---
        st.markdown("### 3. Final Performance Metrics")

        # Get final losses
        final_results = doe_df.groupby('Model_Name')['Test_Loss'].last().reset_index()
        final_results = final_results.sort_values('Test_Loss', ascending=False)

        col_r1, col_r2, col_r3 = st.columns(3)

        model_a_loss = doe_df[doe_df['Model_Name'] == 'Model_A_PixelWise']['Test_Loss'].iloc[-1]
        model_b_loss = doe_df[doe_df['Model_Name'] == 'Model_B_Standard']['Test_Loss'].iloc[-1]
        model_c_loss = doe_df[doe_df['Model_Name'] == 'Model_C_DeepReceptive']['Test_Loss'].iloc[-1]

        with col_r1:
            st.metric(
                "Model A (1x1)",
                f"{model_a_loss:.6f}",
                delta=f"{(model_a_loss/model_c_loss):.0f}x worse",
                delta_color="inverse"
            )

        with col_r2:
            st.metric(
                "Model B (3x3)",
                f"{model_b_loss:.6f}",
                delta=f"{(model_b_loss/model_c_loss):.1f}x worse",
                delta_color="inverse"
            )

        with col_r3:
            st.metric(
                "Model C (Deep) ✓",
                f"{model_c_loss:.6f}",
                delta="Best",
                delta_color="normal"
            )

        # Bar chart comparison
        fig_bar = px.bar(
            final_results,
            x='Model_Name',
            y='Test_Loss',
            color='Model_Name',
            title="Final Test Loss Comparison",
            log_y=True,
            text_auto='.6f'
        )
        fig_bar.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # --- KEY INSIGHTS ---
        st.markdown("### 4. Key Insights")

        st.error(
            "**Why Model A Fails (1x1 Kernels)**\n\n"
            "Model A uses 1x1 convolutions, meaning each output pixel only sees "
            "the corresponding input pixel. It **cannot** capture the physics because:\n\n"
            "- Light propagation is **spatially dependent** (left-to-right energy flow)\n"
            "- Efficiency at position (x) affects ALL positions to its right\n"
            "- A 1x1 receptive field sees no neighbors, treating each pixel independently\n\n"
            "**Result:** 90x worse than Model C"
        )

        st.success(
            "**Why Model C Wins (5-Layer, 11x11 Receptive Field)**\n\n"
            "Model C's larger receptive field allows it to:\n\n"
            "- See ~11 columns of the input simultaneously\n"
            "- Learn the cumulative energy depletion pattern\n"
            "- Capture long-range dependencies in the physics\n\n"
            "**Result:** 6x better than Model B, proving that **spatial context matters** "
            "for physics-based problems."
        )

        # --- RAW DATA ---
        with st.expander("View Raw DOE Data"):
            st.dataframe(doe_df, use_container_width=True)

    else:
        st.warning(
            "**DOE results not found!** The file `data/p3_doe_results.csv` is missing. "
            "Run `python p3_doe_train.py` to generate the results."
        )
