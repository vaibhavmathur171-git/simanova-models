# -*- coding: utf-8 -*-
"""
Project 3: Holographic Waveguide Uniformity
Interactive dashboard for grating design and light uniformity prediction.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --- 1. Page Config ---
st.set_page_config(
    page_title="P3: Waveguide Uniformity",
    page_icon="🔦",
    layout="wide"
)

# --- 2. CONSTANTS ---
GRID_SIZE = 64
ABSORPTION_FACTOR = 0.99
MODEL_PATH = "models/p3_cnn_model.pth"


# --- 3. MODEL DEFINITION (Must match training architecture) ---
class WaveguideCNN(nn.Module):
    """
    Lightweight CNN for Image-to-Image regression.
    Architecture: Conv(1->16) -> Conv(16->32) -> Conv(32->1)
    """
    def __init__(self):
        super(WaveguideCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x


# --- 4. PHYSICS ENGINE (Ground Truth) ---
def solve_light_propagation(efficiency_map: np.ndarray) -> np.ndarray:
    """
    Solve the light propagation using the "Leaky Bucket" Energy Transport model.

    Physics:
    - Light propagates from Left (x=0) to Right (x=63)
    - At each pixel, light is extracted based on local grating efficiency
    - Remaining energy undergoes 1% absorption loss per column

    Algorithm (per column):
    1. Extracted_Light = Current_Energy * Efficiency_Pixel
    2. Remaining_Energy = (Current_Energy - Extracted_Light) * 0.99
    3. Pass Remaining_Energy to next column (x+1)
    """
    grid_size = efficiency_map.shape[0]
    extracted_light = np.zeros((grid_size, grid_size), dtype=np.float32)
    current_energy = np.ones(grid_size, dtype=np.float32)

    for x in range(grid_size):
        extracted_light[:, x] = current_energy * efficiency_map[:, x]
        remaining = current_energy - extracted_light[:, x]
        current_energy = remaining * ABSORPTION_FACTOR

    return extracted_light


# --- 5. GRATING GENERATOR (From User Controls) ---
def generate_grating_from_params(gradient_slope: float, gradient_bias: float,
                                  blob_intensity: float, blob_x: float,
                                  blob_y: float, blob_size: float) -> np.ndarray:
    """
    Generate an efficiency map based on user-controlled parameters.
    Combines a linear gradient with a Gaussian blob.
    """
    y, x = np.meshgrid(np.linspace(0, 1, GRID_SIZE),
                       np.linspace(0, 1, GRID_SIZE), indexing='ij')

    # Linear gradient (left to right)
    gradient = gradient_bias + gradient_slope * x

    # Gaussian blob
    center_x = blob_x * GRID_SIZE
    center_y = blob_y * GRID_SIZE
    sigma = blob_size * 15 + 5  # Range: 5 to 20

    y_idx, x_idx = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing='ij')
    blob = blob_intensity * np.exp(-((x_idx - center_x)**2 + (y_idx - center_y)**2) / (2 * sigma**2))

    # Combine and clip
    efficiency_map = gradient + blob
    efficiency_map = np.clip(efficiency_map, 0.0, 0.8)

    return efficiency_map.astype(np.float32)


# --- 6. MODEL LOADING ---
@st.cache_resource
def load_model():
    """Load the trained CNN model with error handling."""
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        model = WaveguideCNN()
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# --- 7. VISUALIZATION ---
def create_heatmap_figure(data: np.ndarray, title: str, cmap: str = 'viridis'):
    """Create a matplotlib heatmap with consistent styling."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0, origin='upper')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("X (Propagation Direction)")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 32, 63])
    ax.set_yticks([0, 32, 63])
    plt.tight_layout()
    return fig


def compute_uniformity_metrics(light_field: np.ndarray) -> dict:
    """Compute uniformity metrics for the light output."""
    mean_val = np.mean(light_field)
    std_val = np.std(light_field)
    min_val = np.min(light_field)
    max_val = np.max(light_field)
    uniformity = 1.0 - (std_val / (mean_val + 1e-8))  # Higher is better
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'uniformity': max(0, uniformity)
    }


# --- 8. MAIN APP ---
st.title("🔦 P3: Holographic Waveguide Uniformity")

# --- EXECUTIVE SUMMARY ---
with st.container():
    st.markdown("### Project Executive Summary")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
            "**1. The Engineering Goal**\n\n"
            "Achieve **Uniform Light Output** across an AR waveguide display. "
            "Design grating patterns that extract light evenly from left to right, "
            "preventing bright/dark spots."
        )

    with c2:
        st.info(
            "**2. The Physics**\n\n"
            "**Leaky Bucket Model:** Light enters from the left edge and propagates right. "
            "Each pixel extracts light based on local grating efficiency. "
            "Energy depletes as it travels (1% absorption per column)."
        )

    with c3:
        st.info(
            "**3. The AI Strategy**\n\n"
            "**CNN Image-to-Image:** A 3-layer Convolutional Neural Network learns to predict "
            "the 2D light output field from any input grating design. "
            "Enables instant design iteration without running physics simulations."
        )

st.divider()

# --- SIDEBAR: GRATING DESIGNER ---
st.sidebar.header("Grating Designer")
st.sidebar.markdown("Adjust parameters to design your grating pattern.")

st.sidebar.subheader("Gradient Controls")
gradient_slope = st.sidebar.slider(
    "Gradient Slope",
    min_value=-0.5, max_value=1.0, value=0.3, step=0.05,
    help="Controls how efficiency changes from left to right"
)
gradient_bias = st.sidebar.slider(
    "Gradient Bias",
    min_value=0.0, max_value=0.5, value=0.2, step=0.05,
    help="Base efficiency level across the grating"
)

st.sidebar.subheader("Blob Controls")
blob_intensity = st.sidebar.slider(
    "Blob Intensity",
    min_value=0.0, max_value=0.5, value=0.2, step=0.05,
    help="Strength of the local efficiency boost"
)
blob_x = st.sidebar.slider(
    "Blob X Position",
    min_value=0.0, max_value=1.0, value=0.5, step=0.1,
    help="Horizontal position (0=left, 1=right)"
)
blob_y = st.sidebar.slider(
    "Blob Y Position",
    min_value=0.0, max_value=1.0, value=0.5, step=0.1,
    help="Vertical position (0=top, 1=bottom)"
)
blob_size = st.sidebar.slider(
    "Blob Size",
    min_value=0.1, max_value=1.0, value=0.5, step=0.1,
    help="Size of the efficiency blob"
)

# --- LOAD MODEL ---
model = load_model()

if model is None:
    st.warning(
        "**Model not found!** The file `models/p3_cnn_model.pth` is missing. "
        "Run `python p3_train_model.py` to train the model first. "
        "Showing Physics Ground Truth only."
    )

# --- GENERATE GRATING ---
efficiency_map = generate_grating_from_params(
    gradient_slope, gradient_bias,
    blob_intensity, blob_x, blob_y, blob_size
)

# --- RUN PHYSICS SIMULATION ---
physics_output = solve_light_propagation(efficiency_map)

# --- RUN AI PREDICTION ---
if model is not None:
    # Prepare input: (64, 64) -> (1, 1, 64, 64)
    input_tensor = torch.from_numpy(efficiency_map).float()
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 64, 64)

    with torch.no_grad():
        ai_output_tensor = model(input_tensor)

    # Convert back: (1, 1, 64, 64) -> (64, 64)
    ai_output = ai_output_tensor.squeeze().numpy()
else:
    ai_output = None

# --- MAIN VISUALIZATION ---
st.subheader("Design Visualization")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Input Grating (Efficiency Map)**")
    fig1 = create_heatmap_figure(efficiency_map, "Input Grating", cmap='plasma')
    st.pyplot(fig1)
    plt.close(fig1)

    # Metrics
    st.caption(f"Min: {efficiency_map.min():.3f} | Max: {efficiency_map.max():.3f} | Mean: {efficiency_map.mean():.3f}")

with col2:
    st.markdown("**AI Prediction (CNN Output)**")
    if ai_output is not None:
        fig2 = create_heatmap_figure(ai_output, "AI Prediction", cmap='viridis')
        st.pyplot(fig2)
        plt.close(fig2)

        metrics_ai = compute_uniformity_metrics(ai_output)
        st.caption(f"Mean: {metrics_ai['mean']:.4f} | Uniformity: {metrics_ai['uniformity']:.2%}")
    else:
        st.info("Model not available. Train the model to see AI predictions.")

with col3:
    st.markdown("**Physics Ground Truth**")
    fig3 = create_heatmap_figure(physics_output, "Physics Truth", cmap='viridis')
    st.pyplot(fig3)
    plt.close(fig3)

    metrics_physics = compute_uniformity_metrics(physics_output)
    st.caption(f"Mean: {metrics_physics['mean']:.4f} | Uniformity: {metrics_physics['uniformity']:.2%}")

# --- COMPARISON METRICS ---
if ai_output is not None:
    st.divider()
    st.subheader("Model Accuracy")

    # Compute error
    mse = np.mean((ai_output - physics_output) ** 2)
    mae = np.mean(np.abs(ai_output - physics_output))
    max_error = np.max(np.abs(ai_output - physics_output))

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric("MSE", f"{mse:.6f}")
    with col_m2:
        st.metric("MAE", f"{mae:.6f}")
    with col_m3:
        st.metric("Max Error", f"{max_error:.4f}")
    with col_m4:
        # Correlation
        correlation = np.corrcoef(ai_output.flatten(), physics_output.flatten())[0, 1]
        st.metric("Correlation", f"{correlation:.4f}")

    # Error heatmap
    st.markdown("**Prediction Error Map** (AI - Physics)")
    error_map = ai_output - physics_output

    fig_err, ax_err = plt.subplots(figsize=(8, 3))
    im = ax_err.imshow(error_map, cmap='RdBu_r', vmin=-0.1, vmax=0.1, origin='upper')
    ax_err.set_title("Error Map (Blue=Under, Red=Over)", fontsize=11)
    ax_err.set_xlabel("X (Propagation Direction)")
    ax_err.set_ylabel("Y")
    plt.colorbar(im, ax=ax_err, fraction=0.02, pad=0.04)
    plt.tight_layout()
    st.pyplot(fig_err)
    plt.close(fig_err)

# --- PHYSICS EXPLANATION ---
st.divider()
with st.expander("How the Physics Simulation Works"):
    st.markdown("""
    ### Leaky Bucket Energy Transport Model

    The simulation models light propagation through a waveguide with extraction gratings:

    1. **Initialization**: Light enters from the left edge with energy = 1.0 for all rows

    2. **Column-by-Column Propagation** (x = 0 to 63):
       - `Extracted_Light[y, x] = Current_Energy[y] * Efficiency[y, x]`
       - `Remaining_Energy[y] = (Current_Energy[y] - Extracted_Light[y, x]) * 0.99`
       - The 0.99 factor represents 1% absorption loss per column

    3. **Output**: The extracted light field represents what a user would see

    **Design Challenge**: Create efficiency patterns that extract light uniformly across
    the entire display, compensating for energy depletion as light travels right.

    **Optimal Strategy**: Increase efficiency from left to right to compensate for depleting energy.
    """)

# --- FOOTER ---
st.sidebar.divider()
st.sidebar.caption("P3: Waveguide Uniformity | SimaNova")
