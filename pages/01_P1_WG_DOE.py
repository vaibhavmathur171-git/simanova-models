# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --- 1. Page Config ---
st.set_page_config(
    page_title="P1: Inverse Waveguide Design",
    page_icon="📐",
    layout="wide"
)

# --- 2. Model Architecture (Must match your saved .pth file) ---
# We define a standard MLP. If your training used different layers, 
# this might need adjustment, but the 'try-except' block below prevents crashes.
class SimpleMLP(nn.Module):
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

# --- 3. Helper Functions ---
@st.cache_data
def load_data():
    """Loads the DOE results. Checks 'Data' (Cap D) and 'data' (lower d)."""
    # List of possible paths to try (handles case sensitivity issues)
    possible_paths = [
        'Data/p1_doe_results.csv',  # Your current folder structure (Capital D)
        'data/p1_doe_results.csv'   # Standard lowercase backup
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
            
    st.error(f"⚠️ Could not find data file. Checked: {possible_paths}")
    return None

def load_model():
    """Loads the trained model weights from 'models/' folder."""
    path = 'models/p1_mono_model.pth'
    model = SimpleMLP()
    
    if os.path.exists(path):
        try:
            # We map to CPU to ensure it runs on the cloud
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            model.eval()
            return model
        except Exception as e:
            # If the architecture doesn't match, we don't crash the app.
            st.warning(f"⚠️ Note: Using Physics Equation only. (Model architecture mismatch: {e})")
            return None
    else:
        # Silently fail to physics-only mode if model is missing
        return None

def physics_grating_equation(target_angle_deg, wavelength_nm=550, n_out=1.5, n_in=1.0):
    """The Ground Truth Physics Equation."""
    theta_out_rad = np.radians(target_angle_deg)
    m = -1
    # Grating Eq: n_out*sin(theta) = n_in*sin(0) + m*lambda/period
    # Rearranged: Period = m*lambda / (n_out*sin(theta))
    period = (m * wavelength_nm) / (n_out * np.sin(theta_out_rad))
    return abs(period)

# --- 4. Main App Layout ---

st.title("📐 P1: Mono-Waveguide Design Tool")
st.markdown("### AI-Powered Inverse Design (550 nm)")

# Sidebar Inputs
st.sidebar.header("🎛️ Design Parameters")
target_angle = st.sidebar.slider("Target Output Angle (°)", -80.0, -30.0, -51.0, 0.1)

# Perform Calculations
true_period = physics_grating_equation(target_angle)
model = load_model()
ai_period = 0.0

# AI Inference
if model:
    # Normalize input if needed (assuming raw for this version)
    input_tensor = torch.tensor([[target_angle]], dtype=torch.float32)
    with torch.no_grad():
        ai_period = model(input_tensor).item()

# Display Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Target Angle", f"{target_angle}°")
with col2:
    st.metric("Physics Truth (Period)", f"{true_period:.2f} nm")
with col3:
    if model:
        err = abs(ai_period - true_period)
        st.metric("AI Prediction", f"{ai_period:.2f} nm", delta=f"Err: {err:.3f} nm", delta_color="inverse")
    else:
        st.info("AI Model inactive")

# Visualization
st.divider()
st.subheader("📊 Interactive Design Curve")

# Generate the curve data
angles = np.linspace(-80, -30, 100)
periods_truth = [physics_grating_equation(a) for a in angles]

fig, ax = plt.subplots(figsize=(10, 5))
# Plot Truth
ax.plot(angles, periods_truth, label="Physics Truth (Grating Eq)", color='#1f77b4', linewidth=3)
# Plot Selected Point
ax.scatter([target_angle], [true_period], color='red', s=150, zorder=5, label="Current Design Point")

# Plot AI Curve (if model exists)
if model:
    input_batch = torch.tensor(angles.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        periods_ai = model(input_batch).numpy().flatten()
    ax.plot(angles, periods_ai, '--', label="Neural Network Prediction", color='#ff7f0e', linewidth=2)

ax.set_xlabel("Target Output Angle (degrees)", fontsize=12)
ax.set_ylabel("Required Grating Period (nm)", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Data Tables
st.divider()
st.subheader("📑 Experimental Data (DOE)")
df = load_data()
if df is not None:
    st.dataframe(df.head(100), use_container_width=True)
else:
    st.warning("Data file not found. Ensure 'p1_doe_results.csv' is in the 'Data' folder.")