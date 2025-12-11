# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="P1: Inverse Waveguide Design",
    page_icon="📐",
    layout="wide"
)

# --- 2. Define the Neural Network Architecture (Must match your training!) ---
# Assuming a standard 3-layer MLP based on your description
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),  # 1 Input (Angle)
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)   # 1 Output (Period)
        )

    def forward(self, x):
        return self.layers(x)

# --- 3. Helper Functions ---
@st.cache_data
def load_data():
    """Loads the DOE results from the data folder."""
    path = 'data/p1_doe_results.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None

def load_model():
    """Loads the trained model weights."""
    path = 'models/p1_mono_model.pth'
    model = SimpleMLP()
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            model.eval()
            return model
        except Exception as e:
            st.warning(f"⚠️ Could not load trained model: {e}. Using Physics Equation only.")
            return None
    else:
        st.warning(f"⚠️ Model file not found at {path}. Please upload it to 'models/' folder.")
        return None

def grating_equation(target_angle_deg, wavelength_nm=550, n_out=1.5, n_in=1.0):
    """Calculates the Physical Ground Truth Period."""
    # Period = m * lambda / (n_out*sin(theta_out) - n_in*sin(theta_in))
    # Assuming m = -1 and theta_in = 0
    theta_out_rad = np.radians(target_angle_deg)
    m = -1
    period = (m * wavelength_nm) / (n_out * np.sin(theta_out_rad))
    return abs(period)

# --- 4. Main App Layout ---

st.title("📐 P1: Mono-Waveguide Inverse Design")
st.markdown("### Neural Surrogate vs. Physics Truth (550nm)")

# --- Sidebar ---
st.sidebar.header("🎛️ Design Parameters")
target_angle = st.sidebar.slider("Target Output Angle (°)", -80.0, -30.0, -51.0, 0.1)

# --- Inference ---
# 1. Physics Truth
true_period = grating_equation(target_angle)

# 2. AI Prediction
model = load_model()
ai_period = 0.0
if model:
    # Normalize input if your training used scaling (Assuming raw for now, or add scaler logic)
    input_tensor = torch.tensor([[target_angle]], dtype=torch.float32)
    with torch.no_grad():
        ai_period = model(input_tensor).item()
        # NOTE: If you used a Scaler during training, you must inverse_transform this result!

# --- Display Results ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Target Angle", f"{target_angle}°")
with col2:
    st.metric("Physics Truth (Period)", f"{true_period:.2f} nm")
with col3:
    if model:
        error = abs(ai_period - true_period)
        st.metric("AI Prediction", f"{ai_period:.2f} nm", delta=f"Err: {error:.3f} nm")
    else:
        st.info("Model not loaded")

# --- Visualization ---
st.subheader("📊 Interactive Design Curve")
angles = np.linspace(-80, -30, 100)
periods_truth = [grating_equation(a) for a in angles]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(angles, periods_truth, label="Physics Truth (Grating Eq)", color='blue', linewidth=2)
ax.scatter([target_angle], [true_period], color='red', s=100, label="Current Design Point", zorder=5)

if model:
    # Batch prediction for the curve
    inputs = torch.tensor(angles.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        periods_ai = model(inputs).numpy().flatten()
    ax.plot(angles, periods_ai, '--', label="Neural Network Prediction", color='orange')

ax.set_xlabel("Output Angle (deg)")
ax.set_ylabel("Grating Period (nm)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# --- Data Section ---
st.divider()
st.subheader("📑 Design of Experiments (DOE) Results")
df = load_data()
if df is not None:
    st.dataframe(df.head(50), use_container_width=True)
else:
    st.error("Data file 'data/p1_doe_results.csv' not found.")