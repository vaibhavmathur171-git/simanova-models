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

# --- 2. EXECUTIVE SUMMARY (NEW) ---
st.title("📐 P1: Mono-Waveguide Design Tool")

with st.container():
    st.markdown("### 🎯 Project Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.info("**1. The Engineering Goal**\n\nFind the exact **Grating Period ($\Lambda$)** required to steer light to a specific target angle (e.g., -50°).")
    
    with c2:
        st.info("**2. The Physics**\n\n**The Grating Equation:**\n$n_{out} \sin(\\theta_m) = n_{in} \sin(\\theta_{in}) + \\frac{m \lambda}{\Lambda}$")
    
    with c3:
        st.info("**3. The AI Strategy**\n\n**Neural Surrogate:** Train a small Neural Net to learn this physics equation perfectly, replacing the need for explicit math solvers.")
        
    with c4:
        st.success("**4. Why This Matters?**\n\nDemonstrates that AI can 'learn' physics laws. If it works for this simple equation, it can work for complex simulations where no equation exists.")

st.divider()

# --- 3. Model Architecture ---
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

# --- 4. Helper Functions ---
@st.cache_data
def load_data():
    paths = ['data/p1_doe_results.csv', 'Data/p1_doe_results.csv']
    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

def load_model():
    path = 'models/p1_mono_model.pth'
    model = SimpleMLP()
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            model.eval()
            return model
        except:
            return None
    return None

def physics_grating_equation(target_angle_deg, wavelength_nm=550, n_out=1.5):
    theta_out_rad = np.radians(target_angle_deg)
    m = -1
    if np.sin(theta_out_rad) == 0: return 0
    period = (m * wavelength_nm) / (n_out * np.sin(theta_out_rad))
    return abs(period)

# --- 5. Main App Logic ---
st.markdown("### 🛠️ Interactive Design Lab")

# Sidebar
st.sidebar.header("🎛️ Design Parameters")
target_angle = st.sidebar.slider("Target Output Angle (°)", -80.0, -30.0, -51.0, 0.1)

# Calculations
true_period = physics_grating_equation(target_angle)
model = load_model()
ai_period = 0.0

if model:
    input_tensor = torch.tensor([[target_angle]], dtype=torch.float32)
    with torch.no_grad():
        ai_period = model(input_tensor).item()

# Metrics
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
        st.info("Using Physics Eq Only")

# Visualization
st.divider()
st.subheader("📊 Interactive Design Curve")
angles = np.linspace(-80, -30, 100)
periods_truth = [physics_grating_equation(a) for a in angles]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(angles, periods_truth, label="Physics Truth", color='#1f77b4', linewidth=3)
ax.scatter([target_angle], [true_period], color='red', s=150, zorder=5, label="Current Design")

if model:
    input_batch = torch.tensor(angles.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        periods_ai = model(input_batch).numpy().flatten()
    ax.plot(angles, periods_ai, '--', label="AI Prediction", color='#ff7f0e', linewidth=2)

ax.set_xlabel("Output Angle (deg)")
ax.set_ylabel("Grating Period (nm)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Data
st.divider()
st.subheader("📑 DOE Data")
df = load_data()
if df is not None:
    st.dataframe(df.head(50), use_container_width=True)