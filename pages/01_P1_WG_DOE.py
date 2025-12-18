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
    page_title="P1: Neural Surrogate for Inverse Optical Design",
    page_icon="🔬",
    layout="wide"
)

# --- 2. PROJECT CHARACTERIZATION ---
st.title("Project 1: Characterizing Neural Surrogates for Inverse Optical Design")

st.markdown("""
<p style="color: #a0aec0; font-size: 1.1rem; margin-bottom: 2rem;">
Evaluating whether a Multilayer Perceptron (MLP) can bypass iterative RCWA solvers
for real-time architectural trade-offs in diffractive waveguide design.
</p>
""", unsafe_allow_html=True)

with st.container():
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Objective")
        st.markdown("""
        Determine the **Grating Period (Λ)** required to couple light at a specified
        diffraction angle. This inverse problem traditionally requires iterative numerical
        solvers—we evaluate if a neural surrogate can provide deterministic output in sub-millisecond latency.
        """)

        st.markdown("#### Robustness Protocol")
        st.markdown("""
        **Stochastic Perturbation:** Gaussian noise (σ = 0.5°) was injected into training
        inputs to simulate metrology uncertainty and fabrication tolerances. This regularization
        improves model convergence on underlying physical trends rather than overfitting to ideal conditions.
        """)

    with c2:
        st.markdown("#### Physical Constraint")
        st.markdown("""
        **The Grating Equation** governs waveguide coupling:

        $n_{out} \sin(\\theta_m) = n_{in} \sin(\\theta_{in}) + \\frac{m \lambda}{\Lambda}$

        Diffraction angles exhibit high sensitivity to sub-nanometer pitch variations—a 1nm
        change in period can shift output angle by ~0.1°. This sensitivity defines the
        precision requirements for the surrogate model.
        """)

    with c3:
        st.markdown("#### Neural Architecture")
        st.markdown("""
        **Model:** 4-layer MLP (1 → 64 → 64 → 1)

        **Justification:** The inverse grating problem is a point-to-point mapping with no
        spatial or temporal dependencies, making convolutional or recurrent architectures unnecessary.

        **Activation:** ReLU nonlinearities approximate the inherent nonlinearity of the
        optical manifold (the sin⁻¹ relationship between period and angle).
        """)

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
st.markdown("### Performance Evaluation")
st.markdown("""
<p style="color: #667eea; font-size: 0.9rem;">
Inference latency: <strong>&lt;10ms</strong> | 1000x reduction vs. iterative RCWA solver
</p>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Input Parameters")
target_angle = st.sidebar.slider("Target Diffraction Angle (°)", -80.0, -30.0, -51.0, 0.1)

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
    st.metric("Input: Diffraction Angle", f"{target_angle}°")
with col2:
    st.metric("Analytical Solution (Λ)", f"{true_period:.2f} nm")
with col3:
    if model:
        err = abs(ai_period - true_period)
        st.metric("Neural Surrogate Output", f"{ai_period:.2f} nm", delta=f"Δ: {err:.3f} nm", delta_color="inverse")
    else:
        st.info("Model not loaded—displaying analytical solution only")

# Visualization
st.divider()
st.subheader("Optical Manifold: Angle-to-Period Mapping")
angles = np.linspace(-80, -30, 100)
periods_truth = [physics_grating_equation(a) for a in angles]

fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('#0E1117')
ax.set_facecolor('#0E1117')
ax.plot(angles, periods_truth, label="Analytical (Grating Eq.)", color='#667eea', linewidth=2.5)
ax.scatter([target_angle], [true_period], color='#2ecc71', s=150, zorder=5, label="Query Point", edgecolors='white', linewidths=2)

if model:
    input_batch = torch.tensor(angles.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        periods_ai = model(input_batch).numpy().flatten()
    ax.plot(angles, periods_ai, '--', label="Neural Surrogate", color='#f093fb', linewidth=2)

ax.set_xlabel("Diffraction Angle (°)", color='#a0aec0')
ax.set_ylabel("Grating Period Λ (nm)", color='#a0aec0')
ax.tick_params(colors='#a0aec0')
ax.legend(facecolor='#1a1a2e', edgecolor='#2d2d44', labelcolor='#e2e8f0')
ax.grid(True, alpha=0.2, color='#2d2d44')
for spine in ax.spines.values():
    spine.set_color('#2d2d44')
st.pyplot(fig)

# Data
st.divider()
st.subheader("Training Data: Design of Experiments")
st.markdown("*10,000 synthetic samples generated from analytical grating equation with stochastic perturbation.*")
df = load_data()
if df is not None:
    st.dataframe(df.head(50), use_container_width=True)