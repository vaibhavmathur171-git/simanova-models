# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os

# --- 1. Page Config ---
st.set_page_config(
    page_title="P2: Rainbow Surrogate",
    page_icon="🌈",
    layout="wide"
)

# --- 2. Dynamic Neural Architecture ---
class DynamicMLP(nn.Module):
    """
    A Neural Net that adjusts its size based on the DOE winner.
    """
    def __init__(self, hidden_neurons=64):
        super(DynamicMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, hidden_neurons),  # 2 Inputs: Angle, Wavelength
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons), # Assuming 3 hidden layers per DOE
            nn.ReLU(),
            nn.Linear(hidden_neurons, 1)   # 1 Output: Period
        )

    def forward(self, x):
        return self.layers(x)

# --- 3. Helper Functions ---
@st.cache_data
def load_doe_results():
    """Loads the DOE CSV to find the best model parameters."""
    path = 'data/p2_doe_results.csv' # Adjust if using 'Data/'
    if not os.path.exists(path):
        # Fallback for case sensitivity
        path = 'Data/p2_doe_results.csv'

    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_scalers():
    """Loads the MinMax scalers used during training."""
    path = 'models/p2_scalers.json'
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def load_best_model(best_neurons):
    """
    Loads the model weights, dynamically sizing the network
    to match the best-performing DOE run.
    """
    path = 'models/p2_rainbow_model.pth'

    # Instantiate the architecture that WON the DOE
    model = DynamicMLP(hidden_neurons=int(best_neurons))

    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            model.eval()
            return model
        except Exception as e:
            st.error(f"Architecture Mismatch: {e}")
            return None
    return None

def physics_grating_equation(target_angle_deg, wavelength_nm, n_out=1.5):
    """Calculates Period (Inverse) or Angle (Forward)."""
    # Inverse: Given Angle -> Find Period
    theta_rad = np.radians(target_angle_deg)
    m = -1
    if np.sin(theta_rad) == 0: return 0
    period = (m * wavelength_nm) / (n_out * np.sin(theta_rad))
    return abs(period)

# --- 4. Main Application ---

st.title("🌈 P2: Rainbow Surrogate & DOE")

# Load Data & Identify Winner
doe_df = load_doe_results()
best_neurons = 64 # Default
if doe_df is not None:
    # Find row with minimum loss
    best_run = doe_df.loc[doe_df['Final_Test_Loss'].idxmin()]
    best_neurons = best_run['Neurons']
    best_loss = best_run['Final_Test_Loss']
else:
    st.warning("DOE Results not found. Using defaults.")

# Tabs for Engineering vs Research
tab1, tab2 = st.tabs(["🛠️ Engineering (Rainbow Surrogate)", "🔬 Research (DOE Lab)"])

# --- TAB 1: ENGINEERING ---
with tab1:
    st.markdown("### Multi-Objective Design")
    st.markdown("Design a grating for **Green (550nm)** and see how **Red** and **Blue** behave.")

    col_in, col_viz = st.columns([1, 2])

    with col_in:
        st.subheader("1. Design Inputs")
        target_angle = st.slider("Target Angle (deg)", -80.0, -30.0, -50.0)

        # Load Model & Scalers
        model = load_best_model(best_neurons)
        scalers = load_scalers()

        # AI Prediction Logic
        if model and scalers:
            # Prepare Input: Scale [Angle, Wavelength]
            # Note: We need to replicate the exact scaling logic from training
            # Let's assume standard MinMax: (Val - Min) / (Max - Min)

            # Helper to scale single input
            def scale_input(val, col_name):
                mn = scalers['inputs'][col_name]['min']
                mx = scalers['inputs'][col_name]['max']
                return (val - mn) / (mx - mn)

            # Predict for GREEN (550)
            wl_green_norm = scale_input(550, 'Wavelength_nm')
            ang_norm = scale_input(target_angle, 'Target_Angle')

            input_tensor = torch.tensor([[ang_norm, wl_green_norm]], dtype=torch.float32)

            with torch.no_grad():
                pred_norm = model(input_tensor).item()

            # Unscale Output
            p_min = scalers['output']['Period_nm']['min']
            p_max = scalers['output']['Period_nm']['max']
            ai_period = pred_norm * (p_max - p_min) + p_min

            st.success(f"🤖 AI Suggests Period: **{ai_period:.2f} nm**")

            # Physics Truth Check
            phys_period = physics_grating_equation(target_angle, 550)
            err = abs(ai_period - phys_period)
            st.caption(f"Physics Truth: {phys_period:.2f} nm (Error: {err:.3f} nm)")

        else:
            st.warning("Model or Scalers missing. Using Physics fallback.")
            ai_period = physics_grating_equation(target_angle, 550)

    with col_viz:
        st.subheader("2. Chromatic Dispersion Preview")

        # We now have the Period (ai_period).
        # Let's see where R, G, B light actually goes with this grating.
        # Grating Eq Rearranged for Angle: sin(theta) = m*lambda / (n*Period)

        def get_angle(wl, period):
            val = (-1 * wl) / (1.5 * period)
            # Clip for arcsin safety
            val = max(-1, min(1, val))
            return np.degrees(np.arcsin(val))

        ang_R = get_angle(650, ai_period) # Red
        ang_G = get_angle(550, ai_period) # Green
        ang_B = get_angle(450, ai_period) # Blue

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 4))

        # Draw the Waveguide
        ax.axhline(0, color='gray', linewidth=4, alpha=0.3, label="Waveguide")

        # Draw Target
        ax.axvline(target_angle, color='black', linestyle='--', alpha=0.5, label="Target Angle")

        # Draw Rays
        y_origin = 0

        # Red Ray
        ax.plot([0, ang_R], [0, -1], color='red', linewidth=2, label=f"Red (650nm): {ang_R:.1f}°")
        ax.scatter([ang_R], [-1], color='red', s=100)

        # Green Ray
        ax.plot([0, ang_G], [0, -1], color='green', linewidth=3, label=f"Green (550nm): {ang_G:.1f}°")
        ax.scatter([ang_G], [-1], color='green', s=100, zorder=5)

        # Blue Ray
        ax.plot([0, ang_B], [0, -1], color='blue', linewidth=2, label=f"Blue (450nm): {ang_B:.1f}°")
        ax.scatter([ang_B], [-1], color='blue', s=100)

        ax.set_xlim(-90, -20)
        ax.set_ylim(-1.2, 0.2)
        ax.set_yticks([])
        ax.set_xlabel("Output Angle (degrees)")
        ax.set_title(f"Dispersion using AI-Designed Grating ($\Lambda$ = {ai_period:.1f} nm)")
        ax.legend(loc='upper left')
        st.pyplot(fig)

# --- TAB 2: RESEARCH (DOE) ---
with tab2:
    st.markdown("### 🔬 Experiment Results")
    st.markdown(f"**Winner:** The best model used **{int(best_neurons)} neurons** with a loss of **{best_loss:.6f}**.")

    if doe_df is not None:
        # 1. Table
        st.dataframe(doe_df.style.highlight_min(subset=['Final_Test_Loss'], color='#d6f5d6'), use_container_width=True)

        # 2. Interactive Charts
        col_charts1, col_charts2 = st.columns(2)

        with col_charts1:
            st.caption("Impact of Model Size (Neurons) on Accuracy")
            st.scatter_chart(
                doe_df,
                x='Neurons',
                y='Final_Test_Loss',
                color='Size', # Color by dataset size
                size='Epochs'
            )

        with col_charts2:
            st.caption("Training Time vs Data Size")
            st.line_chart(
                doe_df,
                x='Size',
                y='Training_Time_Sec',
                color='Neurons'
            )
    else:
        st.error("Load the 'p2_doe_results.csv' file to see charts.")
