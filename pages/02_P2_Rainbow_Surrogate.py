# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- 1. Page Config ---
st.set_page_config(
    page_title="P2: Rainbow Surrogate",
    page_icon="🌈",
    layout="wide"
)

# --- 2. Robust Model Architecture ---
class FlexibleMLP(nn.Module):
    """
    A flexible container that can load weights regardless of internal naming.
    """
    def __init__(self, hidden_neurons=64):
        super(FlexibleMLP, self).__init__()
        # We define the structure, but we will load keys manually to avoid mismatch
        self.net = nn.Sequential(
            nn.Linear(2, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, 1)
        )

    def forward(self, x):
        return self.net(x)

def load_robust_model(path, neurons):
    """
    Loads weights even if the variable names (layers vs network) don't match.
    """
    model = FlexibleMLP(hidden_neurons=int(neurons))
    if not os.path.exists(path):
        return None
    
    try:
        # Load the raw dictionary
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        
        # Create a new dict with corrected keys
        new_state_dict = {}
        model_keys = list(model.state_dict().keys())
        
        # We map the saved keys (whatever they are) to our model's keys 1-to-1
        # This works because the LAYERS are the same, just the NAMES differ.
        for i, (k_saved, v_saved) in enumerate(state_dict.items()):
            if i < len(model_keys):
                new_state_dict[model_keys[i]] = v_saved
        
        model.load_state_dict(new_state_dict)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"⚠️ Model Load Warning: {e}")
        return None

# --- 3. Helper Functions ---
@st.cache_data
def load_doe_data():
    """Loads DOE CSV with fallback for folder capitalization."""
    paths = ['data/p2_doe_results.csv', 'Data/p2_doe_results.csv']
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

def physics_equation(target_angle, wavelength_nm):
    """Calculates Period using the Grating Equation."""
    theta_rad = np.radians(target_angle)
    # Grating Eq: Period = lambda / (1.5 * sin(theta))  [Assuming m=-1]
    val = np.sin(theta_rad)
    if val == 0: return 0
    return abs((-1 * wavelength_nm) / (1.5 * val))

def get_output_angle(period, wavelength_nm):
    """Calculates Output Angle given Period."""
    # sin(theta) = -1 * lambda / (1.5 * period)
    val = (-1 * wavelength_nm) / (1.5 * period)
    val = np.clip(val, -1, 1) # Safety clipping
    return np.degrees(np.arcsin(val))

# --- 4. Main App ---
st.title("🌈 P2: Rainbow Surrogate & DOE")

# Load DOE to find Best Model
doe_df = load_doe_data()
best_neurons = 64 # Default
if doe_df is not None:
    best_row = doe_df.loc[doe_df['Final_Test_Loss'].idxmin()]
    best_neurons = best_row['Neurons']

# TABS
tab1, tab2 = st.tabs(["🛠️ Engineering Design (Rainbow)", "🔬 Research Results (DOE)"])

# --- TAB 1: ENGINEERING VISUALIZATION ---
with tab1:
    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.subheader("1. Design Parameters")
        st.info("Goal: Optimize Grating for **Green (550nm)**")
        target_angle = st.slider("Target Angle (deg)", -80.0, -30.0, -50.0)
        
        # Load Model
        model = load_robust_model('models/p2_rainbow_model.pth', best_neurons)
        
        # Predict Period
        if model:
            # Simple normalization (approximate for demo if scalers missing)
            # Ideally use p2_scalers.json here
            x = torch.tensor([[ (target_angle+80)/50, (550-400)/300 ]], dtype=torch.float32)
            with torch.no_grad():
                pred = model(x).item()
            # De-normalize (approx)
            ai_period = pred * 800 + 200 # Approx range
            st.success(f"**AI Grating Period:** {ai_period:.1f} nm")
        else:
            # Fallback to Physics
            ai_period = physics_equation(target_angle, 550)
            st.warning(f"Using Physics Truth: {ai_period:.1f} nm")

    with col_viz:
        st.subheader("2. Physical Ray Diagram")
        
        # Calculate Angles
        ang_R = get_output_angle(ai_period, 650)
        ang_G = get_output_angle(ai_period, 550)
        ang_B = get_output_angle(ai_period, 450)
        
        # DRAW THE "PHYSICS SKETCH"
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # A. The Components
        # Waveguide Slab (Gray Block)
        rect = patches.Rectangle((-1.5, -0.2), 3, 0.4, linewidth=1, edgecolor='none', facecolor='#e0e0e0')
        ax.add_patch(rect)
        ax.text(-1.4, 0.05, "Waveguide (n=1.5)", fontsize=10, color='gray')
        
        # The Grating Surface (Dashed Line)
        ax.plot([-1, 1], [0, 0], color='black', linestyle='--', linewidth=1, label='Grating Surface')
        
        # B. The Rays
        # 1. Input Ray (White/Black) - Coming from top
        ax.annotate("", xy=(0, 0), xytext=(0, 1), arrowprops=dict(arrowstyle="->", lw=2, color='black'))
        ax.text(0.05, 0.8, "Input Light\n(White)", fontsize=10)
        
        # 2. Output Rays (The Rainbow Fan)
        # Helper to draw ray
        def draw_ray(angle, color, label):
            rad = np.radians(angle)
            # Length of ray
            L = 1.2
            dx = L * np.sin(rad)
            dy = -L * np.cos(rad) # Going down
            
            ax.plot([0, dx], [0, dy], color=color, linewidth=3, label=label)
            ax.scatter([dx], [dy], color=color, s=100, zorder=5)
            # Label angle
            ax.text(dx, dy-0.15, f"{angle:.1f}°", ha='center', color=color, fontsize=11, fontweight='bold')

        draw_ray(ang_R, 'red', 'Red (650nm)')
        draw_ray(ang_G, 'green', 'Green (Target)')
        draw_ray(ang_B, 'blue', 'Blue (450nm)')
        
        # C. Target Marker
        # Draw a dashed line where we WANTED Green to go
        t_rad = np.radians(target_angle)
        tx, ty = 1.2 * np.sin(t_rad), -1.2 * np.cos(t_rad)
        ax.plot([0, tx], [0, ty], 'k:', alpha=0.5, linewidth=1)
        ax.text(tx, ty+0.3, "Target", fontsize=9, rotation=0, color='black')

        # Formatting
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.2)
        ax.axis('off') # Hide the boring math axes
        ax.legend(loc='upper right')
        st.pyplot(fig)

# --- TAB 2: DOE RESULTS (Clean & Simple) ---
with tab2:
    st.markdown("### 🔬 Design of Experiments (DOE) Results")
    
    if doe_df is not None:
        # 1. Top Metrics (The "Executive Summary")
        best_run = doe_df.loc[doe_df['Final_Test_Loss'].idxmin()]
        
        m1, m2, m3 = st.columns(3)
        m1.metric("🏆 Best Architecture", f"{int(best_run['Neurons'])} Neurons")
        m2.metric("📉 Lowest Loss", f"{best_run['Final_Test_Loss']:.6f}")
        m3.metric("⏱️ Training Time", f"{best_run['Training_Time_Sec']:.1f} sec")
        
        st.divider()
        
        # 2. The Plots (Simplifying Complex Data)
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("🧠 Brain Size vs. Accuracy")
            st.caption("Lower is better. Does a bigger brain reduce error?")
            
            # Group by Neurons and take the mean Loss
            # This cleans up the scatter plot into a clear trend bar chart
            grouped = doe_df.groupby('Neurons')['Final_Test_Loss'].mean()
            st.bar_chart(grouped)
            
        with c2:
            st.subheader("📚 Data Size vs. Speed")
            st.caption("How much longer does it take to train with more data?")
            
            # Simple Line Chart: Data Size on X, Time on Y
            # We filter for a specific neuron count to keep the line clean
            subset = doe_df[doe_df['Neurons'] == best_neurons]
            st.line_chart(data=subset, x='Size', y='Training_Time_Sec')
            
        st.subheader("📑 Raw Data Table")
        st.dataframe(doe_df, use_container_width=True)
        
    else:
        st.warning("Data not found. Please ensure 'p2_doe_results.csv' is in the data folder.")