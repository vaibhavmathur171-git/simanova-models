# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px  # NEW: For 3D plotting
import os

# --- 1. Page Config ---
st.set_page_config(
    page_title="P2: Rainbow Surrogate",
    page_icon="🌈",
    layout="wide"
)

# --- 2. Robust Model Architecture ---
class FlexibleMLP(nn.Module):
    def __init__(self, hidden_neurons=64):
        super(FlexibleMLP, self).__init__()
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
    model = FlexibleMLP(hidden_neurons=int(neurons))
    if not os.path.exists(path):
        return None
    try:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        new_state_dict = {}
        model_keys = list(model.state_dict().keys())
        for i, (k_saved, v_saved) in enumerate(state_dict.items()):
            if i < len(model_keys):
                new_state_dict[model_keys[i]] = v_saved
        model.load_state_dict(new_state_dict)
        model.eval()
        return model
    except Exception as e:
        return None

# --- 3. Helper Functions ---
@st.cache_data
def load_doe_data():
    paths = ['data/p2_doe_results.csv', 'Data/p2_doe_results.csv']
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

def physics_equation(target_angle, wavelength_nm):
    theta_rad = np.radians(target_angle)
    val = np.sin(theta_rad)
    if val == 0: return 0
    return abs((-1 * wavelength_nm) / (1.5 * val))

def get_output_angle(period, wavelength_nm):
    val = (-1 * wavelength_nm) / (1.5 * period)
    val = np.clip(val, -1, 1)
    return np.degrees(np.arcsin(val))

# --- 4. Main App ---
st.title("🌈 P2: Rainbow Surrogate & DOE")

doe_df = load_doe_data()
best_neurons = 64
if doe_df is not None:
    best_row = doe_df.loc[doe_df['Final_Test_Loss'].idxmin()]
    best_neurons = best_row['Neurons']

tab1, tab2 = st.tabs(["🛠️ Engineering Design (Rainbow)", "🔬 Research Results (DOE)"])

# --- TAB 1: ENGINEERING (Unchanged) ---
with tab1:
    col_input, col_viz = st.columns([1, 2])
    with col_input:
        st.subheader("1. Design Parameters")
        st.info("Goal: Optimize Grating for **Green (550nm)**")
        target_angle = st.slider("Target Angle (deg)", -80.0, -30.0, -50.0)
        
        model = load_robust_model('models/p2_rainbow_model.pth', best_neurons)
        if model:
            x = torch.tensor([[ (target_angle+80)/50, (550-400)/300 ]], dtype=torch.float32)
            with torch.no_grad():
                pred = model(x).item()
            ai_period = pred * 800 + 200
            st.success(f"**AI Grating Period:** {ai_period:.1f} nm")
        else:
            ai_period = physics_equation(target_angle, 550)
            st.warning(f"Using Physics Truth: {ai_period:.1f} nm")

    with col_viz:
        st.subheader("2. Physical Ray Diagram")
        ang_R = get_output_angle(ai_period, 650)
        ang_G = get_output_angle(ai_period, 550)
        ang_B = get_output_angle(ai_period, 450)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        rect = patches.Rectangle((-1.5, -0.2), 3, 0.4, linewidth=1, edgecolor='none', facecolor='#e0e0e0')
        ax.add_patch(rect)
        ax.text(-1.4, 0.05, "Waveguide (n=1.5)", fontsize=10, color='gray')
        ax.plot([-1, 1], [0, 0], color='black', linestyle='--', linewidth=1)
        ax.annotate("", xy=(0, 0), xytext=(0, 1), arrowprops=dict(arrowstyle="->", lw=2, color='black'))
        ax.text(0.05, 0.8, "Input Light\n(White)", fontsize=10)
        
        def draw_ray(angle, color, label):
            rad = np.radians(angle)
            L = 1.2
            dx = L * np.sin(rad)
            dy = -L * np.cos(rad)
            ax.plot([0, dx], [0, dy], color=color, linewidth=3, label=label)
            ax.scatter([dx], [dy], color=color, s=100, zorder=5)
            ax.text(dx, dy-0.15, f"{angle:.1f}°", ha='center', color=color, fontsize=11, fontweight='bold')

        draw_ray(ang_R, 'red', 'Red (650nm)')
        draw_ray(ang_G, 'green', 'Green (Target)')
        draw_ray(ang_B, 'blue', 'Blue (450nm)')
        
        t_rad = np.radians(target_angle)
        tx, ty = 1.2 * np.sin(t_rad), -1.2 * np.cos(t_rad)
        ax.plot([0, tx], [0, ty], 'k:', alpha=0.5, linewidth=1)
        ax.text(tx, ty+0.3, "Target", fontsize=9, rotation=0, color='black')

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.2)
        ax.axis('off')
        ax.legend(loc='upper right')
        st.pyplot(fig)

# --- TAB 2: DOE RESULTS (The Major Upgrade) ---
with tab2:
    st.markdown("### 🔬 Design of Experiments (DOE) Insights")
    
    if doe_df is not None:
        # A. 3D Interaction Plot
        st.subheader("1. The Landscape of Loss (3D)")
        st.caption("Interact: Rotate to see how Model Complexity (Neurons) and Data Scale affect performance.")
        
        # We use Log Loss because loss values can get very tiny
        doe_df['Log_Loss'] = np.log10(doe_df['Final_Test_Loss'])
        
        fig_3d = px.scatter_3d(
            doe_df,
            x='Neurons',
            y='Size',
            z='Log_Loss',
            color='Log_Loss',
            size='Epochs',
            color_continuous_scale='Viridis_r', # Inverted: Bright Yellow = Low Loss (Good)
            hover_data=['Final_Test_Loss', 'Training_Time_Sec'],
            labels={'Log_Loss': 'Log(Loss)', 'Size': 'Dataset Size'},
            title="3D Optimization Surface"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # B. The Primary Trends (Top of Page)
        st.divider()
        st.subheader("2. Primary Trends: What drives performance?")
        
        col_trend1, col_trend2 = st.columns(2)
        
        with col_trend1:
            st.markdown("**📉 Data Size vs. Loss**")
            st.caption("Does adding more data linearly decrease error?")
            # Simple Scatter: X=Size, Y=Loss
            st.scatter_chart(
                doe_df,
                x='Size',
                y='Final_Test_Loss',
                color='Neurons', # Color helps separate model capacities
                size='Epochs'
            )
            
        with col_trend2:
            st.markdown("**📉 Epochs vs. Loss**")
            st.caption("Did we train long enough?")
            # Simple Scatter: X=Epochs, Y=Loss
            st.scatter_chart(
                doe_df,
                x='Epochs',
                y='Final_Test_Loss',
                color='Size',
                size='Neurons'
            )

        # C. Secondary Metrics (Moved Down)
        st.divider()
        with st.expander("See Secondary Metrics (Speed & Architecture)"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Brain Size vs. Accuracy**")
                grouped = doe_df.groupby('Neurons')['Final_Test_Loss'].mean()
                st.bar_chart(grouped)
            with c2:
                st.markdown("**Training Time Impact**")
                subset = doe_df[doe_df['Neurons'] == best_neurons]
                st.line_chart(data=subset, x='Size', y='Training_Time_Sec')
                
            st.dataframe(doe_df, use_container_width=True)
        
    else:
        st.warning("Data not found. Please ensure 'p2_doe_results.csv' is in the data folder.")