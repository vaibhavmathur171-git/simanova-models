# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import os

# --- 1. Page Config ---
st.set_page_config(
    page_title="P2: Rainbow Surrogate",
    page_icon="🌈",
    layout="wide"
)

# --- 2. EXECUTIVE SUMMARY (NEW) ---
st.title("🌈 P2: Rainbow Surrogate & DOE")

with st.container():
    st.markdown("### 🎯 Project Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.info("**1. The Engineering Goal**\n\nManage **Chromatic Dispersion** (Rainbows) in AR glasses. Design a grating that works for Green but minimizes error for Red and Blue.")
    
    with c2:
        st.info("**2. The Physics**\n\nLight bends differently based on color ($\lambda$).\n**Red bends more, Blue bends less.** This creates a 'fan' of angles that blurs the image.")
    
    with c3:
        st.info("**3. The AI Strategy**\n\n**Multi-Variable Surrogate:** Train a Neural Net on 50,000 ray-traces to instantly predict the 'Rainbow Spread' for any design.")
        
    with c4:
        st.success("**4. The DOE Goal**\n\n**Optimization:** We ran a Design of Experiments to find the smallest, fastest 'Brain' (Neural Net) that still captures the physics accurately.")

st.divider()

# --- 3. Robust Model Architecture ---
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
    if not os.path.exists(path): return None
    try:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        new_state_dict = {}
        model_keys = list(model.state_dict().keys())
        for i, (k_saved, v_saved) in enumerate(state_dict.items()):
            if i < len(model_keys): new_state_dict[model_keys[i]] = v_saved
        model.load_state_dict(new_state_dict)
        model.eval()
        return model
    except: return None

# --- 4. Helper Functions ---
@st.cache_data
def load_doe_data():
    paths = ['data/p2_doe_results.csv', 'Data/p2_doe_results.csv']
    for p in paths:
        if os.path.exists(p): return pd.read_csv(p)
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

# --- 5. Main App Tabs ---
doe_df = load_doe_data()
best_neurons = 64
if doe_df is not None:
    best_row = doe_df.loc[doe_df['Final_Test_Loss'].idxmin()]
    best_neurons = best_row['Neurons']

tab1, tab2 = st.tabs(["🛠️ Engineering Design (Rainbow)", "🔬 Research Results (DOE)"])

# --- TAB 1: ENGINEERING ---
with tab1:
    col_input, col_viz = st.columns([1, 2])
    with col_input:
        st.subheader("1. Design Inputs")
        target_angle = st.slider("Target Angle (deg)", -80.0, -30.0, -50.0)
        model = load_robust_model('models/p2_rainbow_model.pth', best_neurons)
        
        if model:
            x = torch.tensor([[ (target_angle+80)/50, (550-400)/300 ]], dtype=torch.float32)
            with torch.no_grad(): pred = model(x).item()
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

# --- TAB 2: DOE RESULTS ---
with tab2:
    if doe_df is not None:
        # A. SCOPE
        st.markdown("### 📋 1. Experimental Scope (Variables)")
        sizes = sorted(doe_df['Size'].unique())
        neurons = sorted(doe_df['Neurons'].unique())
        epochs = sorted(doe_df['Epochs'].unique())
        
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.info(f"**Data Sizes**\n\n" + ", ".join([f"{x:,}" for x in sizes]))
        with c2: st.info(f"**Neurons**\n\n" + ", ".join(map(str, neurons)))
        with c3: st.info(f"**Epochs**\n\n" + ", ".join(map(str, epochs)))
        with c4: st.metric("Total Runs", len(doe_df))
        st.divider()

        # B. 3D PLOT
        st.subheader("2. The Landscape of Loss (3D)")
        doe_df['Log_Loss'] = np.log10(doe_df['Final_Test_Loss'])
        fig_3d = px.scatter_3d(doe_df, x='Neurons', y='Size', z='Log_Loss', color='Log_Loss', size='Epochs', color_continuous_scale='Viridis_r')
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # C. 2D TRENDS
        st.divider()
        st.subheader("3. Primary Trends")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Data Size Sensitivity (at 50 Epochs)")
            df_t1 = doe_df[doe_df['Epochs'] == 50].sort_values('Size')
            fig1 = px.line(df_t1, x='Size', y='Final_Test_Loss', color='Neurons', markers=True, log_y=True)
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            st.caption("Training Duration Sensitivity (at Max Data)")
            df_t2 = doe_df[doe_df['Size'] == doe_df['Size'].max()].sort_values('Epochs')
            fig2 = px.line(df_t2, x='Epochs', y='Final_Test_Loss', color='Neurons', markers=True, log_y=True)
            st.plotly_chart(fig2, use_container_width=True)
            
    else:
        st.warning("Data not found.")