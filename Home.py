# -*- coding: utf-8 -*-
"""
Simanova: 10 Days, 10 Engines - The Physics-AI Sprint
System Architect Dashboard
"""
import streamlit as st
from pathlib import Path
import os

# Get the directory where this script is located
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Simanova | Physics-AI Sprint",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM CSS - DARK MODE AESTHETIC
# =============================================================================
st.markdown("""
<style>
    /* Dark mode base */
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #121218 100%);
    }

    /* Hero section */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.25rem;
        color: #a0aec0;
        text-align: center;
        font-weight: 300;
        letter-spacing: 0.05em;
        margin-bottom: 2rem;
    }

    /* Manifesto */
    .manifesto-box {
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.05));
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
    }

    .manifesto-text {
        color: #e2e8f0;
        font-size: 1.1rem;
        line-height: 1.8;
        text-align: center;
    }

    /* Project cards */
    .project-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16161a 100%);
        border: 1px solid #2d2d44;
        border-radius: 16px;
        padding: 1.5rem;
        height: 200px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .project-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }

    .project-card-live {
        background: linear-gradient(145deg, #1a2e1a 0%, #162016 100%);
        border: 1px solid #2ecc71;
    }

    .project-card-live:hover {
        border-color: #27ae60;
        box-shadow: 0 8px 30px rgba(46, 204, 113, 0.2);
    }

    .project-card-locked {
        background: linear-gradient(145deg, #1a1a1a 0%, #0d0d0d 100%);
        border: 1px solid #333;
        opacity: 0.7;
    }

    .card-number {
        font-size: 0.75rem;
        color: #667eea;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .card-number-live {
        color: #2ecc71;
    }

    .card-number-locked {
        color: #666;
    }

    .card-title {
        font-size: 1.1rem;
        color: #f7fafc;
        font-weight: 600;
        margin: 0.5rem 0;
    }

    .card-title-locked {
        color: #666;
    }

    .card-desc {
        font-size: 0.85rem;
        color: #a0aec0;
        line-height: 1.5;
    }

    .card-desc-locked {
        color: #555;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .status-live {
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }

    .status-locked {
        background: rgba(100, 100, 100, 0.2);
        color: #666;
        border: 1px solid rgba(100, 100, 100, 0.3);
    }

    /* Tech stack */
    .tech-icon {
        text-align: center;
        padding: 1rem;
    }

    .tech-icon img {
        width: 48px;
        height: 48px;
        margin-bottom: 0.5rem;
    }

    .tech-name {
        color: #a0aec0;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Section headers */
    .section-header {
        color: #667eea;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #2d2d44, transparent);
        margin: 3rem 0;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HERO SECTION
# =============================================================================
st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

# Hero image - Neural sphere visualization (compact)
hero_image_path = SCRIPT_DIR / "assets" / "hero_neural_sphere.png"
col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    st.image(str(hero_image_path), use_container_width=True)

st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

st.markdown("""
<h1 class="hero-title">10 Days. 10 Physics-AI Engines.</h1>
<p class="hero-subtitle">Neural Surrogates for Hardware Design | Differentiable Physics at Scale</p>
""", unsafe_allow_html=True)

# =============================================================================
# THE MANIFESTO
# =============================================================================
st.markdown("""
<div class="manifesto-box">
    <p class="manifesto-text">
        A proof-of-concept applying the <strong>Universal Approximation Theorem</strong> to real-world physics simulation.
        Each engine is a <strong>differentiable neural surrogate</strong>—trained on high-fidelity simulation data—capable of
        replacing months of analytical derivation across <strong>AR/VR · Photonics · MEMS · CFD · Aerodynamics</strong>.
        The thesis: learned representations will surpass hand-crafted physics models in speed, accuracy, and generalization.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# =============================================================================
# THE INTERACTIVE ROADMAP
# =============================================================================
st.markdown("<p class='section-header'>The Roadmap</p>", unsafe_allow_html=True)

# Project definitions
projects = [
    {"id": "P1", "title": "Inverse Waveguide Design", "desc": "Mono-objective Neural Surrogate for optimizing grating periods in AR waveguides.", "status": "live"},
    {"id": "P2", "title": "Rainbow Surrogate", "desc": "Multi-objective chromatic dispersion correction for see-through optics.", "status": "live"},
    {"id": "P3", "title": "Waveguide Uniformity", "desc": "Spatial intensity optimization across the eyebox field of view.", "status": "locked"},
    {"id": "P4", "title": "Thermal PINN", "desc": "Physics-Informed Neural Network for thermal management in photonics.", "status": "locked"},
    {"id": "P5", "title": "Diffractive Lens Design", "desc": "Inverse design of meta-surfaces for wavefront shaping.", "status": "locked"},
    {"id": "P6", "title": "Spectral Encoder", "desc": "Autoencoder for spectral response compression and reconstruction.", "status": "locked"},
    {"id": "P7", "title": "Tolerance Predictor", "desc": "Manufacturing tolerance impact prediction via ensemble models.", "status": "locked"},
    {"id": "P8", "title": "Multi-Physics Fusion", "desc": "Joint optical-thermal-mechanical co-optimization engine.", "status": "locked"},
    {"id": "P9", "title": "Real-Time Inference", "desc": "Edge-deployed surrogate for in-line process control.", "status": "locked"},
    {"id": "P10", "title": "Foundation Model", "desc": "Pre-trained photonics transformer for few-shot adaptation.", "status": "locked"},
]

# Create 2x5 grid
for row in range(2):
    cols = st.columns(5, gap="medium")
    for col_idx, col in enumerate(cols):
        project_idx = row * 5 + col_idx
        project = projects[project_idx]

        with col:
            if project["status"] == "live":
                st.markdown(f"""
                <div class="project-card project-card-live">
                    <span class="card-number card-number-live">{project['id']}</span>
                    <span class="status-badge status-live">LIVE</span>
                    <h3 class="card-title">{project['title']}</h3>
                    <p class="card-desc">{project['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Launch Engine", key=f"btn_{project['id']}", use_container_width=True):
                    page_map = {
                        "P1": "pages/01_P1_WG_DOE.py",
                        "P2": "pages/02_P2_Rainbow_Solver.py",
                    }
                    st.switch_page(page_map.get(project['id'], "pages/01_P1_WG_DOE.py"))
            else:
                st.markdown(f"""
                <div class="project-card project-card-locked">
                    <span class="card-number card-number-locked">{project['id']}</span>
                    <span class="status-badge status-locked">LOCKED</span>
                    <h3 class="card-title card-title-locked">{project['title']}</h3>
                    <p class="card-desc card-desc-locked">{project['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# =============================================================================
# THE TECH STACK
# =============================================================================
st.markdown("<p class='section-header'>Tech Stack</p>", unsafe_allow_html=True)

tech_cols = st.columns(4, gap="large")

tech_stack = [
    {"name": "Gemini", "icon": "https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg", "desc": "AI Reasoning"},
    {"name": "Claude", "icon": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Anthropic_logo.svg/1200px-Anthropic_logo.svg.png", "desc": "Code Generation"},
    {"name": "PyTorch", "icon": "https://pytorch.org/assets/images/pytorch-logo.png", "desc": "Neural Networks"},
    {"name": "Streamlit", "icon": "https://streamlit.io/images/brand/streamlit-mark-color.svg", "desc": "Deployment"},
]

for col, tech in zip(tech_cols, tech_stack):
    with col:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem; background: rgba(26, 26, 46, 0.5); border-radius: 12px; border: 1px solid #2d2d44;">
            <img src="{tech['icon']}" style="width: 48px; height: 48px; margin-bottom: 0.75rem; filter: grayscale(20%);" onerror="this.style.display='none'">
            <p style="color: #f7fafc; font-weight: 600; margin: 0;">{tech['name']}</p>
            <p style="color: #667eea; font-size: 0.75rem; margin: 0;">{tech['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <p style="color: #a0aec0; font-size: 0.9rem; margin-bottom: 0.5rem;">
        Built by <strong style="color: #FFFFFF;">Vaibhav Mathur</strong>
    </p>
    <p style="color: #4a5568; font-size: 0.85rem; margin: 0;">
        <a href="https://x.com/vaibhavmathur91" target="_blank" style="color: #667eea; text-decoration: none; margin-right: 1rem;">X (Twitter)</a>
        <a href="https://linkedin.com/in/vaibhavmathur91" target="_blank" style="color: #667eea; text-decoration: none; margin-right: 1rem;">LinkedIn</a>
        <a href="https://github.com/vaibhavmathur171-git/simanova-models" target="_blank" style="color: #667eea; text-decoration: none;">GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)
