# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(
    page_title="Simanova Portfolio",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Simanova: AI-Powered Optical Engineering")
st.markdown("""
### Welcome to the Neural Physics Portfolio

This platform demonstrates the application of **Artificial Intelligence** to solve complex problems in **Optical Engineering** and **Nanophotonics**.

#### 📂 Live Projects
* **👈 Select a project from the sidebar to begin.**
* **01 P1: Inverse Waveguide Design:** A Mono-objective Neural Surrogate for optimizing grating periods.
* **02 P2: Rainbow Surrogate:** A Multi-objective solution for chromatic dispersion correction in AR glasses.

#### 🛠 Tech Stack
* **Core:** Python, PyTorch (Neural Networks)
* **Physics:** RCWA, FDTD (Ground Truth Data)
* **Deploy:** Streamlit Cloud, GitHub Actions
""")

st.info("💡 **Navigation Tip:** Use the sidebar on the left to switch between different physics solvers.")