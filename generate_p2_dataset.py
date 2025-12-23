"""
AR Glasses Diffractive Waveguide Dataset Generator
=====================================================
Generates 100,000 synthetic datapoints for neural surrogate training.

Physics: Sellmeier Equation + Grating Equation
Materials: AR-grade optical materials
  - N-BK7: Standard crown glass (n ≈ 1.52)
  - S-LAH79: High-index lanthanum glass (n ≈ 2.00)
  - LiNbO3: Lithium niobate (n ≈ 2.20-2.29)
  - TiO2: Titanium dioxide high-index coating (n ≈ 2.4-2.6)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# 1. MATERIALS DATABASE (Sellmeier Coefficients)
# ============================================================================

MATERIALS = {
    'N-BK7': {
        # Standard borosilicate crown glass (Schott) - n ≈ 1.52
        'B1': 1.03961212,
        'B2': 0.231792344,
        'B3': 1.01046945,
        'C1': 0.00600069867,  # (μm²)
        'C2': 0.0200179144,   # (μm²)
        'C3': 103.560653,     # (μm²)
    },
    'S-LAH79': {
        # High-index lanthanum glass (Ohara/Schott) - n ≈ 2.00
        # Used in AR waveguides for high refractive index
        'B1': 2.4206623,
        'B2': 0.3502465,
        'B3': 1.5186460,
        'C1': 0.0134871,      # (μm²)
        'C2': 0.0591074,      # (μm²)
        'C3': 147.4887,       # (μm²)
    },
    'LiNbO3': {
        # Lithium Niobate (ordinary axis) - n ≈ 2.20-2.29
        # Electro-optic material for tunable AR gratings
        'B1': 2.6734,
        'B2': 1.2290,
        'B3': 12.614,
        'C1': 0.01764,        # (μm²)
        'C2': 0.05914,        # (μm²)
        'C3': 474.6,          # (μm²)
    },
    'TiO2': {
        # Titanium Dioxide thin film - n ≈ 2.4-2.6
        # High-index AR coating material
        'B1': 5.913,
        'B2': 0.2441,
        'B3': 0.0,
        'C1': 0.0803,         # (μm²)
        'C2': 0.0,
        'C3': 0.0,
    }
}

# ============================================================================
# 2. PHYSICS: SELLMEIER EQUATION
# ============================================================================

def get_refractive_index(wavelength_nm, material_name):
    """
    Calculate refractive index using Sellmeier equation.

    n²(λ) = 1 + (B1·λ² / (λ² - C1)) + (B2·λ² / (λ² - C2)) + (B3·λ² / (λ² - C3))

    Parameters:
    -----------
    wavelength_nm : float
        Wavelength in nanometers (e.g., 550 nm)
    material_name : str
        Material key from MATERIALS dict

    Returns:
    --------
    n : float
        Refractive index at specified wavelength
    """
    mat = MATERIALS[material_name]
    wl_um = wavelength_nm / 1000.0  # Convert nm to μm
    wl_um_sq = wl_um ** 2

    # Standard Sellmeier equation
    n_squared = 1.0
    n_squared += (mat['B1'] * wl_um_sq) / (wl_um_sq - mat['C1'])
    n_squared += (mat['B2'] * wl_um_sq) / (wl_um_sq - mat['C2'])
    n_squared += (mat['B3'] * wl_um_sq) / (wl_um_sq - mat['C3'])

    return np.sqrt(n_squared)

# ============================================================================
# 3. PHYSICS: GRATING EQUATION
# ============================================================================

def calculate_diffracted_angle(wavelength_nm, incident_angle_deg, period_nm, n):
    """
    Calculate diffracted angle using the grating equation.

    n·sin(θ_out) = n_in·sin(θ_in) - m·λ/Λ

    For m=1 (first order), n_in=1 (air):
    sin(θ_out) = sin(θ_in) - λ/Λ

    Parameters:
    -----------
    wavelength_nm : float
        Wavelength in nanometers
    incident_angle_deg : float
        Incident angle in degrees
    period_nm : float
        Grating period in nanometers
    n : float
        Refractive index of waveguide material

    Returns:
    --------
    theta_out_deg : float or None
        Diffracted angle in degrees (None if TIR/impossible)
    """
    theta_in_rad = np.radians(incident_angle_deg)

    # Grating equation (m=1, n_in=1 for air)
    sin_theta_out = np.sin(theta_in_rad) - (wavelength_nm / period_nm)

    # Check for total internal reflection or physical impossibility
    if abs(sin_theta_out) > 1.0:
        return None

    theta_out_rad = np.arcsin(sin_theta_out)
    return np.degrees(theta_out_rad)

# ============================================================================
# 4. DATASET GENERATION
# ============================================================================

def generate_dataset(num_samples=100000, seed=42):
    """
    Generate synthetic AR waveguide dataset.

    Parameters:
    -----------
    num_samples : int
        Target number of valid datapoints
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    df : pd.DataFrame
        Dataset with columns: [wavelength, incident_angle, period,
                               refractive_index, material_name, diffracted_angle]
    """
    np.random.seed(seed)

    material_names = list(MATERIALS.keys())
    data = []
    attempts = 0
    max_attempts = num_samples * 5  # Higher safety limit due to ~71% rejection

    print(f"Generating {num_samples:,} valid datapoints...")
    print("=" * 60)

    while len(data) < num_samples and attempts < max_attempts:
        # Random sampling
        material = np.random.choice(material_names)
        wavelength = np.random.uniform(400, 700)      # nm (visible spectrum)
        incident_angle = np.random.uniform(-30, 30)   # degrees
        period = np.random.uniform(300, 600)          # nm (typical diffractive)

        # Calculate physics
        n = get_refractive_index(wavelength, material)
        theta_out = calculate_diffracted_angle(wavelength, incident_angle, period, n)

        # Filter: Keep only physically valid solutions
        if theta_out is not None:
            data.append({
                'wavelength': wavelength,
                'incident_angle': incident_angle,
                'period': period,
                'refractive_index': n,
                'material_name': material,
                'diffracted_angle': theta_out
            })

        attempts += 1

        # Progress indicator
        if len(data) % 10000 == 0 and len(data) > 0:
            print(f"[OK] {len(data):,} valid samples generated "
                  f"({attempts - len(data):,} filtered)")

    print("=" * 60)
    print(f"[OK] Complete: {len(data):,} samples generated")
    print(f"  Rejection rate: {100 * (1 - len(data)/attempts):.2f}%")

    df = pd.DataFrame(data)
    return df

# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def create_visualization(df, output_path):
    """
    Create scatter plot: Wavelength vs Diffracted Angle (colored by Material).
    """
    plt.figure(figsize=(12, 7))

    materials = df['material_name'].unique()
    colors = {
        'N-BK7': '#1f77b4',      # Blue - standard glass
        'S-LAH79': '#ff7f0e',    # Orange - high-index glass
        'LiNbO3': '#2ca02c',     # Green - lithium niobate
        'TiO2': '#d62728'        # Red - titanium dioxide
    }

    for material in materials:
        subset = df[df['material_name'] == material]
        plt.scatter(subset['wavelength'], subset['diffracted_angle'],
                   alpha=0.3, s=1, label=material, color=colors.get(material))

    plt.xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    plt.ylabel('Diffracted Angle (degrees)', fontsize=12, fontweight='bold')
    plt.title('AR Waveguide Physics: Wavelength vs Diffracted Angle\n'
              'Grating Equation + Sellmeier Dispersion',
              fontsize=14, fontweight='bold')
    plt.legend(title='Material', loc='best', framealpha=0.9)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Visualization saved: {output_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Generate dataset
    df = generate_dataset(num_samples=100000, seed=42)

    # Save to CSV
    csv_path = output_dir / "p2_ar_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Dataset saved: {csv_path}")
    print(f"  Shape: {df.shape}")
    print(f"\n  Dataset statistics:")
    print(df.describe())

    # Material distribution
    print(f"\n  Material distribution:")
    print(df['material_name'].value_counts())

    # Create visualization
    viz_path = output_dir / "data_viz.png"
    create_visualization(df, viz_path)

    print("\n" + "=" * 60)
    print("[OK] DATASET GENERATION COMPLETE")
    print("=" * 60)
