"""
P2 AR Waveguide Dataset Generator
Physics-based simulation for grating-based AR waveguide displays.

Physics: Grating Equation (inside waveguide medium)
n * sin(theta_out) = n * sin(theta_in) + (m * wavelength) / Period

Where:
- n: refractive index at given wavelength (calculated via Sellmeier)
- theta_in: incident angle (input)
- theta_out: diffracted angle (target output)
- m: diffraction order (-1 for outcoupling)
- wavelength: RGB range 400-700nm
- Period: grating period (300-600nm)
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List


# Sellmeier coefficients for AR optical materials
# Format: n^2 = 1 + Σ(B_i * λ^2) / (λ^2 - C_i)
# Wavelength λ in micrometers
SELLMEIER_COEFFICIENTS: Dict[str, Dict[str, List[float]]] = {
    "N-BK7": {  # Standard optical glass (n ≈ 1.52)
        "B": [1.03961212, 0.231792344, 1.01046945],
        "C": [0.00600069867, 0.0200179144, 103.560653],
    },
    "Polycarbonate": {  # Common AR plastic (n ≈ 1.58)
        # Approximate coefficients for PC
        "B": [1.1948, 0.0, 0.0],
        "C": [0.0093, 0.0, 0.0],
    },
    "N-SF11": {  # High-index glass (n ≈ 1.78)
        "B": [1.73759695, 0.313747346, 1.89878101],
        "C": [0.013188707, 0.0623068142, 155.23629],
    },
    "TiO2": {  # Very high-index material (n ≈ 2.2-2.4)
        # Simplified model for Titanium Dioxide
        "B": [5.913, 0.2441, 0.0],
        "C": [0.0803, 0.0, 0.0],
    },
}

# Material ID mapping
MATERIAL_IDS = {
    "N-BK7": 0,
    "Polycarbonate": 1,
    "N-SF11": 2,
    "TiO2": 3,
}

# Generation parameters
N_SAMPLES = 100000
WAVELENGTH_MIN = 400.0  # nm (blue)
WAVELENGTH_MAX = 700.0  # nm (red)
INCIDENT_ANGLE_MIN = -30.0  # degrees
INCIDENT_ANGLE_MAX = 30.0   # degrees
GRATING_PERIOD_MIN = 300.0  # nm
GRATING_PERIOD_MAX = 600.0  # nm
DIFFRACTION_ORDER = -1      # Outcoupling grating


def sellmeier_equation(wavelength_nm: float, material: str) -> float:
    """
    Calculate refractive index using Sellmeier equation.

    Args:
        wavelength_nm: Wavelength in nanometers
        material: Material name (N-BK7, Polycarbonate, N-SF11, TiO2)

    Returns:
        Refractive index at the given wavelength
    """
    if material not in SELLMEIER_COEFFICIENTS:
        raise ValueError(f"Unknown material: {material}. Available: {list(SELLMEIER_COEFFICIENTS.keys())}")

    # Convert wavelength from nm to micrometers for Sellmeier equation
    wavelength_um = wavelength_nm / 1000.0
    lambda_sq = wavelength_um ** 2

    # Get Sellmeier coefficients
    B = SELLMEIER_COEFFICIENTS[material]["B"]
    C = SELLMEIER_COEFFICIENTS[material]["C"]

    # Calculate n^2 using 3-term Sellmeier equation
    # n^2 = 1 + Σ(B_i * λ^2) / (λ^2 - C_i)
    n_squared = 1.0
    for i in range(len(B)):
        if B[i] != 0.0:  # Skip zero terms
            n_squared += (B[i] * lambda_sq) / (lambda_sq - C[i])

    return np.sqrt(n_squared)


def calculate_diffracted_angle(
    wavelength_nm: float,
    incident_angle_deg: float,
    grating_period_nm: float,
    refractive_index: float,
    diffraction_order: int = DIFFRACTION_ORDER
) -> float:
    """
    Calculate diffracted angle using the grating equation.

    Grating equation (in medium):
    n * sin(theta_out) = n * sin(theta_in) + m * λ / P

    Simplifies to:
    sin(theta_out) = sin(theta_in) + (m * λ) / (n * P)

    Args:
        wavelength_nm: Wavelength in nanometers
        incident_angle_deg: Incident angle in degrees
        grating_period_nm: Grating period in nanometers
        refractive_index: Refractive index of waveguide material
        diffraction_order: Diffraction order (default -1 for outcoupling)

    Returns:
        Diffracted angle in degrees (NaN if unphysical)
    """
    # Convert incident angle to radians
    theta_in_rad = np.radians(incident_angle_deg)

    # Grating equation: sin(theta_out) = sin(theta_in) + (m * λ) / (n * P)
    sin_theta_out = np.sin(theta_in_rad) + (diffraction_order * wavelength_nm) / (refractive_index * grating_period_nm)

    # Check if result is physical (|sin| <= 1)
    if abs(sin_theta_out) > 1.0:
        return np.nan

    # Calculate diffracted angle
    theta_out_rad = np.arcsin(sin_theta_out)
    theta_out_deg = np.degrees(theta_out_rad)

    return theta_out_deg


def generate_ar_dataset():
    """
    Generate AR waveguide dataset with physics-based grating simulation.

    Returns:
        DataFrame with columns: [wavelength, incident_angle, grating_period,
                                 refractive_index, material_id, diffracted_angle]
    """

    print(f"Generating {N_SAMPLES:,} AR waveguide samples...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Pre-allocate arrays for efficiency
    wavelengths = np.zeros(N_SAMPLES)
    incident_angles = np.zeros(N_SAMPLES)
    grating_periods = np.zeros(N_SAMPLES)
    refractive_indices = np.zeros(N_SAMPLES)
    material_ids = np.zeros(N_SAMPLES, dtype=int)
    diffracted_angles = np.zeros(N_SAMPLES)

    material_names = list(SELLMEIER_COEFFICIENTS.keys())

    # Generate samples
    for i in range(N_SAMPLES):
        # 1. Randomly select material
        material = np.random.choice(material_names)
        material_id = MATERIAL_IDS[material]

        # 2. Randomly select wavelength (RGB range)
        wavelength = np.random.uniform(WAVELENGTH_MIN, WAVELENGTH_MAX)

        # 3. Calculate refractive index for this wavelength and material
        n = sellmeier_equation(wavelength, material)

        # 4. Randomly select incident angle
        incident_angle = np.random.uniform(INCIDENT_ANGLE_MIN, INCIDENT_ANGLE_MAX)

        # 5. Randomly select grating period
        grating_period = np.random.uniform(GRATING_PERIOD_MIN, GRATING_PERIOD_MAX)

        # 6. Calculate diffracted angle using grating equation
        diffracted_angle = calculate_diffracted_angle(
            wavelength, incident_angle, grating_period, n, DIFFRACTION_ORDER
        )

        # Store results
        wavelengths[i] = wavelength
        incident_angles[i] = incident_angle
        grating_periods[i] = grating_period
        refractive_indices[i] = n
        material_ids[i] = material_id
        diffracted_angles[i] = diffracted_angle

        # Progress indicator
        if (i + 1) % 20000 == 0:
            print(f"  Generated {i + 1:,}/{N_SAMPLES:,} samples...")

    # Create DataFrame
    df = pd.DataFrame({
        'wavelength': wavelengths,
        'incident_angle': incident_angles,
        'grating_period': grating_periods,
        'refractive_index': refractive_indices,
        'material_id': material_ids,
        'diffracted_angle': diffracted_angles
    })

    # Filter out unphysical results (NaN diffracted angles)
    initial_count = len(df)
    df = df.dropna()
    filtered_count = initial_count - len(df)

    if filtered_count > 0:
        print(f"\nFiltered out {filtered_count:,} unphysical samples (|sin(theta)| > 1)")
        print(f"Valid samples: {len(df):,}")

    return df


def print_material_properties():
    """Print refractive index properties for all materials at RGB wavelengths."""
    print("\nMaterial Properties at RGB Wavelengths:")
    print("=" * 70)

    rgb_wavelengths = {
        "Red (635nm)": 635.0,
        "Green (532nm)": 532.0,
        "Blue (450nm)": 450.0,
    }

    for material in SELLMEIER_COEFFICIENTS.keys():
        print(f"\n{material} (ID: {MATERIAL_IDS[material]}):")
        for color, wavelength in rgb_wavelengths.items():
            n = sellmeier_equation(wavelength, material)
            print(f"  {color}: n = {n:.4f}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("P2 AR Waveguide Dataset Generator")
    print("=" * 70)
    print(f"\nPhysics: Grating Equation (in waveguide medium)")
    print(f"  n * sin(theta_out) = n * sin(theta_in) + (m * lambda) / P")
    print(f"\nParameters:")
    print(f"  Samples: {N_SAMPLES:,}")
    print(f"  Wavelength range: {WAVELENGTH_MIN}-{WAVELENGTH_MAX} nm (RGB)")
    print(f"  Incident angle range: {INCIDENT_ANGLE_MIN} to {INCIDENT_ANGLE_MAX} deg")
    print(f"  Grating period range: {GRATING_PERIOD_MIN}-{GRATING_PERIOD_MAX} nm")
    print(f"  Diffraction order: {DIFFRACTION_ORDER} (outcoupling)")
    print(f"  Materials: {len(SELLMEIER_COEFFICIENTS)}")

    # Print material properties
    print_material_properties()

    print("\n" + "-" * 70)

    # Generate dataset
    df = generate_ar_dataset()

    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Save to CSV
    output_path = os.path.join(data_dir, 'p2_ar_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")

    # Print statistics
    print("\n" + "=" * 70)
    print("Dataset Statistics:")
    print("=" * 70)
    print(f"\nShape: {df.shape}")
    print(f"\nColumn ranges:")
    for col in df.columns:
        print(f"  {col}:")
        print(f"    Min: {df[col].min():.4f}")
        print(f"    Max: {df[col].max():.4f}")
        print(f"    Mean: {df[col].mean():.4f}")

    print("\n" + "=" * 70)
    print("Generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
