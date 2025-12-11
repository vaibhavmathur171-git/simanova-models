"""
P2 Chromatic Dispersion Dataset Generator

Physics: Grating Equation
n_out * sin(theta_out) = n_in * sin(theta_in) + (m * wavelength) / Period

Solving for Period:
Period = (m * wavelength) / (n_out * sin(theta_out) - n_in * sin(theta_in))

With theta_in = 0 (normal incidence):
Period = (m * wavelength) / (n_out * sin(theta_out))
"""

import numpy as np
import pandas as pd
import os

# Constants
N_OUT = 1.5        # Waveguide Index
N_IN = 1.0         # Air Index
M = -1             # Diffraction Order
THETA_IN = 0       # Normal incidence (degrees)

# Generation parameters
N_SAMPLES = 50000
ANGLE_MIN = -30.0  # degrees
ANGLE_MAX = -80.0  # degrees
WAVELENGTH_MIN = 400.0  # nm
WAVELENGTH_MAX = 700.0  # nm

def generate_dataset():
    """Generate chromatic dispersion dataset using the grating equation."""

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random inputs
    target_angles = np.random.uniform(ANGLE_MIN, ANGLE_MAX, N_SAMPLES)  # degrees
    wavelengths = np.random.uniform(WAVELENGTH_MIN, WAVELENGTH_MAX, N_SAMPLES)  # nm

    # Convert angle to radians for calculation
    theta_out_rad = np.deg2rad(target_angles)

    # Calculate Period using the grating equation
    # Period = (m * wavelength) / (n_out * sin(theta_out))
    # Note: with theta_in = 0, the n_in * sin(theta_in) term = 0
    denominator = N_OUT * np.sin(theta_out_rad)

    # Calculate period (wavelength and period both in nm)
    period = (M * wavelengths) / denominator

    # Create DataFrame
    df = pd.DataFrame({
        'Target_Angle': target_angles,
        'Wavelength_nm': wavelengths,
        'Period_nm': period
    })

    # Filter out physical errors
    # Remove rows with NaN, Inf, or zero/negative periods
    initial_count = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df[df['Period_nm'] > 0]  # Period must be positive

    filtered_count = initial_count - len(df)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} invalid samples")

    return df

def main():
    print("=" * 60)
    print("P2 Chromatic Dispersion Dataset Generator")
    print("=" * 60)
    print(f"\nPhysics: Grating Equation")
    print(f"  n_out * sin(theta_out) = n_in * sin(theta_in) + (m * wavelength) / Period")
    print(f"\nConstants:")
    print(f"  n_out = {N_OUT} (Waveguide Index)")
    print(f"  n_in = {N_IN} (Air Index)")
    print(f"  m = {M} (Diffraction Order)")
    print(f"  theta_in = {THETA_IN}° (Normal incidence)")
    print("-" * 60)

    # Generate dataset
    print(f"\nGenerating {N_SAMPLES:,} samples...")
    df = generate_dataset()
    print(f"Valid samples after filtering: {len(df):,}")

    # Create data folder if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Save to CSV
    output_path = os.path.join(data_dir, 'p2_rainbow_data.csv')
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    print(f"\nTarget_Angle (degrees):")
    print(f"  Min: {df['Target_Angle'].min():.4f}")
    print(f"  Max: {df['Target_Angle'].max():.4f}")

    print(f"\nWavelength_nm:")
    print(f"  Min: {df['Wavelength_nm'].min():.4f}")
    print(f"  Max: {df['Wavelength_nm'].max():.4f}")

    print(f"\nPeriod_nm:")
    print(f"  Min: {df['Period_nm'].min():.4f}")
    print(f"  Max: {df['Period_nm'].max():.4f}")

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
