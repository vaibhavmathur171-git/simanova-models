import numpy as np
import pandas as pd

# Physics constants
N_OUT = 1.5
N_IN = 1.0
M = -1
THETA_IN = 0.0

# Number of samples
NUM_SAMPLES = 20000

# Seed for reproducibility
np.random.seed(42)

def generate_waveguide_data(num_samples):
    """
    Generate synthetic waveguide grating data using the Grating Equation.

    Grating Equation: Period = (m * lambda) / (n_out * sin(theta_out) - n_in * sin(theta_in))
    """
    data = []

    for _ in range(num_samples):
        # Randomly sample inputs
        target_angle_deg = np.random.uniform(-30.0, -80.0)
        wavelength_nm = np.random.uniform(400.0, 700.0)

        # Convert angle to radians
        theta_out_rad = np.deg2rad(target_angle_deg)
        theta_in_rad = np.deg2rad(THETA_IN)

        # Calculate denominator
        denominator = N_OUT * np.sin(theta_out_rad) - N_IN * np.sin(theta_in_rad)

        # Validate: check if sine values are in valid range
        # Also check denominator is not zero
        if abs(denominator) < 1e-10:
            continue

        # Calculate Period using Grating Equation
        period_nm = (M * wavelength_nm) / denominator

        # Additional validation: Period should be positive for physical systems
        if period_nm <= 0:
            continue

        data.append({
            'Wavelength_nm': wavelength_nm,
            'Target_Angle': target_angle_deg,
            'Period_nm': period_nm
        })

    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating P2 Rainbow Waveguide dataset...")
    print(f"Target samples: {NUM_SAMPLES}")
    print(f"Physics constants: n_out={N_OUT}, n_in={N_IN}, m={M}, theta_in={THETA_IN}")
    print()

    # Generate data
    df = generate_waveguide_data(NUM_SAMPLES)

    print(f"Valid samples generated: {len(df)}")
    print()

    # Print min/max values for UI scalers
    print("=== DATASET STATISTICS (for UI scalers) ===")
    print(f"Wavelength_nm - Min: {df['Wavelength_nm'].min():.2f}, Max: {df['Wavelength_nm'].max():.2f}")
    print(f"Target_Angle - Min: {df['Target_Angle'].min():.2f}, Max: {df['Target_Angle'].max():.2f}")
    print(f"Period_nm - Min: {df['Period_nm'].min():.2f}, Max: {df['Period_nm'].max():.2f}")
    print()

    # Save to CSV
    output_file = 'p2_rainbow_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to: {output_file}")
