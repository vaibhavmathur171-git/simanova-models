"""
Project 3: Waveguide Uniformity - Synthetic Data Generator
===========================================================
Generates training data for CNN to predict light uniformity in AR waveguides.

Physics Model: Surface Relief Grating (SRG) with Kogelnik Approximation
- Input: Etch Depth Maps (nanometers)
- Output: Light Field (extracted intensity)

Author: SimaNova Team
"""

import numpy as np
import os
from typing import Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_SAMPLES = 3000
GRID_SIZE = 64
RANDOM_SEED = 42

# =============================================================================
# PHYSICS CONSTANTS (Surface Relief Grating)
# =============================================================================
# Etch Depth Range (nanometers)
DEPTH_MIN = 50.0    # nm - Minimum manufacturable depth
DEPTH_MAX = 350.0   # nm - Maximum before structural issues

# Material Properties
REFRACTIVE_INDEX = 1.7  # Typical for high-index polymer/glass

# Kogelnik Parameter
# Peak efficiency occurs at depth ~ 200nm (quarter-wave condition)
# Efficiency(d) = 0.8 * sin^2(PI * d / 800)
KOGELNIK_PERIOD = 800.0  # nm - Full period of sin^2 response
MAX_EFFICIENCY = 0.8     # Maximum achievable efficiency

# Propagation
ABSORPTION_FACTOR = 0.99  # 1% absorption loss per column


# =============================================================================
# PHYSICS FUNCTIONS
# =============================================================================
def depth_to_efficiency(depth_nm: np.ndarray) -> np.ndarray:
    """
    Convert etch depth to grating efficiency using Kogelnik approximation.

    The Kogelnik coupled-wave theory gives a sinusoidal relationship between
    grating depth and diffraction efficiency for surface relief gratings.

    Formula: Efficiency = 0.8 * sin^2(PI * depth / 800)

    This creates a realistic non-linear response where:
    - ~200nm depth gives peak efficiency (~0.8)
    - ~400nm depth returns to zero (over-etched)
    - Depths between 50-350nm span the useful range

    Parameters:
        depth_nm: Etch depth in nanometers (2D array)

    Returns:
        efficiency: Diffraction efficiency [0, 0.8] (2D array)
    """
    phase = np.pi * depth_nm / KOGELNIK_PERIOD
    efficiency = MAX_EFFICIENCY * np.sin(phase) ** 2
    return efficiency.astype(np.float32)


def solve_light_propagation(efficiency_map: np.ndarray) -> np.ndarray:
    """
    Solve the light propagation using the "Leaky Bucket" Energy Transport model.

    Physics:
    - Light propagates from Left (x=0) to Right (x=63)
    - At each pixel, light is extracted based on local grating efficiency
    - Remaining energy undergoes 1% absorption loss per column

    Algorithm (per column):
    1. Extracted_Light = Current_Energy * Efficiency_Pixel
    2. Remaining_Energy = (Current_Energy - Extracted_Light) * 0.99
    3. Pass Remaining_Energy to next column (x+1)

    Parameters:
        efficiency_map: 2D array of grating efficiencies (64x64)

    Returns:
        extracted_light: 2D array of light field (what we observe)
    """
    grid_size = efficiency_map.shape[0]

    # Output: extracted light at each pixel (the observable uniformity)
    extracted_light = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Initialize: energy entering from the left edge (all rows start with 1.0)
    current_energy = np.ones(grid_size, dtype=np.float32)

    # Propagate column by column (Left to Right)
    for x in range(grid_size):
        # Extract light at this column based on efficiency
        extracted_light[:, x] = current_energy * efficiency_map[:, x]

        # Calculate remaining energy after extraction
        remaining = current_energy - extracted_light[:, x]

        # Apply absorption loss (1% loss per column)
        current_energy = remaining * ABSORPTION_FACTOR

    return extracted_light


# =============================================================================
# DEPTH MAP GENERATION (Engineering Designs)
# =============================================================================
def generate_depth_gradient(grid_size: int) -> np.ndarray:
    """
    Generate a linear depth gradient (Uniformity Correction Design).

    In real SRG waveguides, engineers intentionally vary the etch depth
    from left to right to compensate for energy depletion. This creates
    more uniform light output across the display.

    The gradient is primarily in the x-direction (propagation direction)
    with slight random variations.
    """
    y, x = np.meshgrid(
        np.linspace(0, 1, grid_size),
        np.linspace(0, 1, grid_size),
        indexing='ij'
    )

    # Primary gradient: left-to-right (uniformity correction)
    # Deeper etch on the right compensates for depleted energy
    base_depth = np.random.uniform(100, 180)  # nm - starting depth
    depth_increase = np.random.uniform(50, 150)  # nm - total increase

    # Add slight angular variation (manufacturing alignment)
    angle_offset = np.random.uniform(-0.1, 0.1)

    gradient = base_depth + depth_increase * (x + angle_offset * y)

    return gradient


def generate_manufacturing_hotspots(grid_size: int, num_hotspots: int = None) -> np.ndarray:
    """
    Generate Gaussian "hotspots" representing manufacturing errors.

    Real SRG fabrication (e.g., nanoimprint lithography) produces
    local depth variations due to:
    - Stamp defects
    - Uneven pressure distribution
    - Material flow variations
    - Temperature gradients

    These appear as localized depth deviations (both positive and negative).
    """
    if num_hotspots is None:
        num_hotspots = np.random.randint(3, 10)

    y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
    hotspot_map = np.zeros((grid_size, grid_size))

    for _ in range(num_hotspots):
        # Random hotspot parameters
        center_x = np.random.uniform(0, grid_size)
        center_y = np.random.uniform(0, grid_size)
        sigma = np.random.uniform(4, 15)  # Hotspot width (pixels)

        # Amplitude: can be positive (over-etch) or negative (under-etch)
        amplitude = np.random.uniform(-40, 40)  # nm deviation

        # Add Gaussian hotspot
        hotspot = amplitude * np.exp(
            -((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2)
        )
        hotspot_map += hotspot

    return hotspot_map


def generate_depth_map(grid_size: int) -> np.ndarray:
    """
    Generate a complete etch depth map (grating design).

    Combines:
    1. Intentional gradient (uniformity correction by design)
    2. Manufacturing hotspots (fabrication errors)
    3. Small-scale noise (surface roughness)

    Returns depth values in nanometers, clipped to [50, 350] nm.
    """
    # Intentional design: uniformity correction gradient
    gradient = generate_depth_gradient(grid_size)

    # Manufacturing errors: localized hotspots
    hotspots = generate_manufacturing_hotspots(grid_size)

    # Surface roughness: small random noise
    roughness = np.random.normal(0, 5, (grid_size, grid_size))  # nm

    # Combine all contributions
    depth_map = gradient + hotspots + roughness

    # Clip to valid manufacturing range
    depth_map = np.clip(depth_map, DEPTH_MIN, DEPTH_MAX)

    return depth_map.astype(np.float32)


# =============================================================================
# DATASET GENERATION
# =============================================================================
def generate_dataset(num_samples: int, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the complete dataset of depth maps and corresponding light fields.

    Pipeline for each sample:
    1. Generate Depth Map (nm) - The design input
    2. Convert Depth -> Efficiency using Kogelnik approximation
    3. Solve Light Propagation (Leaky Bucket model)
    4. Output Light Field - What we want to predict

    Parameters:
        num_samples: Number of samples to generate
        grid_size: Size of each sample grid (64x64)

    Returns:
        depth_maps: Array of etch depths in nm (num_samples, 64, 64)
        light_fields: Array of light outputs (num_samples, 64, 64)
    """
    print(f"Generating {num_samples} samples...")

    depth_maps = np.zeros((num_samples, grid_size, grid_size), dtype=np.float32)
    light_fields = np.zeros((num_samples, grid_size, grid_size), dtype=np.float32)

    for i in range(num_samples):
        # Step 1: Generate depth map (the design)
        depth_map = generate_depth_map(grid_size)
        depth_maps[i] = depth_map

        # Step 2: Convert depth to efficiency (Kogelnik)
        efficiency_map = depth_to_efficiency(depth_map)

        # Step 3: Solve physics to get light field
        light_field = solve_light_propagation(efficiency_map)
        light_fields[i] = light_field

        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

    return depth_maps, light_fields


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main function to generate and save the dataset."""
    print("=" * 70)
    print("Project 3: SRG Waveguide Uniformity - Data Generator")
    print("=" * 70)

    print(f"\n{'CONFIGURATION':=^70}")
    print(f"  Number of samples:    {NUM_SAMPLES}")
    print(f"  Grid size:            {GRID_SIZE}x{GRID_SIZE}")

    print(f"\n{'PHYSICS PARAMETERS':=^70}")
    print(f"  Etch Depth Range:     [{DEPTH_MIN}, {DEPTH_MAX}] nm")
    print(f"  Refractive Index:     {REFRACTIVE_INDEX}")
    print(f"  Kogelnik Period:      {KOGELNIK_PERIOD} nm")
    print(f"  Max Efficiency:       {MAX_EFFICIENCY}")
    print(f"  Absorption Factor:    {ABSORPTION_FACTOR}")

    print(f"\n{'KOGELNIK APPROXIMATION':=^70}")
    print(f"  Efficiency(d) = {MAX_EFFICIENCY} * sin^2(PI * d / {KOGELNIK_PERIOD})")
    print(f"  Peak efficiency at:   ~{KOGELNIK_PERIOD/4:.0f} nm depth")
    print()

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Generate dataset
    depth_maps, light_fields = generate_dataset(NUM_SAMPLES, GRID_SIZE)

    # Create data folder if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"\nCreated '{data_dir}' folder")

    # Save arrays
    depth_path = os.path.join(data_dir, "p3_depth_maps.npy")
    light_path = os.path.join(data_dir, "p3_light_fields.npy")

    np.save(depth_path, depth_maps)
    np.save(light_path, light_fields)

    print(f"\nDataset saved successfully!")
    print(f"  - Depth Maps:   {depth_path}")
    print(f"  - Light Fields: {light_path}")

    # Verification
    print(f"\n{'VERIFICATION':=^70}")
    print(f"Depth maps shape:   {depth_maps.shape}  (dtype: {depth_maps.dtype})")
    print(f"Light fields shape: {light_fields.shape}  (dtype: {light_fields.dtype})")

    print(f"\nDepth Map Statistics (nm):")
    print(f"  - Min:  {depth_maps.min():.1f} nm")
    print(f"  - Max:  {depth_maps.max():.1f} nm")
    print(f"  - Mean: {depth_maps.mean():.1f} nm")

    # Show efficiency statistics (derived)
    efficiencies = depth_to_efficiency(depth_maps)
    print(f"\nDerived Efficiency Statistics:")
    print(f"  - Min:  {efficiencies.min():.4f}")
    print(f"  - Max:  {efficiencies.max():.4f}")
    print(f"  - Mean: {efficiencies.mean():.4f}")

    print(f"\nLight Field Statistics:")
    print(f"  - Min:  {light_fields.min():.4f}")
    print(f"  - Max:  {light_fields.max():.4f}")
    print(f"  - Mean: {light_fields.mean():.4f}")

    print(f"\n{'DATA GENERATION COMPLETE':=^70}")


if __name__ == "__main__":
    main()
