"""
Project 3: Waveguide Uniformity - Synthetic Data Generator
===========================================================
Generates training data for CNN to predict light uniformity in AR waveguides.
Uses a "Leaky Bucket" Energy Transport model for physics simulation.

Author: SimaNova Team
"""

import numpy as np
import os
from typing import Tuple

# Configuration
NUM_SAMPLES = 2000
GRID_SIZE = 64
EFFICIENCY_MIN = 0.0
EFFICIENCY_MAX = 0.8
ABSORPTION_FACTOR = 0.99
RANDOM_SEED = 42


def generate_gradient_map(grid_size: int) -> np.ndarray:
    """
    Generate a random linear gradient efficiency map.

    Creates gradients in random directions (horizontal, vertical, or diagonal)
    with random slopes to simulate different grating designs.
    """
    y, x = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), indexing='ij')

    # Random gradient parameters
    angle = np.random.uniform(0, 2 * np.pi)
    slope = np.random.uniform(0.2, 1.0)
    offset = np.random.uniform(0.1, 0.5)

    # Create directional gradient
    gradient = offset + slope * (np.cos(angle) * x + np.sin(angle) * y)

    return gradient


def generate_gaussian_blobs(grid_size: int, num_blobs: int = None) -> np.ndarray:
    """
    Generate random Gaussian blobs to simulate local grating variations.

    Parameters:
        grid_size: Size of the output grid
        num_blobs: Number of Gaussian blobs (random if None)

    Returns:
        2D array with Gaussian blob pattern
    """
    if num_blobs is None:
        num_blobs = np.random.randint(2, 8)

    y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
    blob_map = np.zeros((grid_size, grid_size))

    for _ in range(num_blobs):
        # Random blob parameters
        center_x = np.random.uniform(0, grid_size)
        center_y = np.random.uniform(0, grid_size)
        sigma = np.random.uniform(5, 20)  # Blob width
        amplitude = np.random.uniform(0.1, 0.4)

        # Add Gaussian blob
        blob = amplitude * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        blob_map += blob

    return blob_map


def generate_efficiency_map(grid_size: int) -> np.ndarray:
    """
    Generate a random efficiency map (grating design) by combining
    gradients and Gaussian blobs.

    The efficiency map represents how much light each pixel extracts
    from the waveguide.
    """
    # Combine gradient and blob patterns
    gradient = generate_gradient_map(grid_size)
    blobs = generate_gaussian_blobs(grid_size)

    # Mix with random weights
    gradient_weight = np.random.uniform(0.3, 0.7)
    efficiency_map = gradient_weight * gradient + (1 - gradient_weight) * blobs

    # Add small random noise for realism
    noise = np.random.normal(0, 0.05, (grid_size, grid_size))
    efficiency_map += noise

    # CRITICAL: Clip to valid efficiency range [0.0, 0.8]
    efficiency_map = np.clip(efficiency_map, EFFICIENCY_MIN, EFFICIENCY_MAX)

    return efficiency_map.astype(np.float32)


def solve_light_propagation(efficiency_map: np.ndarray) -> np.ndarray:
    """
    Solve the light propagation using the "Leaky Bucket" Energy Transport model.

    Physics:
    - Light propagates from Left (x=0) to Right (x=63)
    - At each pixel, light is extracted based on local grating efficiency
    - Remaining energy undergoes 1% absorption loss per column

    Algorithm (per column):
    1. Extracted_Light = Current_Energy * Efficiency_Pixel
    2. Remaining_Energy = (Current_Energy - Extracted_Light) * 0.99 (Absorption)
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
        # extracted = current_energy * efficiency for each row
        extracted_light[:, x] = current_energy * efficiency_map[:, x]

        # Calculate remaining energy after extraction
        remaining = current_energy - extracted_light[:, x]

        # Apply absorption loss (1% loss per column)
        current_energy = remaining * ABSORPTION_FACTOR

    return extracted_light


def generate_dataset(num_samples: int, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the complete dataset of input efficiency maps and output light fields.

    Parameters:
        num_samples: Number of samples to generate
        grid_size: Size of each sample grid (64x64)

    Returns:
        inputs: Array of efficiency maps (num_samples, grid_size, grid_size)
        outputs: Array of light fields (num_samples, grid_size, grid_size)
    """
    print(f"Generating {num_samples} samples...")

    inputs = np.zeros((num_samples, grid_size, grid_size), dtype=np.float32)
    outputs = np.zeros((num_samples, grid_size, grid_size), dtype=np.float32)

    for i in range(num_samples):
        # Generate random efficiency map (input)
        efficiency_map = generate_efficiency_map(grid_size)
        inputs[i] = efficiency_map

        # Solve physics to get light field (output)
        light_field = solve_light_propagation(efficiency_map)
        outputs[i] = light_field

        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

    return inputs, outputs


def main():
    """Main function to generate and save the dataset."""
    print("=" * 60)
    print("Project 3: Waveguide Uniformity - Data Generator")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Number of samples: {NUM_SAMPLES}")
    print(f"  - Grid size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  - Efficiency range: [{EFFICIENCY_MIN}, {EFFICIENCY_MAX}]")
    print(f"  - Absorption factor: {ABSORPTION_FACTOR}")
    print()

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Generate dataset
    inputs, outputs = generate_dataset(NUM_SAMPLES, GRID_SIZE)

    # Create data folder if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"\nCreated '{data_dir}' folder")

    # Save arrays
    input_path = os.path.join(data_dir, "p3_input_gratings.npy")
    output_path = os.path.join(data_dir, "p3_output_fields.npy")

    np.save(input_path, inputs)
    np.save(output_path, outputs)

    print(f"\nDataset saved successfully!")
    print(f"  - Inputs:  {input_path}")
    print(f"  - Outputs: {output_path}")

    # Verification: Print shapes
    print(f"\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Input gratings shape:  {inputs.shape}  (dtype: {inputs.dtype})")
    print(f"Output fields shape:   {outputs.shape}  (dtype: {outputs.dtype})")

    # Additional statistics
    print(f"\nInput statistics:")
    print(f"  - Min: {inputs.min():.4f}, Max: {inputs.max():.4f}, Mean: {inputs.mean():.4f}")
    print(f"\nOutput statistics:")
    print(f"  - Min: {outputs.min():.4f}, Max: {outputs.max():.4f}, Mean: {outputs.mean():.4f}")

    print(f"\nData generation complete!")


if __name__ == "__main__":
    main()
