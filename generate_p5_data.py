# -*- coding: utf-8 -*-
"""
P5: LCOS Fringing Surrogate - Data Generation Script
=====================================================
Simulates Liquid Crystal (LC) director profiles across a row of pixels
using 1D Finite Difference Relaxation to capture the "fringing" effect.

Physics: LC molecules tilt under electric field, but elastic coupling
between neighbors causes smooth transitions at pixel boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
# Grid parameters
N_PIXELS = 8                    # Number of pixels in the row
POINTS_PER_PIXEL = 16           # Spatial resolution per pixel
GRID_SIZE = N_PIXELS * POINTS_PER_PIXEL  # Total grid points (128)

# LC Physics parameters
PRETILT_DEG = 3.0               # Pre-tilt angle at boundaries (degrees)
PRETILT_RAD = np.radians(PRETILT_DEG)
MAX_VOLTAGE = 5.0               # Maximum voltage (V)
TARGET_TILT_DEG = 85.0          # Target tilt at max voltage (degrees)

# Relaxation solver parameters
MAX_ITERATIONS = 5000           # Maximum relaxation iterations
CONVERGENCE_TOL = 1e-6          # Convergence tolerance (radians)
RELAXATION_FACTOR = 0.8         # Under-relaxation for stability

# Dataset parameters
N_SAMPLES = 2000                # Number of training samples
RANDOM_SEED = 42

# Output paths
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "p5_lcos_dataset.npz"
VIZ_PATH = DATA_DIR / "p5_physics_check.png"


# =============================================================================
# LC PHYSICS ENGINE - 1D Finite Difference Relaxation
# =============================================================================
class LCOSSimulator:
    """
    Liquid Crystal On Silicon (LCOS) Fringing Simulator.

    Solves the 1D LC director equation using iterative relaxation:
        Elastic Torque + Electric Torque = 0

    The elastic torque pulls theta toward the average of neighbors.
    The electric torque pulls theta toward 90 degrees (perpendicular to field).
    """

    def __init__(self, grid_size: int = 128, n_pixels: int = 8):
        self.grid_size = grid_size
        self.n_pixels = n_pixels
        self.points_per_pixel = grid_size // n_pixels

        # Calibrate coefficient so 5V gives ~85 degree tilt
        # The balance equation: (theta[i-1] + theta[i+1])/2 + coeff * V^2 * sin(2*theta) = theta
        # At equilibrium with uniform V, theta ≈ arcsin(sqrt(coeff * V^2))
        # For 5V -> 85 deg: sin(2*85°) ≈ 0.17, so coeff * 25 * 0.17 ≈ 0.5 rad shift
        # Tuned empirically for the relaxation dynamics
        self.coeff = 0.012  # Tuned for 5V -> ~85 deg at center

        print(f"LCOS Simulator initialized:")
        print(f"  Grid: {grid_size} points ({n_pixels} pixels × {self.points_per_pixel} pts/pixel)")
        print(f"  Pre-tilt: {PRETILT_DEG}°")
        print(f"  Coefficient: {self.coeff} (tuned for {MAX_VOLTAGE}V -> {TARGET_TILT_DEG}°)")

    def expand_voltage_to_grid(self, pixel_voltages: np.ndarray) -> np.ndarray:
        """
        Expand low-resolution pixel commands to high-resolution grid.
        Creates step-function voltage profile.

        Args:
            pixel_voltages: (n_pixels,) array of voltage per pixel

        Returns:
            voltage_grid: (grid_size,) array
        """
        voltage_grid = np.zeros(self.grid_size)

        for i, v in enumerate(pixel_voltages):
            start = i * self.points_per_pixel
            end = start + self.points_per_pixel
            voltage_grid[start:end] = v

        return voltage_grid

    def solve_relaxation(self, voltage_grid: np.ndarray,
                         max_iter: int = MAX_ITERATIONS,
                         tol: float = CONVERGENCE_TOL) -> np.ndarray:
        """
        Solve for steady-state LC tilt profile using VECTORIZED relaxation.

        Physics equation (at each interior point):
            theta_new[i] = (theta[i-1] + theta[i+1])/2 + coeff * V[i]^2 * sin(2*theta[i])

        This balances:
            - Elastic torque (neighbors pull toward their average)
            - Electric torque (field pulls toward 90 degrees)

        Args:
            voltage_grid: (grid_size,) voltage at each point
            max_iter: maximum iterations
            tol: convergence tolerance

        Returns:
            theta: (grid_size,) tilt angle in radians
        """
        # Initialize with pre-tilt everywhere
        theta = np.full(self.grid_size, PRETILT_RAD)

        # Fixed boundary conditions
        theta[0] = PRETILT_RAD
        theta[-1] = PRETILT_RAD

        # Precompute V^2 for interior points
        V_squared = voltage_grid[1:-1] ** 2

        # Vectorized relaxation loop
        for iteration in range(max_iter):
            theta_old = theta.copy()

            # VECTORIZED: Update all interior points at once
            # Elastic contribution (average of neighbors)
            elastic = (theta[:-2] + theta[2:]) / 2

            # Electric contribution (pulls toward 90 deg)
            electric = self.coeff * V_squared * np.sin(2 * theta[1:-1])

            # New value with under-relaxation for stability
            theta_new = elastic + electric
            theta[1:-1] = RELAXATION_FACTOR * theta_new + (1 - RELAXATION_FACTOR) * theta[1:-1]

            # Clamp to physical range [0, pi/2]
            theta[1:-1] = np.clip(theta[1:-1], 0, np.pi/2)

            # Check convergence
            max_change = np.max(np.abs(theta - theta_old))
            if max_change < tol:
                break

        return theta

    def tilt_to_phase(self, theta: np.ndarray) -> np.ndarray:
        """
        Convert tilt angle to phase retardation.

        Full physics: Phase ∝ ∫(n_eff(θ) - n_o) dz
        Simplified: Phase = sin²(θ) (captures the optical effect shape)

        This gives 0 phase at θ=0 (no tilt) and max phase at θ=90° (full tilt).

        Args:
            theta: tilt angle in radians

        Returns:
            phase: normalized phase retardation [0, 1]
        """
        return np.sin(theta) ** 2

    def simulate(self, pixel_voltages: np.ndarray) -> tuple:
        """
        Full simulation: voltage pattern -> tilt profile -> phase response.

        Args:
            pixel_voltages: (n_pixels,) voltage per pixel

        Returns:
            voltage_grid: (grid_size,) expanded voltage
            theta: (grid_size,) tilt angle in radians
            phase: (grid_size,) phase retardation
        """
        # Expand to grid
        voltage_grid = self.expand_voltage_to_grid(pixel_voltages)

        # Solve for tilt
        theta = self.solve_relaxation(voltage_grid)

        # Convert to phase
        phase = self.tilt_to_phase(theta)

        return voltage_grid, theta, phase


# =============================================================================
# SIGNAL GENERATOR - Random Voltage Patterns
# =============================================================================
def generate_random_patterns(n_samples: int, n_pixels: int,
                             voltage_levels: list = None) -> np.ndarray:
    """
    Generate random voltage patterns for training.

    Patterns include:
        - Pure random (continuous 0-5V)
        - Quantized levels (0, 2.5, 5V)
        - Binary patterns (0 or 5V)
        - Gradient patterns

    Args:
        n_samples: number of patterns to generate
        n_pixels: number of pixels
        voltage_levels: discrete voltage levels to use

    Returns:
        patterns: (n_samples, n_pixels) voltage patterns
    """
    if voltage_levels is None:
        voltage_levels = [0, 2.5, 5.0]

    patterns = []

    # Distribution of pattern types
    n_random = n_samples // 4
    n_quantized = n_samples // 4
    n_binary = n_samples // 4
    n_gradient = n_samples - n_random - n_quantized - n_binary

    # Random continuous patterns
    for _ in range(n_random):
        p = np.random.uniform(0, MAX_VOLTAGE, n_pixels)
        patterns.append(p)

    # Quantized patterns (discrete levels)
    for _ in range(n_quantized):
        p = np.random.choice(voltage_levels, n_pixels)
        patterns.append(p)

    # Binary patterns (on/off)
    for _ in range(n_binary):
        p = np.random.choice([0, MAX_VOLTAGE], n_pixels)
        patterns.append(p)

    # Gradient patterns
    for _ in range(n_gradient):
        start = np.random.uniform(0, MAX_VOLTAGE)
        end = np.random.uniform(0, MAX_VOLTAGE)
        p = np.linspace(start, end, n_pixels)
        # Add some noise
        p += np.random.normal(0, 0.2, n_pixels)
        p = np.clip(p, 0, MAX_VOLTAGE)
        patterns.append(p)

    return np.array(patterns)


# =============================================================================
# DATASET GENERATION
# =============================================================================
def generate_dataset():
    """Generate the full P5 training dataset."""
    print("=" * 60)
    print("P5: LCOS Fringing Surrogate - Data Generation")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)

    # Initialize simulator
    sim = LCOSSimulator(grid_size=GRID_SIZE, n_pixels=N_PIXELS)

    # Generate voltage patterns
    print(f"\nGenerating {N_SAMPLES} random voltage patterns...")
    pixel_patterns = generate_random_patterns(N_SAMPLES, N_PIXELS)
    print(f"  Pattern shape: {pixel_patterns.shape}")
    print(f"  Voltage range: [{pixel_patterns.min():.2f}, {pixel_patterns.max():.2f}] V")

    # Simulate each pattern
    print(f"\nRunning LC relaxation solver for {N_SAMPLES} patterns...")

    voltage_grids = []
    phase_responses = []

    for i in tqdm(range(N_SAMPLES), desc="Simulating"):
        voltage_grid, theta, phase = sim.simulate(pixel_patterns[i])
        voltage_grids.append(voltage_grid)
        phase_responses.append(phase)

    # Convert to arrays with shape (N, 128, 1)
    voltage_commands = np.array(voltage_grids)[:, :, np.newaxis].astype(np.float32)
    phase_responses = np.array(phase_responses)[:, :, np.newaxis].astype(np.float32)

    print(f"\nDataset shapes:")
    print(f"  voltage_commands: {voltage_commands.shape}")
    print(f"  phase_responses: {phase_responses.shape}")

    # Compute statistics
    print(f"\nPhase statistics:")
    print(f"  Min: {phase_responses.min():.4f}")
    print(f"  Max: {phase_responses.max():.4f}")
    print(f"  Mean: {phase_responses.mean():.4f}")
    print(f"  Std: {phase_responses.std():.4f}")

    # Save dataset
    np.savez(
        OUTPUT_PATH,
        voltage_commands=voltage_commands,
        phase_responses=phase_responses,
        pixel_patterns=pixel_patterns,
        config={
            'n_samples': N_SAMPLES,
            'grid_size': GRID_SIZE,
            'n_pixels': N_PIXELS,
            'points_per_pixel': POINTS_PER_PIXEL,
            'max_voltage': MAX_VOLTAGE,
            'pretilt_deg': PRETILT_DEG,
            'coeff': sim.coeff
        }
    )
    print(f"\n[OK] Dataset saved to: {OUTPUT_PATH}")

    return voltage_commands, phase_responses, pixel_patterns, sim


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_physics_visualization(sim: LCOSSimulator, pixel_patterns: np.ndarray):
    """
    Create visualization showing the fringing effect.

    Shows 3 example patterns:
        - Top: Square wave voltage input
        - Bottom: Smooth, blurry phase output (the "fringing")
    """
    print("\nCreating physics visualization...")

    # Select interesting patterns to visualize
    examples = [
        np.array([0, 5, 5, 0, 2.5, 0, 5, 0]),      # Mixed pattern
        np.array([5, 0, 5, 0, 5, 0, 5, 0]),        # Alternating
        np.array([0, 0, 5, 5, 5, 5, 0, 0]),        # Center block
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    x = np.arange(GRID_SIZE)
    pixel_boundaries = np.arange(0, GRID_SIZE + 1, POINTS_PER_PIXEL)

    for row, pixel_voltages in enumerate(examples):
        voltage_grid, theta, phase = sim.simulate(pixel_voltages)

        # Left column: Voltage input
        ax1 = axes[row, 0]
        ax1.step(x, voltage_grid, where='mid', color='#667eea', linewidth=2.5, label='Voltage')
        ax1.fill_between(x, 0, voltage_grid, step='mid', alpha=0.3, color='#667eea')

        # Mark pixel boundaries
        for pb in pixel_boundaries:
            ax1.axvline(x=pb, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

        ax1.set_ylabel('Voltage (V)', fontsize=11)
        ax1.set_ylim([-0.5, MAX_VOLTAGE + 0.5])
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        if row == 0:
            ax1.set_title('Input: Pixel Voltage Commands (Step Function)', fontsize=12, fontweight='bold')

        # Right column: Phase output
        ax2 = axes[row, 1]
        ax2.plot(x, phase, color='#2ecc71', linewidth=2.5, label='Phase')
        ax2.fill_between(x, 0, phase, alpha=0.3, color='#2ecc71')

        # Overlay the "ideal" step response for comparison
        ideal_phase = sim.tilt_to_phase(np.radians(TARGET_TILT_DEG) * (voltage_grid / MAX_VOLTAGE))
        ax2.step(x, ideal_phase, where='mid', color='red', linewidth=1, linestyle='--',
                 alpha=0.5, label='Ideal (no fringing)')

        # Mark pixel boundaries
        for pb in pixel_boundaries:
            ax2.axvline(x=pb, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

        ax2.set_ylabel('Phase Retardation', fontsize=11)
        ax2.set_ylim([-0.05, 1.05])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        if row == 0:
            ax2.set_title('Output: Phase Profile (Smooth = Fringing)', fontsize=12, fontweight='bold')

        # Add pattern label
        pattern_str = '[' + ', '.join(f'{v:.1f}' for v in pixel_voltages) + ']'
        ax1.text(0.02, 0.98, f'Pattern: {pattern_str}', transform=ax1.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Set x labels on bottom row
    axes[2, 0].set_xlabel('Grid Position', fontsize=11)
    axes[2, 1].set_xlabel('Grid Position', fontsize=11)

    # Add pixel labels
    for col in range(2):
        ax = axes[2, col]
        for i in range(N_PIXELS):
            center = (i + 0.5) * POINTS_PER_PIXEL
            ax.text(center, -0.15, f'P{i}', ha='center', fontsize=8, color='gray')

    plt.suptitle('P5: LCOS Fringing Effect - LC Elastic Coupling Smooths Pixel Edges',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(VIZ_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] Visualization saved to: {VIZ_PATH}")


def verify_calibration(sim: LCOSSimulator):
    """Verify that 5V gives ~85 degree tilt at center."""
    print("\nCalibration Check:")

    # Uniform 5V pattern
    uniform_5v = np.full(N_PIXELS, MAX_VOLTAGE)
    voltage_grid, theta, phase = sim.simulate(uniform_5v)

    # Check center tilt
    center_idx = GRID_SIZE // 2
    center_tilt_deg = np.degrees(theta[center_idx])
    center_phase = phase[center_idx]

    print(f"  Uniform 5V pattern:")
    print(f"    Center tilt: {center_tilt_deg:.1f}° (target: {TARGET_TILT_DEG}°)")
    print(f"    Center phase: {center_phase:.4f}")
    print(f"    Edge tilt: {np.degrees(theta[1]):.1f}° (pre-tilt: {PRETILT_DEG}°)")

    # Check fringing width
    # Find where tilt drops to 50% of center value
    half_tilt = theta[center_idx] / 2
    fringe_width = 0
    for i in range(center_idx, 0, -1):
        if theta[i] < half_tilt:
            fringe_width = center_idx - i
            break

    print(f"    Fringe width: ~{fringe_width} points ({fringe_width/POINTS_PER_PIXEL:.2f} pixels)")

    if abs(center_tilt_deg - TARGET_TILT_DEG) > 10:
        print(f"  [!] Warning: Calibration off target. Adjust 'coeff' parameter.")
    else:
        print(f"  [OK] Calibration within tolerance.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Generate dataset
    voltage_commands, phase_responses, pixel_patterns, sim = generate_dataset()

    # Verify calibration
    verify_calibration(sim)

    # Create visualization
    create_physics_visualization(sim, pixel_patterns)

    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Dataset: {OUTPUT_PATH}")
    print(f"  Visualization: {VIZ_PATH}")
    print(f"\nDataset ready for neural surrogate training.")


if __name__ == "__main__":
    main()
