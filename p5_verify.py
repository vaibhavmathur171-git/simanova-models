# -*- coding: utf-8 -*-
"""
P5 LCOS Fringing Surrogate - Verification Script
=================================================
Tests the "Point Spread Function" of the display by illuminating
only Pixel #4 and comparing Ground Truth vs AI Prediction.

The key check: Does the AI correctly predict the SLOPE of the
phase change at the pixel edge? That slope IS the fringing field.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = Path("models/best_lcos_model.pth")
OUTPUT_PATH = Path("models/p5_crosstalk_check.png")

# Grid parameters (must match training)
N_PIXELS = 8
POINTS_PER_PIXEL = 16
GRID_SIZE = N_PIXELS * POINTS_PER_PIXEL  # 128

# Physics parameters (must match data generation)
PRETILT_RAD = np.radians(3.0)
MAX_VOLTAGE = 5.0
COEFF = 0.012  # LC elastic/electric coefficient
MAX_ITERATIONS = 5000
CONVERGENCE_TOL = 1e-6
RELAXATION_FACTOR = 0.8


# =============================================================================
# U-NET MODEL (Must match training architecture)
# =============================================================================
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock1D(in_channels, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock1D(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.size(2) != skip.size(2):
            diff = skip.size(2) - x.size(2)
            x = nn.functional.pad(x, [diff // 2, diff - diff // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LCOS_UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=2, base_filters=16):
        super().__init__()
        self.depth = depth
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            self.encoders.append(EncoderBlock1D(in_ch, filters[i]))
            in_ch = filters[i]

        self.bottleneck = ConvBlock1D(filters[depth - 1], filters[depth])

        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.decoders.append(DecoderBlock1D(filters[i + 1], filters[i]))

        self.output = nn.Conv1d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            features, x = encoder(x)
            skips.append(features)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return self.output(x)


# =============================================================================
# PHYSICS SIMULATOR (Ground Truth)
# =============================================================================
class LCOSSimulator:
    """LC relaxation solver - identical to data generation."""

    def __init__(self):
        self.coeff = COEFF

    def expand_voltage_to_grid(self, pixel_voltages):
        voltage_grid = np.zeros(GRID_SIZE)
        for i, v in enumerate(pixel_voltages):
            start = i * POINTS_PER_PIXEL
            end = start + POINTS_PER_PIXEL
            voltage_grid[start:end] = v
        return voltage_grid

    def solve_relaxation(self, voltage_grid):
        theta = np.full(GRID_SIZE, PRETILT_RAD)
        theta[0] = PRETILT_RAD
        theta[-1] = PRETILT_RAD

        V_squared = voltage_grid[1:-1] ** 2

        for iteration in range(MAX_ITERATIONS):
            theta_old = theta.copy()

            elastic = (theta[:-2] + theta[2:]) / 2
            electric = self.coeff * V_squared * np.sin(2 * theta[1:-1])

            theta_new = elastic + electric
            theta[1:-1] = RELAXATION_FACTOR * theta_new + (1 - RELAXATION_FACTOR) * theta[1:-1]
            theta[1:-1] = np.clip(theta[1:-1], 0, np.pi/2)

            if np.max(np.abs(theta - theta_old)) < CONVERGENCE_TOL:
                break

        return theta

    def tilt_to_phase(self, theta):
        return np.sin(theta) ** 2

    def simulate(self, pixel_voltages):
        voltage_grid = self.expand_voltage_to_grid(pixel_voltages)
        theta = self.solve_relaxation(voltage_grid)
        phase = self.tilt_to_phase(theta)
        return voltage_grid, phase


# =============================================================================
# VERIFICATION
# =============================================================================
def load_model():
    """Load trained U-Net model."""
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

    config = checkpoint['config']
    v_max = checkpoint['v_max']

    print(f"  Config: depth={config['depth']}, base_filters={config['base_filters']}")
    print(f"  Parameters: {config['n_parameters']:,}")

    model = LCOS_UNet(
        in_channels=1,
        out_channels=1,
        depth=config['depth'],
        base_filters=config['base_filters']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, v_max


def compute_edge_slopes(x, y, pixel_idx, edge='left'):
    """
    Compute the slope at the edge of a pixel.

    Args:
        x: grid positions
        y: phase values
        pixel_idx: which pixel (0-7)
        edge: 'left' or 'right' edge

    Returns:
        slope: dy/dx at the edge
    """
    if edge == 'left':
        # Left edge of pixel
        edge_pos = pixel_idx * POINTS_PER_PIXEL
        # Use 5 points around the edge for slope calculation
        start = max(0, edge_pos - 3)
        end = min(GRID_SIZE, edge_pos + 3)
    else:
        # Right edge of pixel
        edge_pos = (pixel_idx + 1) * POINTS_PER_PIXEL - 1
        start = max(0, edge_pos - 3)
        end = min(GRID_SIZE, edge_pos + 3)

    # Linear regression for slope
    x_region = x[start:end]
    y_region = y[start:end]

    if len(x_region) > 1:
        slope = np.polyfit(x_region, y_region, 1)[0]
    else:
        slope = 0

    return slope, edge_pos


def create_verification_plot(voltage, phase_gt, phase_ai):
    """Create the verification visualization."""
    print("\nCreating visualization...")

    fig = plt.figure(figsize=(16, 12))

    # Grid positions
    x = np.arange(GRID_SIZE)
    pixel_boundaries = np.arange(0, GRID_SIZE + 1, POINTS_PER_PIXEL)

    # ==========================================================================
    # PLOT 1: Full Overview
    # ==========================================================================
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.step(x, voltage, where='mid', color='#667eea', linewidth=2, label='Voltage Input')
    ax1.fill_between(x, 0, voltage, step='mid', alpha=0.3, color='#667eea')

    for pb in pixel_boundaries:
        ax1.axvline(x=pb, color='gray', linestyle=':', alpha=0.5)

    ax1.set_ylabel('Voltage (V)', fontsize=11)
    ax1.set_title('Test Case: Single Pixel ON (Pixel #4 = 5V)', fontsize=12, fontweight='bold')
    ax1.set_ylim([-0.5, MAX_VOLTAGE + 0.5])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Add pixel labels
    for i in range(N_PIXELS):
        center = (i + 0.5) * POINTS_PER_PIXEL
        color = 'red' if i == 4 else 'gray'
        weight = 'bold' if i == 4 else 'normal'
        ax1.text(center, -0.3, f'P{i}', ha='center', fontsize=9, color=color, fontweight=weight)

    # ==========================================================================
    # PLOT 2: Phase Comparison (Full)
    # ==========================================================================
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(x, phase_gt, 'g-', linewidth=2.5, label='Ground Truth (Physics)', alpha=0.9)
    ax2.plot(x, phase_ai, 'r--', linewidth=2, label='AI Prediction (U-Net)', alpha=0.9)

    for pb in pixel_boundaries:
        ax2.axvline(x=pb, color='gray', linestyle=':', alpha=0.5)

    ax2.set_ylabel('Phase Retardation', fontsize=11)
    ax2.set_title('Phase Response: Ground Truth vs AI', fontsize=12, fontweight='bold')
    ax2.set_ylim([-0.05, 1.1])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Highlight the active pixel region
    ax2.axvspan(4 * POINTS_PER_PIXEL, 5 * POINTS_PER_PIXEL, alpha=0.1, color='red')

    # ==========================================================================
    # PLOT 3: LEFT Edge Zoom (Pixel 4 left boundary)
    # ==========================================================================
    ax3 = fig.add_subplot(3, 2, 3)

    # Zoom region: around left edge of pixel 4
    left_edge = 4 * POINTS_PER_PIXEL
    zoom_start = left_edge - 10
    zoom_end = left_edge + 10
    zoom_slice = slice(zoom_start, zoom_end)

    ax3.plot(x[zoom_slice], phase_gt[zoom_slice], 'g-', linewidth=3,
             label='Ground Truth', marker='o', markersize=6)
    ax3.plot(x[zoom_slice], phase_ai[zoom_slice], 'r--', linewidth=2.5,
             label='AI Prediction', marker='s', markersize=5)

    ax3.axvline(x=left_edge, color='black', linestyle='-', linewidth=2, alpha=0.7, label='Pixel Edge')

    # Compute slopes
    slope_gt_left, _ = compute_edge_slopes(x, phase_gt, 4, 'left')
    slope_ai_left, _ = compute_edge_slopes(x, phase_ai, 4, 'left')

    ax3.set_xlabel('Grid Position', fontsize=11)
    ax3.set_ylabel('Phase', fontsize=11)
    ax3.set_title(f'LEFT Edge Zoom (Fringing Slope)\nGT slope: {slope_gt_left:.4f}, AI slope: {slope_ai_left:.4f}',
                  fontsize=11, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Add slope comparison annotation
    slope_error_left = abs(slope_ai_left - slope_gt_left) / (abs(slope_gt_left) + 1e-8) * 100
    ax3.text(0.05, 0.95, f'Slope Error: {slope_error_left:.1f}%',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # ==========================================================================
    # PLOT 4: RIGHT Edge Zoom (Pixel 4 right boundary)
    # ==========================================================================
    ax4 = fig.add_subplot(3, 2, 4)

    # Zoom region: around right edge of pixel 4
    right_edge = 5 * POINTS_PER_PIXEL
    zoom_start = right_edge - 10
    zoom_end = right_edge + 10
    zoom_slice = slice(zoom_start, zoom_end)

    ax4.plot(x[zoom_slice], phase_gt[zoom_slice], 'g-', linewidth=3,
             label='Ground Truth', marker='o', markersize=6)
    ax4.plot(x[zoom_slice], phase_ai[zoom_slice], 'r--', linewidth=2.5,
             label='AI Prediction', marker='s', markersize=5)

    ax4.axvline(x=right_edge, color='black', linestyle='-', linewidth=2, alpha=0.7, label='Pixel Edge')

    # Compute slopes
    slope_gt_right, _ = compute_edge_slopes(x, phase_gt, 4, 'right')
    slope_ai_right, _ = compute_edge_slopes(x, phase_ai, 4, 'right')

    ax4.set_xlabel('Grid Position', fontsize=11)
    ax4.set_ylabel('Phase', fontsize=11)
    ax4.set_title(f'RIGHT Edge Zoom (Fringing Slope)\nGT slope: {slope_gt_right:.4f}, AI slope: {slope_ai_right:.4f}',
                  fontsize=11, fontweight='bold')
    ax4.legend(loc='lower left', fontsize=9)
    ax4.grid(True, alpha=0.3)

    slope_error_right = abs(slope_ai_right - slope_gt_right) / (abs(slope_gt_right) + 1e-8) * 100
    ax4.text(0.95, 0.95, f'Slope Error: {slope_error_right:.1f}%',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # ==========================================================================
    # PLOT 5: Error Analysis
    # ==========================================================================
    ax5 = fig.add_subplot(3, 2, 5)

    error = phase_ai - phase_gt
    ax5.fill_between(x, 0, error, where=(error >= 0), color='red', alpha=0.5, label='Overprediction')
    ax5.fill_between(x, 0, error, where=(error < 0), color='blue', alpha=0.5, label='Underprediction')
    ax5.axhline(y=0, color='black', linewidth=1)

    for pb in pixel_boundaries:
        ax5.axvline(x=pb, color='gray', linestyle=':', alpha=0.5)

    ax5.set_xlabel('Grid Position', fontsize=11)
    ax5.set_ylabel('Error (AI - GT)', fontsize=11)
    ax5.set_title('Point-wise Prediction Error', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ==========================================================================
    # PLOT 6: Metrics Summary
    # ==========================================================================
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')

    # Compute metrics
    mse = np.mean((phase_ai - phase_gt) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(phase_ai - phase_gt))
    max_error = np.max(np.abs(phase_ai - phase_gt))

    # Correlation
    correlation = np.corrcoef(phase_gt, phase_ai)[0, 1]

    # Peak phase comparison
    peak_gt = np.max(phase_gt)
    peak_ai = np.max(phase_ai)
    peak_error = abs(peak_ai - peak_gt) / peak_gt * 100

    # Average slope error
    avg_slope_error = (slope_error_left + slope_error_right) / 2

    metrics_text = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║          P5 LCOS VERIFICATION: CROSSTALK CHECK           ║
    ╠══════════════════════════════════════════════════════════╣
    ║  Test Case: Single Pixel ON (Pixel #4 = 5V)              ║
    ║  Purpose: Check Point Spread Function / Fringing Field   ║
    ╠══════════════════════════════════════════════════════════╣
    ║                    ERROR METRICS                         ║
    ║  ────────────────────────────────────────────────────    ║
    ║  MSE:          {mse:.6f}                                  ║
    ║  RMSE:         {rmse:.6f}                                  ║
    ║  MAE:          {mae:.6f}                                  ║
    ║  Max Error:    {max_error:.6f}                                  ║
    ║  Correlation:  {correlation:.6f}                                  ║
    ╠══════════════════════════════════════════════════════════╣
    ║                  CRITICAL CHECKS                         ║
    ║  ────────────────────────────────────────────────────    ║
    ║  Peak Phase GT:     {peak_gt:.4f}                              ║
    ║  Peak Phase AI:     {peak_ai:.4f}                              ║
    ║  Peak Error:        {peak_error:.2f}%                               ║
    ║                                                          ║
    ║  Left Edge Slope Error:  {slope_error_left:>6.1f}%                       ║
    ║  Right Edge Slope Error: {slope_error_right:>6.1f}%                       ║
    ║  Average Slope Error:    {avg_slope_error:>6.1f}%                       ║
    ╠══════════════════════════════════════════════════════════╣
    ║                     VERDICT                              ║
    """

    if avg_slope_error < 20 and correlation > 0.95:
        verdict = "║  ✓ PASS: AI correctly predicts fringing field slopes      ║"
        verdict_color = 'green'
    elif avg_slope_error < 50:
        verdict = "║  ~ ACCEPTABLE: Slopes captured with moderate accuracy     ║"
        verdict_color = 'orange'
    else:
        verdict = "║  ✗ NEEDS IMPROVEMENT: Slope prediction inaccurate         ║"
        verdict_color = 'red'

    metrics_text += verdict + "\n    ╚══════════════════════════════════════════════════════════╝"

    ax6.text(0.5, 0.5, metrics_text, transform=ax6.transAxes,
             fontsize=10, family='monospace', verticalalignment='center',
             horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.suptitle('P5 LCOS Fringing Verification: Point Spread Function Test',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n[OK] Verification plot saved to: {OUTPUT_PATH}")

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'correlation': correlation,
        'slope_error_left': slope_error_left,
        'slope_error_right': slope_error_right,
        'avg_slope_error': avg_slope_error
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("P5 LCOS Verification: Crosstalk / Point Spread Function Test")
    print("=" * 60)

    # 1. Create test case: Single pixel ON (Pixel #4)
    print("\n1. Creating test case: Single Pixel ON")
    pixel_voltages = np.zeros(N_PIXELS)
    pixel_voltages[4] = MAX_VOLTAGE  # Only Pixel #4 is ON

    print(f"   Voltage pattern: {pixel_voltages}")
    print(f"   This tests the 'Point Spread Function' of the display")

    # 2. Run Ground Truth (Physics Simulator)
    print("\n2. Running Ground Truth physics simulation...")
    sim = LCOSSimulator()
    voltage_grid, phase_gt = sim.simulate(pixel_voltages)
    print(f"   GT Phase range: [{phase_gt.min():.4f}, {phase_gt.max():.4f}]")

    # 3. Load AI model and run prediction
    print("\n3. Running AI prediction...")
    model, v_max = load_model()

    # Prepare input tensor
    v_tensor = torch.tensor(voltage_grid / v_max, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        phase_ai = model(v_tensor).numpy()[0, 0]

    print(f"   AI Phase range: [{phase_ai.min():.4f}, {phase_ai.max():.4f}]")

    # 4. Create verification plot
    print("\n4. Creating verification visualization...")
    metrics = create_verification_plot(voltage_grid, phase_gt, phase_ai)

    # 5. Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"\nError Metrics:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  Correlation: {metrics['correlation']:.6f}")

    print(f"\nFringing Field (Slope) Analysis:")
    print(f"  Left Edge Slope Error:  {metrics['slope_error_left']:.1f}%")
    print(f"  Right Edge Slope Error: {metrics['slope_error_right']:.1f}%")
    print(f"  Average Slope Error:    {metrics['avg_slope_error']:.1f}%")

    print(f"\nConclusion:")
    if metrics['avg_slope_error'] < 20 and metrics['correlation'] > 0.95:
        print("  [OK] The AI correctly learned the fringing field behavior!")
        print("  [OK] Edge slopes (crosstalk) are accurately predicted.")
    elif metrics['avg_slope_error'] < 50:
        print("  [~] The AI captures the general fringing behavior.")
        print("  [~] Edge slopes have moderate accuracy.")
    else:
        print("  [!] The AI needs improvement in predicting edge slopes.")
        print("  [!] Consider more training or different architecture.")

    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
