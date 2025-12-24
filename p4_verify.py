# -*- coding: utf-8 -*-
"""
P4 MEMS Neural Surrogate - Verification Script ("The Boing Test")
=================================================================
Compares the trained LSTM model against ground truth physics
for a step input response, testing if the AI learned the ringing.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import odeint

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = Path("models/best_mems_model.pth")
DATA_PATH = Path("data/p4_mems_dataset.npz")
OUTPUT_PATH = Path("models/p4_step_response.png")

# Physics parameters (must match training data)
F0 = 2000.0  # Resonant frequency (Hz)
Q = 50.0     # Quality factor
FS = 100000  # Sampling frequency (Hz)

# Test signal: 0V for 10ms, 50V for 20ms, 0V for 10ms
VOLTAGE_LOW = 0.0
VOLTAGE_HIGH = 50.0
T_BEFORE = 0.010   # 10ms before step
T_STEP = 0.020     # 20ms at high voltage
T_AFTER = 0.010    # 10ms after step

# =============================================================================
# MEMS PHYSICS SIMULATOR (Ground Truth)
# =============================================================================
class MEMSSimulator:
    """MEMS electrostatic mirror physics simulator."""

    def __init__(self, f0: float = 2000.0, Q: float = 50.0, k_torque: float = None):
        self.f0 = f0
        self.Q = Q
        self.omega_n = 2 * np.pi * f0
        self.zeta = 1.0 / (2.0 * Q)
        self.omega_d = self.omega_n * np.sqrt(1 - self.zeta**2)
        self.tau = 1.0 / (self.zeta * self.omega_n)

        # Auto-scale torque constant
        if k_torque is None:
            target_theta = 0.1
            self.k_torque = target_theta * self.omega_n**2
        else:
            self.k_torque = k_torque

    def simulate_fast(self, t: np.ndarray, V: np.ndarray,
                      theta0: float = 0.0, theta_dot0: float = 0.0) -> np.ndarray:
        """Fast Euler integration."""
        n = len(t)
        dt = t[1] - t[0]

        theta = np.zeros(n)
        theta_dot = np.zeros(n)

        theta[0] = theta0
        theta_dot[0] = theta_dot0

        c1 = 2 * self.zeta * self.omega_n
        c2 = self.omega_n**2

        for i in range(1, n):
            torque = self.k_torque * V[i-1] * np.abs(V[i-1])
            theta_ddot = -c1 * theta_dot[i-1] - c2 * theta[i-1] + torque
            theta_dot[i] = theta_dot[i-1] + theta_ddot * dt
            theta[i] = theta[i-1] + theta_dot[i] * dt

        return theta


# =============================================================================
# LSTM MODEL (Must match training architecture)
# =============================================================================
class MEMS_LSTM(nn.Module):
    """LSTM model for MEMS dynamics prediction."""

    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.decoder(last_hidden)
        return out


# =============================================================================
# STEP INPUT GENERATOR
# =============================================================================
def generate_step_test_signal():
    """
    Generate the "Boing Test" step signal.

    Profile:
    - 0V for 10ms
    - Jump to 50V for 20ms
    - Drop to 0V for 10ms

    Returns normalized voltage for the model.
    """
    total_duration = T_BEFORE + T_STEP + T_AFTER
    n_samples = int(total_duration * FS)
    t = np.linspace(0, total_duration, n_samples)
    dt = t[1] - t[0]

    # Create step profile
    V_raw = np.zeros(n_samples)

    n_before = int(T_BEFORE * FS)
    n_step = int(T_STEP * FS)

    V_raw[:n_before] = VOLTAGE_LOW
    V_raw[n_before:n_before + n_step] = VOLTAGE_HIGH
    V_raw[n_before + n_step:] = VOLTAGE_LOW

    print(f"Step Test Signal Generated:")
    print(f"  Duration: {total_duration*1000:.0f} ms")
    print(f"  Samples: {n_samples}")
    print(f"  Profile: 0V -> {VOLTAGE_HIGH}V -> 0V")

    return t, V_raw


# =============================================================================
# LOAD MODEL AND NORMALIZATION
# =============================================================================
def load_model_and_stats():
    """Load trained LSTM model and get normalization stats from dataset."""
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    config = checkpoint['config']
    seq_length = config['seq_length']
    hidden_dim = config['hidden_dim']

    print(f"  seq_length: {seq_length}")
    print(f"  hidden_dim: {hidden_dim}")

    model = MEMS_LSTM(input_dim=2, hidden_dim=hidden_dim, num_layers=2, dropout=0.1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # Load normalization stats from dataset
    print(f"\nLoading normalization stats from {DATA_PATH}...")
    data = np.load(DATA_PATH, allow_pickle=True)

    # Get the norm_factor used during data generation
    norm_factor = float(data['norm_factor'])
    print(f"  Response norm_factor: {norm_factor:.6f}")

    # For input normalization, we need to compute from training data
    # Training used voltage in [-1, 1], so we normalize to same range
    # 50V -> normalized_voltage (we'll scale by max expected voltage)
    voltage_max = 100.0  # Assume max 100V for normalization

    return model, seq_length, norm_factor, voltage_max


# =============================================================================
# AI PREDICTION (Point-by-Point)
# =============================================================================
def run_ai_prediction(model, seq_length, t, V_normalized, theta_normalized_gt):
    """
    Run AI prediction point-by-point, simulating real-time inference.

    The model takes a sliding window of (voltage, theta) history and predicts
    the next theta value.
    """
    print(f"\nRunning AI prediction point-by-point...")

    n_samples = len(t)
    theta_ai = np.zeros(n_samples)

    # Initialize: use ground truth for first seq_length samples
    # (We need history to start predicting)
    theta_ai[:seq_length] = theta_normalized_gt[:seq_length]

    # Compute normalization stats for input (approximate from the window)
    # We'll normalize voltage and theta to have zero mean and unit std
    V_mean = 0.0  # Voltage is roughly centered
    V_std = 0.5   # Typical std of normalized voltage
    th_mean = 0.0
    th_std = 0.1  # Typical std of normalized theta

    with torch.no_grad():
        for i in range(seq_length, n_samples):
            # Build input window: last seq_length of (voltage, theta_predicted)
            v_window = V_normalized[i-seq_length:i]
            th_window = theta_ai[i-seq_length:i]

            # Stack into (1, seq_length, 2) tensor
            x = np.stack([v_window, th_window], axis=1)

            # Normalize (matching training normalization)
            x[:, 0] = (x[:, 0] - V_mean) / (V_std + 1e-8)
            x[:, 1] = (x[:, 1] - th_mean) / (th_std + 1e-8)

            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Predict
            y_pred_norm = model(x_tensor).cpu().numpy()[0, 0]

            # Denormalize prediction
            theta_ai[i] = y_pred_norm * th_std + th_mean

            # Progress
            if i % 1000 == 0:
                print(f"  Step {i}/{n_samples}")

    print(f"  Prediction complete.")
    return theta_ai


# =============================================================================
# VISUALIZATION
# =============================================================================
def create_visualization(t, V_raw, theta_gt, theta_ai, norm_factor):
    """Create the comparison plot."""
    print(f"\nCreating visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    t_ms = t * 1000  # Convert to milliseconds

    # --- Top Plot: Voltage Input ---
    ax1 = axes[0]
    ax1.plot(t_ms, V_raw, 'b-', linewidth=2, label='Voltage Input')
    ax1.fill_between(t_ms, 0, V_raw, alpha=0.3, color='blue')
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    # Mark step transitions
    ax1.axvline(x=T_BEFORE*1000, color='red', linewidth=1, linestyle=':', alpha=0.7)
    ax1.axvline(x=(T_BEFORE+T_STEP)*1000, color='red', linewidth=1, linestyle=':', alpha=0.7)

    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title('P4 MEMS Verification: Step Response ("The Boing Test")',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-5, VOLTAGE_HIGH * 1.1])

    # --- Bottom Plot: Angle Response ---
    ax2 = axes[1]

    # Denormalize theta for display (convert back to actual angle)
    theta_gt_actual = theta_gt * norm_factor
    theta_ai_actual = theta_ai * norm_factor

    # Ground truth (solid)
    ax2.plot(t_ms, theta_gt_actual, 'g-', linewidth=2.5,
             label='Ground Truth (Physics)', alpha=0.9)

    # AI prediction (dashed)
    ax2.plot(t_ms, theta_ai_actual, 'r--', linewidth=2,
             label='AI Prediction (LSTM)', alpha=0.9)

    ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    # Mark step transitions
    ax2.axvline(x=T_BEFORE*1000, color='gray', linewidth=1, linestyle=':', alpha=0.5)
    ax2.axvline(x=(T_BEFORE+T_STEP)*1000, color='gray', linewidth=1, linestyle=':', alpha=0.5)

    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Angle (rad)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add annotation for ringing
    # Find peak of first oscillation after step-up
    step_up_idx = int(T_BEFORE * FS)
    search_range = slice(step_up_idx, step_up_idx + int(0.005 * FS))  # First 5ms after step
    peak_idx = step_up_idx + np.argmax(theta_gt_actual[search_range])
    peak_time = t[peak_idx] * 1000
    peak_val = theta_gt_actual[peak_idx]

    ax2.annotate('Ringing\nOscillations',
                 xy=(peak_time, peak_val),
                 xytext=(peak_time + 3, peak_val * 1.2),
                 fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Compute error metrics
    # Only compare where AI has made predictions (after initial seq_length)
    seq_length = 50  # Best model uses seq_length=50
    valid_slice = slice(seq_length, len(theta_gt))

    mse = np.mean((theta_gt_actual[valid_slice] - theta_ai_actual[valid_slice])**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(theta_gt_actual[valid_slice] - theta_ai_actual[valid_slice]))

    # Add metrics box
    metrics_text = (
        f"Error Metrics:\n"
        f"  RMSE: {rmse:.6f} rad\n"
        f"  MAE:  {mae:.6f} rad\n\n"
        f"Physics:\n"
        f"  f0 = {F0:.0f} Hz\n"
        f"  Q = {Q:.0f}"
    )
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] Visualization saved: {OUTPUT_PATH}")

    return rmse, mae


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("P4 MEMS Verification: The Boing Test")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # 1. Generate step test signal
    t, V_raw = generate_step_test_signal()

    # 2. Load model and normalization stats
    model, seq_length, norm_factor, voltage_max = load_model_and_stats()

    # 3. Run ground truth physics simulation
    print(f"\nRunning ground truth physics simulation...")
    sim = MEMSSimulator(f0=F0, Q=Q)

    # Normalize voltage to [-1, 1] range (same as training data)
    V_normalized = V_raw / voltage_max

    theta_gt_raw = sim.simulate_fast(t, V_normalized)

    # Normalize theta to match training data range
    theta_gt_normalized = theta_gt_raw / norm_factor

    print(f"  Ground truth computed.")
    print(f"  Theta range: [{theta_gt_raw.min():.6f}, {theta_gt_raw.max():.6f}] rad")

    # 4. Run AI prediction (point-by-point)
    theta_ai_normalized = run_ai_prediction(
        model, seq_length, t, V_normalized, theta_gt_normalized
    )

    # 5. Create visualization
    rmse, mae = create_visualization(
        t, V_raw, theta_gt_normalized, theta_ai_normalized, norm_factor
    )

    # 6. Summary
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.6f} rad ({rmse*1000:.3f} mrad)")
    print(f"  MAE:  {mae:.6f} rad ({mae*1000:.3f} mrad)")
    print(f"\nOutput: {OUTPUT_PATH}")

    # Check if AI captured ringing
    print(f"\nDoes the AI predict the ringing oscillations?")

    # Simple check: look for oscillations in AI prediction
    # After step-up, check for sign changes in derivative
    step_up_idx = int(T_BEFORE * FS)
    theta_ai_after_step = theta_ai_normalized[step_up_idx:step_up_idx + int(0.01 * FS)]
    theta_diff = np.diff(theta_ai_after_step)
    sign_changes = np.sum(np.diff(np.sign(theta_diff)) != 0)

    if sign_changes >= 2:
        print(f"  YES - Detected {sign_changes} direction changes (ringing pattern)")
    else:
        print(f"  UNCLEAR - Only {sign_changes} direction changes detected")


if __name__ == "__main__":
    main()
