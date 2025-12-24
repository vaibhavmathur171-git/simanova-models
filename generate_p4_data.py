# -*- coding: utf-8 -*-
"""
P4: MEMS Neural Surrogate - Dataset Generator
==============================================
Simulates the time-domain physics of an electrostatic MEMS mirror
using ODE integration. Generates training data for LSTM surrogate.

Physics: 2nd-order damped oscillator with electrostatic torque
         I * theta''(t) + c * theta'(t) + k * theta(t) = Torque(V)

Output: (100, 10000, 1) voltage signals and angle responses
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import chirp
from pathlib import Path
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. MEMS PHYSICS SIMULATOR
# =============================================================================

class MEMSSimulator:
    """
    Electrostatic MEMS Mirror Simulator.

    Solves the 2nd-order equation of motion:
        I * theta''(t) + c * theta'(t) + k * theta(t) = Torque(V)

    Rewritten in normalized form:
        theta'' + 2*zeta*omega_n*theta' + omega_n^2*theta = (Torque/I)

    Where:
        omega_n = 2*pi*f0 (natural frequency)
        zeta = 1/(2*Q) (damping ratio)
        Torque = k_torque * V^2 * sign(V) (electrostatic)

    Parameters:
    -----------
    f0 : float
        Resonant frequency in Hz (default: 2000 Hz)
    Q : float
        Quality factor (default: 50, underdamped)
    k_torque : float
        Torque constant (V^2 to torque conversion)
    """

    def __init__(self, f0: float = 2000.0, Q: float = 50.0, k_torque: float = None):
        self.f0 = f0
        self.Q = Q

        # Derived parameters
        self.omega_n = 2 * np.pi * f0  # Natural frequency (rad/s)
        self.zeta = 1.0 / (2.0 * Q)    # Damping ratio

        # For underdamped system (zeta < 1):
        # Damped frequency: omega_d = omega_n * sqrt(1 - zeta^2)
        self.omega_d = self.omega_n * np.sqrt(1 - self.zeta**2)

        # Time constant for decay: tau = 1 / (zeta * omega_n)
        self.tau = 1.0 / (self.zeta * self.omega_n)

        # Auto-scale torque constant for reasonable deflection
        # Steady-state: theta_ss = k_torque * V^2 / omega_n^2
        # For V=1, we want theta_ss ~ 0.1 rad (about 6 degrees)
        if k_torque is None:
            target_theta = 0.1  # Target steady-state deflection
            self.k_torque = target_theta * self.omega_n**2
        else:
            self.k_torque = k_torque

        print(f"[MEMS] f0={f0:.0f}Hz, Q={Q:.0f}, zeta={self.zeta:.4f}")
        print(f"       omega_n={self.omega_n:.1f} rad/s, omega_d={self.omega_d:.1f} rad/s")
        print(f"       Decay time constant: {self.tau*1000:.2f} ms")
        print(f"       Torque constant: {self.k_torque:.2e}")

    def _equations(self, state: np.ndarray, t: float, V: float) -> List[float]:
        """
        State-space equations for ODE solver.

        State: [theta, theta_dot]
        Returns: [theta_dot, theta_ddot]
        """
        theta, theta_dot = state

        # Electrostatic torque: proportional to V^2, direction follows sign(V)
        # This allows bidirectional actuation
        torque = self.k_torque * V * np.abs(V)

        # theta'' = -2*zeta*omega_n*theta' - omega_n^2*theta + torque
        theta_ddot = (
            -2 * self.zeta * self.omega_n * theta_dot
            - self.omega_n**2 * theta
            + torque
        )

        return [theta_dot, theta_ddot]

    def simulate(self, t: np.ndarray, V: np.ndarray,
                 theta0: float = 0.0, theta_dot0: float = 0.0) -> np.ndarray:
        """
        Simulate MEMS response to voltage input.

        Parameters:
        -----------
        t : np.ndarray
            Time array (seconds)
        V : np.ndarray
            Voltage signal (same length as t)
        theta0 : float
            Initial angle
        theta_dot0 : float
            Initial angular velocity

        Returns:
        --------
        theta : np.ndarray
            Angle response (same length as t)
        """
        n = len(t)
        theta = np.zeros(n)
        theta[0] = theta0

        state = [theta0, theta_dot0]

        # Use piecewise integration with interpolated voltage
        for i in range(1, n):
            dt = t[i] - t[i-1]

            # Use voltage at current step
            V_current = V[i-1]

            # Integrate one step using odeint
            t_span = [t[i-1], t[i]]
            result = odeint(self._equations, state, t_span, args=(V_current,))

            state = result[-1]
            theta[i] = state[0]

        return theta

    def simulate_fast(self, t: np.ndarray, V: np.ndarray,
                      theta0: float = 0.0, theta_dot0: float = 0.0) -> np.ndarray:
        """
        Fast simulation using vectorized Euler integration.
        More efficient for large datasets.
        """
        n = len(t)
        dt = t[1] - t[0]  # Assuming uniform time step

        theta = np.zeros(n)
        theta_dot = np.zeros(n)

        theta[0] = theta0
        theta_dot[0] = theta_dot0

        # Precompute constants
        c1 = 2 * self.zeta * self.omega_n
        c2 = self.omega_n**2

        for i in range(1, n):
            # Electrostatic torque
            torque = self.k_torque * V[i-1] * np.abs(V[i-1])

            # Euler integration (semi-implicit for stability)
            theta_ddot = -c1 * theta_dot[i-1] - c2 * theta[i-1] + torque

            theta_dot[i] = theta_dot[i-1] + theta_ddot * dt
            theta[i] = theta[i-1] + theta_dot[i] * dt

        return theta


# =============================================================================
# 2. SIGNAL GENERATORS
# =============================================================================

def generate_step_signal(t: np.ndarray,
                         n_steps: int = None,
                         amplitude_range: Tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """
    Generate random step function signal.

    Parameters:
    -----------
    t : np.ndarray
        Time array
    n_steps : int
        Number of step transitions (random if None)
    amplitude_range : tuple
        Range for step amplitudes

    Returns:
    --------
    V : np.ndarray
        Step voltage signal
    """
    n = len(t)
    T = t[-1] - t[0]

    if n_steps is None:
        n_steps = np.random.randint(3, 10)

    # Random step times
    step_times = np.sort(np.random.uniform(0, T, n_steps))
    step_times = np.concatenate([[0], step_times, [T + 0.001]])

    # Random amplitudes
    amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], n_steps + 1)

    # Build signal
    V = np.zeros(n)
    for i in range(len(step_times) - 1):
        mask = (t >= step_times[i]) & (t < step_times[i + 1])
        V[mask] = amplitudes[i]

    return V


def generate_chirp_signal(t: np.ndarray,
                          f0_range: Tuple[float, float] = (100, 5000),
                          amplitude: float = 1.0) -> np.ndarray:
    """
    Generate chirp (frequency sweep) signal.

    Parameters:
    -----------
    t : np.ndarray
        Time array
    f0_range : tuple
        Start and end frequencies (Hz)
    amplitude : float
        Signal amplitude

    Returns:
    --------
    V : np.ndarray
        Chirp voltage signal
    """
    f0 = np.random.uniform(f0_range[0], f0_range[0] + 500)
    f1 = np.random.uniform(f0_range[1] - 1000, f0_range[1])

    # Random amplitude
    amp = np.random.uniform(0.5, 1.0) * amplitude

    # Linear chirp
    V = amp * chirp(t, f0=f0, f1=f1, t1=t[-1], method='linear')

    return V


def generate_noise_signal(t: np.ndarray,
                          amplitude: float = 1.0,
                          cutoff_hz: float = 3000) -> np.ndarray:
    """
    Generate band-limited random noise signal.

    Parameters:
    -----------
    t : np.ndarray
        Time array
    amplitude : float
        RMS amplitude
    cutoff_hz : float
        Low-pass cutoff frequency

    Returns:
    --------
    V : np.ndarray
        Noise voltage signal
    """
    n = len(t)
    dt = t[1] - t[0]
    fs = 1.0 / dt

    # Generate white noise
    noise = np.random.randn(n)

    # Simple moving average filter for band-limiting
    # Filter width corresponds to cutoff frequency
    filter_width = max(1, int(fs / cutoff_hz / 2))
    kernel = np.ones(filter_width) / filter_width

    # Apply filter
    V = np.convolve(noise, kernel, mode='same')

    # Normalize to desired amplitude
    V = V / np.std(V) * amplitude * np.random.uniform(0.3, 0.8)

    return V


def generate_sine_burst(t: np.ndarray,
                        freq_range: Tuple[float, float] = (500, 4000),
                        amplitude: float = 1.0) -> np.ndarray:
    """
    Generate sine wave burst signal (on/off periods).
    """
    n = len(t)
    T = t[-1] - t[0]

    # Random frequency
    freq = np.random.uniform(freq_range[0], freq_range[1])
    amp = np.random.uniform(0.5, 1.0) * amplitude

    # Random burst timing
    n_bursts = np.random.randint(2, 5)
    burst_starts = np.sort(np.random.uniform(0, T * 0.7, n_bursts))
    burst_durations = np.random.uniform(T * 0.05, T * 0.2, n_bursts)

    V = np.zeros(n)
    for start, duration in zip(burst_starts, burst_durations):
        mask = (t >= start) & (t < start + duration)
        V[mask] = amp * np.sin(2 * np.pi * freq * (t[mask] - start))

    return V


def generate_mixed_signal(t: np.ndarray) -> np.ndarray:
    """
    Generate a mixed signal combining multiple types.
    """
    signal_type = np.random.choice(['step', 'chirp', 'noise', 'sine', 'mixed'])

    if signal_type == 'step':
        return generate_step_signal(t)
    elif signal_type == 'chirp':
        return generate_chirp_signal(t)
    elif signal_type == 'noise':
        return generate_noise_signal(t)
    elif signal_type == 'sine':
        return generate_sine_burst(t)
    else:
        # Mixed: combine step with noise or sine
        base = generate_step_signal(t, n_steps=np.random.randint(2, 5))
        overlay_type = np.random.choice(['noise', 'sine'])
        if overlay_type == 'noise':
            overlay = generate_noise_signal(t, amplitude=0.3)
        else:
            overlay = generate_sine_burst(t, amplitude=0.3)
        return np.clip(base + overlay, -1, 1)


# =============================================================================
# 3. DATA GENERATION
# =============================================================================

def generate_dataset(n_experiments: int = 100,
                     duration: float = 0.1,
                     fs: float = 100000,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate complete MEMS dataset.

    Parameters:
    -----------
    n_experiments : int
        Number of sequences to generate
    duration : float
        Duration of each sequence (seconds)
    fs : float
        Sampling frequency (Hz)
    seed : int
        Random seed

    Returns:
    --------
    signals : np.ndarray, shape (n_experiments, n_samples, 1)
        Normalized voltage signals
    responses : np.ndarray, shape (n_experiments, n_samples, 1)
        Normalized angle responses
    """
    np.random.seed(seed)

    # Time array
    dt = 1.0 / fs
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)

    print(f"\n{'='*60}")
    print(f"P4: MEMS DATASET GENERATION")
    print(f"{'='*60}")
    print(f"\nParameters:")
    print(f"  Experiments: {n_experiments}")
    print(f"  Duration: {duration*1000:.1f} ms")
    print(f"  Samples: {n_samples}")
    print(f"  Sample Rate: {fs/1000:.0f} kHz")
    print(f"  Time Step: {dt*1e6:.1f} us")

    # Initialize simulator
    print(f"\nInitializing MEMS Simulator...")
    sim = MEMSSimulator(f0=2000.0, Q=50.0)  # k_torque auto-calculated

    # Storage arrays
    all_signals = np.zeros((n_experiments, n_samples))
    all_responses = np.zeros((n_experiments, n_samples))

    # Track statistics for normalization
    max_angle = 0.0

    print(f"\nGenerating {n_experiments} experiments...")
    print("-" * 60)

    for i in range(n_experiments):
        # Generate random input signal (pre-normalized to [-1, 1])
        V = generate_mixed_signal(t)

        # Ensure signal is in [-1, 1]
        V = np.clip(V, -1, 1)

        # Run simulation
        theta = sim.simulate_fast(t, V)

        # Store raw data
        all_signals[i] = V
        all_responses[i] = theta

        # Track max angle for normalization
        max_angle = max(max_angle, np.abs(theta).max())

        # Progress
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:3d}/{n_experiments}] V: [{V.min():.2f}, {V.max():.2f}] | "
                  f"theta: [{theta.min():.4f}, {theta.max():.4f}]")

    # Normalize responses to [-1, 1]
    print(f"\nNormalizing...")
    print(f"  Max angle before normalization: {max_angle:.6f}")

    # Add small epsilon to avoid division by zero
    norm_factor = max_angle + 1e-10
    all_responses = all_responses / norm_factor

    print(f"  Response range after normalization: [{all_responses.min():.4f}, {all_responses.max():.4f}]")

    # Reshape to (n_experiments, n_samples, 1) for LSTM
    signals = all_signals[:, :, np.newaxis]
    responses = all_responses[:, :, np.newaxis]

    print(f"\nFinal shapes:")
    print(f"  signals: {signals.shape}")
    print(f"  responses: {responses.shape}")

    return signals, responses, t, sim, norm_factor


# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def create_visualization(t: np.ndarray, signals: np.ndarray, responses: np.ndarray,
                         sim: MEMSSimulator, output_path: str):
    """
    Create physics visualization showing step response with ringing.
    """
    print(f"\nCreating visualization...")

    # Generate a clean step response for visualization
    duration = 0.01  # 10ms to see ringing clearly
    n_viz = int(duration * 100000)  # 100kHz
    t_viz = np.linspace(0, duration, n_viz)

    # Create a simple step signal
    V_step = np.zeros(n_viz)
    V_step[int(0.001 * 100000):] = 0.8  # Step at 1ms

    # Simulate
    theta_step = sim.simulate_fast(t_viz, V_step)

    # Normalize for display
    theta_norm = theta_step / (np.abs(theta_step).max() + 1e-10)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Plot 1: Step Response (main viz) ---
    ax1 = axes[0, 0]
    ax1.plot(t_viz * 1000, V_step, 'b-', linewidth=2, label='Voltage Input', alpha=0.8)
    ax1.plot(t_viz * 1000, theta_norm, 'r-', linewidth=2, label='Angle Response (norm)')
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax1.axvline(x=1.0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_ylabel('Amplitude (normalized)', fontsize=11)
    ax1.set_title('Step Response - Underdamped Ringing (Q=50)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, duration * 1000])

    # Add annotation for ringing
    ax1.annotate('Ringing\n(Q=50)', xy=(2.5, 0.8), fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # --- Plot 2: Frequency Response Concept ---
    ax2 = axes[0, 1]
    freq = np.linspace(100, 5000, 1000)
    omega = 2 * np.pi * freq
    omega_n = sim.omega_n
    zeta = sim.zeta

    # Transfer function magnitude: |H(jw)| = 1 / sqrt((1-(w/wn)^2)^2 + (2*zeta*w/wn)^2)
    r = omega / omega_n
    H_mag = 1.0 / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
    H_db = 20 * np.log10(H_mag)

    ax2.plot(freq, H_db, 'g-', linewidth=2)
    ax2.axvline(x=sim.f0, color='red', linewidth=1.5, linestyle='--', label=f'f0 = {sim.f0:.0f} Hz')
    ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('Magnitude (dB)', fontsize=11)
    ax2.set_title('Frequency Response (Bode Magnitude)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([100, 5000])
    ax2.set_ylim([-40, 40])

    # --- Plot 3: Sample from Dataset ---
    ax3 = axes[1, 0]
    # Use first 1000 samples (10ms) of first experiment
    idx = 0
    n_show = 1000
    t_show = t[:n_show] * 1000

    ax3.plot(t_show, signals[idx, :n_show, 0], 'b-', linewidth=1.5, label='Voltage', alpha=0.8)
    ax3.plot(t_show, responses[idx, :n_show, 0], 'r-', linewidth=1.5, label='Angle (norm)')
    ax3.set_xlabel('Time (ms)', fontsize=11)
    ax3.set_ylabel('Amplitude', fontsize=11)
    ax3.set_title(f'Dataset Sample (Experiment #{idx+1})', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Dataset Statistics ---
    ax4 = axes[1, 1]

    # Histogram of signal types
    signal_energies = np.sqrt(np.mean(signals**2, axis=(1, 2)))
    response_energies = np.sqrt(np.mean(responses**2, axis=(1, 2)))

    ax4.hist(signal_energies, bins=20, alpha=0.6, label='Signal RMS', color='blue')
    ax4.hist(response_energies, bins=20, alpha=0.6, label='Response RMS', color='red')
    ax4.set_xlabel('RMS Amplitude', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Dataset Distribution', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add text box with parameters
    params_text = (
        f"MEMS Parameters:\n"
        f"  f0 = {sim.f0:.0f} Hz\n"
        f"  Q = {sim.Q:.0f}\n"
        f"  zeta = {sim.zeta:.4f}\n"
        f"  tau = {sim.tau*1000:.2f} ms\n\n"
        f"Dataset:\n"
        f"  {signals.shape[0]} experiments\n"
        f"  {signals.shape[1]} samples each\n"
        f"  100 kHz sampling"
    )
    fig.text(0.98, 0.02, params_text, fontsize=9, family='monospace',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] Visualization saved: {output_path}")


# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    # Paths
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    output_path = data_dir / "p4_mems_dataset.npz"
    viz_path = data_dir / "p4_physics_viz.png"

    # Generate dataset
    signals, responses, t, sim, norm_factor = generate_dataset(
        n_experiments=100,
        duration=0.1,      # 0.1 seconds = 100ms
        fs=100000,         # 100 kHz
        seed=42
    )

    # Save dataset
    print(f"\nSaving dataset...")
    np.savez_compressed(
        output_path,
        signals=signals,
        responses=responses,
        time=t,
        norm_factor=norm_factor,
        f0=sim.f0,
        Q=sim.Q,
        metadata={
            'description': 'MEMS electrostatic mirror simulation',
            'n_experiments': signals.shape[0],
            'n_samples': signals.shape[1],
            'duration_s': 0.1,
            'fs_hz': 100000,
            'f0_hz': sim.f0,
            'Q': sim.Q,
            'zeta': sim.zeta
        }
    )

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Dataset saved: {output_path} ({file_size:.2f} MB)")

    # Create visualization
    create_visualization(t, signals, responses, sim, str(viz_path))

    # Summary
    print(f"\n{'='*60}")
    print(f"P4 DATASET GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  - {output_path}")
    print(f"  - {viz_path}")
    print(f"\nDataset Shape:")
    print(f"  - signals:   {signals.shape} (experiments, samples, features)")
    print(f"  - responses: {responses.shape} (experiments, samples, features)")
    print(f"\nPhysics Parameters:")
    print(f"  - Resonant Frequency: {sim.f0:.0f} Hz")
    print(f"  - Q-Factor: {sim.Q:.0f}")
    print(f"  - Damping Ratio: {sim.zeta:.4f}")
    print(f"  - Decay Time: {sim.tau*1000:.2f} ms")
    print(f"\nNormalization:")
    print(f"  - Voltage: already in [-1, 1]")
    print(f"  - Angle: divided by {norm_factor:.6f}")


if __name__ == "__main__":
    main()
