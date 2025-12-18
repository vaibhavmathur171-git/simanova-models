# -*- coding: utf-8 -*-
"""
P1 Neural Surrogate: Design of Experiments (DOE) Sweep
======================================================
Systematically evaluate model capacity for the inverse grating problem.

Sweeps:
    - Model Depth: [2, 4, 6] hidden layers
    - Model Width: [32, 64, 128] neurons per layer
    - Dataset Size: [1k, 10k, 50k] samples
    - Noise Level: [0, 0.5, 1.0] degrees Gaussian

Outputs:
    - p1_doe_results.csv with MAE, RMSE, training time
    - Best model saved to models/p1_mono_model.pth

Usage:
    python p1_doe_sweep.py
"""

import os
import time
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # DOE Parameters
    "depths": [2, 4, 6],
    "widths": [32, 64, 128],
    "dataset_sizes": [1_000, 10_000, 50_000],
    "noise_levels": [0.0, 0.5, 1.0],

    # Training Hyperparameters
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "test_split": 0.2,
    "random_seed": 42,

    # Physics Parameters
    "wavelength_nm": 532,
    "n_out": 1.5,
    "angle_min": -80.0,
    "angle_max": -30.0,

    # Output
    "output_dir": "data",
    "model_dir": "models",
    "results_file": "p1_doe_results.csv",
}

# Set random seeds for reproducibility
torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# PHYSICS: GRATING EQUATION
# =============================================================================
def grating_equation(angle_deg: np.ndarray, wavelength_nm: float = 532, n_out: float = 1.5) -> np.ndarray:
    """
    Analytical solution to the Grating Equation for first-order diffraction.

    Physics:
        n_out * sin(θ_out) = n_in * sin(θ_in) + m * λ / Λ

    For normal incidence (θ_in = 0) and m = -1:
        Λ = -λ / (n_out * sin(θ_out))

    Args:
        angle_deg: Output diffraction angle(s) in degrees
        wavelength_nm: Operating wavelength in nanometers
        n_out: Output medium refractive index

    Returns:
        Grating period(s) Λ in nanometers
    """
    theta_rad = np.radians(angle_deg)
    m = -1  # First-order diffraction

    # Avoid division by zero
    sin_theta = np.sin(theta_rad)
    sin_theta = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)

    period = (m * wavelength_nm) / (n_out * sin_theta)
    return np.abs(period)


def generate_dataset(
    n_samples: int,
    noise_deg: float = 0.0,
    angle_range: tuple = (-80.0, -30.0),
    wavelength_nm: float = 532,
    n_out: float = 1.5
) -> tuple:
    """
    Generate synthetic training data using the Grating Equation.

    Args:
        n_samples: Number of samples to generate
        noise_deg: Gaussian noise standard deviation (degrees)
        angle_range: (min, max) angle range in degrees
        wavelength_nm: Operating wavelength
        n_out: Output refractive index

    Returns:
        (angles, periods) as numpy arrays
    """
    # Uniform sampling of angles
    angles = np.random.uniform(angle_range[0], angle_range[1], n_samples)

    # Add Gaussian noise to simulate measurement uncertainty
    if noise_deg > 0:
        noise = np.random.normal(0, noise_deg, n_samples)
        angles_noisy = angles + noise
        # Clip to valid range
        angles_noisy = np.clip(angles_noisy, angle_range[0], angle_range[1])
    else:
        angles_noisy = angles

    # Compute ground truth periods
    periods = grating_equation(angles_noisy, wavelength_nm, n_out)

    return angles_noisy.astype(np.float32), periods.astype(np.float32)


# =============================================================================
# MODEL: FLEXIBLE MLP ARCHITECTURE
# =============================================================================
class FlexibleMLP(nn.Module):
    """
    Multilayer Perceptron with configurable depth and width.

    Architecture:
        Input(1) -> [Linear(width) -> ReLU] x depth -> Linear(1)
    """

    def __init__(self, depth: int = 4, width: int = 64):
        """
        Args:
            depth: Number of hidden layers
            width: Neurons per hidden layer
        """
        super(FlexibleMLP, self).__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(1, width))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(width, 1))

        self.network = nn.Sequential(*layers)

        # Store architecture info
        self.depth = depth
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def count_parameters(self) -> int:
        """Return total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# TRAINING PIPELINE
# =============================================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    verbose: bool = False
) -> float:
    """
    Train the model and return training time.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        verbose: Print progress

    Returns:
        Training time in seconds
    """
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if verbose and (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")

    training_time = time.time() - start_time
    return training_time


def evaluate_model(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor) -> dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained PyTorch model
        X_test: Test inputs
        y_test: Test targets

    Returns:
        Dictionary with MAE, RMSE, and other metrics
    """
    model.eval()
    model.to(DEVICE)

    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    with torch.no_grad():
        predictions = model(X_test)

    # Move back to CPU for numpy operations
    predictions = predictions.cpu().numpy().flatten()
    targets = y_test.cpu().numpy().flatten()

    # Calculate metrics
    errors = np.abs(predictions - targets)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)

    return {
        "mae_nm": mae,
        "rmse_nm": rmse,
        "max_error_nm": max_error,
    }


# =============================================================================
# DOE SWEEP
# =============================================================================
def run_doe_sweep(config: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Run full Design of Experiments sweep.

    Args:
        config: Configuration dictionary
        verbose: Print progress

    Returns:
        DataFrame with all experiment results
    """
    results = []

    # Generate all parameter combinations
    param_grid = list(itertools.product(
        config["depths"],
        config["widths"],
        config["dataset_sizes"],
        config["noise_levels"]
    ))

    total_experiments = len(param_grid)
    best_mae = float("inf")
    best_model = None
    best_config = None

    print("=" * 70)
    print("P1 NEURAL SURROGATE: DESIGN OF EXPERIMENTS")
    print("=" * 70)
    print(f"Total experiments: {total_experiments}")
    print(f"Device: {DEVICE}")
    print(f"Parameters: depths={config['depths']}, widths={config['widths']}")
    print(f"            samples={config['dataset_sizes']}, noise={config['noise_levels']}")
    print("=" * 70)

    for idx, (depth, width, n_samples, noise) in enumerate(param_grid, 1):
        exp_id = f"EXP_{idx:03d}"

        if verbose:
            print(f"\n[{idx}/{total_experiments}] {exp_id}")
            print(f"  Config: depth={depth}, width={width}, samples={n_samples:,}, noise={noise}°")

        # Generate dataset
        angles, periods = generate_dataset(
            n_samples=n_samples,
            noise_deg=noise,
            angle_range=(config["angle_min"], config["angle_max"]),
            wavelength_nm=config["wavelength_nm"],
            n_out=config["n_out"]
        )

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            angles.reshape(-1, 1),
            periods.reshape(-1, 1),
            test_size=config["test_split"],
            random_state=config["random_seed"]
        )

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)

        # Create data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True
        )

        # Initialize model
        model = FlexibleMLP(depth=depth, width=width)
        n_params = model.count_parameters()

        # Train
        train_time = train_model(
            model,
            train_loader,
            epochs=config["epochs"],
            lr=config["learning_rate"],
            verbose=False
        )

        # Evaluate
        metrics = evaluate_model(model, X_test_t, y_test_t)

        if verbose:
            print(f"  Parameters: {n_params:,}")
            print(f"  Training time: {train_time:.2f}s")
            print(f"  MAE: {metrics['mae_nm']:.4f} nm | RMSE: {metrics['rmse_nm']:.4f} nm")

        # Track best model
        if metrics["mae_nm"] < best_mae:
            best_mae = metrics["mae_nm"]
            best_model = model
            best_config = {
                "depth": depth,
                "width": width,
                "n_samples": n_samples,
                "noise": noise
            }

        # Record results
        results.append({
            "experiment_id": exp_id,
            "depth": depth,
            "width": width,
            "n_samples": n_samples,
            "noise_deg": noise,
            "n_parameters": n_params,
            "train_time_s": round(train_time, 2),
            "mae_nm": round(metrics["mae_nm"], 4),
            "rmse_nm": round(metrics["rmse_nm"], 4),
            "max_error_nm": round(metrics["max_error_nm"], 4),
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 70)
    print("DOE COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\nBest Configuration:")
    print(f"  Depth: {best_config['depth']} layers")
    print(f"  Width: {best_config['width']} neurons")
    print(f"  Samples: {best_config['n_samples']:,}")
    print(f"  Noise: {best_config['noise']}°")
    print(f"  MAE: {best_mae:.4f} nm")

    return df, best_model


def save_results(df: pd.DataFrame, model: nn.Module, config: dict):
    """Save DOE results and best model."""

    # Create directories
    output_dir = Path(config["output_dir"])
    model_dir = Path(config["model_dir"])
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # Save CSV
    csv_path = output_dir / config["results_file"]
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Save best model
    model_path = model_dir / "p1_mono_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved to: {model_path}")

    # Save summary statistics
    summary_path = output_dir / "p1_doe_summary.txt"
    with open(summary_path, "w") as f:
        f.write("P1 Neural Surrogate - DOE Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Experiments: {len(df)}\n\n")
        f.write("Best Results:\n")
        f.write(df.nsmallest(5, "mae_nm").to_string())
        f.write("\n\nWorst Results:\n")
        f.write(df.nlargest(5, "mae_nm").to_string())
        f.write("\n\nStatistics:\n")
        f.write(df.describe().to_string())
    print(f"Summary saved to: {summary_path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Run the full DOE sweep."""

    print("\n" + "=" * 70)
    print("  P1 INVERSE WAVEGUIDE SOLVER - DESIGN OF EXPERIMENTS")
    print("  Neural Surrogate Capacity Characterization")
    print("=" * 70 + "\n")

    # Run sweep
    df, best_model = run_doe_sweep(CONFIG, verbose=True)

    # Save results
    save_results(df, best_model, CONFIG)

    # Display final table
    print("\n" + "=" * 70)
    print("FULL RESULTS TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\n✓ DOE sweep complete!")


if __name__ == "__main__":
    main()
