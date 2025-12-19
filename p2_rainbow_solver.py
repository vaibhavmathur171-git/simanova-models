# -*- coding: utf-8 -*-
"""
P2: The Rainbow Solver - Multi-Spectral Grating Optimization
=============================================================
Physics-Informed Neural Network for chromatic dispersion correction.

Finds optimal grating pitch (Lambda) that minimizes angular error across
RGB spectrum (450nm, 532nm, 635nm) while accounting for material dispersion.

Author: Vaibhav Mathur
"""

import os
import time
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Spectral wavelengths (nm)
    "lambda_blue": 450.0,
    "lambda_green": 532.0,
    "lambda_red": 635.0,

    # Photopic weights (human eye sensitivity)
    "weight_blue": 0.2,
    "weight_green": 0.6,
    "weight_red": 0.2,

    # Training parameters
    "n_samples": 50_000,
    "batch_size": 128,
    "epochs": 150,
    "test_split": 0.2,
    "random_seed": 42,

    # Angle range (degrees)
    "angle_min": -75.0,
    "angle_max": -25.0,

    # DOE sweep parameters
    "doe_num_blocks": [2, 4, 8],
    "doe_learning_rates": [1e-3, 1e-4],

    # Output paths
    "output_dir": "data",
    "model_dir": "models",
}

torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# SELLMEIER PHYSICS: MATERIAL DISPERSION
# =============================================================================
# Glass coefficients for three-term Sellmeier equation:
# n^2 - 1 = B1*L^2/(L^2-C1) + B2*L^2/(L^2-C2) + B3*L^2/(L^2-C3)
# where L = wavelength in micrometers

GLASS_COEFFICIENTS = {
    "BK7": {
        "B1": 1.03961212,
        "B2": 0.231792344,
        "B3": 1.01046945,
        "C1": 0.00600069867,  # um^2
        "C2": 0.0200179144,   # um^2
        "C3": 103.560653,     # um^2
    },
    "HIGH_INDEX": {
        # Generic high-index glass (n ~ 1.9 at 532nm)
        "B1": 1.43134930,
        "B2": 0.65054713,
        "B3": 5.34164330,
        "C1": 0.00795267,
        "C2": 0.02190550,
        "C3": 82.5693560,
    },
}

# Material ID mapping
MATERIAL_IDS = {"BK7": 0, "HIGH_INDEX": 1}


def get_refractive_index(lambda_nm: float, glass_type: str = "BK7") -> float:
    """
    Calculate refractive index using the three-term Sellmeier Equation.

    The Sellmeier equation models dispersion in optical glasses:
    n^2(lambda) - 1 = sum_i [ B_i * lambda^2 / (lambda^2 - C_i) ]

    Args:
        lambda_nm: Wavelength in nanometers
        glass_type: "BK7" or "HIGH_INDEX"

    Returns:
        Refractive index n at the specified wavelength
    """
    # Convert nm to micrometers for Sellmeier equation
    L = lambda_nm / 1000.0
    L_sq = L ** 2

    coeff = GLASS_COEFFICIENTS.get(glass_type, GLASS_COEFFICIENTS["BK7"])

    # Three-term Sellmeier formula
    n_sq_minus_1 = (
        (coeff["B1"] * L_sq) / (L_sq - coeff["C1"]) +
        (coeff["B2"] * L_sq) / (L_sq - coeff["C2"]) +
        (coeff["B3"] * L_sq) / (L_sq - coeff["C3"])
    )

    n = np.sqrt(1 + n_sq_minus_1)
    return n


def get_refractive_index_batch(lambda_nm: np.ndarray, material_ids: np.ndarray) -> np.ndarray:
    """
    Vectorized refractive index calculation for batch processing.

    Args:
        lambda_nm: Array of wavelengths in nm
        material_ids: Array of material IDs (0=BK7, 1=HIGH_INDEX)

    Returns:
        Array of refractive indices
    """
    n_values = np.zeros_like(lambda_nm, dtype=np.float64)

    for glass_type, mat_id in MATERIAL_IDS.items():
        mask = (material_ids == mat_id)
        if np.any(mask):
            L = lambda_nm[mask] / 1000.0
            L_sq = L ** 2
            coeff = GLASS_COEFFICIENTS[glass_type]

            n_sq_minus_1 = (
                (coeff["B1"] * L_sq) / (L_sq - coeff["C1"]) +
                (coeff["B2"] * L_sq) / (L_sq - coeff["C2"]) +
                (coeff["B3"] * L_sq) / (L_sq - coeff["C3"])
            )
            n_values[mask] = np.sqrt(1 + n_sq_minus_1)

    return n_values


# =============================================================================
# GRATING PHYSICS
# =============================================================================
def grating_pitch_from_angle(
    angle_deg: np.ndarray,
    lambda_nm: float,
    n_out: np.ndarray,
    m: int = -1
) -> np.ndarray:
    """
    Calculate grating pitch from diffraction angle using the Grating Equation.

    Grating Equation: n_out * sin(theta_out) = n_in * sin(theta_in) + m * lambda / Lambda

    For normal incidence (theta_in = 0):
        Lambda = m * lambda / (n_out * sin(theta_out))

    Args:
        angle_deg: Output diffraction angle in degrees
        lambda_nm: Wavelength in nanometers
        n_out: Output medium refractive index (can be array)
        m: Diffraction order (default -1)

    Returns:
        Grating pitch Lambda in nanometers
    """
    theta_rad = np.radians(angle_deg)
    sin_theta = np.sin(theta_rad)

    # Avoid division by zero
    sin_theta = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)

    pitch = (m * lambda_nm) / (n_out * sin_theta)
    return np.abs(pitch)


def angle_from_pitch(
    pitch_nm: np.ndarray,
    lambda_nm: float,
    n_out: np.ndarray,
    m: int = -1
) -> np.ndarray:
    """
    Calculate diffraction angle from grating pitch (inverse of above).

    Args:
        pitch_nm: Grating pitch in nanometers
        lambda_nm: Wavelength in nanometers
        n_out: Output medium refractive index
        m: Diffraction order

    Returns:
        Diffraction angle in degrees
    """
    sin_theta = (m * lambda_nm) / (n_out * pitch_nm)

    # Clamp to valid range for arcsin
    sin_theta = np.clip(sin_theta, -1.0, 1.0)

    theta_rad = np.arcsin(sin_theta)
    return np.degrees(theta_rad)


# =============================================================================
# DATA GENERATION
# =============================================================================
def generate_rainbow_dataset(
    n_samples: int,
    angle_range: tuple,
    lambda_ref: float = 532.0,  # Reference wavelength (Green)
) -> tuple:
    """
    Generate training data for the Rainbow Solver.

    Input: [target_angle, material_id]
    Output: pitch_nm that satisfies grating equation for Green (532nm) reference

    Args:
        n_samples: Number of samples to generate
        angle_range: (min, max) angle in degrees
        lambda_ref: Reference wavelength for pitch calculation

    Returns:
        (X, y) where X is [angle, material_id] and y is pitch_nm
    """
    # Random target angles
    angles = np.random.uniform(angle_range[0], angle_range[1], n_samples)

    # Random material selection (0=BK7, 1=HIGH_INDEX)
    material_ids = np.random.randint(0, 2, n_samples)

    # Get refractive index for reference wavelength (Green 532nm)
    n_out = get_refractive_index_batch(
        np.full(n_samples, lambda_ref),
        material_ids
    )

    # Calculate pitch that gives the target angle at reference wavelength
    pitch_nm = grating_pitch_from_angle(angles, lambda_ref, n_out)

    # Create input features: [angle, material_id]
    X = np.column_stack([angles, material_ids]).astype(np.float32)
    y = pitch_nm.astype(np.float32).reshape(-1, 1)

    return X, y


# =============================================================================
# MODEL ARCHITECTURE: SPECTRAL RESNET
# =============================================================================
class ResidualBlock(nn.Module):
    """
    Residual Block with LayerNorm for stable training.

    Architecture:
        Input -> Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm -> (+Input) -> ReLU
    """

    def __init__(self, hidden_dim: int):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.relu(out)
        return out


class SpectralResNet(nn.Module):
    """
    ResNet for multi-spectral grating optimization.

    Architecture:
        Input(2) -> Embedding(128) -> N Residual Blocks -> Output(1)

    The network learns to predict optimal grating pitch from
    (target_angle, material_id) inputs.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_blocks: int = 6):
        super(SpectralResNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # Input embedding layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Stack of residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input embedding
        x = self.input_layer(x)

        # Residual blocks
        x = self.residual_blocks(x)

        # Output prediction
        x = self.output_layer(x)

        return x

    def count_parameters(self) -> int:
        """Return total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# CUSTOM LOSS FUNCTION: WEIGHTED SPECTRAL LOSS
# =============================================================================
class WeightedSpectralLoss(nn.Module):
    """
    Physics-informed loss function for multi-spectral optimization.

    Calculates MSE of predicted pitch against theoretical requirement
    for Red, Green, and Blue wavelengths separately, then applies
    photopic sensitivity weights.

    Weights mirror human eye sensitivity:
        - Green (532nm): 0.6 (peak sensitivity)
        - Red (635nm): 0.2
        - Blue (450nm): 0.2
    """

    def __init__(
        self,
        lambda_blue: float = 450.0,
        lambda_green: float = 532.0,
        lambda_red: float = 635.0,
        weight_blue: float = 0.2,
        weight_green: float = 0.6,
        weight_red: float = 0.2,
    ):
        super(WeightedSpectralLoss, self).__init__()

        self.lambda_blue = lambda_blue
        self.lambda_green = lambda_green
        self.lambda_red = lambda_red

        self.weight_blue = weight_blue
        self.weight_green = weight_green
        self.weight_red = weight_red

        self.mse = nn.MSELoss(reduction='none')

    def forward(
        self,
        pred_pitch: torch.Tensor,
        target_pitch: torch.Tensor,
        material_ids: torch.Tensor,
        target_angles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted spectral loss.

        Args:
            pred_pitch: Predicted grating pitch (N, 1)
            target_pitch: Ground truth pitch for Green reference (N, 1)
            material_ids: Material IDs for each sample (N,)
            target_angles: Target diffraction angles (N,)

        Returns:
            Weighted scalar loss value
        """
        # Move to numpy for physics calculations
        pred_np = pred_pitch.detach().cpu().numpy().flatten()
        mat_ids_np = material_ids.detach().cpu().numpy().flatten()
        angles_np = target_angles.detach().cpu().numpy().flatten()

        batch_size = pred_pitch.shape[0]

        # Get refractive indices for each wavelength
        n_blue = get_refractive_index_batch(
            np.full(batch_size, self.lambda_blue), mat_ids_np
        )
        n_green = get_refractive_index_batch(
            np.full(batch_size, self.lambda_green), mat_ids_np
        )
        n_red = get_refractive_index_batch(
            np.full(batch_size, self.lambda_red), mat_ids_np
        )

        # Calculate ideal pitch for each wavelength to achieve target angle
        pitch_blue_ideal = grating_pitch_from_angle(angles_np, self.lambda_blue, n_blue)
        pitch_green_ideal = grating_pitch_from_angle(angles_np, self.lambda_green, n_green)
        pitch_red_ideal = grating_pitch_from_angle(angles_np, self.lambda_red, n_red)

        # Convert to tensors
        pitch_blue_t = torch.tensor(pitch_blue_ideal, dtype=torch.float32, device=pred_pitch.device).reshape(-1, 1)
        pitch_green_t = torch.tensor(pitch_green_ideal, dtype=torch.float32, device=pred_pitch.device).reshape(-1, 1)
        pitch_red_t = torch.tensor(pitch_red_ideal, dtype=torch.float32, device=pred_pitch.device).reshape(-1, 1)

        # Calculate MSE for each wavelength
        loss_blue = self.mse(pred_pitch, pitch_blue_t).mean()
        loss_green = self.mse(pred_pitch, pitch_green_t).mean()
        loss_red = self.mse(pred_pitch, pitch_red_t).mean()

        # Weighted combination
        total_loss = (
            self.weight_blue * loss_blue +
            self.weight_green * loss_green +
            self.weight_red * loss_red
        )

        return total_loss


class SimpleMSELoss(nn.Module):
    """Simple MSE loss for baseline comparison"""

    def __init__(self):
        super(SimpleMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_pitch, target_pitch, material_ids=None, target_angles=None):
        return self.mse(pred_pitch, target_pitch)


# =============================================================================
# TRAINING PIPELINE
# =============================================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    epochs: int = 100,
    lr: float = 1e-3,
    use_spectral_loss: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Train the SpectralResNet model.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function (WeightedSpectralLoss or SimpleMSELoss)
        epochs: Number of training epochs
        lr: Learning rate
        use_spectral_loss: Whether loss_fn expects extra args
        verbose: Print progress

    Returns:
        Dictionary with training history
    """
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
    }

    start_time = time.time()
    best_val_mae = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(X_batch)

            if use_spectral_loss:
                # Extract angle and material_id from input
                angles = X_batch[:, 0]
                mat_ids = X_batch[:, 1]
                loss = loss_fn(predictions, y_batch, mat_ids, angles)
            else:
                loss = loss_fn(predictions, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        n_val_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                predictions = model(X_batch)

                if use_spectral_loss:
                    angles = X_batch[:, 0]
                    mat_ids = X_batch[:, 1]
                    loss = loss_fn(predictions, y_batch, mat_ids, angles)
                else:
                    loss = loss_fn(predictions, y_batch)

                val_loss += loss.item()

                # Calculate MAE in nm
                mae = torch.abs(predictions - y_batch).sum().item()
                val_mae += mae
                n_val_samples += y_batch.shape[0]

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / n_val_samples

        history["val_loss"].append(avg_val_loss)
        history["val_mae"].append(avg_val_mae)

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train_loss:.4f} | "
                  f"Val: {avg_val_loss:.4f} | "
                  f"MAE: {avg_val_mae:.3f} nm")

    training_time = time.time() - start_time
    history["training_time"] = training_time
    history["best_val_mae"] = best_val_mae

    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    scaler_y: StandardScaler = None,
) -> dict:
    """
    Evaluate model on test set.

    Returns:
        Dictionary with MAE, RMSE, and spectral errors
    """
    model.eval()
    model.to(DEVICE)

    all_preds = []
    all_targets = []
    all_angles = []
    all_mat_ids = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            predictions = model(X_batch).cpu().numpy()

            all_preds.append(predictions)
            all_targets.append(y_batch.numpy())
            all_angles.append(X_batch[:, 0].cpu().numpy())
            all_mat_ids.append(X_batch[:, 1].cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    angles = np.concatenate(all_angles)
    mat_ids = np.concatenate(all_mat_ids).astype(int)

    # If scaling was applied, inverse transform
    if scaler_y is not None:
        preds = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
        targets = scaler_y.inverse_transform(targets.reshape(-1, 1)).flatten()

    # Calculate metrics
    errors = np.abs(preds - targets)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)

    # Calculate spectral errors (angular deviation at each wavelength)
    spectral_errors = {}
    for name, lam in [("blue", 450.0), ("green", 532.0), ("red", 635.0)]:
        n_out = get_refractive_index_batch(np.full(len(preds), lam), mat_ids)
        pred_angles = angle_from_pitch(preds, lam, n_out)
        angle_errors = np.abs(pred_angles - angles)
        spectral_errors[f"mae_angle_{name}_deg"] = np.mean(angle_errors)

    return {
        "mae_nm": mae,
        "rmse_nm": rmse,
        "max_error_nm": max_error,
        **spectral_errors,
    }


# =============================================================================
# DOE: DESIGN OF EXPERIMENTS
# =============================================================================
def run_doe_sweep(config: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Run DOE sweep over num_blocks and learning_rate.

    Args:
        config: Configuration dictionary
        verbose: Print progress

    Returns:
        DataFrame with DOE results
    """
    results = []

    # Generate dataset once
    print("Generating dataset...")
    X, y = generate_rainbow_dataset(
        n_samples=config["n_samples"],
        angle_range=(config["angle_min"], config["angle_max"]),
        lambda_ref=config["lambda_green"],
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_split"], random_state=config["random_seed"]
    )

    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test_scaled, dtype=torch.float32)
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # DOE grid
    param_grid = list(itertools.product(
        config["doe_num_blocks"],
        config["doe_learning_rates"]
    ))

    print("=" * 70)
    print("P2 RAINBOW SOLVER: DESIGN OF EXPERIMENTS")
    print("=" * 70)
    print(f"Total experiments: {len(param_grid)}")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {config['n_samples']:,} samples")
    print("=" * 70)

    best_mae = float('inf')
    best_model = None
    best_config_result = None

    for idx, (num_blocks, lr) in enumerate(param_grid, 1):
        exp_id = f"EXP_{idx:02d}"

        if verbose:
            print(f"\n[{idx}/{len(param_grid)}] {exp_id}")
            print(f"  Config: num_blocks={num_blocks}, lr={lr}")

        # Initialize model
        model = SpectralResNet(
            input_dim=2,
            hidden_dim=128,
            num_blocks=num_blocks
        )
        n_params = model.count_parameters()

        # Use simple MSE loss for scaled data (spectral loss requires unscaled)
        loss_fn = SimpleMSELoss()

        # Train
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            loss_fn=loss_fn,
            epochs=config["epochs"],
            lr=lr,
            use_spectral_loss=False,
            verbose=False,
        )

        # Evaluate (need to create unscaled test loader for proper metrics)
        test_dataset_unscaled = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        test_loader_unscaled = DataLoader(test_dataset_unscaled, batch_size=config["batch_size"])

        # For evaluation, we need to create a wrapper that unscales predictions
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                predictions = model(X_batch).cpu().numpy()
                all_preds.append(predictions)
                all_targets.append(y_batch.numpy())

        preds_scaled = np.concatenate(all_preds)
        preds_unscaled = scaler_y.inverse_transform(preds_scaled).flatten()
        targets_unscaled = y_test.flatten()

        # Calculate metrics in physical units (nm)
        errors = np.abs(preds_unscaled - targets_unscaled)
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors ** 2))

        if verbose:
            print(f"  Parameters: {n_params:,}")
            print(f"  Training time: {history['training_time']:.2f}s")
            print(f"  MAE: {mae:.4f} nm | RMSE: {rmse:.4f} nm")

        # Track best model
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_config_result = {"num_blocks": num_blocks, "lr": lr}

        # Record results
        results.append({
            "experiment_id": exp_id,
            "num_blocks": num_blocks,
            "learning_rate": lr,
            "n_parameters": n_params,
            "train_time_s": round(history["training_time"], 2),
            "mae_nm": round(mae, 4),
            "rmse_nm": round(rmse, 4),
            "final_train_loss": round(history["train_loss"][-1], 6),
            "final_val_loss": round(history["val_loss"][-1], 6),
        })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 70)
    print("DOE COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\nBest Configuration:")
    print(f"  Num Blocks: {best_config_result['num_blocks']}")
    print(f"  Learning Rate: {best_config_result['lr']}")
    print(f"  MAE: {best_mae:.4f} nm")

    return df, best_model, scaler_X, scaler_y


def save_results(
    df: pd.DataFrame,
    model: nn.Module,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    config: dict
):
    """Save DOE results and best model."""

    # Create directories
    output_dir = Path(config["output_dir"])
    model_dir = Path(config["model_dir"])
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # Save DOE results
    csv_path = output_dir / "p2_doe_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDOE results saved to: {csv_path}")

    # Save best model
    model_path = model_dir / "p2_rainbow_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved to: {model_path}")

    # Save scalers for inference
    import json
    scalers_path = model_dir / "p2_scalers.json"
    scalers_data = {
        "scaler_X_mean": scaler_X.mean_.tolist(),
        "scaler_X_scale": scaler_X.scale_.tolist(),
        "scaler_y_mean": float(scaler_y.mean_[0]),
        "scaler_y_scale": float(scaler_y.scale_[0]),
    }
    with open(scalers_path, "w") as f:
        json.dump(scalers_data, f, indent=2)
    print(f"Scalers saved to: {scalers_path}")


# =============================================================================
# DEMONSTRATION: DISPERSION VISUALIZATION
# =============================================================================
def demonstrate_dispersion():
    """Show chromatic dispersion across wavelengths."""

    print("\n" + "=" * 70)
    print("SELLMEIER DISPERSION DEMONSTRATION")
    print("=" * 70)

    wavelengths = [450, 500, 532, 600, 635, 700]

    print("\nRefractive Index vs Wavelength:")
    print("-" * 50)
    print(f"{'Wavelength (nm)':<18} {'BK7':<12} {'High-Index':<12}")
    print("-" * 50)

    for lam in wavelengths:
        n_bk7 = get_refractive_index(lam, "BK7")
        n_hi = get_refractive_index(lam, "HIGH_INDEX")
        print(f"{lam:<18} {n_bk7:<12.6f} {n_hi:<12.6f}")

    print("-" * 50)

    # Show impact on grating pitch
    print("\nGrating Pitch for -50deg output (m=-1):")
    print("-" * 50)
    target_angle = -50.0

    for glass in ["BK7", "HIGH_INDEX"]:
        print(f"\n{glass}:")
        for name, lam in [("Blue", 450), ("Green", 532), ("Red", 635)]:
            n = get_refractive_index(lam, glass)
            pitch = abs((-1 * lam) / (n * np.sin(np.radians(target_angle))))
            print(f"  {name} ({lam}nm): {pitch:.2f} nm")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Run the full P2 Rainbow Solver training pipeline."""

    print("\n" + "=" * 70)
    print("  P2: THE RAINBOW SOLVER")
    print("  Multi-Spectral Grating Optimization with Material Dispersion")
    print("=" * 70 + "\n")

    # Demonstrate dispersion physics
    demonstrate_dispersion()

    # Run DOE sweep
    df, best_model, scaler_X, scaler_y = run_doe_sweep(CONFIG, verbose=True)

    # Save results
    save_results(df, best_model, scaler_X, scaler_y, CONFIG)

    # Display final results table
    print("\n" + "=" * 70)
    print("FULL DOE RESULTS TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\nP2 Rainbow Solver training complete!")


if __name__ == "__main__":
    main()
