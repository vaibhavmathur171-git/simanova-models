# -*- coding: utf-8 -*-
"""
Train P1 Optimal Model - Quick single configuration training
Uses optimal parameters from DOE: 4 layers, 10K samples, 200 epochs
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path

# =============================================================================
# CONFIGURATION (Optimal from DOE)
# =============================================================================
CONFIG = {
    "n_layers": 4,
    "n_samples": 10_000,
    "n_epochs": 200,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "noise_deg": 0.5,
    "wavelength_nm": 532,
    "n_out": 1.5,
    "angle_min": -80.0,
    "angle_max": -30.0,
}

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# PHYSICS
# =============================================================================
def grating_equation(angle_deg, wavelength_nm=532, n_out=1.5):
    theta_rad = np.radians(angle_deg)
    m = -1
    sin_theta = np.sin(theta_rad)
    sin_theta = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
    period = (m * wavelength_nm) / (n_out * sin_theta)
    return np.abs(period)

def generate_dataset(n_samples, noise_deg, angle_range, wavelength_nm, n_out):
    angles = np.random.uniform(angle_range[0], angle_range[1], n_samples)
    if noise_deg > 0:
        noise = np.random.normal(0, noise_deg, n_samples)
        angles_noisy = np.clip(angles + noise, angle_range[0], angle_range[1])
    else:
        angles_noisy = angles
    periods = grating_equation(angles_noisy, wavelength_nm, n_out)
    return angles_noisy.astype(np.float32), periods.astype(np.float32)

# =============================================================================
# MODEL
# =============================================================================
class SimpleMLP(nn.Module):
    def __init__(self, n_layers=4, width=64):
        super(SimpleMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(1, width))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# =============================================================================
# TRAINING
# =============================================================================
def train_model(model, train_loader, epochs, lr):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"Training on {DEVICE}...")
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

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")

    return model

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("P1 OPTIMAL MODEL TRAINING")
    print("=" * 70)
    print(f"Config: {CONFIG['n_layers']} layers, {CONFIG['n_samples']:,} samples, {CONFIG['n_epochs']} epochs")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Generate data
    print("\nGenerating dataset...")
    angles, periods = generate_dataset(
        n_samples=CONFIG["n_samples"],
        noise_deg=CONFIG["noise_deg"],
        angle_range=(CONFIG["angle_min"], CONFIG["angle_max"]),
        wavelength_nm=CONFIG["wavelength_nm"],
        n_out=CONFIG["n_out"]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        angles.reshape(-1, 1),
        periods.reshape(-1, 1),
        test_size=0.2,
        random_state=42
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # Initialize and train model
    print("\nInitializing model...")
    model = SimpleMLP(n_layers=CONFIG["n_layers"], width=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print("\nTraining...")
    model = train_model(model, train_loader, CONFIG["n_epochs"], CONFIG["learning_rate"])

    # Evaluate
    print("\nEvaluating...")
    model.eval()
    X_test_t = X_test_t.to(DEVICE)
    y_test_t = y_test_t.to(DEVICE)

    with torch.no_grad():
        predictions = model(X_test_t)

    predictions = predictions.cpu().numpy().flatten()
    targets = y_test_t.cpu().numpy().flatten()

    errors = np.abs(predictions - targets)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors ** 2))

    print(f"\nFinal Performance:")
    print(f"  MAE:  {mae:.4f} nm")
    print(f"  RMSE: {rmse:.4f} nm")

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "p1_mono_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    print("\n✓ Training complete!")

if __name__ == "__main__":
    main()
