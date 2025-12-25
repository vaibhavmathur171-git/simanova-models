# -*- coding: utf-8 -*-
"""
P5: LCOS Fringing Surrogate - IMPROVED Training Script
=======================================================
Fixes edge slope prediction with:
1. Gradient Loss - penalizes derivative differences at edges
2. More epochs (50) for better convergence
3. Learning rate scheduling
4. Combined loss: MSE + lambda * GradientLoss
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = Path("data/p5_lcos_dataset.npz")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# IMPROVED Training Parameters
DEPTH = 3              # Best from DOE
BASE_FILTERS = 16      # Best from DOE
EPOCHS = 100           # Even more epochs for precise edge learning
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
GRADIENT_LOSS_WEIGHT = 1.0  # Higher weight for gradient loss
TRAIN_SPLIT = 0.8

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# 1D U-NET ARCHITECTURE (Same as before)
# =============================================================================
class ConvBlock1D(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

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
    """Encoder block: ConvBlock -> MaxPool (downsample by 2)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock1D(in_channels, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock1D(nn.Module):
    """Decoder block: Upsample -> Concat skip -> ConvBlock"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )
        self.conv = ConvBlock1D(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.size(2) != skip.size(2):
            diff = skip.size(2) - x.size(2)
            x = nn.functional.pad(x, [diff // 2, diff - diff // 2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class LCOS_UNet(nn.Module):
    """1D U-Net for LCOS Fringing Prediction."""

    def __init__(self, in_channels=1, out_channels=1, depth=2, base_filters=16):
        super().__init__()
        self.depth = depth
        self.base_filters = base_filters

        filters = [base_filters * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            self.encoders.append(EncoderBlock1D(in_ch, filters[i]))
            in_ch = filters[i]

        self.bottleneck = ConvBlock1D(filters[depth - 1], filters[depth])

        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = filters[i + 1]
            out_ch = filters[i]
            self.decoders.append(DecoderBlock1D(in_ch, out_ch))

        self.output = nn.Conv1d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            features, x = encoder(x)
            skips.append(features)

        x = self.bottleneck(x)

        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        x = self.output(x)
        return x


# =============================================================================
# EDGE-AWARE LOSS FUNCTIONS
# =============================================================================
class GradientLoss(nn.Module):
    """
    Gradient Loss: Penalizes differences in first derivative (slope).

    This forces the model to learn correct edge slopes - critical for
    fringing field prediction at pixel boundaries.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Compute first derivative (gradient) using finite differences
        # Shape: (batch, 1, 128) -> gradient has shape (batch, 1, 127)
        pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
        target_grad = target[:, :, 1:] - target[:, :, :-1]

        # MSE on gradients
        return torch.mean((pred_grad - target_grad) ** 2)


class EdgeWeightedMSELoss(nn.Module):
    """
    Edge-Weighted MSE: Higher weight on pixel boundary regions.

    Pixel boundaries are at positions: 16, 32, 48, 64, 80, 96, 112
    Weight is higher near these positions.
    """

    def __init__(self, grid_size=128, pixels=8, edge_weight=3.0, edge_width=4):
        super().__init__()
        self.grid_size = grid_size

        # Create weight mask
        weights = torch.ones(grid_size)
        pixel_width = grid_size // pixels  # 16

        for i in range(1, pixels):
            boundary = i * pixel_width
            # Add higher weights around boundary
            start = max(0, boundary - edge_width)
            end = min(grid_size, boundary + edge_width)
            weights[start:end] = edge_weight

        # Normalize weights
        weights = weights / weights.mean()
        self.register_buffer('weights', weights.view(1, 1, -1))

    def forward(self, pred, target):
        diff_sq = (pred - target) ** 2
        weighted_diff = diff_sq * self.weights.to(pred.device)
        return torch.mean(weighted_diff)


class CombinedLoss(nn.Module):
    """
    Combined Loss = MSE + lambda * GradientLoss + mu * EdgeWeightedMSE

    This learns both:
    1. Overall shape (MSE)
    2. Correct slopes at all points (GradientLoss)
    3. Accuracy at pixel boundaries (EdgeWeightedMSE)
    """

    def __init__(self, gradient_weight=0.5, edge_weight=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.gradient = GradientLoss()
        self.edge_mse = EdgeWeightedMSELoss()
        self.gradient_weight = gradient_weight
        self.edge_weight = edge_weight

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        grad_loss = self.gradient(pred, target)
        edge_loss = self.edge_mse(pred, target)

        total = mse_loss + self.gradient_weight * grad_loss + self.edge_weight * edge_loss
        return total, mse_loss, grad_loss, edge_loss


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load and prepare the LCOS dataset."""
    print(f"Loading data from {DATA_PATH}...")
    data = np.load(DATA_PATH, allow_pickle=True)

    voltage = data['voltage_commands']  # (N, 128, 1)
    phase = data['phase_responses']      # (N, 128, 1)

    # Transpose to (N, 1, 128) for Conv1d
    voltage = voltage.transpose(0, 2, 1)
    phase = phase.transpose(0, 2, 1)

    # Normalize voltage to [0, 1]
    v_max = voltage.max()
    voltage = voltage / v_max

    print(f"  Voltage shape: {voltage.shape}, range: [{voltage.min():.3f}, {voltage.max():.3f}]")
    print(f"  Phase shape: {phase.shape}, range: [{phase.min():.3f}, {phase.max():.3f}]")

    # Train/validation split
    n_samples = len(voltage)
    n_train = int(n_samples * TRAIN_SPLIT)

    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = torch.tensor(voltage[train_idx], dtype=torch.float32)
    y_train = torch.tensor(phase[train_idx], dtype=torch.float32)
    X_val = torch.tensor(voltage[val_idx], dtype=torch.float32)
    y_val = torch.tensor(phase[val_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}")

    return train_loader, val_loader, v_max


# =============================================================================
# IMPROVED TRAINING FUNCTION
# =============================================================================
def train_model(model, train_loader, val_loader, epochs):
    """Train with combined loss and LR scheduling."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = CombinedLoss(gradient_weight=GRADIENT_LOSS_WEIGHT, edge_weight=0.3)

    history = {
        'train_loss': [], 'val_loss': [],
        'mse': [], 'grad': [], 'edge': [],
        'lr': []
    }

    best_val_loss = float('inf')
    best_state = None

    print(f"\nTraining with Combined Loss:")
    print(f"  MSE Weight: 1.0")
    print(f"  Gradient Weight: {GRADIENT_LOSS_WEIGHT}")
    print(f"  Edge Weight: 0.3")
    print(f"  Learning Rate: {LEARNING_RATE} (with ReduceLROnPlateau)")
    print("-" * 60)

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            total_loss, mse, grad, edge = criterion(y_pred, y_batch)
            total_loss.backward()
            optimizer.step()

            train_losses.append(total_loss.item())

        # Validation
        model.eval()
        val_losses = []
        val_mse, val_grad, val_edge = [], [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                y_pred = model(X_batch)
                total_loss, mse, grad, edge = criterion(y_pred, y_batch)

                val_losses.append(total_loss.item())
                val_mse.append(mse.item())
                val_grad.append(grad.item())
                val_edge.append(edge.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mse'].append(np.mean(val_mse))
        history['grad'].append(np.mean(val_grad))
        history['edge'].append(np.mean(val_edge))
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # LR scheduling
        scheduler.step(val_loss)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        # Progress output
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"train={train_loss:.6f}, val={val_loss:.6f} | "
                  f"mse={np.mean(val_mse):.6f}, grad={np.mean(val_grad):.6f}, edge={np.mean(val_edge):.6f}")

    # Restore best model
    model.load_state_dict(best_state)

    return history, best_val_loss


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# SAVE & VISUALIZE
# =============================================================================
def save_model(model, v_max):
    """Save the trained model."""
    model_path = MODELS_DIR / "best_lcos_model.pth"

    config = {
        'depth': DEPTH,
        'base_filters': BASE_FILTERS,
        'n_parameters': count_parameters(model)
    }

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'v_max': v_max
    }, model_path)

    print(f"\nModel saved to {model_path}")
    print(f"  Config: {config}")


def create_training_chart(history):
    """Create training visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Total Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train', alpha=0.7)
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Combined Loss', fontsize=11)
    ax1.set_title('Total Loss (MSE + Gradient + Edge)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Individual Loss Components
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['mse'], 'g-', linewidth=2, label='MSE')
    ax2.plot(epochs, history['grad'], 'm-', linewidth=2, label='Gradient')
    ax2.plot(epochs, history['edge'], 'c-', linewidth=2, label='Edge-Weighted')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Loss Value', fontsize=11)
    ax2.set_title('Loss Components (Validation)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: Learning Rate
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['lr'], 'k-', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Learning Rate', fontsize=11)
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Plot 4: Final metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    final_metrics = f"""
TRAINING SUMMARY
================

Configuration:
  Depth: {DEPTH}
  Base Filters: {BASE_FILTERS}
  Epochs: {EPOCHS}

Loss Weights:
  MSE: 1.0
  Gradient: {GRADIENT_LOSS_WEIGHT}
  Edge: 0.3

Final Validation Losses:
  Total: {history['val_loss'][-1]:.6f}
  MSE: {history['mse'][-1]:.6f}
  Gradient: {history['grad'][-1]:.6f}
  Edge: {history['edge'][-1]:.6f}

Best Total Loss: {min(history['val_loss']):.6f}
"""
    ax4.text(0.1, 0.9, final_metrics, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('P5 LCOS Improved Training (Edge-Aware Loss)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    chart_path = MODELS_DIR / "p5_improved_training.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Training chart saved to {chart_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("P5: LCOS Fringing - IMPROVED Training (Edge-Aware)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Architecture: depth={DEPTH}, base_filters={BASE_FILTERS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Gradient Loss Weight: {GRADIENT_LOSS_WEIGHT}")
    print("=" * 60)

    # Set seeds
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # Load data
    train_loader, val_loader, v_max = load_data()

    # Create model
    model = LCOS_UNet(
        in_channels=1,
        out_channels=1,
        depth=DEPTH,
        base_filters=BASE_FILTERS
    )
    n_params = count_parameters(model)
    print(f"\nModel parameters: {n_params:,}")

    # Train
    start_time = time.time()
    history, best_val_loss = train_model(model, train_loader, val_loader, EPOCHS)
    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Save
    save_model(model, v_max)
    create_training_chart(history)

    print("\n" + "=" * 60)
    print("IMPROVED TRAINING COMPLETE")
    print("=" * 60)
    print("\nNow run p5_verify.py to check edge slope accuracy!")


if __name__ == "__main__":
    main()
