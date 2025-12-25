# -*- coding: utf-8 -*-
"""
P5: LCOS Fringing Surrogate - DOE Training Script
==================================================
1D U-Net for Signal-to-Signal translation:
Sharp Voltage Pattern -> Smooth Phase Profile (Fringing Effect)

Grid Search over:
- depth: [2, 3] - Number of encoder/decoder layers
- base_filters: [16, 32] - Network width
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

# DOE Grid Parameters
DEPTHS = [2, 3]
BASE_FILTERS = [16, 32]
EPOCHS_PER_EXPERIMENT = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8

# Random seed for reproducibility
RANDOM_SEED = 42


# =============================================================================
# 1D U-NET ARCHITECTURE
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
        return features, pooled  # Return both for skip connection


class DecoderBlock1D(nn.Module):
    """Decoder block: Upsample -> Concat skip -> ConvBlock"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsample: doubles spatial dimension
        self.upsample = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )
        # After concat with skip, we have out_channels * 2 channels
        self.conv = ConvBlock1D(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)

        # Handle size mismatch (if input size not perfectly divisible)
        if x.size(2) != skip.size(2):
            diff = skip.size(2) - x.size(2)
            x = nn.functional.pad(x, [diff // 2, diff - diff // 2])

        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class LCOS_UNet(nn.Module):
    """
    1D U-Net for LCOS Fringing Prediction.

    Signal-to-Signal translation:
    Input: (batch, 1, 128) - Voltage pattern
    Output: (batch, 1, 128) - Phase profile

    Architecture:
    - Encoder: depth x (ConvBlock + MaxPool)
    - Bottleneck: ConvBlock at lowest resolution
    - Decoder: depth x (Upsample + Skip + ConvBlock)
    - Output: 1x1 Conv to single channel

    Args:
        depth: Number of encoder/decoder stages (2 or 3)
        base_filters: Number of filters in first layer (doubles each stage)
    """

    def __init__(self, in_channels=1, out_channels=1, depth=2, base_filters=16):
        super().__init__()
        self.depth = depth
        self.base_filters = base_filters

        # Calculate filter sizes for each level
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]
        # e.g., depth=2, base=16: [16, 32, 64]
        # e.g., depth=3, base=16: [16, 32, 64, 128]

        # Encoder path
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            self.encoders.append(EncoderBlock1D(in_ch, filters[i]))
            in_ch = filters[i]

        # Bottleneck
        self.bottleneck = ConvBlock1D(filters[depth - 1], filters[depth])

        # Decoder path
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = filters[i + 1]
            out_ch = filters[i]
            self.decoders.append(DecoderBlock1D(in_ch, out_ch))

        # Final output layer
        self.output = nn.Conv1d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder: collect skip connections
        skips = []
        for encoder in self.encoders:
            features, x = encoder(x)
            skips.append(features)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: use skip connections in reverse order
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Output
        x = self.output(x)
        return x


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load and prepare the LCOS dataset."""
    print(f"Loading data from {DATA_PATH}...")
    data = np.load(DATA_PATH, allow_pickle=True)

    # Shape: (2000, 128, 1) -> need (2000, 1, 128) for Conv1d
    voltage = data['voltage_commands']  # (N, 128, 1)
    phase = data['phase_responses']      # (N, 128, 1)

    # Transpose to (N, 1, 128) for Conv1d: (batch, channels, length)
    voltage = voltage.transpose(0, 2, 1)  # (N, 1, 128)
    phase = phase.transpose(0, 2, 1)      # (N, 1, 128)

    # Normalize voltage to [0, 1]
    v_max = voltage.max()
    voltage = voltage / v_max

    print(f"  Voltage shape: {voltage.shape}, range: [{voltage.min():.3f}, {voltage.max():.3f}]")
    print(f"  Phase shape: {phase.shape}, range: [{phase.min():.3f}, {phase.max():.3f}]")

    # Train/validation split
    n_samples = len(voltage)
    n_train = int(n_samples * TRAIN_SPLIT)

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
# TRAINING FUNCTION
# =============================================================================
def train_model(model, train_loader, val_loader, epochs):
    """Train the U-Net model and return metrics."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"    Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    return history


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# DOE LOOP - GRID SEARCH
# =============================================================================
def run_doe():
    """Run Design of Experiments grid search."""
    print("=" * 60)
    print("P5: LCOS Fringing Surrogate - DOE Training (1D U-Net)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Grid: depth={DEPTHS}, base_filters={BASE_FILTERS}")
    print(f"Epochs per experiment: {EPOCHS_PER_EXPERIMENT}")
    print("=" * 60)

    # Load data once
    train_loader, val_loader, v_max = load_data()

    results = []
    best_val_loss = float('inf')
    best_model_state = None
    best_config = None

    exp_id = 0
    total_experiments = len(DEPTHS) * len(BASE_FILTERS)

    for depth in DEPTHS:
        for base_filters in BASE_FILTERS:
            exp_id += 1
            print(f"\n{'='*60}")
            print(f"Experiment {exp_id}/{total_experiments}")
            print(f"  depth={depth}, base_filters={base_filters}")
            print("=" * 60)

            # Create model
            model = LCOS_UNet(
                in_channels=1,
                out_channels=1,
                depth=depth,
                base_filters=base_filters
            )
            n_params = count_parameters(model)
            print(f"  Parameters: {n_params:,}")

            # Print architecture summary
            filters = [base_filters * (2 ** i) for i in range(depth + 1)]
            print(f"  Filter progression: {filters}")

            # Train
            start_time = time.time()
            history = train_model(model, train_loader, val_loader, EPOCHS_PER_EXPERIMENT)
            train_time = time.time() - start_time

            final_train_loss = history['train_loss'][-1]
            final_val_loss = history['val_loss'][-1]

            print(f"  Training time: {train_time:.1f}s")
            print(f"  Final val_loss: {final_val_loss:.6f}")

            # Record results
            result = {
                'experiment_id': exp_id,
                'depth': depth,
                'base_filters': base_filters,
                'n_parameters': n_params,
                'train_loss': final_train_loss,
                'val_loss': final_val_loss,
                'train_time_s': round(train_time, 2),
                'history': history
            }
            results.append(result)

            # Track best model
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_model_state = model.state_dict().copy()
                best_config = {
                    'depth': depth,
                    'base_filters': base_filters,
                    'n_parameters': n_params
                }
                print(f"  *** New best model! ***")

    return results, best_model_state, best_config, v_max


# =============================================================================
# SAVE RESULTS
# =============================================================================
def save_results(results, best_model_state, best_config, v_max):
    """Save DOE results and best model."""
    # Save best model
    model_path = MODELS_DIR / "best_lcos_model.pth"
    torch.save({
        'model_state_dict': best_model_state,
        'config': best_config,
        'v_max': v_max
    }, model_path)
    print(f"\nBest model saved to {model_path}")
    print(f"  Config: {best_config}")

    # Save DOE results (without history for JSON)
    results_for_json = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != 'history'}
        results_for_json.append(r_copy)

    json_path = MODELS_DIR / "p5_doe_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'experiments': results_for_json,
            'best_config': best_config,
            'grid': {
                'depths': DEPTHS,
                'base_filters': BASE_FILTERS
            }
        }, f, indent=2)
    print(f"DOE results saved to {json_path}")

    # Create visualization
    create_doe_chart(results)


def create_doe_chart(results):
    """Create DOE results visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prepare data
    labels = [f"d={r['depth']}, f={r['base_filters']}" for r in results]
    val_losses = [r['val_loss'] for r in results]
    n_params = [r['n_parameters'] for r in results]
    train_times = [r['train_time_s'] for r in results]

    # Find best (lowest val loss)
    best_idx = np.argmin(val_losses)
    bar_colors = ['#2ecc71' if i == best_idx else '#667eea' for i in range(len(results))]

    # Plot 1: Model Complexity (Parameters) vs Validation Loss
    ax1 = axes[0]
    x_pos = np.arange(len(results))

    # Bar chart for val loss
    bars = ax1.bar(x_pos, val_losses, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('Validation Loss (MSE)', fontsize=11, color='#667eea')
    ax1.set_title('P5 LCOS U-Net DOE: Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='#667eea')

    # Add value labels on bars
    for bar, val in zip(bars, val_losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, color='#667eea')

    # Secondary axis for parameter count
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_pos, n_params, 'ro-', linewidth=2, markersize=8, label='Parameters')
    ax1_twin.set_ylabel('Parameters', fontsize=11, color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')

    # Format parameter labels with K suffix
    ax1_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Training Time
    ax2 = axes[1]
    bars2 = ax2.bar(x_pos, train_times, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylabel('Training Time (s)', fontsize=11)
    ax2.set_title('P5 LCOS U-Net DOE: Training Time', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')

    # Add value labels
    for bar, val in zip(bars2, train_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9)

    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    chart_path = MODELS_DIR / "p5_doe_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"DOE chart saved to {chart_path}")


# =============================================================================
# VERIFICATION
# =============================================================================
def verify_best_model(best_config, v_max):
    """Quick verification of the best model on example patterns."""
    print("\nVerifying best model...")

    # Load model
    model_path = MODELS_DIR / "best_lcos_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = LCOS_UNet(
        in_channels=1,
        out_channels=1,
        depth=best_config['depth'],
        base_filters=best_config['base_filters']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Test patterns
    test_patterns = [
        [0, 5, 5, 0, 2.5, 0, 5, 0],   # Mixed
        [5, 0, 5, 0, 5, 0, 5, 0],     # Alternating
        [0, 0, 5, 5, 5, 5, 0, 0],     # Center block
    ]

    # Create visualization
    fig, axes = plt.subplots(len(test_patterns), 2, figsize=(12, 3*len(test_patterns)))

    for row, pattern in enumerate(test_patterns):
        # Expand pattern to grid (128 points)
        voltage = np.zeros(128)
        for i, v in enumerate(pattern):
            start = i * 16
            voltage[start:start+16] = v

        # Normalize and prepare tensor
        v_tensor = torch.tensor(voltage / v_max, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Predict
        with torch.no_grad():
            phase_pred = model(v_tensor).numpy()[0, 0]

        # Plot
        x = np.arange(128)

        # Voltage input
        ax1 = axes[row, 0]
        ax1.step(x, voltage, where='mid', color='#667eea', linewidth=2)
        ax1.fill_between(x, 0, voltage, step='mid', alpha=0.3, color='#667eea')
        ax1.set_ylabel('Voltage (V)', fontsize=10)
        ax1.set_ylim([-0.5, 5.5])
        ax1.grid(True, alpha=0.3)
        if row == 0:
            ax1.set_title('Input: Voltage Pattern', fontsize=11, fontweight='bold')

        # Phase output
        ax2 = axes[row, 1]
        ax2.plot(x, phase_pred, color='#2ecc71', linewidth=2, label='U-Net Prediction')
        ax2.fill_between(x, 0, phase_pred, alpha=0.3, color='#2ecc71')
        ax2.set_ylabel('Phase', fontsize=10)
        ax2.set_ylim([-0.05, 1.1])
        ax2.grid(True, alpha=0.3)
        if row == 0:
            ax2.set_title('Output: Predicted Phase (Fringing)', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8)

        # Pattern label
        pattern_str = str(pattern)
        ax1.text(0.02, 0.98, pattern_str, transform=ax1.transAxes, fontsize=8,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1, 0].set_xlabel('Grid Position', fontsize=10)
    axes[-1, 1].set_xlabel('Grid Position', fontsize=10)

    plt.suptitle('P5 LCOS U-Net: Best Model Predictions', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    verify_path = MODELS_DIR / "p5_verification.png"
    plt.savefig(verify_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Verification plot saved to {verify_path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Set random seeds
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)

    # Run DOE
    results, best_model_state, best_config, v_max = run_doe()

    # Save everything
    save_results(results, best_model_state, best_config, v_max)

    # Verify best model
    verify_best_model(best_config, v_max)

    print("\n" + "=" * 60)
    print("DOE COMPLETE")
    print("=" * 60)
    print(f"\nBest configuration: {best_config}")
    print(f"Best validation loss: {min(r['val_loss'] for r in results):.6f}")
    print(f"\nFiles created:")
    print(f"  - models/best_lcos_model.pth")
    print(f"  - models/p5_doe_results.json")
    print(f"  - models/p5_doe_chart.png")
    print(f"  - models/p5_verification.png")
