# -*- coding: utf-8 -*-
"""
P3: Virtual Wind Tunnel - DOE Training Script
==============================================
Design of Experiments for 1D CNN Aerodynamics Surrogate.

Grid Search:
  - kernel_size: [3, 7]
  - num_filters: [16, 32]
  - num_layers: [2, 3]

Total: 2 x 2 x 2 = 8 experiments @ 15 epochs each
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# 1. DATA LOADER
# =============================================================================

def load_dataset(data_path: str = "data/p3_aero_dataset.npz",
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load airfoil dataset and create train/val/test dataloaders.

    Parameters:
    -----------
    data_path : str
        Path to the NPZ file
    train_ratio, val_ratio : float
        Split ratios (test_ratio = 1 - train - val)
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    train_loader, val_loader, test_loader : DataLoader
    """
    print(f"\n[DATA] Loading dataset from {data_path}...")

    # Load data
    data = np.load(data_path)
    X = data['X_shapes']      # (N, 2, 100) - x and y coordinates
    y = data['y_pressures']   # (N, 100) - Cp values

    print(f"       X_shapes: {X.shape}")
    print(f"       y_pressures: {y.shape}")

    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    # Shuffle and split
    n_samples = len(X)
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # Create datasets
    train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
    test_dataset = TensorDataset(X_tensor[test_idx], y_tensor[test_idx])

    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"       Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    return train_loader, val_loader, test_loader


# =============================================================================
# 2. 1D CNN MODEL
# =============================================================================

class AeroCNN(nn.Module):
    """
    1D Convolutional Neural Network for Airfoil Pressure Prediction.

    Architecture:
    -------------
    Encoder: Stack of Conv1d layers with ReLU and BatchNorm
    Decoder: Flatten -> Linear layers -> Output (100 Cp values)

    Parameters:
    -----------
    kernel_size : int
        Convolution kernel size (3 = local, 7 = wide context)
    num_filters : int
        Number of filters in first conv layer (doubles each layer)
    num_layers : int
        Number of convolutional layers in encoder
    """

    def __init__(self, kernel_size: int = 5, num_filters: int = 32, num_layers: int = 3):
        super().__init__()

        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers

        # Input: (Batch, 2, 100) - 2 channels (x, y coords)
        in_channels = 2
        seq_len = 100

        # Build encoder (Conv1d layers)
        encoder_layers = []
        current_channels = in_channels
        current_len = seq_len

        for i in range(num_layers):
            out_channels = num_filters * (2 ** i)  # 16, 32, 64, ...
            padding = kernel_size // 2  # Same padding

            encoder_layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ])

            # Add pooling every other layer (except last)
            if i < num_layers - 1 and i % 2 == 0:
                encoder_layers.append(nn.MaxPool1d(2))
                current_len = current_len // 2

            current_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate flattened size
        self.flat_size = current_channels * current_len

        # Build decoder (Linear layers)
        hidden_dim = 256
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 100),  # Output: 100 Cp values
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters:
        -----------
        x : torch.Tensor, shape (Batch, 2, 100)
            Airfoil coordinates [x_coords, y_coords]

        Returns:
        --------
        Cp : torch.Tensor, shape (Batch, 100)
            Predicted pressure coefficients
        """
        features = self.encoder(x)
        Cp = self.decoder(features)
        return Cp

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 3. TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model: nn.Module, loader: DataLoader,
                criterion: nn.Module, optimizer: torch.optim.Optimizer,
                device: torch.device) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def validate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> float:
    """Evaluate on validation set, return average loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, device: torch.device, verbose: bool = True) -> Dict:
    """
    Train model and return training history.

    Returns:
    --------
    history : dict
        Contains 'train_loss', 'val_loss', 'best_val_loss', 'train_time'
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }

    start_time = time.time()

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Track best
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch + 1

        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"       Epoch {epoch+1:2d}/{epochs}: "
                  f"Train={train_loss:.6f} | Val={val_loss:.6f}")

    history['train_time'] = time.time() - start_time
    return history


# =============================================================================
# 4. DOE GRID SEARCH
# =============================================================================

@dataclass
class DOEResult:
    """Store results from one DOE experiment."""
    experiment_id: int
    kernel_size: int
    num_filters: int
    num_layers: int
    n_parameters: int
    train_time: float
    best_val_loss: float
    final_train_loss: float
    final_val_loss: float


def run_doe(train_loader: DataLoader, val_loader: DataLoader,
            device: torch.device, epochs: int = 15) -> List[DOEResult]:
    """
    Run Design of Experiments grid search.

    Grid:
    -----
    kernel_size: [3, 7]
    num_filters: [16, 32]
    num_layers: [2, 3]

    Total: 8 experiments
    """
    # DOE grid
    kernel_sizes = [3, 7]
    num_filters_list = [16, 32]
    num_layers_list = [2, 3]

    grid = list(product(kernel_sizes, num_filters_list, num_layers_list))

    print("\n" + "=" * 70)
    print("P3 DOE: GRID SEARCH")
    print("=" * 70)
    print(f"\nGrid Parameters:")
    print(f"  kernel_size: {kernel_sizes}")
    print(f"  num_filters: {num_filters_list}")
    print(f"  num_layers:  {num_layers_list}")
    print(f"  Total experiments: {len(grid)}")
    print(f"  Epochs per experiment: {epochs}")
    print("\n" + "-" * 70)

    results = []

    for exp_id, (ks, nf, nl) in enumerate(grid, 1):
        print(f"\n[EXP {exp_id}/{len(grid)}] kernel={ks}, filters={nf}, layers={nl}")

        # Create model
        model = AeroCNN(kernel_size=ks, num_filters=nf, num_layers=nl)
        model = model.to(device)
        n_params = model.count_parameters()
        print(f"       Parameters: {n_params:,}")

        # Train
        history = train_model(
            model, train_loader, val_loader,
            epochs=epochs, device=device, verbose=True
        )

        # Store result
        result = DOEResult(
            experiment_id=exp_id,
            kernel_size=ks,
            num_filters=nf,
            num_layers=nl,
            n_parameters=n_params,
            train_time=history['train_time'],
            best_val_loss=history['best_val_loss'],
            final_train_loss=history['train_loss'][-1],
            final_val_loss=history['val_loss'][-1]
        )
        results.append(result)

        print(f"       Best Val Loss: {result.best_val_loss:.6f} "
              f"| Time: {result.train_time:.1f}s")

    return results


# =============================================================================
# 5. VISUALIZATION AND OUTPUT
# =============================================================================

def plot_doe_results(results: List[DOEResult], output_path: Path):
    """Create bar chart of DOE results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sort by validation loss
    sorted_results = sorted(results, key=lambda x: x.best_val_loss)

    # Labels
    labels = [f"K{r.kernel_size}_F{r.num_filters}_L{r.num_layers}" for r in sorted_results]
    val_losses = [r.best_val_loss for r in sorted_results]
    n_params = [r.n_parameters / 1000 for r in sorted_results]  # In thousands

    # Color by rank
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(results)))

    # Plot 1: Validation Loss
    ax1 = axes[0]
    bars1 = ax1.barh(labels, val_losses, color=colors)
    ax1.set_xlabel('Best Validation Loss (MSE)', fontsize=12, fontweight='bold')
    ax1.set_title('DOE Results: Validation Loss', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars1, val_losses):
        ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    # Highlight best
    best_idx = 0
    bars1[best_idx].set_color('#2ecc71')
    bars1[best_idx].set_edgecolor('black')
    bars1[best_idx].set_linewidth(2)

    # Plot 2: Parameters vs Loss
    ax2 = axes[1]

    for i, r in enumerate(sorted_results):
        marker = '*' if i == 0 else 'o'
        size = 200 if i == 0 else 100
        ax2.scatter(r.n_parameters / 1000, r.best_val_loss,
                   s=size, c=[colors[i]], marker=marker,
                   edgecolors='black', linewidths=1,
                   label=f"K{r.kernel_size}_F{r.num_filters}_L{r.num_layers}")

    ax2.set_xlabel('Parameters (K)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Best Validation Loss (MSE)', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy vs Model Complexity', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n[OK] DOE chart saved: {output_path}")


def save_results(results: List[DOEResult], output_path: Path):
    """Save DOE results to JSON."""
    results_dict = {
        'experiments': [
            {
                'experiment_id': r.experiment_id,
                'kernel_size': r.kernel_size,
                'num_filters': r.num_filters,
                'num_layers': r.num_layers,
                'n_parameters': r.n_parameters,
                'train_time_s': round(r.train_time, 2),
                'best_val_loss': round(r.best_val_loss, 8),
                'final_train_loss': round(r.final_train_loss, 8),
                'final_val_loss': round(r.final_val_loss, 8),
            }
            for r in results
        ],
        'best_config': None
    }

    # Find best
    best = min(results, key=lambda x: x.best_val_loss)
    results_dict['best_config'] = {
        'kernel_size': best.kernel_size,
        'num_filters': best.num_filters,
        'num_layers': best.num_layers,
        'val_loss': round(best.best_val_loss, 8),
        'n_parameters': best.n_parameters
    }

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"[OK] DOE results saved: {output_path}")


def train_final_model(train_loader: DataLoader, val_loader: DataLoader,
                      test_loader: DataLoader, best_config: Dict,
                      device: torch.device, epochs: int = 30) -> Tuple[nn.Module, Dict, float]:
    """Train final model with best configuration."""
    print("\n" + "=" * 70)
    print("FINAL MODEL TRAINING")
    print("=" * 70)
    print(f"\nBest Configuration:")
    print(f"  kernel_size: {best_config['kernel_size']}")
    print(f"  num_filters: {best_config['num_filters']}")
    print(f"  num_layers:  {best_config['num_layers']}")
    print(f"\nTraining for {epochs} epochs...")

    model = AeroCNN(
        kernel_size=best_config['kernel_size'],
        num_filters=best_config['num_filters'],
        num_layers=best_config['num_layers']
    )
    model = model.to(device)

    print(f"Parameters: {model.count_parameters():,}")
    print("-" * 70)

    # Train with more epochs
    history = train_model(
        model, train_loader, val_loader,
        epochs=epochs, device=device, verbose=True
    )

    # Final test evaluation
    criterion = nn.MSELoss()
    test_loss = validate(model, test_loader, criterion, device)

    print("-" * 70)
    print(f"\n[FINAL RESULTS]")
    print(f"  Best Val Loss: {history['best_val_loss']:.6f}")
    print(f"  Test Loss:     {test_loss:.6f}")
    print(f"  Train Time:    {history['train_time']:.1f}s")

    return model, history, test_loss


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("P3: VIRTUAL WIND TUNNEL - DOE TRAINING")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Output directories
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader = load_dataset(
        data_path="data/p3_aero_dataset.npz",
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )

    # Run DOE grid search
    doe_results = run_doe(
        train_loader, val_loader,
        device=device,
        epochs=15
    )

    # Find best configuration
    best_result = min(doe_results, key=lambda x: x.best_val_loss)
    best_config = {
        'kernel_size': best_result.kernel_size,
        'num_filters': best_result.num_filters,
        'num_layers': best_result.num_layers
    }

    # Save DOE results
    save_results(doe_results, models_dir / "p3_doe_results.json")

    # Plot DOE results
    plot_doe_results(doe_results, models_dir / "p3_doe_chart.png")

    # Print DOE summary
    print("\n" + "=" * 70)
    print("DOE SUMMARY")
    print("=" * 70)
    print(f"\n{'Exp':<4} {'Kernel':<7} {'Filters':<8} {'Layers':<7} {'Params':<10} {'Val Loss':<12}")
    print("-" * 60)

    sorted_results = sorted(doe_results, key=lambda x: x.best_val_loss)
    for r in sorted_results:
        marker = " *" if r == best_result else ""
        print(f"{r.experiment_id:<4} {r.kernel_size:<7} {r.num_filters:<8} {r.num_layers:<7} "
              f"{r.n_parameters:<10,} {r.best_val_loss:<12.6f}{marker}")

    print(f"\n[BEST] kernel={best_result.kernel_size}, filters={best_result.num_filters}, "
          f"layers={best_result.num_layers} -> Val Loss: {best_result.best_val_loss:.6f}")

    # Train final model with more epochs
    final_model, final_history, test_loss = train_final_model(
        train_loader, val_loader, test_loader,
        best_config, device, epochs=30
    )

    # Save best model
    model_path = models_dir / "best_aero_model.pth"
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'config': best_config,
        'val_loss': final_history['best_val_loss'],
        'test_loss': test_loss,
        'n_parameters': final_model.count_parameters()
    }, model_path)

    print(f"\n[OK] Best model saved: {model_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("[OK] P3 DOE TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - models/best_aero_model.pth")
    print(f"  - models/p3_doe_results.json")
    print(f"  - models/p3_doe_chart.png")
    print(f"\nBest Model Performance:")
    print(f"  - Validation MSE: {final_history['best_val_loss']:.6f}")
    print(f"  - Test MSE:       {test_loss:.6f}")
    print(f"  - Parameters:     {final_model.count_parameters():,}")
