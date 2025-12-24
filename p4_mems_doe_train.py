# -*- coding: utf-8 -*-
"""
P4 MEMS Neural Surrogate - DOE Training Script
Grid search over sequence_length and hidden_dim for LSTM model
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
DATA_PATH = Path("data/p4_mems_dataset.npz")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# DOE Grid Parameters
SEQUENCE_LENGTHS = [20, 50, 100]
HIDDEN_DIMS = [32, 64]
EPOCHS_PER_EXPERIMENT = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8

# Optimization: Limit data for faster training
MAX_EXPERIMENTS = 20  # Use first 20 experiments instead of 100
SEQUENCE_STRIDE = 5   # Take every 5th sample instead of every sample

# =============================================================================
# DATA PREPARATION - SLIDING WINDOW
# =============================================================================
def create_sequences(voltage, theta, seq_length, stride=1):
    """
    Create sliding window sequences for LSTM training.

    Input: voltage[t-seq:t], theta[t-seq:t-1]
    Output: theta[t]

    Args:
        voltage: (n_experiments, n_samples) array
        theta: (n_experiments, n_samples) array
        seq_length: number of timesteps in sequence
        stride: step size for sliding window (default 1)

    Returns:
        X: (n_sequences, seq_length, 2) - [voltage, theta_history]
        y: (n_sequences, 1) - theta_next
    """
    X_list = []
    y_list = []

    n_experiments, n_samples = voltage.shape

    for exp_idx in range(n_experiments):
        v = voltage[exp_idx]
        th = theta[exp_idx]

        for t in range(seq_length, n_samples, stride):  # Use stride
            # Input: voltage[t-seq:t] and theta[t-seq:t-1] (shifted by 1)
            v_seq = v[t-seq_length:t]
            th_seq = th[t-seq_length:t]

            # Stack into (seq_length, 2) feature matrix
            x = np.stack([v_seq, th_seq], axis=1)
            X_list.append(x)

            # Target: theta at time t
            y_list.append(th[t])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)

    return X, y


def load_and_prepare_data(seq_length):
    """Load dataset and create train/val splits."""
    print(f"  Loading data from {DATA_PATH}...")
    data = np.load(DATA_PATH, allow_pickle=True)
    voltage = data['signals'].squeeze(-1)    # (100, 10000, 1) -> (100, 10000)
    theta = data['responses'].squeeze(-1)    # (100, 10000, 1) -> (100, 10000)

    # Limit experiments for faster training
    voltage = voltage[:MAX_EXPERIMENTS]
    theta = theta[:MAX_EXPERIMENTS]
    print(f"  Using {MAX_EXPERIMENTS} experiments, stride={SEQUENCE_STRIDE}")

    print(f"  Creating sequences with seq_length={seq_length}...")
    X, y = create_sequences(voltage, theta, seq_length, stride=SEQUENCE_STRIDE)

    # Normalize inputs
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Normalize outputs
    y_mean = y.mean()
    y_std = y.std() + 1e-8
    y_norm = (y - y_mean) / y_std

    # Train/val split
    n_samples = len(X_norm)
    n_train = int(n_samples * TRAIN_SPLIT)

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = torch.tensor(X_norm[train_idx])
    y_train = torch.tensor(y_norm[train_idx])
    X_val = torch.tensor(X_norm[val_idx])
    y_val = torch.tensor(y_norm[val_idx])

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

    stats = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std
    }

    print(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    return train_loader, val_loader, stats


# =============================================================================
# THE MODEL - LSTM
# =============================================================================
class MEMS_LSTM(nn.Module):
    """
    LSTM model for MEMS dynamics prediction.

    Architecture:
        - LSTM encoder (2 input features -> hidden_dim)
        - Linear decoder (hidden_dim -> 1 output)
    """
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
        # x: (batch, seq_length, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # Decode to output
        out = self.decoder(last_hidden)
        return out


# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_model(model, train_loader, val_loader, epochs):
    """Train model and return metrics."""
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
    print("P4 MEMS Neural Surrogate - DOE Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Grid: seq_length={SEQUENCE_LENGTHS}, hidden_dim={HIDDEN_DIMS}")
    print(f"Epochs per experiment: {EPOCHS_PER_EXPERIMENT}")
    print("=" * 60)

    results = []
    best_val_loss = float('inf')
    best_model_state = None
    best_config = None

    exp_id = 0
    total_experiments = len(SEQUENCE_LENGTHS) * len(HIDDEN_DIMS)

    for seq_length in SEQUENCE_LENGTHS:
        # Load data for this sequence length
        print(f"\nPreparing data for seq_length={seq_length}...")
        train_loader, val_loader, stats = load_and_prepare_data(seq_length)

        for hidden_dim in HIDDEN_DIMS:
            exp_id += 1
            print(f"\n{'='*60}")
            print(f"Experiment {exp_id}/{total_experiments}")
            print(f"  seq_length={seq_length}, hidden_dim={hidden_dim}")
            print("=" * 60)

            # Create model
            model = MEMS_LSTM(
                input_dim=2,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=0.1
            )
            n_params = count_parameters(model)
            print(f"  Parameters: {n_params:,}")

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
                'seq_length': seq_length,
                'hidden_dim': hidden_dim,
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
                    'seq_length': seq_length,
                    'hidden_dim': hidden_dim,
                    'n_parameters': n_params
                }
                print(f"  *** New best model! ***")

    return results, best_model_state, best_config


def save_results(results, best_model_state, best_config):
    """Save DOE results and best model."""
    # Save best model
    model_path = MODELS_DIR / "best_mems_model.pth"
    torch.save({
        'model_state_dict': best_model_state,
        'config': best_config
    }, model_path)
    print(f"\nBest model saved to {model_path}")
    print(f"  Config: {best_config}")

    # Save DOE results (without history for JSON)
    results_for_json = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != 'history'}
        results_for_json.append(r_copy)

    json_path = MODELS_DIR / "p4_mems_doe_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'experiments': results_for_json,
            'best_config': best_config,
            'grid': {
                'sequence_lengths': SEQUENCE_LENGTHS,
                'hidden_dims': HIDDEN_DIMS
            }
        }, f, indent=2)
    print(f"DOE results saved to {json_path}")

    # Create visualization
    create_doe_chart(results)


def create_doe_chart(results):
    """Create DOE results bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prepare data
    labels = [f"seq={r['seq_length']}\nhid={r['hidden_dim']}" for r in results]
    val_losses = [r['val_loss'] for r in results]
    train_times = [r['train_time_s'] for r in results]

    # Color by performance (lower is better)
    sorted_indices = np.argsort(val_losses)
    bar_colors = ['#2ecc71' if i == sorted_indices[0] else '#667eea' for i in range(len(results))]

    # Plot 1: Validation Loss
    ax1 = axes[0]
    bars1 = ax1.bar(labels, val_losses, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('Validation Loss (MSE)', fontsize=11)
    ax1.set_title('P4 MEMS LSTM - DOE Results: Validation Loss', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, val in zip(bars1, val_losses):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Training Time
    ax2 = axes[1]
    bars2 = ax2.bar(labels, train_times, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylabel('Training Time (s)', fontsize=11)
    ax2.set_title('P4 MEMS LSTM - DOE Results: Training Time', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels
    for bar, val in zip(bars2, train_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    chart_path = MODELS_DIR / "p4_mems_doe_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"DOE chart saved to {chart_path}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Run DOE
    results, best_model_state, best_config = run_doe()

    # Save everything
    save_results(results, best_model_state, best_config)

    print("\n" + "=" * 60)
    print("DOE COMPLETE")
    print("=" * 60)
    print(f"Best configuration: {best_config}")
    print(f"Best validation loss: {min(r['val_loss'] for r in results):.6f}")
