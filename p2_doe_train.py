"""
P2 DOE Training Script - Design of Experiments for Neural Architecture Search
==============================================================================
Systematically tests hyperparameter combinations to find optimal ResNet architecture
for AR waveguide diffraction angle prediction.

DOE Grid:
- hidden_dim: [32, 64, 128]
- num_residual_blocks: [2, 4, 6]
Total: 9 experiments
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 1. NEURAL NETWORK ARCHITECTURE
# ============================================================================


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and skip connection."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class ARWaveguideResNet(nn.Module):
    """
    Configurable ResNet for AR waveguide diffraction prediction.

    Inputs: [wavelength, incident_angle, period, refractive_index, material_encoded]
    Output: diffracted_angle
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_residual_blocks: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_residual_blocks = num_residual_blocks

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)]
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return x.squeeze(-1)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# 2. DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray, np.ndarray,
                                                       StandardScaler, StandardScaler, LabelEncoder]:
    """
    Load AR dataset and split into train/val/test sets.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y, label_encoder
    """
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape[0]:,} samples")
    print(f"Columns: {list(df.columns)}")

    # Encode material names
    label_encoder = LabelEncoder()
    material_encoded = label_encoder.fit_transform(df['material_name'])

    # Prepare features (X) and target (y)
    X = np.column_stack([
        df['wavelength'].values,
        df['incident_angle'].values,
        df['period'].values,
        df['refractive_index'].values,
        material_encoded
    ])
    y = df['diffracted_angle'].values

    # Split: 80% train, 10% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    print(f"\nFeature scaling applied (StandardScaler)")
    print(f"  Material encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            scaler_X, scaler_y, label_encoder)


# ============================================================================
# 3. TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> float:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, lr: float, device: torch.device, verbose: bool = False) -> float:
    """
    Train model for specified epochs and return final validation loss.

    Returns:
        Final validation MSE loss
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs}: Train Loss = {train_loss:.6f}, "
                  f"Val Loss = {val_loss:.6f}")

    return best_val_loss


# ============================================================================
# 4. DESIGN OF EXPERIMENTS (DOE)
# ============================================================================

def run_doe(X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            hyperparameter_grid: Dict[str, List],
            device: torch.device) -> List[Dict]:
    """
    Run DOE grid search over hyperparameter combinations.

    Returns:
        List of experiment results with config and metrics
    """
    print("\n" + "=" * 70)
    print("DESIGN OF EXPERIMENTS - HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(f"Grid: {hyperparameter_grid}")

    # Calculate total experiments
    total_experiments = 1
    for values in hyperparameter_grid.values():
        total_experiments *= len(values)
    print(f"Total experiments: {total_experiments}")

    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # DOE loop
    results = []
    experiment_num = 1

    print("\n" + "-" * 70)
    print("Starting experiments...")
    print("-" * 70)

    for hidden_dim in hyperparameter_grid['hidden_dim']:
        for num_blocks in hyperparameter_grid['num_residual_blocks']:
            config = {
                'hidden_dim': hidden_dim,
                'num_residual_blocks': num_blocks
            }

            print(f"\n[Experiment {experiment_num}/{total_experiments}] Config: {config}")

            # Initialize model
            model = ARWaveguideResNet(
                input_dim=5,
                hidden_dim=hidden_dim,
                num_residual_blocks=num_blocks
            ).to(device)

            n_params = model.count_parameters()
            print(f"  Parameters: {n_params:,}")

            # Train
            start_time = time.time()
            final_val_loss = train_model(
                model, train_loader, val_loader,
                epochs=20, lr=0.001, device=device, verbose=False
            )
            train_time = time.time() - start_time

            print(f"  Val Loss: {final_val_loss:.6f}")
            print(f"  Time: {train_time:.1f}s")

            # Store results
            results.append({
                'experiment_num': experiment_num,
                'config': config,
                'hidden_dim': hidden_dim,
                'num_residual_blocks': num_blocks,
                'val_loss': final_val_loss,
                'n_parameters': n_params,
                'train_time_s': train_time
            })

            experiment_num += 1

    return results


# ============================================================================
# 5. RESULTS ANALYSIS
# ============================================================================

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze DOE results and identify best configuration."""
    print("\n" + "=" * 70)
    print("DOE RESULTS SUMMARY")
    print("=" * 70)

    # Sort by validation loss
    sorted_results = sorted(results, key=lambda x: x['val_loss'])

    # Print table
    print(f"\n{'Rank':<6} {'Hidden':<8} {'Blocks':<8} {'Params':<10} {'Val Loss':<12} {'Time (s)':<10}")
    print("-" * 70)
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<6} {result['hidden_dim']:<8} {result['num_residual_blocks']:<8} "
              f"{result['n_parameters']:<10,} {result['val_loss']:<12.6f} "
              f"{result['train_time_s']:<10.1f}")

    # Best configuration
    best_result = sorted_results[0]
    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"  Hidden Dim: {best_result['hidden_dim']}")
    print(f"  Num Blocks: {best_result['num_residual_blocks']}")
    print(f"  Val Loss: {best_result['val_loss']:.6f}")
    print(f"  Parameters: {best_result['n_parameters']:,}")

    return best_result


def plot_doe_results(results: List[Dict], output_path: str):
    """Generate bar chart comparing all configurations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    configs = [f"H={r['hidden_dim']}, B={r['num_residual_blocks']}" for r in results]
    val_losses = [r['val_loss'] for r in results]

    # Sort by loss
    sorted_indices = np.argsort(val_losses)
    configs = [configs[i] for i in sorted_indices]
    val_losses = [val_losses[i] for i in sorted_indices]

    # Color code: best is green, worst is red
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(val_losses)))

    # Plot
    bars = ax.bar(range(len(configs)), val_losses, color=colors, edgecolor='black', linewidth=1.5)

    # Highlight best
    bars[0].set_edgecolor('darkgreen')
    bars[0].set_linewidth(3)

    ax.set_xlabel('Configuration (Hidden Dim, Num Blocks)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss (MSE)', fontsize=12, fontweight='bold')
    ax.set_title('DOE: Neural Architecture Search Results\n'
                 'AR Waveguide Diffraction Angle Prediction',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, loss) in enumerate(zip(bars, val_losses)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] DOE plot saved: {output_path}")
    plt.close()


# ============================================================================
# 6. FINAL MODEL TRAINING
# ============================================================================

def train_final_model(best_config: Dict, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray,
                      y_test: np.ndarray, device: torch.device) -> nn.Module:
    """
    Re-train best configuration for 50 epochs on Train+Val data.
    """
    print("\n" + "=" * 70)
    print("FINAL MODEL TRAINING (50 EPOCHS ON TRAIN+VAL)")
    print("=" * 70)

    # Combine train and validation sets
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])

    print(f"Combined training set: {len(X_combined):,} samples")

    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_combined),
        torch.FloatTensor(y_combined)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Initialize best model
    model = ARWaveguideResNet(
        input_dim=5,
        hidden_dim=best_config['hidden_dim'],
        num_residual_blocks=best_config['num_residual_blocks']
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Train
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining progress:")
    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        if (epoch + 1) % 10 == 0:
            test_loss = evaluate(model, test_loader, criterion, device)
            print(f"  Epoch {epoch+1:2d}/50: Train Loss = {train_loss:.6f}, "
                  f"Test Loss = {test_loss:.6f}")

    # Final evaluation
    final_test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\n[OK] Final Test Loss: {final_test_loss:.6f}")

    return model


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("P2 DOE TRAINING - NEURAL ARCHITECTURE SEARCH")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create output directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # 1. Load and preprocess data
    (X_train, X_val, X_test, y_train, y_val, y_test,
     scaler_X, scaler_y, label_encoder) = load_and_preprocess_data("data/p2_ar_dataset.csv")

    # Save scalers
    scalers = {
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'label_encoder': label_encoder
    }
    with open(models_dir / "p2_scalers.pkl", 'wb') as f:
        pickle.dump(scalers, f)
    print(f"\n[OK] Scalers saved: models/p2_scalers.pkl")

    # 2. Define hyperparameter grid
    hyperparameter_grid = {
        'hidden_dim': [32, 64, 128],
        'num_residual_blocks': [2, 4, 6]
    }

    # 3. Run DOE
    results = run_doe(X_train, y_train, X_val, y_val, hyperparameter_grid, device)

    # 4. Analyze results
    best_result = analyze_results(results)

    # 5. Plot results
    plot_doe_results(results, str(models_dir / "doe_performance.png"))

    # 6. Save DOE results
    doe_results_path = models_dir / "doe_results.json"
    with open(doe_results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] DOE results saved: {doe_results_path}")

    # 7. Train final model
    final_model = train_final_model(
        best_result, X_train, y_train, X_val, y_val, X_test, y_test, device
    )

    # 8. Save final model
    model_path = models_dir / "best_rainbow_model.pth"
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'config': best_result['config'],
        'val_loss': best_result['val_loss'],
    }, model_path)
    print(f"[OK] Best model saved: {model_path}")

    print("\n" + "=" * 70)
    print("[OK] DOE TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
