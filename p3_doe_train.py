"""
Project 3: Waveguide Uniformity - DOE Training Script
======================================================
Design of Experiments to benchmark three CNN architectures:
- Model A: Pixel-Wise (1x1 kernels) - Baseline
- Model B: Standard (3-layer, 3x3 kernels) - Benchmark
- Model C: Deep Receptive (5-layer, 3x3 kernels) - High Capacity

Author: SimaNova Team
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Normalization constant (max depth in nm)
DEPTH_NORM = 400.0

# Paths
DEPTH_PATH = "data/p3_depth_maps.npy"
LIGHT_PATH = "data/p3_light_fields.npy"
MODEL_DIR = "models"
RESULTS_PATH = "data/p3_doe_results.csv"


# =============================================================================
# DEVICE DETECTION
# =============================================================================
def get_device():
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class ModelA_PixelWise(nn.Module):
    """
    Model A: "The Pixel-Wise" (Baseline)

    Architecture: Conv2d(1x1) -> ReLU -> Conv2d(1x1)

    Hypothesis: This should perform POORLY because 1x1 kernels cannot see
    neighboring pixels. The physics involves energy propagation from left
    to right, which requires spatial context.

    Receptive Field: 1x1 (single pixel)
    """
    def __init__(self):
        super(ModelA_PixelWise, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class ModelB_Standard(nn.Module):
    """
    Model B: "The Standard" (Benchmark)

    Architecture: 3 Layers of Conv2d(3x3, padding=1)
    Conv(1->16) -> ReLU -> Conv(16->32) -> ReLU -> Conv(32->1) -> Sigmoid

    This is the standard approach with moderate receptive field.

    Receptive Field: 7x7 (3 layers * 2 pixels growth + 1)
    """
    def __init__(self):
        super(ModelB_Standard, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class ModelC_DeepReceptive(nn.Module):
    """
    Model C: "The Deep Receptive" (High Capacity)

    Architecture: 5 Layers of Conv2d(3x3, padding=1)
    Conv(1->16) -> Conv(16->32) -> Conv(32->64) -> Conv(64->32) -> Conv(32->1)

    Should theoretically capture longer-range energy depletion effects
    due to larger receptive field.

    Receptive Field: 11x11 (5 layers * 2 pixels growth + 1)
    """
    def __init__(self):
        super(ModelC_DeepReceptive, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data():
    """Load and prepare the dataset with normalization."""
    print("Loading data...")

    # Load numpy arrays
    depth_maps = np.load(DEPTH_PATH)
    light_fields = np.load(LIGHT_PATH)

    print(f"  Raw depth maps shape:  {depth_maps.shape}")
    print(f"  Raw light fields shape: {light_fields.shape}")

    # Normalize depth maps (divide by 400nm)
    depth_maps_norm = depth_maps / DEPTH_NORM
    print(f"  Normalized depth range: [{depth_maps_norm.min():.3f}, {depth_maps_norm.max():.3f}]")

    # Reshape from (N, 64, 64) to (N, 1, 64, 64) for Conv2d
    inputs = depth_maps_norm[:, np.newaxis, :, :]
    targets = light_fields[:, np.newaxis, :, :]

    print(f"  Reshaped input:  {inputs.shape}")
    print(f"  Reshaped target: {targets.shape}")

    # Convert to PyTorch tensors
    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(targets).float()

    return inputs, targets


def split_data(inputs, targets, train_split=TRAIN_SPLIT):
    """Split data into training and test sets."""
    torch.manual_seed(RANDOM_SEED)

    n_samples = inputs.shape[0]
    n_train = int(n_samples * train_split)

    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_inputs = inputs[train_indices]
    train_targets = targets[train_indices]
    test_inputs = inputs[test_indices]
    test_targets = targets[test_indices]

    print(f"\nData split:")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Test samples:     {len(test_indices)}")

    return train_inputs, train_targets, test_inputs, test_targets


def create_dataloaders(train_inputs, train_targets, test_inputs, test_targets):
    """Create PyTorch DataLoaders."""
    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test data."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(test_loader)


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, model_name, train_loader, test_loader, device, epochs=EPOCHS):
    """
    Train a model and return training history.

    Returns:
        history: List of (model_name, epoch, test_loss) tuples
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,}")

    history = []

    print(f"\n{'Epoch':>6} | {'Train Loss':>12} | {'Test Loss':>12}")
    print("-" * 36)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        print(f"{epoch:>6} | {train_loss:>12.6f} | {test_loss:>12.6f}")

        # Store results
        history.append({
            'Model_Name': model_name,
            'Epoch': epoch,
            'Test_Loss': test_loss
        })

    print("-" * 36)
    print(f"Final Test Loss: {test_loss:.6f}")

    return model, history


# =============================================================================
# MAIN DOE EXECUTION
# =============================================================================
def main():
    """Main DOE execution function."""
    print("=" * 70)
    print("Project 3: DOE - CNN Architecture Benchmark")
    print("=" * 70)

    # Get device
    device = get_device()
    print()

    # Load and prepare data
    inputs, targets = load_data()
    train_inputs, train_targets, test_inputs, test_targets = split_data(inputs, targets)
    train_loader, test_loader = create_dataloaders(
        train_inputs, train_targets, test_inputs, test_targets
    )

    # Create models directory
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"\nCreated '{MODEL_DIR}' folder")

    # Define experiments
    experiments = [
        ("Model_A_PixelWise", ModelA_PixelWise(), "models/p3_model_A.pth"),
        ("Model_B_Standard", ModelB_Standard(), "models/p3_model_B.pth"),
        ("Model_C_DeepReceptive", ModelC_DeepReceptive(), "models/p3_model_C.pth"),
    ]

    # Print experiment overview
    print(f"\n{'EXPERIMENT OVERVIEW':=^70}")
    print(f"{'Model':<25} {'Architecture':<30} {'Receptive Field':<15}")
    print("-" * 70)
    print(f"{'Model A (Pixel-Wise)':<25} {'2x Conv(1x1)':<30} {'1x1':<15}")
    print(f"{'Model B (Standard)':<25} {'3x Conv(3x3)':<30} {'7x7':<15}")
    print(f"{'Model C (Deep)':<25} {'5x Conv(3x3)':<30} {'11x11':<15}")
    print()

    # Collect all results
    all_results = []

    # Run experiments
    for model_name, model, save_path in experiments:
        trained_model, history = train_model(
            model, model_name, train_loader, test_loader, device, epochs=EPOCHS
        )

        # Save model
        torch.save(trained_model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

        # Collect results
        all_results.extend(history)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n{'RESULTS SAVED':=^70}")
    print(f"Results table: {RESULTS_PATH}")

    # Print summary table
    print(f"\n{'FINAL RESULTS SUMMARY':=^70}")
    print(f"\n{'Model':<25} {'Final Test Loss':>15} {'Parameters':>15}")
    print("-" * 55)

    for model_name, model, _ in experiments:
        final_loss = results_df[results_df['Model_Name'] == model_name]['Test_Loss'].iloc[-1]
        n_params = count_parameters(model)
        print(f"{model_name:<25} {final_loss:>15.6f} {n_params:>15,}")

    # Analysis
    print(f"\n{'HYPOTHESIS VALIDATION':=^70}")

    model_a_loss = results_df[results_df['Model_Name'] == 'Model_A_PixelWise']['Test_Loss'].iloc[-1]
    model_b_loss = results_df[results_df['Model_Name'] == 'Model_B_Standard']['Test_Loss'].iloc[-1]
    model_c_loss = results_df[results_df['Model_Name'] == 'Model_C_DeepReceptive']['Test_Loss'].iloc[-1]

    print(f"\nModel A (1x1 kernel) Loss: {model_a_loss:.6f}")
    print(f"Model B (3x3 kernel) Loss: {model_b_loss:.6f}")
    print(f"Model C (5-layer)    Loss: {model_c_loss:.6f}")

    if model_a_loss > model_b_loss:
        print("\n[CONFIRMED] Model A performs worse than Model B.")
        print("  -> Spatial context (3x3 kernels) is essential for this physics problem.")
    else:
        print("\n[UNEXPECTED] Model A performs similar or better than Model B.")

    if model_c_loss < model_b_loss:
        print("\n[CONFIRMED] Model C outperforms Model B.")
        print("  -> Deeper receptive field captures longer-range depletion effects.")
    else:
        print("\n[OBSERVATION] Model C does not significantly outperform Model B.")
        print("  -> 3-layer network may be sufficient for this grid size (64x64).")

    best_model = min([
        ('Model_A_PixelWise', model_a_loss),
        ('Model_B_Standard', model_b_loss),
        ('Model_C_DeepReceptive', model_c_loss)
    ], key=lambda x: x[1])

    print(f"\nBest Model: {best_model[0]} (Loss: {best_model[1]:.6f})")

    print(f"\n{'DOE COMPLETE':=^70}")


if __name__ == "__main__":
    main()
