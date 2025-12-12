"""
Project 3: Waveguide Uniformity - CNN Training Script
======================================================
Trains a lightweight CNN for Image-to-Image regression to predict
light uniformity patterns from grating efficiency maps.

Author: SimaNova Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# Configuration
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Paths
INPUT_PATH = "data/p3_input_gratings.npy"
TARGET_PATH = "data/p3_output_fields.npy"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "p3_cnn_model.pth")


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


class WaveguideCNN(nn.Module):
    """
    Lightweight CNN for Image-to-Image regression.

    Architecture:
        Input (1, 64, 64) -> Conv(16) -> Conv(32) -> Conv(1) -> Output (1, 64, 64)

    Uses padding=1 with kernel_size=3 to maintain spatial dimensions.
    """

    def __init__(self):
        super(WaveguideCNN, self).__init__()

        # Layer 1: Conv2d(1, 16, kernel=3, padding=1) -> ReLU
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # Layer 2: Conv2d(16, 32, kernel=3, padding=1) -> ReLU
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # Layer 3: Conv2d(32, 1, kernel=3, padding=1) -> Sigmoid
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Layer 1: (N, 1, 64, 64) -> (N, 16, 64, 64)
        x = self.relu1(self.conv1(x))

        # Layer 2: (N, 16, 64, 64) -> (N, 32, 64, 64)
        x = self.relu2(self.conv2(x))

        # Layer 3: (N, 32, 64, 64) -> (N, 1, 64, 64)
        x = self.sigmoid(self.conv3(x))

        return x


def load_data():
    """Load and prepare the dataset."""
    print("Loading data...")

    # Load numpy arrays
    inputs = np.load(INPUT_PATH)
    targets = np.load(TARGET_PATH)

    print(f"  Raw input shape:  {inputs.shape}")
    print(f"  Raw target shape: {targets.shape}")

    # Reshape from (N, 64, 64) to (N, 1, 64, 64) for Conv2d
    inputs = inputs[:, np.newaxis, :, :]
    targets = targets[:, np.newaxis, :, :]

    print(f"  Reshaped input:   {inputs.shape}")
    print(f"  Reshaped target:  {targets.shape}")

    # Convert to PyTorch tensors
    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(targets).float()

    return inputs, targets


def split_data(inputs, targets, train_split=TRAIN_SPLIT):
    """Split data into training and test sets."""
    # Set seed for reproducibility
    torch.manual_seed(RANDOM_SEED)

    n_samples = inputs.shape[0]
    n_train = int(n_samples * train_split)

    # Shuffle indices
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    # Split data
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


def train_model(model, train_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate_model(model, test_loader, criterion, device):
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

    avg_loss = total_loss / len(test_loader)
    return avg_loss


def main():
    """Main training function."""
    print("=" * 60)
    print("Project 3: Waveguide CNN Training")
    print("=" * 60)

    # Get device
    device = get_device()
    print()

    # Load and prepare data
    inputs, targets = load_data()
    train_inputs, train_targets, test_inputs, test_targets = split_data(inputs, targets)
    train_loader, test_loader = create_dataloaders(
        train_inputs, train_targets, test_inputs, test_targets
    )

    # Initialize model
    model = WaveguideCNN().to(device)
    print(f"\nModel Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining Configuration:")
    print(f"  - Loss Function: MSELoss")
    print(f"  - Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Epochs: {EPOCHS}")

    # Training loop
    print(f"\n{'='*60}")
    print("Training Started")
    print("=" * 60)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Test Loss':>12}")
    print("-" * 36)

    best_test_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_model(model, train_loader, criterion, optimizer, device, epoch)
        test_loss = evaluate_model(model, test_loader, criterion, device)

        print(f"{epoch:>6} | {train_loss:>12.6f} | {test_loss:>12.6f}")

        # Track best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss

    print("-" * 36)

    # Create models directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"\nCreated '{MODEL_DIR}' folder")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Final evaluation
    final_test_loss = evaluate_model(model, test_loader, criterion, device)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Test Loss: {final_test_loss:.6f}")
    print(f"Best Test Loss:  {best_test_loss:.6f}")

    # Verify output dimensions
    print(f"\nDimension Verification:")
    model.eval()
    with torch.no_grad():
        sample_input = test_inputs[:1].to(device)
        sample_output = model(sample_input)
        print(f"  Input shape:  {sample_input.shape}")
        print(f"  Output shape: {sample_output.shape}")
        print(f"  Match: {sample_input.shape == sample_output.shape}")


if __name__ == "__main__":
    main()
