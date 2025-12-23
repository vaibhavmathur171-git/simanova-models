"""
P2 AR Waveguide - Google Colab Training Script
===============================================

INSTRUCTIONS:
1. Upload this script to Google Colab
2. Upload p2_ar_dataset.csv to the 'data' folder in Colab
3. Run this script

This will train the ResNet model with DOE and save results.
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from itertools import product

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# DOE Parameters
HIDDEN_DIMS = [64, 128]
NUM_LAYERS = [2, 4]

# Fixed parameters
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2
INPUT_DIM = 4  # wavelength, incident_angle, grating_period, refractive_index
OUTPUT_DIM = 1  # diffracted_angle


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out


class ARWaveguideResNet(nn.Module):
    """ResNet for AR waveguide diffraction angle prediction."""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ARWaveguideResNet, self).__init__()

        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


def load_and_prepare_data(filepath):
    """
    Load AR waveguide dataset and return features and target.

    Features (X): wavelength, incident_angle, grating_period, refractive_index
    Target (Y): diffracted_angle
    """
    df = pd.read_csv(filepath)

    # Features
    X = df[['wavelength', 'incident_angle', 'grating_period', 'refractive_index']].values

    # Target
    y = df[['diffracted_angle']].values

    print(f"\nDataset loaded: {len(df):,} samples")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y


def train_model(model, train_loader, criterion, optimizer, epochs, verbose=True):
    """Train the model and return training history."""
    model.train()
    history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

    return history


def evaluate_model(model, test_loader, criterion):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

    return total_loss / len(test_loader)


def run_experiment(X_train, y_train, X_test, y_test, hidden_dim, num_layers):
    """Run a single training experiment."""
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create model
    model = ARWaveguideResNet(INPUT_DIM, hidden_dim, num_layers, OUTPUT_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    print(f"  Training...")
    start_time = time.time()
    history = train_model(model, train_loader, criterion, optimizer, EPOCHS, verbose=True)
    training_time = time.time() - start_time

    # Evaluate
    test_loss = evaluate_model(model, test_loader, criterion)

    return model, test_loss, training_time, history


def main():
    print("=" * 70)
    print("P2 AR Waveguide - Design of Experiments (DOE)")
    print("=" * 70)

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('Data', exist_ok=True)

    # Load data
    print("\nLoading AR waveguide dataset...")
    data_path = 'data/p2_ar_dataset.csv'

    if not os.path.exists(data_path):
        print(f"\nERROR: Dataset not found at {data_path}")
        print("\nPlease upload p2_ar_dataset.csv to the 'data' folder:")
        print("1. Click the folder icon on the left sidebar")
        print("2. Navigate to 'data' folder")
        print("3. Click upload and select your p2_ar_dataset.csv file")
        return

    X, y = load_and_prepare_data(data_path)

    # Shuffle data
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Train/test split
    test_size = int(len(X) * TEST_SPLIT)
    X_train_raw = X[:-test_size]
    y_train_raw = y[:-test_size]
    X_test_raw = X[-test_size:]
    y_test_raw = y[-test_size:]

    print(f"\nTraining set size: {len(X_train_raw):,}")
    print(f"Test set size: {len(X_test_raw):,}")

    # Standardize features and target
    print("\nStandardizing data...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)

    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    # Generate all experiment combinations
    experiments = list(product(HIDDEN_DIMS, NUM_LAYERS))
    total_experiments = len(experiments)

    print(f"\nRunning {total_experiments} experiments...")
    print(f"DOE Grid: Hidden Dims {HIDDEN_DIMS}, Layers {NUM_LAYERS}")
    print("-" * 70)

    results = []
    best_loss = float('inf')
    best_model = None
    best_config = None

    for exp_idx, (hidden_dim, num_layers) in enumerate(experiments, 1):
        print(f"\nExperiment {exp_idx}/{total_experiments}: "
              f"Hidden Dim={hidden_dim}, Layers={num_layers}")

        # Run experiment
        model, test_loss, training_time, history = run_experiment(
            X_train_scaled, y_train_scaled,
            X_test_scaled, y_test_scaled,
            hidden_dim, num_layers
        )

        print(f"\n  -> Test MSE Loss: {test_loss:.6f}")
        print(f"  -> Training Time: {training_time:.2f}s")

        # Record results
        results.append({
            'Hidden_Dim': hidden_dim,
            'Num_Layers': num_layers,
            'Test_MSE_Loss': test_loss,
            'Training_Time_Sec': round(training_time, 2)
        })

        # Track best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model
            best_config = {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'input_dim': INPUT_DIM,
                'output_dim': OUTPUT_DIM
            }
            print(f"  -> New best model!")

    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")

    results_df = pd.DataFrame(results)
    results_path = 'Data/p2_doe_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"DOE results saved to: {results_path}")

    # Save best model
    model_path = 'models/best_ar_waveguide_model.pth'
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'config': best_config
    }, model_path)
    print(f"Best model saved to: {model_path}")

    # Save scaler as pickle
    scaler_path = 'models/p2_scaler.pkl'
    scaler_dict = {
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_dict, f)
    print(f"Scalers saved to: {scaler_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("DOE Results Summary:")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Find best configuration
    best_idx = results_df['Test_MSE_Loss'].idxmin()
    best_result = results_df.iloc[best_idx]
    print("\n" + "-" * 70)
    print("BEST CONFIGURATION:")
    print(f"  Hidden Dim: {int(best_result['Hidden_Dim'])}")
    print(f"  Num Layers: {int(best_result['Num_Layers'])}")
    print(f"  Test MSE Loss: {best_result['Test_MSE_Loss']:.6f}")
    print(f"  Training Time: {best_result['Training_Time_Sec']:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
