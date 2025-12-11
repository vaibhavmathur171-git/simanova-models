"""
P2 Rainbow Dataset - Design of Experiments (DOE) Training Script

Experiments with:
- Training Set Sizes: [1000, 10000, 40000]
- Neurons per Layer: [8, 32, 128] (3 hidden layers)
- Epochs: [10, 50]

Total: 3 x 3 x 2 = 18 experiments
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import product

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# DOE Parameters
TRAINING_SIZES = [1000, 10000, 40000]
NEURONS_PER_LAYER = [8, 32, 128]
EPOCHS_LIST = [10, 50]

# Fixed parameters
TEST_SIZE = 5000
BATCH_SIZE = 256
LEARNING_RATE = 0.001


class RainbowNet(nn.Module):
    """Neural network for chromatic dispersion prediction."""

    def __init__(self, neurons_per_layer):
        super(RainbowNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.ReLU(),
            nn.Linear(neurons_per_layer, 1)
        )

    def forward(self, x):
        return self.network(x)


class MinMaxScaler:
    """Simple Min-Max scaler."""

    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit(self, data):
        self.min_val = data.min(axis=0)
        self.max_val = data.max(axis=0)
        return self

    def transform(self, data):
        return (data - self.min_val) / (self.max_val - self.min_val + 1e-8)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * (self.max_val - self.min_val + 1e-8) + self.min_val

    def to_dict(self):
        return {
            'min': self.min_val.tolist() if hasattr(self.min_val, 'tolist') else float(self.min_val),
            'max': self.max_val.tolist() if hasattr(self.max_val, 'tolist') else float(self.max_val)
        }


def load_and_prepare_data(filepath):
    """Load dataset and return as numpy arrays."""
    df = pd.read_csv(filepath)
    X = df[['Target_Angle', 'Wavelength_nm']].values
    y = df[['Period_nm']].values
    return X, y


def train_model(model, train_loader, criterion, optimizer, epochs, verbose=False):
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

        if verbose and (epoch + 1) % 10 == 0:
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


def save_scalers(input_scaler, output_scaler, filepath):
    """Save scalers to JSON file."""
    scalers_dict = {
        'input_scaler': input_scaler.to_dict(),
        'output_scaler': output_scaler.to_dict()
    }
    with open(filepath, 'w') as f:
        json.dump(scalers_dict, f, indent=2)


def run_experiment(X_train, y_train, X_test, y_test, neurons, epochs):
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
    model = RainbowNet(neurons).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    start_time = time.time()
    train_model(model, train_loader, criterion, optimizer, epochs)
    training_time = time.time() - start_time

    # Evaluate
    test_loss = evaluate_model(model, test_loader, criterion)

    return model, test_loss, training_time


def main():
    print("=" * 70)
    print("P2 Rainbow Dataset - Design of Experiments (DOE)")
    print("=" * 70)

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Load data
    print("\nLoading dataset...")
    data_path = os.path.join('data', 'p2_rainbow_data.csv')
    X, y = load_and_prepare_data(data_path)
    print(f"Total samples: {len(X):,}")

    # Shuffle data
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Reserve test set (last TEST_SIZE samples)
    X_test_raw = X[-TEST_SIZE:]
    y_test_raw = y[-TEST_SIZE:]
    X_pool = X[:-TEST_SIZE]
    y_pool = y[:-TEST_SIZE]

    print(f"Test set size: {TEST_SIZE:,}")
    print(f"Training pool size: {len(X_pool):,}")

    # Generate all experiment combinations
    experiments = list(product(TRAINING_SIZES, NEURONS_PER_LAYER, EPOCHS_LIST))
    total_experiments = len(experiments)

    print(f"\nRunning {total_experiments} experiments...")
    print("-" * 70)

    results = []
    best_loss = float('inf')
    best_model = None
    best_input_scaler = None
    best_output_scaler = None

    for exp_idx, (train_size, neurons, epochs) in enumerate(experiments, 1):
        print(f"\nExperiment {exp_idx}/{total_experiments}: "
              f"Size={train_size:,}, Neurons={neurons}, Epochs={epochs}")

        # Select training subset
        X_train_raw = X_pool[:train_size]
        y_train_raw = y_pool[:train_size]

        # Create and fit scalers
        input_scaler = MinMaxScaler()
        output_scaler = MinMaxScaler()

        X_train_scaled = input_scaler.fit_transform(X_train_raw)
        y_train_scaled = output_scaler.fit_transform(y_train_raw)

        # Scale test set using training scalers
        X_test_scaled = input_scaler.transform(X_test_raw)
        y_test_scaled = output_scaler.transform(y_test_raw)

        # Run experiment
        model, test_loss, training_time = run_experiment(
            X_train_scaled, y_train_scaled,
            X_test_scaled, y_test_scaled,
            neurons, epochs
        )

        print(f"  -> Test Loss: {test_loss:.6f}, Time: {training_time:.2f}s")

        # Record results
        results.append({
            'Size': train_size,
            'Neurons': neurons,
            'Epochs': epochs,
            'Final_Test_Loss': test_loss,
            'Training_Time_Sec': round(training_time, 2)
        })

        # Track best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model
            best_input_scaler = input_scaler
            best_output_scaler = output_scaler
            print(f"  -> New best model!")

    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")

    results_df = pd.DataFrame(results)
    results_path = os.path.join('data', 'p2_doe_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    # Save best model
    model_path = os.path.join('models', 'p2_rainbow_model.pth')
    torch.save(best_model.state_dict(), model_path)
    print(f"Best model saved to: {model_path}")

    # Save scalers for best model
    scalers_path = os.path.join('models', 'p2_scalers.json')
    save_scalers(best_input_scaler, best_output_scaler, scalers_path)
    print(f"Scalers saved to: {scalers_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("DOE Results Summary:")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Find best configuration
    best_idx = results_df['Final_Test_Loss'].idxmin()
    best_config = results_df.iloc[best_idx]
    print("\n" + "-" * 70)
    print("BEST CONFIGURATION:")
    print(f"  Size: {int(best_config['Size']):,}")
    print(f"  Neurons: {int(best_config['Neurons'])}")
    print(f"  Epochs: {int(best_config['Epochs'])}")
    print(f"  Test Loss: {best_config['Final_Test_Loss']:.6f}")
    print(f"  Training Time: {best_config['Training_Time_Sec']:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
