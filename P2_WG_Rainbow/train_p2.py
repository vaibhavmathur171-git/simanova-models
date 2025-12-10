import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
INPUT_SIZE = 2
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

class RainbowMLP(nn.Module):
    """
    MLP for P2 Rainbow Waveguide prediction.
    Architecture: 2 -> 128 -> 128 -> 128 -> 1
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RainbowMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def load_and_preprocess_data(csv_file):
    """
    Load data and apply scalers.
    - Wavelength: MinMaxScaler (400-700 -> 0-1)
    - Angle: StandardScaler
    - Period: StandardScaler (target)
    """
    print("Loading data from:", csv_file)
    df = pd.read_csv(csv_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print()

    # Separate features and target
    X_wavelength = df['Wavelength_nm'].values.reshape(-1, 1)
    X_angle = df['Target_Angle'].values.reshape(-1, 1)
    y_period = df['Period_nm'].values.reshape(-1, 1)

    # Initialize scalers
    wavelength_scaler = MinMaxScaler(feature_range=(0, 1))
    angle_scaler = StandardScaler()
    period_scaler = StandardScaler()

    # Fit and transform
    X_wavelength_scaled = wavelength_scaler.fit_transform(X_wavelength)
    X_angle_scaled = angle_scaler.fit_transform(X_angle)
    y_period_scaled = period_scaler.fit_transform(y_period)

    # Combine features
    X_scaled = np.hstack([X_angle_scaled, X_wavelength_scaled])

    # Save scaler parameters to JSON
    scaler_params = {
        'wavelength': {
            'type': 'MinMaxScaler',
            'min': float(wavelength_scaler.data_min_[0]),
            'max': float(wavelength_scaler.data_max_[0]),
            'scale': float(wavelength_scaler.scale_[0]),
            'data_min': float(wavelength_scaler.min_[0])
        },
        'angle': {
            'type': 'StandardScaler',
            'mean': float(angle_scaler.mean_[0]),
            'std': float(angle_scaler.scale_[0])
        },
        'period': {
            'type': 'StandardScaler',
            'mean': float(period_scaler.mean_[0]),
            'std': float(period_scaler.scale_[0])
        }
    }

    with open('p2_scalers.json', 'w') as f:
        json.dump(scaler_params, f, indent=4)
    print("Scaler parameters saved to: p2_scalers.json")
    print()

    return X_scaled, y_period_scaled, scaler_params

def create_data_loaders(X, y, test_size=0.2, batch_size=32):
    """Split data and create PyTorch DataLoaders."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()

    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    """Train the MLP model."""
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on test set."""
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss

if __name__ == "__main__":
    print("=" * 60)
    print("P2 RAINBOW WAVEGUIDE - TRAINING SCRIPT")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    # Load and preprocess data
    X_scaled, y_scaled, scaler_params = load_and_preprocess_data('p2_rainbow_data.csv')

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        X_scaled, y_scaled, test_size=0.2, batch_size=BATCH_SIZE
    )

    # Initialize model
    model = RainbowMLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(device)
    print("Model Architecture:")
    print(model)
    print()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training Configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Optimizer: Adam")
    print(f"  Loss Function: MSELoss")
    print()

    # Train the model
    print("Starting training...")
    print("-" * 60)
    train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS, device)
    print("-" * 60)
    print()

    # Evaluate on test set
    test_loss = evaluate_model(model, test_loader, criterion, device)
    print("=" * 60)
    print(f"FINAL TEST LOSS: {test_loss:.6f}")
    print("=" * 60)
    print()

    # Save the model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/p2_rainbow_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    print()

    print("Training completed successfully!")
