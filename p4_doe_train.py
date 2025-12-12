"""
Project 4: Thermal Management - PINN DOE Training Script
=========================================================
Design of Experiments to find optimal Physics-Informed Neural Network
architecture for solving the 2D Heat Equation with source term.

PDE: du/dt = alpha * (d2u/dx2 + d2u/dy2) + Q
Where:
- alpha = 0.01 (thermal diffusivity)
- Q = heat source (Gaussian hotspot)

Author: SimaNova Team
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import List, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================
EPOCHS = 2000
LEARNING_RATE = 0.001
LOG_INTERVAL = 100
RANDOM_SEED = 42

# Physics Constants
ALPHA = 0.01  # Thermal diffusivity

# Domain bounds
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 0.2
T_MIN, T_MAX = 0.0, 1.0

# Collocation points
N_COLLOCATION = 10000  # Interior points for PDE residual
N_BOUNDARY = 1000      # Boundary condition points
N_INITIAL = 1000       # Initial condition points

# Paths
MODEL_DIR = "models"
DATA_DIR = "data"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "p4_pinn_model.pth")
RESULTS_PATH = os.path.join(DATA_DIR, "p4_doe_results.csv")


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
# PINN MODEL ARCHITECTURE
# =============================================================================
class ThermalPINN(nn.Module):
    """
    Physics-Informed Neural Network for 2D Heat Equation.

    Architecture:
    - Input: (x, y, t) -> 3 neurons
    - Hidden: Variable layers with Tanh activation (crucial for smooth derivatives)
    - Output: u (temperature) -> 1 neuron

    Tanh is essential for PINNs because:
    1. It's infinitely differentiable (C∞)
    2. Smooth gradients for autograd
    3. Output bounded, helps stability
    """

    def __init__(self, num_layers: int, neurons_per_layer: int):
        super(ThermalPINN, self).__init__()

        self.num_layers = num_layers
        self.neurons = neurons_per_layer

        # Build network layers
        layers = []

        # Input layer: (x, y, t) -> neurons
        layers.append(nn.Linear(3, neurons_per_layer))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())

        # Output layer: neurons -> u (temperature)
        layers.append(nn.Linear(neurons_per_layer, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better gradient flow."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass: (x, y, t) -> u"""
        inputs = torch.cat([x, y, t], dim=1)
        return self.network(inputs)


# =============================================================================
# PHYSICS ENGINE
# =============================================================================
def heat_source(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Gaussian heat source (hotspot) at (0.2, 0.1).

    Q(x,y) = 5.0 * exp(-((x-0.2)^2 + (y-0.1)^2) / 0.01)

    This represents a localized heat injection point.
    """
    x0, y0 = 0.2, 0.1  # Hotspot center
    sigma2 = 0.01      # Spread parameter
    amplitude = 5.0

    r2 = (x - x0)**2 + (y - y0)**2
    Q = amplitude * torch.exp(-r2 / sigma2)

    return Q


def compute_pde_residual(model: ThermalPINN, x: torch.Tensor, y: torch.Tensor,
                          t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Compute the PDE residual for the 2D Heat Equation.

    PDE: du/dt - alpha * (d2u/dx2 + d2u/dy2) - Q = 0

    Residual should be zero if the network satisfies the physics.
    """
    # Ensure gradients are tracked
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    # Forward pass
    u = model(x, y, t)

    # First derivatives
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]
    du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]

    # Second derivatives (Laplacian)
    d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx),
                                   create_graph=True, retain_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy),
                                   create_graph=True, retain_graph=True)[0]

    # Heat source
    Q = heat_source(x, y)

    # PDE Residual: du/dt - alpha * (d2u/dx2 + d2u/dy2) - Q
    residual = du_dt - ALPHA * (d2u_dx2 + d2u_dy2) - Q

    return residual


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================
def generate_collocation_points(n_points: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """Generate random interior collocation points for PDE residual."""
    x = torch.rand(n_points, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(n_points, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
    t = torch.rand(n_points, 1, device=device) * (T_MAX - T_MIN) + T_MIN

    return x, y, t


def generate_boundary_points(n_points: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """
    Generate boundary condition points.

    BC: u(x=0, y, t) = 0 (Dirichlet at left boundary)
    """
    # Left boundary: x = 0
    x = torch.zeros(n_points, 1, device=device)
    y = torch.rand(n_points, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
    t = torch.rand(n_points, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    u_bc = torch.zeros(n_points, 1, device=device)

    return x, y, t, u_bc


def generate_initial_points(n_points: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """
    Generate initial condition points.

    IC: u(x, y, t=0) = 0 (Initially cold)
    """
    x = torch.rand(n_points, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(n_points, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
    t = torch.zeros(n_points, 1, device=device)
    u_ic = torch.zeros(n_points, 1, device=device)

    return x, y, t, u_ic


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
def compute_loss(model: ThermalPINN, device: torch.device) -> Tuple[torch.Tensor, ...]:
    """
    Compute total PINN loss.

    Total Loss = Loss_PDE + Loss_BC + Loss_IC

    Where:
    - Loss_PDE: MSE of PDE residual (physics constraint)
    - Loss_BC: MSE of boundary condition error
    - Loss_IC: MSE of initial condition error
    """
    # Generate fresh points each iteration (stochastic sampling)
    x_pde, y_pde, t_pde = generate_collocation_points(N_COLLOCATION, device)
    x_bc, y_bc, t_bc, u_bc = generate_boundary_points(N_BOUNDARY, device)
    x_ic, y_ic, t_ic, u_ic = generate_initial_points(N_INITIAL, device)

    # PDE residual loss
    residual = compute_pde_residual(model, x_pde, y_pde, t_pde, device)
    loss_pde = torch.mean(residual ** 2)

    # Boundary condition loss
    u_pred_bc = model(x_bc, y_bc, t_bc)
    loss_bc = torch.mean((u_pred_bc - u_bc) ** 2)

    # Initial condition loss
    u_pred_ic = model(x_ic, y_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic) ** 2)

    # Total loss
    loss_total = loss_pde + loss_bc + loss_ic

    return loss_total, loss_pde, loss_bc + loss_ic


# =============================================================================
# TRAINING FUNCTION
# =============================================================================
def train_model(model: ThermalPINN, model_name: str, device: torch.device,
                epochs: int = EPOCHS) -> List[dict]:
    """
    Train a PINN model and return training history.

    Returns:
        history: List of dicts with [Model_Name, Epoch, Loss_Total, Loss_PDE, Loss_BC]
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    print(f"Architecture: {model.num_layers} layers x {model.neurons} neurons")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []

    print(f"\n{'Epoch':>6} | {'Total Loss':>12} | {'PDE Loss':>12} | {'BC/IC Loss':>12}")
    print("-" * 52)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Compute loss
        loss_total, loss_pde, loss_bc = compute_loss(model, device)

        # Backward pass
        loss_total.backward()
        optimizer.step()

        # Log every LOG_INTERVAL epochs
        if epoch % LOG_INTERVAL == 0 or epoch == 1:
            print(f"{epoch:>6} | {loss_total.item():>12.6f} | {loss_pde.item():>12.6f} | {loss_bc.item():>12.6f}")

            history.append({
                'Model_Name': model_name,
                'Epoch': epoch,
                'Loss_Total': loss_total.item(),
                'Loss_PDE': loss_pde.item(),
                'Loss_BC': loss_bc.item()
            })

    print("-" * 52)
    print(f"Final Loss: {loss_total.item():.6f}")

    return history, loss_total.item()


# =============================================================================
# MAIN DOE EXECUTION
# =============================================================================
def main():
    """Main DOE execution function."""
    print("=" * 70)
    print("Project 4: PINN DOE - Heat Equation Architecture Benchmark")
    print("=" * 70)

    # Set seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Get device
    device = get_device()
    print()

    # Print physics setup
    print(f"{'PHYSICS SETUP':=^70}")
    print(f"  PDE: du/dt = {ALPHA} * (d2u/dx2 + d2u/dy2) + Q")
    print(f"  Domain: x=[{X_MIN}, {X_MAX}], y=[{Y_MIN}, {Y_MAX}], t=[{T_MIN}, {T_MAX}]")
    print(f"  Heat Source: Q = 5.0 * exp(-((x-0.2)^2 + (y-0.1)^2) / 0.01)")
    print(f"  BC: u(x=0) = 0 (Dirichlet)")
    print(f"  IC: u(t=0) = 0")

    print(f"\n{'TRAINING CONFIG':=^70}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Collocation Points: {N_COLLOCATION}")
    print(f"  Boundary Points: {N_BOUNDARY}")
    print(f"  Initial Points: {N_INITIAL}")

    # Define experiments
    experiments = [
        ("Model_A_Shallow", 2, 64),   # 2 layers x 64 neurons
        ("Model_B_Standard", 4, 64),  # 4 layers x 64 neurons
        ("Model_C_Deep", 8, 64),      # 8 layers x 64 neurons
    ]

    print(f"\n{'EXPERIMENT DESIGN':=^70}")
    print(f"{'Model':<20} {'Layers':<10} {'Neurons':<10} {'Hypothesis':<30}")
    print("-" * 70)
    print(f"{'Model A (Shallow)':<20} {'2':<10} {'64':<10} {'May underfit complex dynamics':<30}")
    print(f"{'Model B (Standard)':<20} {'4':<10} {'64':<10} {'Balanced capacity':<30}")
    print(f"{'Model C (Deep)':<20} {'8':<10} {'64':<10} {'May capture subtle physics':<30}")

    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Collect results
    all_history = []
    best_loss = float('inf')
    best_model = None
    best_model_name = None

    # Run experiments
    for model_name, num_layers, neurons in experiments:
        # Create fresh model (re-initialize weights)
        model = ThermalPINN(num_layers=num_layers, neurons_per_layer=neurons)

        # Train
        history, final_loss = train_model(model, model_name, device, epochs=EPOCHS)

        # Collect history
        all_history.extend(history)

        # Track best model
        if final_loss < best_loss:
            best_loss = final_loss
            best_model = model
            best_model_name = model_name

    # Save best model
    torch.save(best_model.state_dict(), BEST_MODEL_PATH)
    print(f"\n{'BEST MODEL SAVED':=^70}")
    print(f"Model: {best_model_name}")
    print(f"Final Loss: {best_loss:.6f}")
    print(f"Saved to: {BEST_MODEL_PATH}")

    # Save results to CSV
    results_df = pd.DataFrame(all_history)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to: {RESULTS_PATH}")

    # Print summary
    print(f"\n{'FINAL RESULTS SUMMARY':=^70}")
    print(f"\n{'Model':<20} {'Final Loss':>15} {'Layers':>10} {'Neurons':>10}")
    print("-" * 55)

    for model_name, num_layers, neurons in experiments:
        final_loss = results_df[results_df['Model_Name'] == model_name]['Loss_Total'].iloc[-1]
        marker = " <-- BEST" if model_name == best_model_name else ""
        print(f"{model_name:<20} {final_loss:>15.6f} {num_layers:>10} {neurons:>10}{marker}")

    # Analysis
    print(f"\n{'ANALYSIS':=^70}")

    model_a_loss = results_df[results_df['Model_Name'] == 'Model_A_Shallow']['Loss_Total'].iloc[-1]
    model_b_loss = results_df[results_df['Model_Name'] == 'Model_B_Standard']['Loss_Total'].iloc[-1]
    model_c_loss = results_df[results_df['Model_Name'] == 'Model_C_Deep']['Loss_Total'].iloc[-1]

    if model_a_loss > model_b_loss:
        print("\n[CONFIRMED] Model A (2-layer) has higher loss than Model B.")
        print("  -> Shallow networks may underfit the heat diffusion dynamics.")

    if model_c_loss < model_b_loss:
        print("\n[CONFIRMED] Model C (8-layer) outperforms Model B.")
        print("  -> Deeper networks better capture the physics of heat propagation.")
    elif model_c_loss > model_b_loss:
        print("\n[OBSERVATION] Model C (8-layer) does not outperform Model B.")
        print("  -> Diminishing returns or potential vanishing gradients in deep PINNs.")

    print(f"\n{'DOE COMPLETE':=^70}")


if __name__ == "__main__":
    main()
