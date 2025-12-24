# -*- coding: utf-8 -*-
"""
P3: Virtual Wind Tunnel - Physics Validation Script
====================================================
Validates the trained AeroCNN model against ground truth physics.

Test Case: NACA 2412 (Classic Cessna wing profile)
  - 2% max camber at 40% chord
  - 12% thickness

Outputs: models/p3_physics_check.png
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import r2_score


# =============================================================================
# 1. NACA GEOMETRY GENERATOR (from generate_p3_data.py)
# =============================================================================

def naca_4digit(m: float, p: float, t: float, n_points: int = 50):
    """
    Generate NACA 4-digit airfoil coordinates.

    Parameters:
    -----------
    m : float - Maximum camber (0-0.09)
    p : float - Position of maximum camber (0.1-0.9)
    t : float - Maximum thickness (0.05-0.30)
    n_points : int - Number of points per surface

    Returns:
    --------
    x, y : np.ndarray - Airfoil coordinates (clockwise from TE)
    """
    # Cosine spacing for better LE/TE resolution
    beta = np.linspace(0, np.pi, n_points)
    xc = 0.5 * (1 - np.cos(beta))

    # Thickness distribution (standard NACA formula)
    yt = 5 * t * (
        0.2969 * np.sqrt(xc + 1e-10)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1015 * xc**4
    )

    # Camber line and gradient
    if p < 0.01 or m < 0.001:
        yc = np.zeros_like(xc)
        dyc_dx = np.zeros_like(xc)
    else:
        yc = np.where(
            xc < p,
            m / p**2 * (2 * p * xc - xc**2),
            m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * xc - xc**2)
        )
        dyc_dx = np.where(
            xc < p,
            2 * m / p**2 * (p - xc),
            2 * m / (1 - p)**2 * (p - xc)
        )

    # Perpendicular offset angle
    theta = np.arctan(dyc_dx)

    # Upper and lower surfaces
    xu = xc - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = xc + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Combine: upper (TE->LE) + lower (LE->TE)
    x_full = np.concatenate([xu[::-1], xl[1:]])
    y_full = np.concatenate([yu[::-1], yl[1:]])

    return x_full, y_full


# =============================================================================
# 2. VORTEX PANEL METHOD SOLVER (from generate_p3_data.py)
# =============================================================================

class VortexPanelMethod:
    """Vortex Panel Method for 2D inviscid flow around airfoils."""

    def __init__(self, x, y, V_inf=1.0, alpha_deg=0.0):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.V_inf = V_inf
        self.alpha = np.radians(alpha_deg)
        self.u_inf = V_inf * np.cos(self.alpha)
        self.v_inf = V_inf * np.sin(self.alpha)
        self.n = len(x) - 1
        self._setup_panels()

    def _setup_panels(self):
        self.x1, self.y1 = self.x[:-1], self.y[:-1]
        self.x2, self.y2 = self.x[1:], self.y[1:]
        self.xc = 0.5 * (self.x1 + self.x2)
        self.yc = 0.5 * (self.y1 + self.y2)
        self.S = np.maximum(np.sqrt((self.x2-self.x1)**2 + (self.y2-self.y1)**2), 1e-10)
        self.theta = np.arctan2(self.y2 - self.y1, self.x2 - self.x1)
        self.nx, self.ny = np.sin(self.theta), -np.cos(self.theta)
        self.tx, self.ty = np.cos(self.theta), np.sin(self.theta)

    def _compute_influence(self, xp, yp, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        S = np.sqrt(dx**2 + dy**2)
        if S < 1e-12:
            return 0.0, 0.0

        theta_p = np.arctan2(dy, dx)
        cos_t, sin_t = np.cos(theta_p), np.sin(theta_p)
        dxp, dyp = xp - x1, yp - y1
        x_loc = dxp * cos_t + dyp * sin_t
        y_loc = -dxp * sin_t + dyp * cos_t

        if abs(y_loc) < 1e-10:
            y_loc = 1e-10 if y_loc >= 0 else -1e-10

        r1_sq = x_loc**2 + y_loc**2
        r2_sq = (x_loc - S)**2 + y_loc**2
        theta1 = np.arctan2(y_loc, x_loc)
        theta2 = np.arctan2(y_loc, x_loc - S)

        u_loc = (theta2 - theta1) / (2 * np.pi)
        v_loc = -np.log(np.sqrt(r2_sq / r1_sq)) / (2 * np.pi)

        u = u_loc * cos_t - v_loc * sin_t
        v = u_loc * sin_t + v_loc * cos_t
        return u, v

    def _build_system(self):
        n = self.n
        A = np.zeros((n + 1, n + 1))
        self.At = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    A[i, j] = 0.5
                    self.At[i, j] = 0.0
                else:
                    u, v = self._compute_influence(
                        self.xc[i], self.yc[i],
                        self.x1[j], self.y1[j],
                        self.x2[j], self.y2[j]
                    )
                    A[i, j] = u * self.nx[i] + v * self.ny[i]
                    self.At[i, j] = u * self.tx[i] + v * self.ty[i]

        A[n, 0], A[n, n-1] = 1.0, 1.0
        b = np.zeros(n + 1)
        for i in range(n):
            b[i] = -(self.u_inf * self.nx[i] + self.v_inf * self.ny[i])
        return A, b

    def solve(self):
        try:
            A, b = self._build_system()
            solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            gamma = solution[:-1]

            Vt = np.zeros(self.n)
            for i in range(self.n):
                Vt[i] = self.u_inf * self.tx[i] + self.v_inf * self.ty[i]
                for j in range(self.n):
                    Vt[i] += self.At[i, j] * gamma[j]
                Vt[i] += 0.5 * gamma[i]

            Cp = 1.0 - (Vt / self.V_inf)**2
            return Cp, gamma, True
        except:
            return None, None, False


# =============================================================================
# 3. AEROCNN MODEL (from p3_doe_train.py)
# =============================================================================

class AeroCNN(nn.Module):
    """1D CNN for Airfoil Pressure Prediction."""

    def __init__(self, kernel_size=5, num_filters=32, num_layers=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers

        in_channels = 2
        seq_len = 100

        encoder_layers = []
        current_channels = in_channels
        current_len = seq_len

        for i in range(num_layers):
            out_channels = num_filters * (2 ** i)
            padding = kernel_size // 2

            encoder_layers.extend([
                nn.Conv1d(current_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ])

            if i < num_layers - 1 and i % 2 == 0:
                encoder_layers.append(nn.MaxPool1d(2))
                current_len = current_len // 2

            current_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)
        self.flat_size = current_channels * current_len

        hidden_dim = 256
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 100),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


# =============================================================================
# 4. RESAMPLE TO 100 POINTS (ARC-LENGTH BASED)
# =============================================================================

def resample_airfoil(x, y, n_target=100):
    """Resample airfoil to fixed number of points using arc-length."""
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    s_norm = s / s[-1]

    s_new = np.linspace(0, 1, n_target)
    x_new = np.interp(s_new, s_norm, x)
    y_new = np.interp(s_new, s_norm, y)
    return x_new, y_new, s_norm, s_new


def resample_cp_arclength(s_cp, Cp, s_new):
    """Resample Cp values using arc-length parameterization."""
    return np.interp(s_new, s_cp, Cp)


# =============================================================================
# 5. MAIN VALIDATION
# =============================================================================

def main():
    print("=" * 70)
    print("P3: VIRTUAL WIND TUNNEL - PHYSICS VALIDATION")
    print("=" * 70)

    # Paths
    model_path = Path("models/best_aero_model.pth")
    output_path = Path("models/p3_physics_check.png")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # -------------------------------------------------------------------------
    # STEP 1: Generate NACA 2412 Airfoil
    # -------------------------------------------------------------------------
    print("\n[1] Generating NACA 2412 airfoil (Cessna wing profile)...")

    # NACA 2412: 2% camber at 40% chord, 12% thickness
    m = 0.02   # 2% max camber
    p = 0.4    # at 40% chord
    t = 0.12   # 12% thickness

    # Generate with more points for accurate physics
    x_raw, y_raw = naca_4digit(m, p, t, n_points=80)
    print(f"    Raw airfoil: {len(x_raw)} points")

    # -------------------------------------------------------------------------
    # STEP 2: Compute Ground Truth Cp (Vortex Panel Method)
    # -------------------------------------------------------------------------
    print("\n[2] Computing Ground Truth Cp (Vortex Panel Method)...")

    # Close the airfoil for panel method
    x_closed = np.append(x_raw, x_raw[0])
    y_closed = np.append(y_raw, y_raw[0])

    solver = VortexPanelMethod(x_closed, y_closed, V_inf=1.0, alpha_deg=2.0)
    Cp_raw, gamma, success = solver.solve()

    if not success:
        print("    ERROR: Panel method failed!")
        return

    print(f"    Success! Cp range: [{Cp_raw.min():.3f}, {Cp_raw.max():.3f}]")

    # -------------------------------------------------------------------------
    # STEP 3: Resample geometry AND Cp consistently using arc-length
    # -------------------------------------------------------------------------
    print("\n[3] Resampling with arc-length parameterization...")

    # Compute arc-length for closed airfoil (used by panel method)
    dx_closed = np.diff(x_closed)
    dy_closed = np.diff(y_closed)
    ds_closed = np.sqrt(dx_closed**2 + dy_closed**2)
    s_closed = np.concatenate([[0], np.cumsum(ds_closed)])
    s_closed_norm = s_closed / s_closed[-1]

    # Control points are at panel midpoints - compute their arc-length
    # Cp_raw has n values where n = len(x_closed) - 1 = number of panels
    s_cp = 0.5 * (s_closed_norm[:-1] + s_closed_norm[1:])

    print(f"    Panels: {len(Cp_raw)}, s_cp length: {len(s_cp)}")

    # Compute arc-length for raw geometry (for resampling shape)
    dx_raw = np.diff(x_raw)
    dy_raw = np.diff(y_raw)
    ds_raw = np.sqrt(dx_raw**2 + dy_raw**2)
    s_raw = np.concatenate([[0], np.cumsum(ds_raw)])
    s_raw_norm = s_raw / s_raw[-1]

    # New uniform arc-length sampling (skip the closure segment)
    # Map s_new to 0->0.99 to avoid the closing panel
    s_new = np.linspace(0, 0.99, 100)

    # Resample geometry to 100 points
    x_100 = np.interp(s_new, s_raw_norm, x_raw)
    y_100 = np.interp(s_new, s_raw_norm, y_raw)

    # Resample Cp to 100 points using arc-length
    Cp_truth = np.interp(s_new, s_cp, Cp_raw)

    print(f"    Resampled to 100 points")
    print(f"    Cp_truth range: [{Cp_truth.min():.3f}, {Cp_truth.max():.3f}]")

    # -------------------------------------------------------------------------
    # STEP 4: Load Trained AeroCNN Model
    # -------------------------------------------------------------------------
    print("\n[4] Loading trained AeroCNN model...")

    if not model_path.exists():
        print(f"    ERROR: Model not found at {model_path}")
        return

    # Best config from DOE: kernel=3, filters=16, layers=2
    model = AeroCNN(kernel_size=3, num_filters=16, num_layers=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"    Model loaded successfully!")
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -------------------------------------------------------------------------
    # STEP 5: AI Prediction
    # -------------------------------------------------------------------------
    print("\n[5] Running AI prediction...")

    # Prepare input: (1, 2, 100)
    X = np.stack([x_100, y_100], axis=0)  # (2, 100)
    X_tensor = torch.FloatTensor(X).unsqueeze(0).to(device)  # (1, 2, 100)

    with torch.no_grad():
        Cp_pred = model(X_tensor).cpu().numpy().squeeze()  # (100,)

    print(f"    Prediction Cp range: [{Cp_pred.min():.3f}, {Cp_pred.max():.3f}]")

    # -------------------------------------------------------------------------
    # STEP 6: Calculate R-squared
    # -------------------------------------------------------------------------
    print("\n[6] Calculating R-squared score...")

    r2 = r2_score(Cp_truth, Cp_pred)
    mse = np.mean((Cp_truth - Cp_pred)**2)
    mae = np.mean(np.abs(Cp_truth - Cp_pred))

    print(f"    R-squared:  {r2:.4f}")
    print(f"    MSE:        {mse:.6f}")
    print(f"    MAE:        {mae:.6f}")

    # -------------------------------------------------------------------------
    # STEP 7: Create Visualization
    # -------------------------------------------------------------------------
    print("\n[7] Creating visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1.5]})

    # -- TOP PLOT: Airfoil Shape --
    ax1 = axes[0]
    ax1.fill(x_100, y_100, alpha=0.3, color='steelblue', label='Airfoil')
    ax1.plot(x_100, y_100, 'b-', linewidth=2, label='NACA 2412')
    ax1.set_xlabel('x/c (Chord Position)', fontsize=12)
    ax1.set_ylabel('y/c (Thickness)', fontsize=12)
    ax1.set_title('NACA 2412 Airfoil Shape (Cessna Wing Profile)', fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.15, 0.15)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax1.legend(loc='upper right')

    # Add airfoil specs annotation
    ax1.text(0.02, 0.98,
             f"NACA 2412\nCamber: {m*100:.0f}%\nPosition: {p*100:.0f}%\nThickness: {t*100:.0f}%",
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # -- BOTTOM PLOT: Pressure Distribution --
    ax2 = axes[1]

    # Plot ground truth (solid) and prediction (dashed)
    ax2.plot(x_100, Cp_truth, 'b-', linewidth=2.5, label='Ground Truth (Physics)')
    ax2.plot(x_100, Cp_pred, 'r--', linewidth=2.5, label='AI Prediction (AeroCNN)')

    # Invert Y-axis (aerodynamics convention: negative Cp on top = suction side)
    ax2.invert_yaxis()

    ax2.set_xlabel('x/c (Chord Position)', fontsize=12)
    ax2.set_ylabel('Pressure Coefficient Cp', fontsize=12)
    ax2.set_title('Pressure Distribution: Physics vs AI', fontsize=14, fontweight='bold')
    ax2.set_xlim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax2.legend(loc='lower right', fontsize=11)

    # Add metrics annotation
    metrics_text = f"R-squared: {r2:.4f}\nMSE: {mse:.6f}\nMAE: {mae:.4f}"
    ax2.text(0.02, 0.02, metrics_text,
             transform=ax2.transAxes, fontsize=11, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if r2 > 0.9 else 'lightyellow', alpha=0.9))

    # Add annotation for Cp regions
    ax2.annotate('Suction Side\n(Upper Surface)', xy=(0.3, Cp_truth[25]),
                 xytext=(0.15, Cp_truth.min() - 0.3),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))
    ax2.annotate('Pressure Side\n(Lower Surface)', xy=(0.5, Cp_truth[75]),
                 xytext=(0.7, 0.8),
                 fontsize=9, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n[OK] Visualization saved: {output_path}")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Test Case:    NACA 2412 (Cessna wing)")
    print(f"  Angle of Attack: 2 degrees")
    print(f"  ")
    print(f"  R-squared:    {r2:.4f}")
    print(f"  MSE:          {mse:.6f}")
    print(f"  MAE:          {mae:.4f}")
    print(f"  ")
    if r2 > 0.95:
        print("  Status:       EXCELLENT - AI closely matches physics")
    elif r2 > 0.85:
        print("  Status:       GOOD - AI captures main features")
    elif r2 > 0.70:
        print("  Status:       FAIR - AI needs improvement")
    else:
        print("  Status:       POOR - Model may need retraining")
    print("=" * 70)


if __name__ == "__main__":
    main()
