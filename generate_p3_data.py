# -*- coding: utf-8 -*-
"""
P3: Virtual Wind Tunnel - Airfoil Dataset Generator
=====================================================
Generates synthetic airfoil shapes and pressure distributions
using a Vortex Panel Method solver.

Physics: Potential Flow + Vortex Panel Method (Kutta Condition)
Geometry: NACA 4-Digit Airfoil Generator
Output: (2000, 2, 100) shapes + (2000, 100) Cp curves
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 1. NACA 4-DIGIT GEOMETRY GENERATOR
# =============================================================================

def naca_4digit(m: float, p: float, t: float, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate NACA 4-digit airfoil coordinates.

    The NACA 4-digit series is defined by:
      - First digit (m): Maximum camber as fraction of chord (0-9%)
      - Second digit (p): Position of max camber in tenths of chord (0-90%)
      - Last two digits (t): Maximum thickness as fraction of chord (01-40%)

    Parameters:
    -----------
    m : float
        Maximum camber (0-0.09)
    p : float
        Position of maximum camber (0.1-0.9)
    t : float
        Maximum thickness (0.05-0.30)
    n_points : int
        Number of points per surface

    Returns:
    --------
    x, y : np.ndarray
        Airfoil coordinates going clockwise from TE upper -> LE -> TE lower
    """
    # Cosine spacing for better resolution at leading/trailing edges
    beta = np.linspace(0, np.pi, n_points)
    xc = 0.5 * (1 - np.cos(beta))

    # Thickness distribution (standard NACA formula)
    # Coefficients for finite TE thickness (original NACA)
    yt = 5 * t * (
        0.2969 * np.sqrt(xc + 1e-10)
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1015 * xc**4
    )

    # Camber line and gradient
    if p < 0.01 or m < 0.001:
        # Symmetric airfoil
        yc = np.zeros_like(xc)
        dyc_dx = np.zeros_like(xc)
    else:
        # Cambered airfoil
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

    # Upper surface
    xu = xc - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    # Lower surface
    xl = xc + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Combine surfaces: upper (TE->LE) + lower (LE->TE), skip duplicate LE
    x_full = np.concatenate([xu[::-1], xl[1:]])
    y_full = np.concatenate([yu[::-1], yl[1:]])

    return x_full, y_full


def naca_code_to_params(code: str) -> Tuple[float, float, float]:
    """Convert NACA 4-digit code (e.g., '2412') to (m, p, t) parameters."""
    m = int(code[0]) / 100.0   # max camber as fraction
    p = int(code[1]) / 10.0    # position of max camber
    t = int(code[2:4]) / 100.0 # thickness as fraction
    return m, p, t


# =============================================================================
# 2. VORTEX PANEL METHOD SOLVER
# =============================================================================

class VortexPanelMethod:
    """
    Vortex Panel Method for 2D inviscid, incompressible flow around airfoils.

    Theory:
    -------
    Each panel has a constant-strength vortex distribution. The vortex strength
    (circulation) is determined by enforcing:
      1. Flow tangency at each panel control point (no penetration)
      2. Kutta condition at trailing edge (smooth flow departure)

    The resulting linear system [A]{gamma} = {b} is solved for vortex strengths,
    then surface velocities and Cp are computed.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, V_inf: float = 1.0, alpha_deg: float = 0.0):
        """
        Initialize panel method solver.

        Parameters:
        -----------
        x, y : np.ndarray
            Airfoil boundary coordinates (closed loop, clockwise from TE)
        V_inf : float
            Freestream velocity magnitude
        alpha_deg : float
            Angle of attack in degrees
        """
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.V_inf = V_inf
        self.alpha = np.radians(alpha_deg)

        # Freestream components
        self.u_inf = V_inf * np.cos(self.alpha)
        self.v_inf = V_inf * np.sin(self.alpha)

        # Number of panels (one less than number of points)
        self.n = len(x) - 1

        # Precompute panel geometry
        self._setup_panels()

    def _setup_panels(self):
        """Compute panel geometry: endpoints, midpoints, lengths, angles."""
        n = self.n

        # Panel endpoints
        self.x1 = self.x[:-1]  # Start points
        self.y1 = self.y[:-1]
        self.x2 = self.x[1:]   # End points
        self.y2 = self.y[1:]

        # Control points (panel midpoints)
        self.xc = 0.5 * (self.x1 + self.x2)
        self.yc = 0.5 * (self.y1 + self.y2)

        # Panel lengths
        self.S = np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
        self.S = np.maximum(self.S, 1e-10)  # Avoid division by zero

        # Panel orientation angles (angle of panel tangent vector)
        self.theta = np.arctan2(self.y2 - self.y1, self.x2 - self.x1)

        # Unit normal vectors (pointing outward for clockwise ordering)
        self.nx = np.sin(self.theta)   # = -(y2-y1)/S, but for outward normal
        self.ny = -np.cos(self.theta)  # = (x2-x1)/S

        # Unit tangent vectors
        self.tx = np.cos(self.theta)
        self.ty = np.sin(self.theta)

    def _compute_influence(self, xp: float, yp: float,
                           x1: float, y1: float,
                           x2: float, y2: float) -> Tuple[float, float]:
        """
        Compute velocity induced at point (xp, yp) by a unit-strength
        vortex panel from (x1, y1) to (x2, y2).

        Uses analytical integration of the vortex sheet.

        Returns:
        --------
        u, v : float
            Induced velocity components
        """
        # Panel properties
        dx = x2 - x1
        dy = y2 - y1
        S = np.sqrt(dx**2 + dy**2)

        if S < 1e-12:
            return 0.0, 0.0

        # Panel angle
        theta_p = np.arctan2(dy, dx)
        cos_t = np.cos(theta_p)
        sin_t = np.sin(theta_p)

        # Transform point to panel-local coordinates
        # Origin at panel start, x-axis along panel
        dxp = xp - x1
        dyp = yp - y1

        x_loc = dxp * cos_t + dyp * sin_t
        y_loc = -dxp * sin_t + dyp * cos_t

        # Avoid singularity when point is on the panel
        if abs(y_loc) < 1e-10:
            y_loc = 1e-10 if y_loc >= 0 else -1e-10

        # Distances to panel endpoints
        r1_sq = x_loc**2 + y_loc**2
        r2_sq = (x_loc - S)**2 + y_loc**2

        # Angles from panel endpoints
        theta1 = np.arctan2(y_loc, x_loc)
        theta2 = np.arctan2(y_loc, x_loc - S)

        # Induced velocity in panel-local coordinates
        # From integration of vortex sheet
        u_loc = (theta2 - theta1) / (2 * np.pi)
        v_loc = -np.log(np.sqrt(r2_sq / r1_sq)) / (2 * np.pi)

        # Transform back to global coordinates
        u = u_loc * cos_t - v_loc * sin_t
        v = u_loc * sin_t + v_loc * cos_t

        return u, v

    def _build_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the linear system for vortex strengths.

        Flow tangency condition at each control point:
            sum_j (A_ij * gamma_j) = -V_inf . n_i

        Plus Kutta condition: gamma_1 + gamma_n = 0
        """
        n = self.n

        # Influence coefficient matrices
        # A_ij = normal velocity at control point i induced by unit vortex on panel j
        A = np.zeros((n + 1, n + 1))

        # Also store tangential influence for velocity calculation
        self.At = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Self-induced velocity (half the vortex strength)
                    A[i, j] = 0.5
                    self.At[i, j] = 0.0
                else:
                    # Influence of panel j on control point i
                    u, v = self._compute_influence(
                        self.xc[i], self.yc[i],
                        self.x1[j], self.y1[j],
                        self.x2[j], self.y2[j]
                    )

                    # Normal component (for tangency condition)
                    A[i, j] = u * self.nx[i] + v * self.ny[i]

                    # Tangential component (for velocity calculation)
                    self.At[i, j] = u * self.tx[i] + v * self.ty[i]

        # Kutta condition: gamma at TE upper + gamma at TE lower = 0
        A[n, 0] = 1.0      # First panel (TE upper)
        A[n, n-1] = 1.0    # Last panel (TE lower)

        # RHS: negative of freestream normal component
        b = np.zeros(n + 1)
        for i in range(n):
            b[i] = -(self.u_inf * self.nx[i] + self.v_inf * self.ny[i])
        b[n] = 0.0  # Kutta condition RHS

        return A, b

    def solve(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        Solve for vortex distribution and compute pressure coefficient.

        Returns:
        --------
        Cp : np.ndarray or None
            Pressure coefficient at each control point
        gamma : np.ndarray or None
            Vortex strength per unit length on each panel
        success : bool
            Whether the solution is valid
        """
        try:
            # Build and solve the linear system
            A, b = self._build_system()

            # Check matrix conditioning
            cond = np.linalg.cond(A[:-1, :-1])
            if cond > 1e12:
                return None, None, False

            # Solve using least squares for robustness
            solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            gamma = solution[:-1]  # Exclude Kutta multiplier

            # Compute tangential velocity at each control point
            Vt = np.zeros(self.n)
            for i in range(self.n):
                # Freestream tangential component
                Vt[i] = self.u_inf * self.tx[i] + self.v_inf * self.ty[i]

                # Induced tangential velocity from all panels
                for j in range(self.n):
                    Vt[i] += self.At[i, j] * gamma[j]

                # Self-induced tangential (vortex sheet velocity jump)
                Vt[i] += 0.5 * gamma[i]

            # Pressure coefficient from Bernoulli equation
            # Cp = 1 - (V/V_inf)^2
            Cp = 1.0 - (Vt / self.V_inf)**2

            # Validate results
            if np.any(np.isnan(Cp)) or np.any(np.isinf(Cp)):
                return None, None, False

            # Physical bounds check (Cp shouldn't be too extreme)
            if np.min(Cp) < -8.0 or np.max(Cp) > 1.5:
                return None, None, False

            return Cp, gamma, True

        except (np.linalg.LinAlgError, ValueError):
            return None, None, False


# =============================================================================
# 3. DATASET GENERATION
# =============================================================================

def generate_dataset(n_samples: int = 2000, n_points: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate airfoil dataset with shapes and Cp distributions.

    Parameters:
    -----------
    n_samples : int
        Target number of valid samples
    n_points : int
        Number of points for final airfoil representation
    seed : int
        Random seed

    Returns:
    --------
    X_shapes : np.ndarray, shape (N, 2, n_points)
        Airfoil coordinates [x, y]
    y_pressures : np.ndarray, shape (N, n_points)
        Pressure coefficients
    metadata : list of dict
        NACA parameters for each sample
    """
    np.random.seed(seed)

    X_shapes = []
    y_pressures = []
    metadata = []

    attempts = 0
    max_attempts = n_samples * 10

    print("=" * 70)
    print("P3: VIRTUAL WIND TUNNEL - Airfoil Dataset Generator")
    print("=" * 70)
    print(f"\nPhysics Engine: Vortex Panel Method (Kutta Condition)")
    print(f"Geometry: NACA 4-Digit Series")
    print(f"Target Samples: {n_samples:,}")
    print(f"Points per Airfoil: {n_points}")
    print("\n" + "-" * 70)

    while len(X_shapes) < n_samples and attempts < max_attempts:
        attempts += 1

        # Random NACA parameters within realistic ranges
        if np.random.random() < 0.15:
            # 15% symmetric airfoils (m=0)
            m = 0.0
            p = 0.0
        else:
            m = np.random.uniform(0.01, 0.08)  # 1-8% camber
            p = np.random.uniform(0.2, 0.6)    # 20-60% chord position

        t = np.random.uniform(0.06, 0.24)      # 6-24% thickness
        alpha = np.random.uniform(-4, 8)        # -4 to +8 degrees AoA

        try:
            # Generate geometry with extra points for panel method accuracy
            n_panel_pts = n_points // 2 + 1
            x_raw, y_raw = naca_4digit(m, p, t, n_panel_pts)

            # Run panel method solver
            solver = VortexPanelMethod(x_raw, y_raw, V_inf=1.0, alpha_deg=alpha)
            Cp, gamma, success = solver.solve()

            if not success:
                continue

            # Resample to exactly n_points using arc-length parameterization
            # Compute cumulative arc length
            dx = np.diff(x_raw)
            dy = np.diff(y_raw)
            ds = np.sqrt(dx**2 + dy**2)
            s = np.concatenate([[0], np.cumsum(ds)])
            s_norm = s / s[-1]

            # New uniform arc-length parameter
            s_new = np.linspace(0, 1, n_points)

            # Interpolate coordinates and Cp
            x_interp = np.interp(s_new, s_norm, x_raw)
            y_interp = np.interp(s_new, s_norm, y_raw)

            # Cp is defined at panel midpoints, need to interpolate
            s_mid = 0.5 * (s_norm[:-1] + s_norm[1:])
            Cp_interp = np.interp(s_new, s_mid, Cp)

            # Final validation
            if np.any(np.isnan(Cp_interp)) or np.any(np.isinf(Cp_interp)):
                continue

            # Store results
            X_shapes.append(np.stack([x_interp, y_interp], axis=0))
            y_pressures.append(Cp_interp)
            metadata.append({
                'm': m, 'p': p, 't': t, 'alpha': alpha,
                'naca': f"{int(m*100)}{int(p*10)}{int(t*100):02d}"
            })

            # Progress update
            if len(X_shapes) % 250 == 0:
                reject_rate = 100 * (1 - len(X_shapes) / attempts)
                print(f"  [OK] {len(X_shapes):,} samples | Attempts: {attempts:,} | Reject: {reject_rate:.1f}%")

        except Exception as e:
            continue

    # Convert to arrays
    X_shapes = np.array(X_shapes, dtype=np.float32)
    y_pressures = np.array(y_pressures, dtype=np.float32)

    print("-" * 70)
    print(f"\n[OK] Generation Complete!")
    print(f"     Valid Samples: {len(X_shapes):,}")
    print(f"     Total Attempts: {attempts:,}")
    print(f"     Success Rate: {100 * len(X_shapes) / attempts:.1f}%")

    return X_shapes, y_pressures, metadata


# =============================================================================
# 4. VISUALIZATION
# =============================================================================

def plot_samples(X_shapes: np.ndarray, y_pressures: np.ndarray,
                 metadata: List[Dict], output_path: Path, n_samples: int = 3):
    """
    Create publication-quality visualization of sample airfoils and Cp curves.
    """
    fig = plt.figure(figsize=(16, 5 * n_samples))

    # Select random samples
    indices = np.random.choice(len(X_shapes), n_samples, replace=False)

    for row, idx in enumerate(indices):
        x = X_shapes[idx, 0, :]
        y = X_shapes[idx, 1, :]
        Cp = y_pressures[idx, :]
        meta = metadata[idx]

        # Find leading edge index (minimum x)
        le_idx = np.argmin(x)

        # =====================================================================
        # Left: Airfoil Shape with pressure coloring
        # =====================================================================
        ax1 = fig.add_subplot(n_samples, 2, 2*row + 1)

        # Create color gradient based on Cp
        from matplotlib.collections import LineCollection

        # Plot filled airfoil
        ax1.fill(x, y, alpha=0.2, color='steelblue', label='Airfoil')

        # Plot outline with Cp coloring
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize Cp for coloring
        norm = plt.Normalize(Cp.min(), Cp.max())
        lc = LineCollection(segments, cmap='coolwarm_r', norm=norm, linewidth=3)
        lc.set_array(Cp[:-1])
        ax1.add_collection(lc)

        # Formatting
        ax1.set_xlim(-0.05, 1.1)
        ax1.set_ylim(-0.25, 0.25)
        ax1.set_aspect('equal')
        ax1.set_xlabel('x/c', fontsize=12, fontweight='bold')
        ax1.set_ylabel('y/c', fontsize=12, fontweight='bold')
        ax1.set_title(f"NACA {meta['naca']} | AoA = {meta['alpha']:.1f} deg",
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)

        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax1, orientation='vertical', pad=0.02)
        cbar.set_label('Cp', fontsize=10)

        # =====================================================================
        # Right: Cp Distribution (Standard Aerodynamics Plot)
        # =====================================================================
        ax2 = fig.add_subplot(n_samples, 2, 2*row + 2)

        # Upper surface (from TE to LE, first half of points)
        x_upper = x[:le_idx+1]
        Cp_upper = Cp[:le_idx+1]

        # Lower surface (from LE to TE, second half of points)
        x_lower = x[le_idx:]
        Cp_lower = Cp[le_idx:]

        # Plot with correct x-ordering (from LE to TE)
        ax2.plot(x_upper[::-1], Cp_upper[::-1], 'b-', linewidth=2.5,
                label='Upper Surface', marker='o', markersize=2, markevery=5)
        ax2.plot(x_lower, Cp_lower, 'r-', linewidth=2.5,
                label='Lower Surface', marker='s', markersize=2, markevery=5)

        # Standard Cp plot convention: negative Cp at top (suction)
        ax2.invert_yaxis()

        # Reference lines
        ax2.axhline(y=0, color='k', linewidth=1, linestyle='-', alpha=0.5)
        ax2.axhline(y=1, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='Stagnation')

        # Mark suction peak
        min_Cp_idx = np.argmin(Cp)
        ax2.scatter([x[min_Cp_idx]], [Cp[min_Cp_idx]], s=100, c='green',
                   marker='*', zorder=5, label=f'Suction Peak: Cp={Cp[min_Cp_idx]:.2f}')

        # Formatting
        ax2.set_xlim(-0.02, 1.02)
        ax2.set_xlabel('x/c', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cp', fontsize=12, fontweight='bold')
        ax2.set_title('Pressure Distribution', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # Add physics annotation
        ax2.annotate('Suction\n(Low Pressure)', xy=(0.2, Cp.min()), fontsize=9,
                    color='blue', ha='center', alpha=0.7)
        ax2.annotate('Compression\n(High Pressure)', xy=(0.8, 0.5), fontsize=9,
                    color='red', ha='center', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n[OK] Visualization saved: {output_path}")


# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Generate dataset
    X_shapes, y_pressures, metadata = generate_dataset(
        n_samples=2000,
        n_points=100,
        seed=42
    )

    # Save to compressed NPZ
    npz_path = output_dir / "p3_aero_dataset.npz"
    np.savez_compressed(
        npz_path,
        X_shapes=X_shapes,      # (2000, 2, 100)
        y_pressures=y_pressures  # (2000, 100)
    )

    print(f"\n[OK] Dataset saved: {npz_path}")
    print(f"     X_shapes shape: {X_shapes.shape}")
    print(f"     y_pressures shape: {y_pressures.shape}")
    print(f"     File size: {npz_path.stat().st_size / 1e6:.2f} MB")

    # Dataset statistics
    print(f"\n  Pressure Coefficient Statistics:")
    print(f"    Cp min:  {y_pressures.min():.3f} (max suction)")
    print(f"    Cp max:  {y_pressures.max():.3f}")
    print(f"    Cp mean: {y_pressures.mean():.3f}")
    print(f"    Cp std:  {y_pressures.std():.3f}")

    # NACA parameter statistics
    cambers = [m['m'] * 100 for m in metadata]
    positions = [m['p'] * 10 for m in metadata if m['p'] > 0]
    thicknesses = [m['t'] * 100 for m in metadata]
    alphas = [m['alpha'] for m in metadata]

    print(f"\n  NACA Parameter Ranges:")
    print(f"    Camber: {min(cambers):.1f}% - {max(cambers):.1f}%")
    print(f"    Position: {min(positions):.0f} - {max(positions):.0f} (tenths)")
    print(f"    Thickness: {min(thicknesses):.1f}% - {max(thicknesses):.1f}%")
    print(f"    AoA: {min(alphas):.1f} deg - {max(alphas):.1f} deg")

    # Visualization
    viz_path = output_dir / "p3_viz.png"
    plot_samples(X_shapes, y_pressures, metadata, viz_path, n_samples=3)

    print("\n" + "=" * 70)
    print("[OK] P3 VIRTUAL WIND TUNNEL - DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nNext Steps:")
    print(f"  1. Train CNN/U-Net to predict Cp from airfoil shape")
    print(f"  2. Input: (2, 100) coordinates -> Output: (100,) Cp values")
    print(f"  3. Loss: MSE on Cp distribution")
