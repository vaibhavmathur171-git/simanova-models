# -*- coding: utf-8 -*-
"""
P2 Physics Module: Sellmeier dispersion + grating equation utilities.

Units:
- Wavelengths are in nanometers (nm) unless stated otherwise.
- Sellmeier coefficients expect wavelengths in micrometers (um).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class SellmeierCoefficients:
    """Three-term Sellmeier coefficients (B1,B2,B3,C1,C2,C3)."""

    B1: float
    B2: float
    B3: float
    C1: float
    C2: float
    C3: float


GLASS_LIBRARY: Dict[str, SellmeierCoefficients] = {
    # Schott N-BK7
    "N-BK7": SellmeierCoefficients(
        B1=1.03961212,
        B2=0.231792344,
        B3=1.01046945,
        C1=0.00600069867,
        C2=0.0200179144,
        C3=103.560653,
    ),
    # Schott N-SF11
    "N-SF11": SellmeierCoefficients(
        B1=1.73759695,
        B2=0.313747346,
        B3=1.89878101,
        C1=0.013188707,
        C2=0.0623068142,
        C3=155.23629,
    ),
}


def sellmeier_n(
    lambda_nm: np.ndarray | float,
    coeffs: SellmeierCoefficients,
) -> np.ndarray | float:
    """
    Compute refractive index from three-term Sellmeier equation.

    n^2(lambda) - 1 = sum_i [ B_i * lambda^2 / (lambda^2 - C_i) ]
    where lambda is in micrometers (um).
    """
    lam_um = np.asarray(lambda_nm, dtype=np.float64) / 1000.0
    lam2 = lam_um ** 2

    n_sq_minus_1 = (
        (coeffs.B1 * lam2) / (lam2 - coeffs.C1) +
        (coeffs.B2 * lam2) / (lam2 - coeffs.C2) +
        (coeffs.B3 * lam2) / (lam2 - coeffs.C3)
    )
    n = np.sqrt(1.0 + n_sq_minus_1)
    return n if isinstance(lambda_nm, np.ndarray) else float(n)


def glass_coeffs(glass_name: str) -> SellmeierCoefficients:
    """Fetch built-in glass coefficients by name."""
    return GLASS_LIBRARY.get(glass_name, GLASS_LIBRARY["N-BK7"])


def grating_pitch_from_angle(
    angle_deg: np.ndarray | float,
    lambda_nm: float,
    n_out: np.ndarray | float,
    order: int = -1,
) -> np.ndarray | float:
    """
    Grating equation for normal incidence:
        Lambda = m * lambda / (n_out * sin(theta_out))
    """
    theta_rad = np.radians(angle_deg)
    sin_theta = np.sin(theta_rad)
    sin_theta = np.where(np.abs(sin_theta) < 1e-10, 1e-10, sin_theta)
    pitch_nm = (order * lambda_nm) / (n_out * sin_theta)
    return np.abs(pitch_nm)


def angle_from_pitch(
    pitch_nm: np.ndarray | float,
    lambda_nm: float,
    n_out: np.ndarray | float,
    order: int = -1,
) -> np.ndarray | float:
    """Inverse grating equation: theta = arcsin(m*lambda/(n_out*Lambda))."""
    sin_theta = (order * lambda_nm) / (n_out * pitch_nm)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    theta_rad = np.arcsin(sin_theta)
    return np.degrees(theta_rad)


def chromatic_penalty(
    pitch_nm: np.ndarray | float,
    target_angle_deg: np.ndarray | float,
    n_blue: np.ndarray | float,
    n_green: np.ndarray | float,
    n_red: np.ndarray | float,
    lambda_blue: float = 450.0,
    lambda_green: float = 532.0,
    lambda_red: float = 635.0,
    weight_blue: float = 0.2,
    weight_green: float = 0.6,
    weight_red: float = 0.2,
    order: int = -1,
) -> Tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Photopic-weighted angular deviation across RGB."""
    angle_b = angle_from_pitch(pitch_nm, lambda_blue, n_blue, order=order)
    angle_g = angle_from_pitch(pitch_nm, lambda_green, n_green, order=order)
    angle_r = angle_from_pitch(pitch_nm, lambda_red, n_red, order=order)

    penalty = (
        weight_blue * np.abs(angle_b - target_angle_deg) +
        weight_green * np.abs(angle_g - target_angle_deg) +
        weight_red * np.abs(angle_r - target_angle_deg)
    )
    return penalty, angle_b, angle_g, angle_r


def optimize_pitch(
    target_angle_deg: float,
    n_blue: float,
    n_green: float,
    n_red: float,
    lambda_blue: float = 450.0,
    lambda_green: float = 532.0,
    lambda_red: float = 635.0,
    weight_blue: float = 0.2,
    weight_green: float = 0.6,
    weight_red: float = 0.2,
    order: int = -1,
    search_span: float = 0.15,
    n_steps: int = 161,
) -> Tuple[float, float, float, float, float]:
    """
    Deterministic 1D search around base pitch to minimize chromatic penalty.
    """
    base_pitch = grating_pitch_from_angle(target_angle_deg, lambda_green, n_green, order=order)
    deltas = np.linspace(-search_span, search_span, n_steps)
    candidates = base_pitch * (1.0 + deltas)

    penalty, ang_b, ang_g, ang_r = chromatic_penalty(
        candidates,
        target_angle_deg,
        n_blue,
        n_green,
        n_red,
        lambda_blue=lambda_blue,
        lambda_green=lambda_green,
        lambda_red=lambda_red,
        weight_blue=weight_blue,
        weight_green=weight_green,
        weight_red=weight_red,
        order=order,
    )
    best_idx = int(np.argmin(penalty))
    return (
        float(candidates[best_idx]),
        float(penalty[best_idx]),
        float(ang_b[best_idx]),
        float(ang_g[best_idx]),
        float(ang_r[best_idx]),
    )


def generate_doe(
    n_samples: int,
    angle_min: float,
    angle_max: float,
    glass_names: Iterable[str],
    seed: int = 42,
    order: int = -1,
    lambda_blue: float = 450.0,
    lambda_green: float = 532.0,
    lambda_red: float = 635.0,
    weight_blue: float = 0.2,
    weight_green: float = 0.6,
    weight_red: float = 0.2,
) -> np.ndarray:
    """
    Generate deterministic DOE samples for the Rainbow Penalty objective.
    """
    rng = np.random.default_rng(seed)
    angles = rng.uniform(angle_min, angle_max, n_samples)
    glass_list = list(glass_names)
    glass_idx = rng.integers(0, len(glass_list), size=n_samples)

    rows = []
    for angle, g_idx in zip(angles, glass_idx):
        glass_name = glass_list[int(g_idx)]
        coeffs = glass_coeffs(glass_name)

        n_b = sellmeier_n(lambda_blue, coeffs)
        n_g = sellmeier_n(lambda_green, coeffs)
        n_r = sellmeier_n(lambda_red, coeffs)

        base_pitch = grating_pitch_from_angle(angle, lambda_green, n_g, order=order)
        opt_pitch, penalty, ang_b, ang_g, ang_r = optimize_pitch(
            angle,
            n_b,
            n_g,
            n_r,
            lambda_blue=lambda_blue,
            lambda_green=lambda_green,
            lambda_red=lambda_red,
            weight_blue=weight_blue,
            weight_green=weight_green,
            weight_red=weight_red,
            order=order,
        )

        rows.append(
            [
                angle,
                order,
                glass_name,
                coeffs.B1,
                coeffs.B2,
                coeffs.B3,
                coeffs.C1,
                coeffs.C2,
                coeffs.C3,
                n_b,
                n_g,
                n_r,
                base_pitch,
                opt_pitch,
                penalty,
                ang_b,
                ang_g,
                ang_r,
            ]
        )

    return np.asarray(rows, dtype=np.float64)
