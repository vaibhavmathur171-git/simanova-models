# -*- coding: utf-8 -*-
"""
P2 Model: Physics-residual ResNet-6 for dispersion correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Normalizer:
    mean: torch.Tensor
    scale: torch.Tensor

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.scale


class ResidualBlock(nn.Module):
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


class RainbowResNet6(nn.Module):
    """
    Physics-residual model.

    Inputs (default order):
        [angle_deg, n_blue, n_green, n_red, order_m]
    Output:
        pitch_nm (base physics + learned residual)
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, num_blocks: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, 1)

        self._normalizer: Optional[Normalizer] = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_normalizer(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        self._normalizer = Normalizer(mean=mean, scale=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [angle_deg, n_blue, n_green, n_red, order_m]
        if self._normalizer is not None:
            x_in = self._normalizer.apply(x)
        else:
            x_in = x

        angle_deg = x[:, 0]
        n_green = x[:, 2]
        order_m = x[:, 4]

        theta = torch.deg2rad(angle_deg)
        sin_theta = torch.sin(theta)
        sin_theta = torch.where(
            torch.abs(sin_theta) < 1e-10,
            torch.full_like(sin_theta, 1e-10),
            sin_theta,
        )

        lambda_green = 532.0
        base_pitch = torch.abs((order_m * lambda_green) / (n_green * sin_theta))
        base_pitch = base_pitch.unsqueeze(-1)

        residual = self.output_layer(self.residual_blocks(self.input_layer(x_in)))
        return base_pitch + residual

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
