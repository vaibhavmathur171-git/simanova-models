# -*- coding: utf-8 -*-
"""
P2 Training: Physics-residual ResNet-6 for Rainbow Solver.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from p2_model import RainbowResNet6
from p2_physics import (
    GLASS_LIBRARY,
    chromatic_penalty,
    generate_doe,
)


@dataclass
class TrainConfig:
    angle_min: float = -75.0
    angle_max: float = -25.0
    n_samples: int = 50_000
    batch_size: int = 256
    epochs: int = 150
    lr: float = 1e-3
    weight_blue: float = 0.2
    weight_green: float = 0.6
    weight_red: float = 0.2
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class WeightedPhotopicLoss(nn.Module):
    def __init__(
        self,
        lambda_blue: float = 450.0,
        lambda_green: float = 532.0,
        lambda_red: float = 635.0,
        weight_blue: float = 0.2,
        weight_green: float = 0.6,
        weight_red: float = 0.2,
    ):
        super().__init__()
        self.lambda_blue = lambda_blue
        self.lambda_green = lambda_green
        self.lambda_red = lambda_red
        self.weight_blue = weight_blue
        self.weight_green = weight_green
        self.weight_red = weight_red
        self.mse = nn.MSELoss()

    def forward(self, pred_pitch, target_angle, n_blue, n_green, n_red, order_m):
        angle_b = torch.rad2deg(
            torch.arcsin(
                torch.clamp((order_m * self.lambda_blue) / (n_blue * pred_pitch), -1.0, 1.0)
            )
        )
        angle_g = torch.rad2deg(
            torch.arcsin(
                torch.clamp((order_m * self.lambda_green) / (n_green * pred_pitch), -1.0, 1.0)
            )
        )
        angle_r = torch.rad2deg(
            torch.arcsin(
                torch.clamp((order_m * self.lambda_red) / (n_red * pred_pitch), -1.0, 1.0)
            )
        )

        loss_b = self.mse(angle_b, target_angle)
        loss_g = self.mse(angle_g, target_angle)
        loss_r = self.mse(angle_r, target_angle)

        return (
            self.weight_blue * loss_b +
            self.weight_green * loss_g +
            self.weight_red * loss_r
        )


def build_dataset(config: TrainConfig, glass_names: List[str]) -> pd.DataFrame:
    rows = generate_doe(
        n_samples=config.n_samples,
        angle_min=config.angle_min,
        angle_max=config.angle_max,
        glass_names=glass_names,
        seed=config.seed,
        weight_blue=config.weight_blue,
        weight_green=config.weight_green,
        weight_red=config.weight_red,
    )
    columns = [
        "target_angle_deg",
        "order_m",
        "glass_name",
        "B1",
        "B2",
        "B3",
        "C1",
        "C2",
        "C3",
        "n_blue",
        "n_green",
        "n_red",
        "pitch_base_nm",
        "pitch_opt_nm",
        "penalty_deg",
        "angle_blue_deg",
        "angle_green_deg",
        "angle_red_deg",
    ]
    return pd.DataFrame(rows, columns=columns)


def to_tensors(df: pd.DataFrame, device: torch.device) -> Tuple[torch.Tensor, ...]:
    features = df[
        ["target_angle_deg", "n_blue", "n_green", "n_red", "order_m"]
    ].values.astype(np.float32)
    target_angle = df["target_angle_deg"].values.astype(np.float32)
    n_blue = df["n_blue"].values.astype(np.float32)
    n_green = df["n_green"].values.astype(np.float32)
    n_red = df["n_red"].values.astype(np.float32)
    order_m = df["order_m"].values.astype(np.float32)
    pitch_opt = df["pitch_opt_nm"].values.astype(np.float32)

    return (
        torch.tensor(features, device=device),
        torch.tensor(target_angle, device=device),
        torch.tensor(n_blue, device=device),
        torch.tensor(n_green, device=device),
        torch.tensor(n_red, device=device),
        torch.tensor(order_m, device=device),
        torch.tensor(pitch_opt, device=device),
    )


def train_one(
    model: RainbowResNet6,
    loader: DataLoader,
    loss_fn: WeightedPhotopicLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        features, target_angle, n_blue, n_green, n_red, order_m, _ = batch
        pred_pitch = model(features)
        loss = loss_fn(pred_pitch, target_angle, n_blue, n_green, n_red, order_m)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(
    model: RainbowResNet6,
    loader: DataLoader,
    loss_fn: WeightedPhotopicLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    mae_nm = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            features, target_angle, n_blue, n_green, n_red, order_m, pitch_opt = batch
            pred_pitch = model(features)
            loss = loss_fn(pred_pitch, target_angle, n_blue, n_green, n_red, order_m)
            total_loss += loss.item()
            mae_nm += torch.mean(torch.abs(pred_pitch.squeeze(-1) - pitch_opt)).item()
            n_samples += 1
    return {
        "loss": total_loss / len(loader),
        "mae_nm": mae_nm / max(n_samples, 1),
    }


def train_pipeline(config: TrainConfig, output_dir: Path, model_dir: Path) -> Dict[str, float]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    glass_names = list(GLASS_LIBRARY.keys())
    df = build_dataset(config, glass_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "p2_rainbow_doe.csv"
    df.to_csv(dataset_path, index=False)

    device = torch.device(config.device)
    tensors = to_tensors(df, device=device)
    features, target_angle, n_blue, n_green, n_red, order_m, pitch_opt = tensors

    # Train/test split (deterministic)
    n_total = features.shape[0]
    split_idx = int(0.8 * n_total)
    train_data = TensorDataset(
        features[:split_idx],
        target_angle[:split_idx],
        n_blue[:split_idx],
        n_green[:split_idx],
        n_red[:split_idx],
        order_m[:split_idx],
        pitch_opt[:split_idx],
    )
    val_data = TensorDataset(
        features[split_idx:],
        target_angle[split_idx:],
        n_blue[split_idx:],
        n_green[split_idx:],
        n_red[split_idx:],
        order_m[split_idx:],
        pitch_opt[split_idx:],
    )

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    model = RainbowResNet6(input_dim=5, hidden_dim=128, num_blocks=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    loss_fn = WeightedPhotopicLoss(
        weight_blue=config.weight_blue,
        weight_green=config.weight_green,
        weight_red=config.weight_red,
    )

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_mae_nm": []}
    start = time.time()
    for epoch in range(config.epochs):
        train_loss = train_one(model, train_loader, loss_fn, optimizer, device)
        metrics = evaluate(model, val_loader, loss_fn, device)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["loss"])
        history["val_mae_nm"].append(metrics["mae_nm"])
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{config.epochs} | "
                f"Train {train_loss:.6f} | "
                f"Val {metrics['loss']:.6f} | "
                f"MAE {metrics['mae_nm']:.4f} nm"
            )

    train_time = time.time() - start
    history_path = output_dir / "p2_train_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    model_path = model_dir / "p2_resnet6_physics.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "input_dim": 5,
                "hidden_dim": 128,
                "num_blocks": 6,
            },
            "normalizer": None,
            "train_config": asdict(config),
        },
        model_path,
    )

    metrics = {
        "train_time_s": train_time,
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "final_val_mae_nm": float(history["val_mae_nm"][-1]),
    }
    metrics_path = output_dir / "p2_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train P2 ResNet-6 Physics Surrogate")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny smoke training")
    args = parser.parse_args()

    config = TrainConfig()
    if args.smoke:
        config.n_samples = 512
        config.epochs = 2
        config.batch_size = 64

    output_dir = Path("data")
    model_dir = Path("models")

    metrics = train_pipeline(config, output_dir, model_dir)
    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
