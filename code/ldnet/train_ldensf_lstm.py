#!/usr/bin/env python3
"""
Train an LSTM encoder that maps hydro observations to latent states.

The input dataset is expected to come from `create_ldensf_dataset.py` and
contain `data_train`, `data_valid`, and `data_test` splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from src.encoder import TimeSeriesLSTM


def _resolve_path(base_path: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_path / path


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return torch.as_tensor(value)


def _flatten_observations(obs: torch.Tensor) -> torch.Tensor:
    if obs.ndim != 4:
        raise ValueError(f"Expected observations with shape (N, T, K, C); got {tuple(obs.shape)}")
    return obs.reshape(obs.shape[0], obs.shape[1], -1)


def _canonicalize_observation_layout(obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """
    Accept either (N, T, K, C) or the accidentally transposed (N, K, T, C)
    layout and return time-first observations.
    """

    if obs.ndim != 4:
        raise ValueError(f"Expected 4D observations; got {tuple(obs.shape)}")
    if latent.ndim != 3:
        raise ValueError(f"Expected latent states with shape (N, T, D); got {tuple(latent.shape)}")

    time_len = int(latent.shape[1])
    if obs.shape[1] == time_len:
        return obs
    if obs.shape[2] == time_len:
        return obs.transpose(1, 2).contiguous()

    raise ValueError(
        f"Could not infer observation time axis from obs shape {tuple(obs.shape)} and latent shape {tuple(latent.shape)}"
    )


def _compute_stats(obs_train: torch.Tensor, latent_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_flat = obs_train.reshape(-1, obs_train.shape[-1])
    latent_flat = latent_train.reshape(-1, latent_train.shape[-1])
    obs_mean = obs_flat.mean(dim=0)
    obs_std = obs_flat.std(dim=0, unbiased=False).clamp_min(1e-6)
    latent_mean = latent_flat.mean(dim=0)
    latent_std = latent_flat.std(dim=0, unbiased=False).clamp_min(1e-6)
    return obs_mean, obs_std, latent_mean, latent_std


def _normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    view_shape = (1,) * (x.ndim - 1) + (mean.shape[0],)
    return (x - mean.reshape(view_shape)) / std.reshape(view_shape)


def _make_loader(obs: torch.Tensor, latent: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(obs.float(), latent.float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def create_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset-path", type=Path, default=Path("data/postprocessed/illinois/observation_ldensf_dataset.pth"))
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/ldensf_lstm"))
    parser.add_argument("--save-name", type=str, default="lstm_ldensf.ckpt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--obs-noise-std", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument("--normalize", dest="normalize", action="store_true")
    norm_group.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=False)
    parser.add_argument("--eval-interval", type=int, default=10)
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "cuda" in device_str and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _train_epoch(
    model: TimeSeriesLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    obs_mean: torch.Tensor | None,
    obs_std: torch.Tensor | None,
    latent_mean: torch.Tensor | None,
    latent_std: torch.Tensor | None,
    obs_noise_std: float,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0

    for obs, latent in tqdm(loader, desc="train", leave=False):
        obs = obs.to(device)
        latent = latent.to(device)
        if obs_noise_std > 0:
            obs = obs + obs_noise_std * torch.randn_like(obs)

        if obs_mean is not None and obs_std is not None:
            obs = _normalize(obs, obs_mean.to(device), obs_std.to(device))
        if latent_mean is not None and latent_std is not None:
            target = _normalize(latent, latent_mean.to(device), latent_std.to(device))
        else:
            target = latent

        optimizer.zero_grad(set_to_none=True)
        pred = model(obs)
        loss = criterion(pred, target)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * obs.shape[0]

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def _eval_epoch(
    model: TimeSeriesLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    obs_mean: torch.Tensor | None,
    obs_std: torch.Tensor | None,
    latent_mean: torch.Tensor | None,
    latent_std: torch.Tensor | None,
) -> float:
    model.eval()
    total_loss = 0.0

    for obs, latent in tqdm(loader, desc="valid", leave=False):
        obs = obs.to(device)
        latent = latent.to(device)
        if obs_mean is not None and obs_std is not None:
            obs = _normalize(obs, obs_mean.to(device), obs_std.to(device))
        if latent_mean is not None and latent_std is not None:
            target = _normalize(latent, latent_mean.to(device), latent_std.to(device))
        else:
            target = latent

        pred = model(obs)
        loss = criterion(pred, target)
        total_loss += float(loss.item()) * obs.shape[0]

    return total_loss / max(1, len(loader.dataset))


def main(opt: argparse.Namespace) -> None:
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    base_path = opt.base_path
    dataset_path = _resolve_path(base_path, opt.dataset_path)
    save_dir = _resolve_path(base_path, opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / opt.save_name

    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    train_split = dataset["data_train"]
    valid_split = dataset["data_valid"]

    latent_train = _as_tensor(train_split["latent_states"]).float()
    latent_valid = _as_tensor(valid_split["latent_states"]).float()
    obs_train_raw = _as_tensor(train_split["observation"]).float()
    obs_valid_raw = _as_tensor(valid_split["observation"]).float()

    obs_train = _flatten_observations(_canonicalize_observation_layout(obs_train_raw, latent_train)).float()
    obs_valid = _flatten_observations(_canonicalize_observation_layout(obs_valid_raw, latent_valid)).float()

    obs_mean = obs_std = latent_mean = latent_std = None
    if opt.normalize:
        obs_mean, obs_std, latent_mean, latent_std = _compute_stats(obs_train, latent_train)

    input_size = int(obs_train.shape[-1])
    output_size = int(latent_train.shape[-1])

    train_loader = _make_loader(obs_train, latent_train, opt.batch_size, shuffle=True)
    valid_loader = _make_loader(obs_valid, latent_valid, opt.batch_size, shuffle=False)

    device = _resolve_device(opt.device)
    model = TimeSeriesLSTM(
        input_size=input_size,
        hidden_size=opt.hidden_size,
        output_size=output_size,
        num_layers=opt.num_layers,
        dropout=opt.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, opt.epochs))
    criterion = nn.MSELoss()

    history = {
        "train_loss": [],
        "valid_loss": [],
    }
    best_valid = float("inf")
    best_epoch = -1

    print(f"Dataset: {dataset_path}")
    print(f"Input dim: {input_size} | output dim: {output_size}")
    print(f"Normalize: {opt.normalize}")
    print(f"Saving checkpoint to: {checkpoint_path}")

    for epoch in range(opt.epochs):
        train_loss = _train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            obs_mean=obs_mean,
            obs_std=obs_std,
            latent_mean=latent_mean,
            latent_std=latent_std,
            obs_noise_std=opt.obs_noise_std,
            grad_clip=opt.grad_clip,
        )
        valid_loss = _eval_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            obs_mean=obs_mean,
            obs_std=obs_std,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_size": input_size,
                    "hidden_size": opt.hidden_size,
                    "output_size": output_size,
                    "num_layers": opt.num_layers,
                    "dropout": opt.dropout,
                },
                "stats": {
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "latent_mean": latent_mean,
                    "latent_std": latent_std,
                },
                "normalize": bool(opt.normalize),
                "obs_noise_std": float(opt.obs_noise_std),
                "history": history,
                "best_epoch": int(best_epoch),
                "best_valid_loss": float(best_valid),
                "dataset_meta": dataset.get("meta", {}),
            }
            torch.save(checkpoint, checkpoint_path)

        if (epoch + 1) % opt.eval_interval == 0 or epoch == 0 or epoch == opt.epochs - 1:
            print(
                f"Epoch {epoch+1:04d}/{opt.epochs} | "
                f"train_loss={train_loss:.6f} | valid_loss={valid_loss:.6f} | "
                f"best_valid={best_valid:.6f} @ epoch {best_epoch+1}"
            )

    final_report = {
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid),
        "normalize": bool(opt.normalize),
        "obs_noise_std": float(opt.obs_noise_std),
    }
    report_path = save_dir / "train_report.json"
    report_path.write_text(json.dumps(final_report, indent=2))
    print(f"Saved training report to {report_path}")


if __name__ == "__main__":
    main(create_options())
