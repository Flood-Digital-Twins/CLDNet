#!/usr/bin/env python3
"""Train a ConvLSTM that propagates rainfall sequences across time."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from src.rain_lstm import RainPropagatorLSTM, normalize_sequence


def _resolve_path(base_path: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_path / path


def _available_traj_ids(data_root: Path) -> list[int]:
    ids: list[int] = []
    for path in data_root.glob("rain_source_traj*.npy"):
        match = re.search(r"traj(\d+)\.npy$", path.name)
        if match is not None:
            ids.append(int(match.group(1)))
    ids = sorted(set(ids))
    if not ids:
        raise FileNotFoundError(f"No rain_source_traj*.npy files found in {data_root}")
    return ids


def _split_ids(
    traj_ids: list[int],
    train_max_traj_id: int,
    valid_max_traj_id: int,
    test_max_traj_id: int,
) -> tuple[list[int], list[int], list[int]]:
    if not (train_max_traj_id < valid_max_traj_id < test_max_traj_id):
        raise ValueError("Expected train_max_traj_id < valid_max_traj_id < test_max_traj_id")

    train_ids = [traj_id for traj_id in traj_ids if traj_id <= train_max_traj_id]
    valid_ids = [traj_id for traj_id in traj_ids if train_max_traj_id < traj_id <= valid_max_traj_id]
    test_ids = [traj_id for traj_id in traj_ids if valid_max_traj_id < traj_id <= test_max_traj_id]

    if not train_ids or not valid_ids or not test_ids:
        raise ValueError(
            "Trajectory split produced an empty partition. "
            f"train={len(train_ids)}, valid={len(valid_ids)}, test={len(test_ids)}"
        )
    return train_ids, valid_ids, test_ids


def _load_rain_sequence(data_root: Path, traj_id: int) -> np.ndarray:
    rain = np.load(data_root / f"rain_source_traj{traj_id}.npy", mmap_mode="r")
    rain_arr = np.asarray(rain, dtype=np.float32)
    if rain_arr.ndim != 3 or rain_arr.shape[0] != 1:
        raise ValueError(f"Expected rain with shape (1, T, D); got {rain_arr.shape}")
    return rain_arr[0]


def _load_split_sequences(data_root: Path, traj_ids: list[int]) -> torch.Tensor:
    sequences = [_load_rain_sequence(data_root, traj_id) for traj_id in traj_ids]
    lengths = {seq.shape[0] for seq in sequences}
    if len(lengths) != 1:
        raise ValueError(f"Rain sequence lengths are not uniform: {sorted(lengths)}")
    return torch.from_numpy(np.stack(sequences, axis=0))


def _compute_stats(train_rain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = train_rain.reshape(-1, train_rain.shape[-1])
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def _make_loader(inputs: torch.Tensor, targets: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(inputs.float(), targets.float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def _train_epoch(
    model: RainPropagatorLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    rain_mean: torch.Tensor | None,
    rain_std: torch.Tensor | None,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0

    for rain_in, rain_target in tqdm(loader, desc="train", leave=False):
        rain_in = rain_in.to(device)
        rain_target = rain_target.to(device)
        if rain_mean is not None and rain_std is not None:
            rain_in = normalize_sequence(rain_in, rain_mean.to(device), rain_std.to(device))
            rain_target = normalize_sequence(rain_target, rain_mean.to(device), rain_std.to(device))

        optimizer.zero_grad(set_to_none=True)
        pred, _ = model(rain_in)
        loss = criterion(pred, rain_target)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item()) * rain_in.shape[0]

    return total_loss / max(1, len(loader.dataset))


@torch.no_grad()
def _eval_epoch(
    model: RainPropagatorLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rain_mean: torch.Tensor | None,
    rain_std: torch.Tensor | None,
) -> float:
    model.eval()
    total_loss = 0.0

    for rain_in, rain_target in tqdm(loader, desc="valid", leave=False):
        rain_in = rain_in.to(device)
        rain_target = rain_target.to(device)
        if rain_mean is not None and rain_std is not None:
            rain_in = normalize_sequence(rain_in, rain_mean.to(device), rain_std.to(device))
            rain_target = normalize_sequence(rain_target, rain_mean.to(device), rain_std.to(device))

        pred, _ = model(rain_in)
        loss = criterion(pred, rain_target)
        total_loss += float(loss.item()) * rain_in.shape[0]

    return total_loss / max(1, len(loader.dataset))


def create_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/rain_lstm"))
    parser.add_argument("--save-name", type=str, default="rain_lstm.ckpt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--rain-height", type=int, default=39)
    parser.add_argument("--rain-width", type=int, default=13)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--train-max-traj-id", type=int, default=100)
    parser.add_argument("--valid-max-traj-id", type=int, default=117)
    parser.add_argument("--test-max-traj-id", type=int, default=120)
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


def main(opt: argparse.Namespace) -> None:
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    base_path = opt.base_path
    data_root = _resolve_path(base_path, opt.data_root)
    save_dir = _resolve_path(base_path, opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / opt.save_name

    traj_ids = _available_traj_ids(data_root)
    train_ids, valid_ids, test_ids = _split_ids(
        traj_ids,
        train_max_traj_id=opt.train_max_traj_id,
        valid_max_traj_id=opt.valid_max_traj_id,
        test_max_traj_id=opt.test_max_traj_id,
    )

    train_rain_full = _load_split_sequences(data_root, train_ids)
    valid_rain_full = _load_split_sequences(data_root, valid_ids)
    test_rain_full = _load_split_sequences(data_root, test_ids)

    if train_rain_full.shape[1] < 2:
        raise ValueError("Rain sequences must contain at least two time steps")

    train_inputs = train_rain_full[:, :-1, :].contiguous()
    train_targets = train_rain_full[:, 1:, :].contiguous()
    valid_inputs = valid_rain_full[:, :-1, :].contiguous()
    valid_targets = valid_rain_full[:, 1:, :].contiguous()
    test_inputs = test_rain_full[:, :-1, :].contiguous()
    test_targets = test_rain_full[:, 1:, :].contiguous()

    rain_mean = rain_std = None
    if opt.normalize:
        rain_mean, rain_std = _compute_stats(train_rain_full)

    input_size = int(train_inputs.shape[-1])
    output_size = int(train_targets.shape[-1])
    grid_shape = (int(opt.rain_height), int(opt.rain_width))
    if grid_shape[0] * grid_shape[1] != input_size:
        raise ValueError(
            f"Rain grid shape {grid_shape} does not match flattened input size {input_size}"
        )

    device = _resolve_device(opt.device)
    model = RainPropagatorLSTM(
        input_size=input_size,
        hidden_size=opt.hidden_size,
        output_size=output_size,
        num_layers=opt.num_layers,
        dropout=opt.dropout,
        grid_shape=grid_shape,
        kernel_size=opt.kernel_size,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.weight_decay,
    )
    criterion = nn.MSELoss()
    train_loader = _make_loader(train_inputs, train_targets, opt.batch_size, shuffle=True)
    valid_loader = _make_loader(valid_inputs, valid_targets, opt.batch_size, shuffle=False)
    test_loader = _make_loader(test_inputs, test_targets, opt.batch_size, shuffle=False)

    history = {
        "train_loss": [],
        "valid_loss": [],
        "best_valid_loss": None,
        "best_epoch": None,
    }
    best_valid = float("inf")
    best_epoch = -1

    print(f"Dataset root: {data_root}")
    print(f"Input dim: {input_size} | output dim: {output_size}")
    print(f"Rain grid shape: {grid_shape[0]} x {grid_shape[1]}")
    print(f"Normalize: {opt.normalize}")
    print(f"Saving checkpoint to: {checkpoint_path}")

    for epoch in range(1, opt.epochs + 1):
        train_loss = _train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            rain_mean=rain_mean,
            rain_std=rain_std,
            grad_clip=opt.grad_clip,
        )
        valid_loss = _eval_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device,
            rain_mean=rain_mean,
            rain_std=rain_std,
        )

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "model_type": "convlstm",
                    "input_size": input_size,
                    "hidden_size": opt.hidden_size,
                    "output_size": output_size,
                    "num_layers": opt.num_layers,
                    "dropout": opt.dropout,
                    "grid_shape": list(grid_shape),
                    "input_channels": 1,
                    "kernel_size": opt.kernel_size,
                },
                "normalize": bool(opt.normalize),
                "train_ids": train_ids,
                "valid_ids": valid_ids,
                "test_ids": test_ids,
            }
            if rain_mean is not None and rain_std is not None:
                checkpoint["rain_mean"] = rain_mean.cpu().numpy()
                checkpoint["rain_std"] = rain_std.cpu().numpy()
            torch.save(checkpoint, checkpoint_path)
            if rain_mean is not None and rain_std is not None:
                np.save(save_dir / "rain_mean.npy", rain_mean.cpu().numpy())
                np.save(save_dir / "rain_std.npy", rain_std.cpu().numpy())

        if epoch == 1 or epoch % opt.eval_interval == 0 or epoch == opt.epochs:
            print(
                f"Epoch {epoch:04d}/{opt.epochs} | "
                f"train_loss={train_loss:.6f} | valid_loss={valid_loss:.6f} | "
                f"best_valid={best_valid:.6f} @ epoch {best_epoch}"
            )

    test_loss = _eval_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        rain_mean=rain_mean,
        rain_std=rain_std,
    )

    history["best_valid_loss"] = best_valid
    history["best_epoch"] = best_epoch

    report = {
        "dataset_root": str(data_root),
        "checkpoint_path": str(checkpoint_path),
        "input_dim": input_size,
        "output_dim": output_size,
        "grid_shape": list(grid_shape),
        "hidden_size": opt.hidden_size,
        "num_layers": opt.num_layers,
        "dropout": opt.dropout,
        "kernel_size": opt.kernel_size,
        "normalize": bool(opt.normalize),
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "test_ids": test_ids,
        "best_valid_loss": best_valid,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "history": history,
    }
    report_path = save_dir / "train_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved training report to {report_path}")


if __name__ == "__main__":
    main(create_options())
