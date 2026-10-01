"""Encode every Texas train/test flow trajectory with the saved VAE.

Only the two latent arrays are replaced. Per-trajectory temporary files make an
interrupted run resumable; the existing arrays stay intact until assembly ends.
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from paths import REPOSITORY_ROOT, TEXAS_DATA_ROOT, VAE_CHECKPOINT_DIR
from data_layout import SPLITS, FLOW_SHAPE, sample_ids_in_rain_order
from vae_texas import VAE_LSTM


LATENT_SHAPE = (192, 16, 25, 25)


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_model(checkpoint_path: Path, device: torch.device) -> VAE_LSTM:
    model = VAE_LSTM(
        resolution=256,
        in_channel=3,
        ch_mult=(1, 4, 16),
        num_res_blocks=3,
        hidden_dim=16,
        embed_dim=8,
        latent_dim=256,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"Loaded VAE epoch {checkpoint['epoch']} from {checkpoint_path}", flush=True)
    return model.to(device).eval()


def load_flow_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as stats:
        mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 3, 1, 1)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError(f"Invalid VAE normalization statistics in {path}")
    return mean, std


def encode_trajectory(
    flow_path: Path,
    model: VAE_LSTM,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    flow = np.load(flow_path, mmap_mode="r")
    if flow.shape != FLOW_SHAPE:
        raise ValueError(f"Expected {FLOW_SHAPE} at {flow_path}; found {flow.shape}")
    result = np.empty(LATENT_SHAPE, dtype=np.float32)
    with torch.inference_mode():
        for start in range(1, FLOW_SHAPE[0], batch_size):
            stop = min(start + batch_size, FLOW_SHAPE[0])
            frames = np.asarray(flow[start:stop], dtype=np.float32)
            frames = np.ascontiguousarray((frames - mean) / std)
            encoded = model.encode(torch.from_numpy(frames).to(device))
            chunk = encoded.cpu().numpy()
            expected = (stop - start, *LATENT_SHAPE[1:])
            if chunk.shape != expected or not np.all(np.isfinite(chunk)):
                raise ValueError(f"Invalid VAE output for {flow_path}: {chunk.shape}")
            result[start - 1 : stop - 1] = chunk
    return result


def save_array_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def generate_split(
    split: str,
    model: VAE_LSTM,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    rain_stats_path: Path = VAE_CHECKPOINT_DIR / "mean_std_rain_source.npz",
) -> None:
    input_dir, sample_ids = SPLITS[split]
    sample_ids = sample_ids_in_rain_order(split, output_dir, list(sample_ids), rain_stats_path)
    cache_dir = output_dir / f".latent_{split}_samples"
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for position, sample_id in enumerate(sample_ids, 1):
        cached = cache_dir / f"sample_{sample_id:05d}.npy"
        if cached.is_file():
            existing = np.load(cached, mmap_mode="r")
            if existing.shape == LATENT_SHAPE and existing.dtype == np.float32:
                print(f"{split} {position}/{len(sample_ids)}: sample {sample_id} cached", flush=True)
                continue
        flow_path = input_dir / f"sample_{sample_id:05d}" / "flow_variables.npy"
        if not flow_path.is_file():
            raise FileNotFoundError(flow_path)
        latent = encode_trajectory(flow_path, model, mean, std, device, batch_size)
        save_array_atomic(cached, latent)
        print(f"{split} {position}/{len(sample_ids)}: encoded sample {sample_id} "
              f"({time.perf_counter()-started:.1f}s elapsed)", flush=True)

    final_path = output_dir / f"latent_{split}.npy"
    temporary = output_dir / f".latent_{split}.partial.npy"
    combined = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32,
        shape=(len(sample_ids), *LATENT_SHAPE),
    )
    for index, sample_id in enumerate(sample_ids):
        combined[index] = np.load(cache_dir / f"sample_{sample_id:05d}.npy", mmap_mode="r")
    combined.flush()
    del combined
    assembled = np.load(temporary, mmap_mode="r")
    if assembled.shape != (len(sample_ids), *LATENT_SHAPE):
        raise ValueError(f"Unexpected assembled shape at {temporary}: {assembled.shape}")
    del assembled
    os.replace(temporary, final_path)
    shutil.rmtree(cache_dir)
    print(f"Saved {final_path} with shape {(len(sample_ids), *LATENT_SHAPE)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--checkpoint", type=Path, default=VAE_CHECKPOINT_DIR / "checkpoint_vae.pth")
    parser.add_argument("--flow-stats", type=Path, default=VAE_CHECKPOINT_DIR / "mean_std.npz")
    parser.add_argument("--rain-stats", type=Path, default=VAE_CHECKPOINT_DIR / "mean_std_rain_source.npz")
    parser.add_argument("--output-dir", type=Path, default=TEXAS_DATA_ROOT / "vae_and_latents_texas")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    device = torch.device(args.device)
    output_dir = repository_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(repository_path(args.checkpoint), device)
    mean, std = load_flow_stats(repository_path(args.flow_stats))
    for split in ("train", "test") if args.split == "all" else (args.split,):
        generate_split(split, model, mean, std, device, args.batch_size, output_dir,
                       repository_path(args.rain_stats))


if __name__ == "__main__":
    main()
