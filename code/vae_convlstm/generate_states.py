"""Build normalized Texas state arrays from raw flow samples.

Rows follow the saved rain and latent arrays. Each split is written to a
temporary memory-mapped array and replaces its old state file only when complete.
An interrupted split resumes at the first unfinished trajectory.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from data_layout import FLOW_SHAPE, SPLITS, sample_ids_in_rain_order
from paths import REPOSITORY_ROOT, TEXAS_DATA_ROOT, VAE_CHECKPOINT_DIR


STATE_SHAPE = (192, 3, 500, 500)


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_flow_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as stats:
        mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 3, 1, 1)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"Invalid flow normalization statistics in {path}")
    return mean, std


def write_progress(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def generate_split(
    split: str,
    output_dir: Path,
    mean: np.ndarray,
    std: np.ndarray,
    rain_stats_path: Path,
    chunk_size: int,
) -> None:
    source_dir, source_ids = SPLITS[split]
    sample_ids = sample_ids_in_rain_order(split, output_dir, list(source_ids), rain_stats_path)
    latent = np.load(output_dir / f"latent_{split}.npy", mmap_mode="r")
    if latent.shape[:2] != (len(sample_ids), STATE_SHAPE[0]):
        raise ValueError(f"Latent and state row/time counts differ for {split}")

    final_path = output_dir / f"state_{split}.npy"
    temporary_path = output_dir / f".state_{split}.partial.npy"
    progress_path = output_dir / f".state_{split}.progress.json"
    shape = (len(sample_ids), *STATE_SHAPE)
    stats_digest = hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()
    expected_record = {"sample_ids": sample_ids, "stats_sha256": stats_digest}

    if temporary_path.exists() or progress_path.exists():
        if not temporary_path.is_file() or not progress_path.is_file():
            raise RuntimeError(f"Incomplete state generation markers for {split}")
        record = json.loads(progress_path.read_text())
        if any(record.get(key) != value for key, value in expected_record.items()):
            raise ValueError(f"State generation inputs changed since {split} was started")
        completed = record["completed"]
        if not isinstance(completed, int) or not 0 <= completed <= len(sample_ids):
            raise ValueError(f"Invalid {split} progress count: {completed}")
        output = np.lib.format.open_memmap(temporary_path, mode="r+")
        if output.shape != shape or output.dtype != np.float32:
            raise ValueError(f"Unexpected temporary state array for {split}")
        print(f"Resuming {split} at row {completed + 1}", flush=True)
    else:
        required_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
        free_bytes = os.statvfs(output_dir).f_bavail * os.statvfs(output_dir).f_frsize
        if free_bytes < required_bytes:
            raise OSError(f"Insufficient free space for state_{split}.npy")
        output = np.lib.format.open_memmap(
            temporary_path, mode="w+", dtype=np.float32, shape=shape,
        )
        completed = 0
        write_progress(progress_path, {**expected_record, "completed": 0})

    started = time.perf_counter()
    for row in range(completed, len(sample_ids)):
        sample_id = sample_ids[row]
        flow_path = source_dir / f"sample_{sample_id:05d}" / "flow_variables.npy"
        flow = np.load(flow_path, mmap_mode="r")
        if flow.shape != FLOW_SHAPE or flow.dtype != np.float32:
            raise ValueError(f"Unexpected raw flow shape or dtype for sample {sample_id}")
        for start in range(1, FLOW_SHAPE[0], chunk_size):
            stop = min(start + chunk_size, FLOW_SHAPE[0])
            frames = np.asarray(flow[start:stop])
            normalized = np.empty(frames.shape, dtype=np.float32)
            np.subtract(frames, mean, out=normalized)
            np.divide(normalized, std, out=normalized)
            if not np.isfinite(normalized).all():
                raise ValueError(f"Non-finite normalized state for sample {sample_id}")
            output[row, start - 1:stop - 1] = normalized
        output.flush()
        write_progress(progress_path, {**expected_record, "completed": row + 1})
        print(f"{split} {row + 1}/{len(sample_ids)}: sample {sample_id} "
              f"({time.perf_counter() - started:.1f}s elapsed)", flush=True)

    output.flush()
    del output
    assembled = np.load(temporary_path, mmap_mode="r")
    if assembled.shape != shape or assembled.dtype != np.float32:
        raise ValueError(f"Unexpected assembled state array for {split}")
    for row in {0, len(sample_ids) // 2, len(sample_ids) - 1}:
        sample_id = sample_ids[row]
        flow = np.load(source_dir / f"sample_{sample_id:05d}" / "flow_variables.npy", mmap_mode="r")
        for time_index in (0, STATE_SHAPE[0] // 2, STATE_SHAPE[0] - 1):
            expected = (flow[time_index + 1] - mean[0]) / std[0]
            if not np.array_equal(assembled[row, time_index], expected):
                raise ValueError(f"State verification failed for {split} sample {sample_id}")
    del assembled
    os.replace(temporary_path, final_path)
    progress_path.unlink()
    print(f"Saved state_{split}.npy with shape {shape}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--flow-stats", type=Path, default=VAE_CHECKPOINT_DIR / "mean_std.npz")
    parser.add_argument("--rain-stats", type=Path, default=VAE_CHECKPOINT_DIR / "mean_std_rain_source.npz")
    parser.add_argument("--output-dir", type=Path, default=TEXAS_DATA_ROOT / "vae_and_latents_texas")
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    output_dir = repository_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mean, std = load_flow_stats(repository_path(args.flow_stats))
    rain_stats_path = repository_path(args.rain_stats)
    for split in ("train", "test") if args.split == "all" else (args.split,):
        generate_split(split, output_dir, mean, std, rain_stats_path, args.chunk_size)


if __name__ == "__main__":
    main()
