"""Texas sample layout and the row order of the saved rain arrays."""

from pathlib import Path

import numpy as np

from paths import TEXAS_TRAIN_DIR, TEXAS_TEST_DIR


SPLITS = {
    "train": (TEXAS_TRAIN_DIR, range(1, 101)),
    "test": (TEXAS_TEST_DIR, range(101, 121)),
}
FLOW_SHAPE = (193, 3, 500, 500)


def sample_ids_in_rain_order(
    split: str, output_dir: Path, sample_ids: list[int], rain_stats_path: Path,
) -> list[int]:
    """Match each saved rain row to its raw sample ID."""
    rain = np.load(output_dir / f"rain_{split}.npy", mmap_mode="r")
    if rain.shape != (len(sample_ids), 192, 1):
        raise ValueError(f"Unexpected rain_{split} shape: {rain.shape}")
    with np.load(rain_stats_path) as stats:
        mean = np.asarray(stats["mean"], dtype=np.float32).item()
        std = np.asarray(stats["std"], dtype=np.float32).item()
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(f"Invalid rain normalization statistics in {rain_stats_path}")

    input_dir, _ = SPLITS[split]
    by_rain = {}
    for sample_id in sample_ids:
        source = np.load(input_dir / f"sample_{sample_id:05d}" / "rain_source.npy", mmap_mode="r")
        expected = np.repeat(source[:-1, 1], 4).astype(np.float32)
        expected = np.ascontiguousarray(((expected - mean) / std).reshape(192, 1))
        key = expected.tobytes()
        if key in by_rain:
            raise ValueError(f"Duplicate normalized rain for samples {by_rain[key]} and {sample_id}")
        by_rain[key] = sample_id

    ordered = []
    for index, row in enumerate(rain):
        sample_id = by_rain.get(np.ascontiguousarray(row).tobytes())
        if sample_id is None:
            raise ValueError(f"Could not match rain_{split} row {index} to a raw sample")
        ordered.append(sample_id)
    if len(set(ordered)) != len(sample_ids):
        raise ValueError(f"rain_{split} rows do not map one-to-one to raw samples")
    return ordered
