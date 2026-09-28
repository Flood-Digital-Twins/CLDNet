#!/usr/bin/env python3
"""Score LDNet and CLDNet predictions on identical finite cells.

Relative RMSE is RMSE / RMS(truth), equivalently sqrt(SSE / sum(truth**2)).
Aggregate metrics pool every valid h, hu, and hv value. The "all" trajectory
row pools all requested trajectories before computing each metric.
An optional spatial mask limits scoring to the model's reduced evaluation grid.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


CHANNELS = ("h", "hu", "hv")


class Sums:
    def __init__(self) -> None:
        self.count = np.zeros(3, dtype=np.int64)
        self.truth_sum = np.zeros(3, dtype=np.float64)
        self.truth_sq_sum = np.zeros(3, dtype=np.float64)
        self.error_sq_sum = np.zeros(3, dtype=np.float64)

    def add(self, truth: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> None:
        truth_valid = np.where(valid, truth, 0.0)
        error = np.zeros_like(prediction)
        np.subtract(prediction, truth, out=error, where=valid)
        axes = (0, 1)
        self.count += valid.sum(axis=axes, dtype=np.int64)
        self.truth_sum += truth_valid.sum(axis=axes, dtype=np.float64)
        self.truth_sq_sum += np.square(truth_valid).sum(axis=axes, dtype=np.float64)
        self.error_sq_sum += np.square(error).sum(axis=axes, dtype=np.float64)

    def merge(self, other: Sums) -> None:
        self.count += other.count
        self.truth_sum += other.truth_sum
        self.truth_sq_sum += other.truth_sq_sum
        self.error_sq_sum += other.error_sq_sum

    def rows(self, model: str, trajectory: str) -> list[dict[str, object]]:
        rows = []
        for variable, channels in (("aggregate", slice(None)), *((name, i) for i, name in enumerate(CHANNELS))):
            count = int(np.sum(self.count[channels]))
            truth_sum = float(np.sum(self.truth_sum[channels]))
            truth_sq_sum = float(np.sum(self.truth_sq_sum[channels]))
            error_sq_sum = float(np.sum(self.error_sq_sum[channels]))
            centered_sq_sum = truth_sq_sum - truth_sum**2 / count if count else 0.0
            rows.append(
                {
                    "model": model,
                    "trajectory": trajectory,
                    "variable": variable,
                    "relative_rmse": np.sqrt(error_sq_sum / truth_sq_sum) if truth_sq_sum > 0 else float("nan"),
                    "rmse": np.sqrt(error_sq_sum / count) if count else float("nan"),
                    "r2": 1.0 - error_sq_sum / centered_sq_sum if centered_sq_sum > 0 else float("nan"),
                    "count": count,
                }
            )
        return rows


def score_trajectory(
    ldnet_dir: Path,
    cldnet_dir: Path,
    trajectory: int,
    row_block: int,
    spatial_mask: np.ndarray | None,
) -> dict[str, Sums]:
    name = f"predictions_allindices_traj{trajectory}.npy"
    truth_name = f"truth_traj{trajectory}.npy"
    ldnet = np.load(ldnet_dir / name, mmap_mode="r")
    cldnet = np.load(cldnet_dir / name, mmap_mode="r")
    truth = np.load(ldnet_dir / truth_name, mmap_mode="r")
    cldnet_truth = np.load(cldnet_dir / truth_name, mmap_mode="r")
    if not (ldnet.shape == cldnet.shape == truth.shape == cldnet_truth.shape):
        raise ValueError(f"Trajectory {trajectory}: prediction/truth shapes differ")
    if ldnet.ndim != 4 or ldnet.shape[-1] != len(CHANNELS):
        raise ValueError(f"Trajectory {trajectory}: expected (T, H, W, 3), got {ldnet.shape}")
    if spatial_mask is not None and spatial_mask.shape != ldnet.shape[1:3]:
        raise ValueError(f"Trajectory {trajectory}: mask shape {spatial_mask.shape} differs from grid {ldnet.shape[1:3]}")

    result = {"LDNet": Sums(), "CLDNet": Sums()}
    t_len, h_dim = ldnet.shape[:2]
    for timestep in range(t_len):
        for start in range(0, h_dim, row_block):
            section = np.s_[timestep, start : start + row_block]
            y = np.asarray(truth[section], dtype=np.float64)
            y_cldnet = np.asarray(cldnet_truth[section], dtype=np.float64)
            if not np.array_equal(y, y_cldnet, equal_nan=True):
                raise ValueError(f"Trajectory {trajectory}: truth differs at t={timestep}, row={start}")
            ld = np.asarray(ldnet[section], dtype=np.float64)
            cl = np.asarray(cldnet[section], dtype=np.float64)
            valid_ld = np.isfinite(y) & np.isfinite(ld)
            valid_cl = np.isfinite(y) & np.isfinite(cl)
            if spatial_mask is not None:
                section_mask = spatial_mask[start : start + row_block, :, None]
                valid_ld &= section_mask
                valid_cl &= section_mask
            if not np.array_equal(valid_ld, valid_cl):
                raise ValueError(f"Trajectory {trajectory}: finite prediction masks differ at t={timestep}, row={start}")
            result["LDNet"].add(y, ld, valid_ld)
            result["CLDNet"].add(y, cl, valid_ld)
        if (timestep + 1) % 24 == 0 or timestep + 1 == t_len:
            print(f"Trajectory {trajectory}: scored {timestep + 1}/{t_len} timesteps", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ldnet-dir", required=True, type=Path)
    parser.add_argument("--cldnet-dir", required=True, type=Path)
    parser.add_argument("--traj-ids", required=True, type=int, nargs="+")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--spatial-mask", type=Path, help="Boolean (H, W) mask restricting scored cells")
    parser.add_argument("--row-block", type=int, default=256)
    opt = parser.parse_args()
    if opt.row_block <= 0:
        parser.error("--row-block must be positive")

    rows = []
    pooled = {"LDNet": Sums(), "CLDNet": Sums()}
    spatial_mask = np.load(opt.spatial_mask, mmap_mode="r") if opt.spatial_mask is not None else None
    if spatial_mask is not None and (spatial_mask.ndim != 2 or spatial_mask.dtype != np.bool_):
        parser.error("--spatial-mask must be a two-dimensional boolean .npy array")
    for trajectory in opt.traj_ids:
        scores = score_trajectory(opt.ldnet_dir, opt.cldnet_dir, trajectory, opt.row_block, spatial_mask)
        for model in ("LDNet", "CLDNet"):
            rows.extend(scores[model].rows(model, str(trajectory)))
            pooled[model].merge(scores[model])
    if len(opt.traj_ids) > 1:
        for model in ("LDNet", "CLDNet"):
            rows.extend(pooled[model].rows(model, "all"))

    opt.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with opt.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model", "trajectory", "variable", "relative_rmse", "rmse", "r2", "count"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {opt.output_csv}")


if __name__ == "__main__":
    main()
