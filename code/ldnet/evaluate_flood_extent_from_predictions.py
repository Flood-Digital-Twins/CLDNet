#!/usr/bin/env python3
"""Compute event flood-extent scores from saved full-grid model predictions.

Matches ``_print_flood_inundation_confusion`` in
``ldnet_chicago_efficient_test.py``: a cell is positive if its depth reaches
the threshold at any timestep. Truth is read from the reduced evaluation data
used by that script; full-grid predictions are selected by the aggregate mask.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    return {
        "csi": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=int, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--mask", type=Path, default=Path("data/postprocessed/illinois/aggregate_mask.npy"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.5])
    parser.add_argument("--output-csv", type=Path, required=True)
    opt = parser.parse_args()

    mask = np.load(opt.mask, mmap_mode="r")
    if mask.ndim != 2 or mask.dtype != np.bool_:
        parser.error("--mask must be a two-dimensional boolean .npy array")
    indices = np.flatnonzero(mask)
    flow = np.load(opt.data_root / f"flow_variables_traj{opt.trajectory}.npy", mmap_mode="r")
    predictions = {
        model: np.load(
            opt.prediction_root / directory / f"predictions_allindices_traj{opt.trajectory}.npy",
            mmap_mode="r",
        )
        for model, directory in (("LDNet", "ldnet"), ("CLDNet", "cldnet"))
    }
    if flow.ndim != 4 or flow.shape[0] != 1 or flow.shape[2] != indices.size:
        raise ValueError(f"Reduced flow shape {flow.shape} does not match mask cell count {indices.size}")
    t_len = flow.shape[1]
    for model, pred in predictions.items():
        if pred.shape[:3] != (t_len, *mask.shape) or pred.shape[-1] < 1:
            raise ValueError(f"{model} prediction shape {pred.shape} does not match truth and mask")

    peak_truth = np.full(indices.size, -np.inf, dtype=np.float32)
    peak_predictions = {
        model: np.full(indices.size, -np.inf, dtype=np.float32) for model in predictions
    }
    for timestep in range(t_len):
        truth_depth = np.asarray(flow[0, timestep, :, 0], dtype=np.float32)
        if not np.isfinite(truth_depth).all():
            raise ValueError(f"Non-finite truth depth at timestep {timestep}")
        np.maximum(peak_truth, truth_depth, out=peak_truth)
        for model, pred in predictions.items():
            depth = np.asarray(pred[timestep, :, :, 0].reshape(-1)[indices], dtype=np.float32)
            if not np.isfinite(depth).all():
                raise ValueError(f"Non-finite {model} depth at timestep {timestep}")
            np.maximum(peak_predictions[model], depth, out=peak_predictions[model])
        if (timestep + 1) % 12 == 0 or timestep + 1 == t_len:
            print(f"Processed {timestep + 1}/{t_len} timesteps", flush=True)

    rows = []
    for threshold in opt.thresholds:
        truth_positive = peak_truth >= threshold
        for model, peak in peak_predictions.items():
            predicted_positive = peak >= threshold
            tp = int(np.count_nonzero(predicted_positive & truth_positive))
            fp = int(np.count_nonzero(predicted_positive & ~truth_positive))
            fn = int(np.count_nonzero(~predicted_positive & truth_positive))
            tn = int(indices.size - tp - fp - fn)
            row = {
                "trajectory": opt.trajectory,
                "model": model,
                "threshold_m": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                **scores(tp, fp, fn),
            }
            rows.append(row)
            print(
                f"{model} threshold {threshold:g} m: "
                + ", ".join(f"{name}={row[name] * 100:.2f}%" for name in ("csi", "f1", "precision", "recall"))
                + f" (TP={tp}, FP={fp}, FN={fn}, TN={tn})",
                flush=True,
            )

    opt.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with opt.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {opt.output_csv}")


if __name__ == "__main__":
    main()
