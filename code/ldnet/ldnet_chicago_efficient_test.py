"""
Efficient Fourier LDNN Test Script for Flood Surrogate Modeling.

Usage from the repository root:
    python code/ldnet/ldnet_chicago_efficient_test.py --traj-id 109 \\
        --checkpoint-epoch 539 --model-path checkpoints/ldnet/illinois \\
        --all-vars --fourier-mapping-size 32

    python code/ldnet/ldnet_chicago_efficient_test.py --traj-id 109 \\
        --checkpoint-epoch 539 --model-path checkpoints/cldnet/illinois \\
        --all-vars --fourier-mapping-size 32 --use-static-features

Full command arrays for Illinois and Texas are in configs/{ldnet,cldnet}/*.json.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm

repo_path = Path(__file__).resolve().parent
sys.path.append(str(repo_path))

from src.logger import Logger
from efficient_fourier_ldnet import EfficientFourierLDNN

dt = 1
CHICAGO_DEM_PATH = (
    Path(__file__).resolve().parents[2]
    / "data/usgs_2013_validation/data_usgs/usgs_plot/GT_sample_data_folder/dem_5070.tif"
)


def create_options():
    parser = argparse.ArgumentParser()
    default_base_path = Path(__file__).resolve().parents[2]
    parser.add_argument("--base-path", type=Path, default=default_base_path)
    parser.add_argument("--log-dir", type=Path, default="log")
    parser.add_argument("--name", type=str, default="ldnet_chicago_efficient_test")
    parser.add_argument("--model-path", type=Path, default="checkpoints/ldnet")
    parser.add_argument("--data-root", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--aggregate-mask-path", type=Path, default=Path("data/postprocessed/illinois/aggregate_mask.npy"))
    parser.add_argument("--use-static-features", action="store_true", default=False)
    parser.add_argument("--static-slope-xy", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--normalize-rain", action="store_true", default=False)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/sims_30"),
    )
    parser.add_argument("--traj-id", type=int, default=0)
    parser.add_argument("--burn-in-length", type=int, default=1)
    parser.add_argument("--depth-threshold", type=float, default=0.1)
    parser.add_argument("--flood-threshold", type=float, default=0.1)
    parser.add_argument("--flood-thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--metrics-only", action="store_true", default=False,
                        help="Print reduced-grid flood metrics and skip plotting.")
    parser.add_argument("--snapshot-metrics-only", action="store_true", default=False,
                        help="Print reduced-grid rising, peak, recession, and combined depth metrics and skip plotting.")
    parser.add_argument("--height-multiplier", type=float, default=6.0)
    parser.add_argument("--width-multiplier", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--num-latent-states", type=int, default=200)
    parser.add_argument("--fourier-mapping-size", type=int, default=32)
    parser.add_argument("--NN-dyn-depth", type=int, default=8)
    parser.add_argument("--NN-dyn-width", type=int, default=50)
    parser.add_argument("--NN-rec-depth", type=int, default=10)
    parser.add_argument("--NN-rec-width", type=int, default=300)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--kernel-initializer", type=str, default="Glorot normal")
    parser.add_argument("--dyn-checkpoint", type=Path, default=None)
    parser.add_argument("--rec-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-epoch", type=int, default=None)
    parser.add_argument(
        "--rec-finetune",
        action="store_true",
        default=False,
        help="Load the latest matched rec_finetune/B_finetune checkpoints for rec and B.",
    )
    parser.add_argument(
        "--use-finetune-rec",
        dest="rec_finetune",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--finetune-path",
        type=Path,
        default=None,
        help="Directory containing rec_finetune_*.ckpt and B_finetune_*.ckpt files. Defaults to --model-path.",
    )
    parser.add_argument(
        "--finetune-rec-path",
        dest="finetune_path",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-map-image",
        type=Path,
        default=Path("plots/easting_northing.png"),
    )
    parser.add_argument("--data-sanity", action="store_true", default=False)
    parser.add_argument("--data-sanity-only", action="store_true", default=False)
    parser.add_argument("--all-vars", action="store_true", default=False)
    return parser.parse_args()


def _load_reduced_data(data_root: Path, traj_id: int, use_static: bool, static_slope_xy: bool = False):
    flow = np.load(data_root / f"flow_variables_traj{traj_id}.npy", mmap_mode="r")
    coords = np.load(data_root / f"coords_traj{traj_id}.npy", mmap_mode="r")
    rain = np.load(data_root / f"rain_source_traj{traj_id}.npy", mmap_mode="r")
    static = None
    if use_static:
        static_prefix = "static_features_xy" if static_slope_xy else "static_features"
        static = np.load(data_root / f"{static_prefix}_traj{traj_id}.npy", mmap_mode="r")
        coords = np.concatenate([coords, static], axis=-1)
    return flow, coords, rain


def _load_full_dem(input_root: Path) -> tuple[np.ndarray, str]:
    elev, _, _, _ = _load_full_dem_georef(input_root)
    return elev, "Elevation (m)"


def _load_full_dem_georef(input_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    dem_path = CHICAGO_DEM_PATH
    if not dem_path.exists():
        raise FileNotFoundError(f"Missing DEM GeoTIFF at {dem_path}")

    import rasterio

    with rasterio.open(dem_path) as dataset:
        elev = dataset.read(1, masked=True).astype(np.float32).filled(np.nan)
        transform = dataset.transform
        cols = np.arange(dataset.width, dtype=np.float64)[None, :] + 0.5
        rows = np.arange(dataset.height, dtype=np.float64)[:, None] + 0.5
        x_grid = (transform.a * cols + transform.b * rows + transform.c).astype(np.float32)
        y_grid = (transform.d * cols + transform.e * rows + transform.f).astype(np.float32)
    return elev, x_grid, y_grid, "Elevation (m)"


def _maybe_load_channel_normalization(
    enabled: bool,
    mean_path: Path,
    std_path: Path,
    label: str,
    logger,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not enabled:
        return None, None

    if not mean_path.exists() or not std_path.exists():
        logger.info(
            f"{label} normalization requested, but normalization files were not both found at "
            f"{mean_path} and {std_path}. Proceeding without {label} normalization."
        )
        return None, None

    mean = np.load(mean_path).astype(np.float32, copy=False)
    std = np.load(std_path).astype(np.float32, copy=False)
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise ValueError(
            f"Normalization files must be matching 1D arrays; got mean {mean.shape} and std {std.shape}"
        )
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)):
        raise ValueError("Normalization files contain non-finite values.")
    if np.any(std <= 0):
        raise ValueError(f"{label} normalization std contains non-positive values.")

    logger.info(f"Applying {label} normalization from {mean_path} and {std_path}")
    return mean, std


def _r2_score(truth: np.ndarray, pred: np.ndarray) -> float:
    truth_flat = np.asarray(truth, dtype=np.float32).reshape(-1)
    pred_flat = np.asarray(pred, dtype=np.float32).reshape(-1)
    valid = np.isfinite(truth_flat) & np.isfinite(pred_flat)
    if not np.any(valid):
        return float("nan")

    truth_valid = truth_flat[valid]
    pred_valid = pred_flat[valid]
    ss_res = float(np.sum((pred_valid - truth_valid) ** 2))
    ss_tot = float(np.sum((truth_valid - truth_valid.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _rmse_score(truth: np.ndarray, pred: np.ndarray) -> float:
    truth_flat = np.asarray(truth, dtype=np.float32).reshape(-1)
    pred_flat = np.asarray(pred, dtype=np.float32).reshape(-1)
    valid = np.isfinite(truth_flat) & np.isfinite(pred_flat)
    if not np.any(valid):
        return float("nan")

    truth_valid = truth_flat[valid]
    pred_valid = pred_flat[valid]
    return float(np.sqrt(np.mean((pred_valid - truth_valid) ** 2)))


def _paired_valid_values(truth: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth_ma = np.ma.array(truth, copy=False)
    pred_ma = np.ma.array(pred, copy=False)
    if truth_ma.shape != pred_ma.shape:
        raise ValueError(f"Metric arrays must have matching shapes; got {truth_ma.shape} and {pred_ma.shape}")

    truth_flat = np.asarray(truth_ma.data, dtype=np.float32).reshape(-1)
    pred_flat = np.asarray(pred_ma.data, dtype=np.float32).reshape(-1)
    truth_mask = np.ma.getmaskarray(truth_ma).reshape(-1)
    pred_mask = np.ma.getmaskarray(pred_ma).reshape(-1)
    valid = (~truth_mask) & (~pred_mask) & np.isfinite(truth_flat) & np.isfinite(pred_flat)
    return truth_flat[valid], pred_flat[valid]


def _print_data_sanity(rain, flow, coords, sample_points=1000):
    print(f"Data sanity: rain {rain.shape} {rain.dtype}")
    print(f"Data sanity: flow {flow.shape} {flow.dtype}")
    print(f"Data sanity: coords {coords.shape} {coords.dtype}")

    max_points = min(flow.shape[2], sample_points)
    rng = np.random.default_rng(0)
    idx = rng.choice(flow.shape[2], size=max_points, replace=False)
    rain_slice = np.asarray(rain[:, :1, :min(rain.shape[2], sample_points)], dtype=np.float32)
    flow_slice = np.asarray(flow[:, :1, idx, :], dtype=np.float32)
    coords_slice = np.asarray(coords[:, idx, :], dtype=np.float32)

    print(f"Rain sample min/max: {rain_slice.min():.6f}/{rain_slice.max():.6f}")
    print(f"Flow sample min/max: {flow_slice.min():.6f}/{flow_slice.max():.6f}")
    print(f"Coords sample min/max: {coords_slice.min():.6f}/{coords_slice.max():.6f}")


def _load_aggregate_mask(mask_path: Path) -> tuple[np.ndarray, int, int]:
    mask = np.load(mask_path)
    if mask.ndim != 2:
        raise ValueError(f"Expected aggregate mask with shape (H, W), got {mask.shape}")
    h_dim, w_dim = int(mask.shape[0]), int(mask.shape[1])
    valid = mask.reshape(-1).astype(bool)
    return valid, h_dim, w_dim


def _build_coords(h_dim: int, w_dim: int) -> np.ndarray:
    x = 2.0 * (np.arange(w_dim, dtype=np.float32) / w_dim - 0.5)
    aspect = h_dim / float(w_dim)
    y = 2.0 * (np.arange(h_dim, dtype=np.float32) / h_dim - 0.5) * aspect
    x_coords = np.tile(x, h_dim)
    y_coords = np.repeat(y, w_dim)
    coords = np.stack([x_coords, y_coords], axis=1)[None, :, :]
    return coords.astype(np.float32)


def _resolve_path(base_path: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_path / path


def _resolve_scatter_indices(valid_mask: np.ndarray, h_dim: int, w_dim: int, coords: np.ndarray) -> np.ndarray:
    mask_indices = np.flatnonzero(valid_mask)
    n_reduced = int(coords.shape[1])
    if mask_indices.size < n_reduced:
        raise ValueError(
            f"Aggregate mask has fewer points than reduced traj data: mask={mask_indices.size}, reduced={n_reduced}"
        )
    if mask_indices.size == n_reduced:
        return mask_indices

    # Reconstruct full-grid coords and align masked positions to reduced traj ordering.
    full_coords = _build_coords(h_dim, w_dim).reshape(-1, 2).astype(np.float16)
    masked_coords = full_coords[mask_indices]
    reduced_xy = coords[0, :, :2].astype(np.float16, copy=False)

    coord_to_flat = {}
    for idx, xy in zip(mask_indices, masked_coords):
        key = (float(xy[0]), float(xy[1]))
        coord_to_flat[key] = int(idx)

    scatter_indices = np.empty(n_reduced, dtype=np.int64)
    for j, xy in enumerate(reduced_xy):
        key = (float(xy[0]), float(xy[1]))
        flat_idx = coord_to_flat.get(key)
        if flat_idx is None:
            raise ValueError(
                "Could not map reduced trajectory coords onto aggregate mask. "
                f"Missing coord {key} at reduced index {j}."
            )
        scatter_indices[j] = flat_idx
    return scatter_indices


def _resolve_checkpoint_paths(opt):
    if opt.checkpoint_epoch is not None:
        dyn_ckpt = opt.base_path / opt.model_path / f"dyn_{opt.checkpoint_epoch}.ckpt"
        rec_ckpt = opt.base_path / opt.model_path / f"rec_{opt.checkpoint_epoch}.ckpt"
        B_ckpt = opt.base_path / opt.model_path / f"B_{opt.checkpoint_epoch}.ckpt"
    else:
        dyn_ckpt = opt.dyn_checkpoint
        rec_ckpt = opt.rec_checkpoint
        B_ckpt = None
    return dyn_ckpt, rec_ckpt, B_ckpt


def _extract_ckpt_epoch(path: Path, prefix: str) -> int | None:
    match = re.fullmatch(rf"{prefix}_(\d+)\.ckpt", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _latest_epoch_for_prefix(save_path: Path, prefix: str) -> int | None:
    epochs = {
        epoch
        for ckpt in save_path.glob(f"{prefix}_*.ckpt")
        if (epoch := _extract_ckpt_epoch(ckpt, prefix)) is not None
    }
    return max(epochs) if epochs else None


def _latest_complete_finetune_epoch(save_path: Path) -> int | None:
    rec_epochs = {
        epoch
        for ckpt in save_path.glob("rec_finetune_*.ckpt")
        if (epoch := _extract_ckpt_epoch(ckpt, "rec_finetune")) is not None
    }
    b_epochs = {
        epoch
        for ckpt in save_path.glob("B_finetune_*.ckpt")
        if (epoch := _extract_ckpt_epoch(ckpt, "B_finetune")) is not None
    }
    complete = sorted(rec_epochs.intersection(b_epochs))
    return complete[-1] if complete else None


def _resolve_finetune_dir(opt) -> Path:
    finetune_dir = opt.finetune_path
    if finetune_dir is None:
        finetune_dir = opt.model_path
    if not finetune_dir.is_absolute():
        finetune_dir = opt.base_path / finetune_dir
    return finetune_dir


def _resolve_finetune_checkpoint_paths(opt) -> tuple[Path | None, Path | None]:
    if not opt.rec_finetune:
        return None, None

    finetune_dir = _resolve_finetune_dir(opt)
    latest_epoch = _latest_complete_finetune_epoch(finetune_dir)
    if latest_epoch is None:
        raise FileNotFoundError(
            f"No matching rec_finetune/B_finetune checkpoint pairs found under {finetune_dir}"
        )
    rec_ckpt = finetune_dir / f"rec_finetune_{latest_epoch}.ckpt"
    B_ckpt = finetune_dir / f"B_finetune_{latest_epoch}.ckpt"
    return rec_ckpt, B_ckpt


def _print_flood_inundation_confusion(
    pred_depth: np.ndarray,
    true_depth: np.ndarray,
    threshold: float,
) -> None:
    pred_positive = np.any(pred_depth >= threshold, axis=0)
    true_positive = np.any(true_depth >= threshold, axis=0)

    tp = int(np.sum(pred_positive & true_positive))
    fp = int(np.sum(pred_positive & ~true_positive))
    tn = int(np.sum(~pred_positive & ~true_positive))
    fn = int(np.sum(~pred_positive & true_positive))
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")

    print(f"Flood inundation event test (height >= {threshold:.3f}, positive if any timestep exceeds threshold):")
    print(f"  TP: {tp}")
    print(f"  FP: {fp}")
    print(f"  TN: {tn}")
    print(f"  FN: {fn}")
    print(f"  Accuracy: {accuracy:.6f}")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall: {recall:.6f}")
    print(f"  IoU: {iou:.6f}")
    print(f"  F1: {f1:.6f}")


def _build_flood_inundation_labels(
    pred_depth: np.ndarray,
    true_depth: np.ndarray,
    threshold: float,
) -> np.ndarray:
    pred_positive = np.any(pred_depth >= threshold, axis=0)
    true_positive = np.any(true_depth >= threshold, axis=0)

    labels = np.zeros(pred_positive.shape, dtype=np.int8)
    labels[pred_positive & true_positive] = 0  # TP
    labels[~pred_positive & ~true_positive] = 1  # TN
    labels[pred_positive & ~true_positive] = 2  # FP
    labels[~pred_positive & true_positive] = 3  # FN
    return labels


def save_flood_inundation_map(
    pred_full: np.ndarray,
    true_full: np.ndarray,
    output_path: Path,
    threshold: float,
    valid_mask: np.ndarray | None = None,
) -> None:
    labels = _build_flood_inundation_labels(
        pred_full[:, :, :, 0],
        true_full[:, :, :, 0],
        threshold,
    )

    if valid_mask is not None:
        mask_2d = valid_mask.astype(bool, copy=False)
        labels = np.ma.array(labels, mask=~mask_2d)
    else:
        labels = np.ma.array(labels)

    cmap = ListedColormap(["green", "blue", "red", "orange"])
    cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(labels, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(f"Flood Inundation Event Map (h >= {threshold:.3f})")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    legend_labels = ["True Positive", "True Negative", "False Positive", "False Negative"]
    legend_colors = ["green", "blue", "red", "orange"]
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="None", markersize=10,
                   markerfacecolor=color, markeredgecolor=color, label=label)
        for color, label in zip(legend_colors, legend_labels)
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def save_peak_inundation_qualitative_figure(
    pred_full: np.ndarray,
    true_full: np.ndarray,
    output_path: Path,
    threshold: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    pred_depth = pred_full[:, :, :, 0]
    true_depth = true_full[:, :, :, 0]

    true_peak = np.max(true_depth, axis=0)
    pred_peak = np.max(pred_depth, axis=0)
    diff_peak = np.abs(pred_peak - true_peak)

    if valid_mask is not None:
        mask_2d = valid_mask.astype(bool, copy=False)
        true_peak = np.ma.array(true_peak, mask=~mask_2d)
        pred_peak = np.ma.array(pred_peak, mask=~mask_2d)
        diff_peak = np.ma.array(diff_peak, mask=~mask_2d)

    finite_max = []
    for arr in (true_peak, pred_peak, diff_peak):
        values = np.asarray(arr.compressed() if np.ma.isMaskedArray(arr) else arr[np.isfinite(arr)], dtype=np.float32)
        if values.size > 0:
            finite_max.append(float(np.max(values)))
    vmax = max(finite_max) if finite_max else 0.1
    vmax = 0.5 * max(vmax, threshold, 0.1)

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 1.0))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 6), constrained_layout=True)
    panels = [
        (true_peak, "Reference Water Depth"),
        (pred_peak, "CLDNet Water Depth"),
        (diff_peak, "Absolute Difference"),
    ]

    ims = []
    for ax, (panel, title) in zip(axes, panels):
        im = ax.imshow(panel, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal")
        ims.append(im)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    peak_true_vals, peak_pred_vals = _paired_valid_values(true_peak, pred_peak)
    peak_r2 = _r2_score(peak_true_vals, peak_pred_vals)
    peak_rmse = _rmse_score(peak_true_vals, peak_pred_vals)

    cbar = fig.colorbar(ims[0], ax=axes, orientation="horizontal", pad=0.06, shrink=0.95)
    cbar.set_label("Water Depth h (m)")
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    return peak_r2, peak_rmse


def _finite_panel_max(arrays: list[np.ndarray | np.ma.MaskedArray], fallback: float) -> float:
    finite_max = []
    for arr in arrays:
        if np.ma.isMaskedArray(arr):
            values = np.asarray(arr.compressed(), dtype=np.float32)
        else:
            arr_np = np.asarray(arr, dtype=np.float32)
            values = arr_np[np.isfinite(arr_np)]
        if values.size > 0:
            finite_max.append(float(np.max(values)))
    vmax = max(finite_max) if finite_max else fallback
    return max(vmax, fallback)


def _select_flood_edge_timesteps(
    true_full: np.ndarray,
    threshold: float,
    valid_mask: np.ndarray | None = None,
) -> dict[str, int]:
    true_depth = true_full[:, :, :, 0]
    if valid_mask is not None:
        mask = valid_mask.astype(bool, copy=False)
        true_depth_valid = true_depth[:, mask]
    else:
        true_depth_valid = true_depth.reshape(true_depth.shape[0], -1)

    aggregate_depth = np.sum(np.clip(true_depth_valid, a_min=0.0, a_max=None), axis=1)
    wet_counts = np.sum(true_depth_valid >= threshold, axis=1)

    if not np.any(np.isfinite(aggregate_depth)):
        raise ValueError("Could not determine flood timesteps because aggregate depth is not finite.")

    peak_idx = int(np.nanargmax(aggregate_depth))
    active = np.flatnonzero(wet_counts > 0)
    if active.size == 0:
        active = np.flatnonzero(aggregate_depth > 0)
    if active.size == 0:
        return {"rising_mid": peak_idx, "peak": peak_idx, "falling_mid": peak_idx}

    start_idx = int(active[0])
    end_idx = int(active[-1])

    rising_end = max(peak_idx, start_idx)
    falling_start = min(peak_idx, end_idx)

    rising_segment = np.arange(start_idx, rising_end + 1, dtype=np.int64)
    falling_segment = np.arange(falling_start, end_idx + 1, dtype=np.int64)

    rising_target = 0.5 * (aggregate_depth[start_idx] + aggregate_depth[peak_idx])
    # Pick a later recession snapshot, closer to the end of the falling limb.
    falling_target = aggregate_depth[end_idx] + 0.25 * (aggregate_depth[peak_idx] - aggregate_depth[end_idx])

    if rising_segment.size > 0:
        rising_mid = int(rising_segment[np.argmin(np.abs(aggregate_depth[rising_segment] - rising_target))])
    else:
        rising_mid = peak_idx

    if falling_segment.size > 0:
        falling_mid = int(falling_segment[np.argmin(np.abs(aggregate_depth[falling_segment] - falling_target))])
    else:
        falling_mid = peak_idx

    return {"rising_mid": rising_mid, "peak": peak_idx, "falling_mid": falling_mid}


def save_flood_stage_snapshot_figure(
    pred_full: np.ndarray,
    true_full: np.ndarray,
    output_path: Path,
    threshold: float,
    valid_mask: np.ndarray | None = None,
    is_texas: bool = False,
) -> dict[str, int]:
    stage_timesteps = _select_flood_edge_timesteps(
        true_full,
        threshold=threshold,
        valid_mask=valid_mask,
    )

    stage_titles = [
        ("rising_mid", "Rising"),
        ("peak", "Peak"),
        ("falling_mid", "Recession"),
    ]
    panel_labels = ["reference", "CLDNet", "error"]
    panel_letters = "abcdefghi"

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 1.0))

    panels_depth = []
    panels_error = []
    for key, _title in stage_titles:
        t_idx = stage_timesteps[key]
        true_panel = true_full[t_idx, :, :, 0]
        pred_panel = pred_full[t_idx, :, :, 0]
        err_panel = np.abs(pred_panel - true_panel)
        if valid_mask is not None:
            mask_2d = valid_mask.astype(bool, copy=False)
            true_panel = np.ma.array(true_panel, mask=~mask_2d)
            pred_panel = np.ma.array(pred_panel, mask=~mask_2d)
            err_panel = np.ma.array(err_panel, mask=~mask_2d)
        panels_depth.extend([true_panel, pred_panel])
        panels_error.append(err_panel)

    depth_vmax = 0.5 * _finite_panel_max(panels_depth, fallback=max(threshold, 0.1))
    error_vmax = depth_vmax

    fig_width = 10 if is_texas else 6
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(fig_width, 12), constrained_layout=True)

    depth_im = None
    for row_idx, (key, stage_label) in enumerate(stage_titles):
        t_idx = stage_timesteps[key]
        true_panel = true_full[t_idx, :, :, 0]
        pred_panel = pred_full[t_idx, :, :, 0]
        err_panel = np.abs(pred_panel - true_panel)
        if valid_mask is not None:
            mask_2d = valid_mask.astype(bool, copy=False)
            true_panel = np.ma.array(true_panel, mask=~mask_2d)
            pred_panel = np.ma.array(pred_panel, mask=~mask_2d)
            err_panel = np.ma.array(err_panel, mask=~mask_2d)

        depth_im = axes[row_idx, 0].imshow(
            true_panel, cmap=cmap, vmin=0.0, vmax=depth_vmax, interpolation="nearest", aspect="equal"
        )
        axes[row_idx, 1].imshow(
            pred_panel, cmap=cmap, vmin=0.0, vmax=depth_vmax, interpolation="nearest", aspect="equal"
        )
        axes[row_idx, 2].imshow(
            err_panel, cmap=cmap, vmin=0.0, vmax=error_vmax, interpolation="nearest", aspect="equal"
        )

        for col_idx in range(3):
            panel_idx = row_idx * 3 + col_idx
            axes[row_idx, col_idx].set_xlabel(f"({panel_letters[panel_idx]}) {stage_label} — {panel_labels[col_idx]}")
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    depth_cbar = fig.colorbar(depth_im, ax=axes, orientation="horizontal", pad=0.04, shrink=0.95)
    depth_cbar.set_label("Water Depth / Absolute Error (m)")
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    return stage_timesteps


def _collect_snapshot_depth_pairs(
    pred_full: np.ndarray,
    true_full: np.ndarray,
    snapshot_timesteps: list[tuple[str, int]],
    valid_mask: np.ndarray | None = None,
) -> tuple[list[tuple[str, int, np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    pred_depth = pred_full[:, :, :, 0]
    true_depth = true_full[:, :, :, 0]
    mask_2d = valid_mask.astype(bool, copy=False) if valid_mask is not None else None

    snapshot_pairs: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    ref_parts = []
    pred_parts = []

    for label, t_idx in snapshot_timesteps:
        ref_depth = true_depth[int(t_idx)]
        cldnet_depth = pred_depth[int(t_idx)]
        if mask_2d is not None:
            ref_depth = ref_depth[mask_2d]
            cldnet_depth = cldnet_depth[mask_2d]

        ref_depth = np.asarray(ref_depth, dtype=np.float32).reshape(-1)
        cldnet_depth = np.asarray(cldnet_depth, dtype=np.float32).reshape(-1)
        finite = np.isfinite(ref_depth) & np.isfinite(cldnet_depth)
        ref_depth = ref_depth[finite]
        cldnet_depth = cldnet_depth[finite]

        snapshot_pairs.append((label, int(t_idx), ref_depth, cldnet_depth))
        if ref_depth.size > 0:
            ref_parts.append(ref_depth)
            pred_parts.append(cldnet_depth)

    combined_ref = np.concatenate(ref_parts) if ref_parts else np.empty(0, dtype=np.float32)
    combined_pred = np.concatenate(pred_parts) if pred_parts else np.empty(0, dtype=np.float32)
    return snapshot_pairs, combined_ref, combined_pred


def _print_snapshot_depth_metrics(
    snapshot_pairs: list[tuple[str, int, np.ndarray, np.ndarray]],
    combined_ref: np.ndarray,
    combined_pred: np.ndarray,
) -> tuple[float, float]:
    if combined_ref.size == 0 or combined_pred.size == 0:
        raise ValueError("No finite samples available for snapshot depth metrics.")

    print("Snapshot depth metrics (Reference vs CLDNet):")
    for label, t_idx, ref_depth, cldnet_depth in snapshot_pairs:
        if ref_depth.size == 0:
            print(f"  {label} (t={t_idx}): no finite samples")
            continue
        r2 = _r2_score(ref_depth, cldnet_depth)
        rmse = _rmse_score(ref_depth, cldnet_depth)
        print(f"  {label} (t={t_idx}): R^2={r2:.6f}, RMSE={rmse:.6f}, N={ref_depth.size}")

    combined_r2 = _r2_score(combined_ref, combined_pred)
    combined_rmse = _rmse_score(combined_ref, combined_pred)
    print(f"  combined: R^2={combined_r2:.6f}, RMSE={combined_rmse:.6f}, N={combined_ref.size}")
    return combined_r2, combined_rmse


def save_reference_vs_cldnet_depth_scatter(
    snapshot_pairs: list[tuple[str, int, np.ndarray, np.ndarray]],
    combined_ref: np.ndarray,
    combined_pred: np.ndarray,
    output_path: Path,
) -> None:
    if combined_ref.size == 0 or combined_pred.size == 0:
        raise ValueError("No finite samples available for the snapshot depth scatter plot.")

    finite_values = np.concatenate([combined_ref, combined_pred])
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("No finite samples available for the snapshot depth scatter plot.")

    depth_limit = max(6.5, float(np.nanpercentile(finite_values, 99.5)))
    overall_r2 = _r2_score(combined_ref, combined_pred)
    overall_rmse = _rmse_score(combined_ref, combined_pred)

    fig, ax = plt.subplots(figsize=(6.5, 6.5), constrained_layout=True)
    ax.scatter(
        combined_ref,
        combined_pred,
        s=8,
        alpha=0.22,
        color="tab:blue",
        edgecolors="none",
        label="Snapshot points",
    )

    ax.plot([0.0, depth_limit], [0.0, depth_limit], color="black", linestyle="--", linewidth=1.5, label="1:1 line")
    ax.set_xlim(0.0, depth_limit)
    ax.set_ylim(0.0, depth_limit)
    ax.set_aspect("equal", "box")
    ax.set_title("Reference vs CLDNet Snapshot Depth Parity")
    ax.set_xlabel("Reference Water Depth h (m)")
    ax.set_ylabel("CLDNet Water Depth h (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.text(
        0.04,
        0.96,
        f"R^2={overall_r2:.4f}\nRMSE={overall_rmse:.4f}\nN={combined_ref.size}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_error_vs_depth_diagnostic(
    pred_full: np.ndarray,
    true_full: np.ndarray,
    output_path: Path,
    valid_mask: np.ndarray | None = None,
    num_bins: int = 30,
) -> None:
    pred_depth = pred_full[:, :, :, 0]
    true_depth = true_full[:, :, :, 0]
    abs_error = np.abs(pred_depth - true_depth)

    if valid_mask is not None:
        mask_2d = valid_mask.astype(bool, copy=False)
        pred_depth = pred_depth[:, mask_2d]
        true_depth = true_depth[:, mask_2d]
        abs_error = abs_error[:, mask_2d]

    ref_depth = true_depth.reshape(-1)
    abs_error = abs_error.reshape(-1)
    finite = np.isfinite(ref_depth) & np.isfinite(abs_error)
    ref_depth = ref_depth[finite]
    abs_error = abs_error[finite]

    if ref_depth.size == 0:
        raise ValueError("No finite samples available for error-vs-depth diagnostic.")

    depth_max = 6.5
    err_max = 2.5

    in_range = (
        (ref_depth >= 0.0)
        & (ref_depth <= depth_max)
        & (abs_error >= 0.0)
        & (abs_error <= err_max)
    )
    ref_depth = ref_depth[in_range]
    abs_error = abs_error[in_range]

    if ref_depth.size == 0:
        raise ValueError("No finite in-range samples available for error-vs-depth diagnostic.")

    depth_edges = np.linspace(0.0, depth_max, num_bins + 1, dtype=np.float32)
    err_edges = np.linspace(0.0, err_max, num_bins + 1, dtype=np.float32)
    hist2d, _, _ = np.histogram2d(
        ref_depth,
        abs_error,
        bins=[depth_edges, err_edges],
    )
    x_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    y_centers = 0.5 * (err_edges[:-1] + err_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    occupied = hist2d > 0

    fig, ax_hist = plt.subplots(figsize=(7, 5), constrained_layout=True)

    occupied_counts = hist2d[occupied]
    scatter = ax_hist.scatter(
        xx[occupied],
        yy[occupied],
        c=occupied_counts,
        s=42,
        cmap="Blues",
        norm=LogNorm(vmin=float(np.min(occupied_counts)), vmax=float(np.max(occupied_counts))),
        marker="s",
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax_hist, pad=0.02)
    cbar.set_label("Sample Count")
    ax_hist.set_title("Absolute Error vs Reference Depth")
    ax_hist.set_xlabel("Reference Depth h (m)")
    ax_hist.set_ylabel("Absolute Error |pred - truth| (m)")
    ax_hist.set_xlim(0.0, 6.5)
    ax_hist.set_ylim(0.0, err_max)
    ax_hist.grid(True, alpha=0.25)

    fig.suptitle("Depth-Conditioned CLDNet Error Diagnostic", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _compute_error_vs_depth_summary(
    pred_full: np.ndarray,
    true_full: np.ndarray,
    valid_mask: np.ndarray | None = None,
    num_bins: int = 30,
    depth_max: float = 6.5,
    err_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_depth = pred_full[:, :, :, 0]
    true_depth = true_full[:, :, :, 0]
    abs_error = np.abs(pred_depth - true_depth)

    if valid_mask is not None:
        mask_2d = valid_mask.astype(bool, copy=False)
        true_depth = true_depth[:, mask_2d]
        abs_error = abs_error[:, mask_2d]

    ref_depth = true_depth.reshape(-1)
    abs_error = abs_error.reshape(-1)
    finite = np.isfinite(ref_depth) & np.isfinite(abs_error)
    ref_depth = ref_depth[finite]
    abs_error = abs_error[finite]

    if ref_depth.size == 0:
        raise ValueError("No finite samples available for error-vs-depth IQR diagnostic.")

    in_range = (
        (ref_depth >= 0.0)
        & (ref_depth <= depth_max)
        & (abs_error >= 0.0)
        & (abs_error <= err_max)
    )
    ref_depth = ref_depth[in_range]
    abs_error = abs_error[in_range]

    if ref_depth.size == 0:
        raise ValueError("No finite in-range samples available for error-vs-depth IQR diagnostic.")

    depth_edges = np.linspace(0.0, depth_max, num_bins + 1, dtype=np.float32)
    bin_ids = np.digitize(ref_depth, depth_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)
    x_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])

    median_error = np.full(num_bins, np.nan, dtype=np.float32)
    q1_error = np.full(num_bins, np.nan, dtype=np.float32)
    q3_error = np.full(num_bins, np.nan, dtype=np.float32)
    counts = np.zeros(num_bins, dtype=np.int32)

    for bin_idx in range(num_bins):
        in_bin = bin_ids == bin_idx
        if not np.any(in_bin):
            continue
        bin_errors = abs_error[in_bin]
        counts[bin_idx] = int(bin_errors.size)
        q1_error[bin_idx], median_error[bin_idx], q3_error[bin_idx] = np.percentile(
            bin_errors,
            [25.0, 50.0, 75.0],
        )

    valid_bins = counts > 0
    if not np.any(valid_bins):
        raise ValueError("No populated bins available for error-vs-depth IQR diagnostic.")

    return x_centers, median_error, q1_error, q3_error, valid_bins


def save_error_vs_depth_iqr_diagnostic(
    pred_full: np.ndarray,
    true_full: np.ndarray,
    output_path: Path,
    valid_mask: np.ndarray | None = None,
    num_bins: int = 30,
) -> None:
    depth_max = 6.5
    err_max = 1.0
    x_centers, median_error, q1_error, q3_error, valid_bins = _compute_error_vs_depth_summary(
        pred_full,
        true_full,
        valid_mask=valid_mask,
        num_bins=num_bins,
        depth_max=depth_max,
        err_max=err_max,
    )

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.fill_between(
        x_centers,
        q1_error,
        q3_error,
        where=valid_bins,
        color="tab:orange",
        alpha=0.25,
        label="Interquartile Range",
    )
    ax.plot(
        x_centers[valid_bins],
        median_error[valid_bins],
        color="tab:orange",
        linewidth=2.0,
        label="Median Absolute Error",
    )
    ax.scatter(
        x_centers[valid_bins],
        median_error[valid_bins],
        color="tab:orange",
        s=28,
        edgecolors="black",
        linewidths=0.3,
        zorder=3,
    )

    ax.set_title("Absolute Error Summary vs Reference Depth")
    ax.set_xlabel("Reference Depth h (m)")
    ax.set_ylabel("Absolute Error |pred - truth| (m)")
    ax.set_xlim(0.0, depth_max)
    ax.set_ylim(0.0, err_max)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    fig.suptitle("Depth-Conditioned CLDNet Error Median and IQR", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_error_vs_depth_overlay_diagnostic(
    pred_full: np.ndarray,
    true_full: np.ndarray,
    output_path: Path,
    valid_mask: np.ndarray | None = None,
    num_bins: int = 30,
) -> None:
    pred_depth = pred_full[:, :, :, 0]
    true_depth = true_full[:, :, :, 0]
    abs_error = np.abs(pred_depth - true_depth)

    if valid_mask is not None:
        mask_2d = valid_mask.astype(bool, copy=False)
        true_depth = true_depth[:, mask_2d]
        abs_error = abs_error[:, mask_2d]

    ref_depth = true_depth.reshape(-1)
    abs_error = abs_error.reshape(-1)
    finite = np.isfinite(ref_depth) & np.isfinite(abs_error)
    ref_depth = ref_depth[finite]
    abs_error = abs_error[finite]

    if ref_depth.size == 0:
        raise ValueError("No finite samples available for overlaid error-vs-depth diagnostic.")

    depth_max = 6.5
    scatter_err_max = 2.5
    summary_err_max = 1.0

    in_range = (
        (ref_depth >= 0.0)
        & (ref_depth <= depth_max)
        & (abs_error >= 0.0)
        & (abs_error <= scatter_err_max)
    )
    ref_depth = ref_depth[in_range]
    abs_error = abs_error[in_range]

    if ref_depth.size == 0:
        raise ValueError("No finite in-range samples available for overlaid error-vs-depth diagnostic.")

    depth_edges = np.linspace(0.0, depth_max, num_bins + 1, dtype=np.float32)
    err_edges = np.linspace(0.0, scatter_err_max, num_bins + 1, dtype=np.float32)
    hist2d, _, _ = np.histogram2d(ref_depth, abs_error, bins=[depth_edges, err_edges])
    x_centers_hist = 0.5 * (depth_edges[:-1] + depth_edges[1:])
    y_centers = 0.5 * (err_edges[:-1] + err_edges[1:])
    xx, yy = np.meshgrid(x_centers_hist, y_centers, indexing="ij")
    occupied = hist2d > 0

    x_centers_summary, median_error, q1_error, q3_error, valid_bins = _compute_error_vs_depth_summary(
        pred_full,
        true_full,
        valid_mask=valid_mask,
        num_bins=num_bins,
        depth_max=depth_max,
        err_max=summary_err_max,
    )

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    occupied_counts = hist2d[occupied]
    scatter = ax.scatter(
        xx[occupied],
        yy[occupied],
        c=occupied_counts,
        s=42,
        cmap="Blues",
        norm=LogNorm(vmin=float(np.min(occupied_counts)), vmax=float(np.max(occupied_counts))),
        marker="s",
        edgecolors="none",
        alpha=0.8,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Sample Count (log-scaled)")

    ax.fill_between(
        x_centers_summary,
        q1_error,
        q3_error,
        where=valid_bins,
        color="tab:orange",
        alpha=0.28,
        label="Interquartile Range",
        zorder=2,
    )
    ax.plot(
        x_centers_summary[valid_bins],
        median_error[valid_bins],
        color="tab:orange",
        linewidth=2.0,
        label="Median Absolute Error",
        zorder=3,
    )
    ax.scatter(
        x_centers_summary[valid_bins],
        median_error[valid_bins],
        color="tab:orange",
        s=28,
        edgecolors="black",
        linewidths=0.3,
        zorder=4,
    )

    ax.set_title("Absolute Error vs Reference Depth")
    ax.set_xlabel("Reference Depth h (m)")
    ax.set_ylabel("Absolute Error |pred - truth| (m)")
    ax.set_xlim(0.0, depth_max)
    ax.set_ylim(0.0, scatter_err_max)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _compute_per_timestep_csi(
    pred_depth: np.ndarray,
    true_depth: np.ndarray,
    threshold: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    pred_positive = pred_depth >= threshold
    true_positive = true_depth >= threshold

    if valid_mask is not None:
        mask = valid_mask.astype(bool, copy=False)
        pred_positive = pred_positive[:, mask]
        true_positive = true_positive[:, mask]
    else:
        pred_positive = pred_positive.reshape(pred_positive.shape[0], -1)
        true_positive = true_positive.reshape(true_positive.shape[0], -1)

    intersection = np.sum(pred_positive & true_positive, axis=1)
    union = np.sum(pred_positive | true_positive, axis=1)
    csi = np.full(intersection.shape, np.nan, dtype=np.float32)
    valid = union > 0
    csi[valid] = intersection[valid] / union[valid]
    return csi


def save_per_timestep_csi_curve(
    pred_full: np.ndarray,
    true_full: np.ndarray,
    output_path: Path,
    threshold: float,
    valid_mask: np.ndarray | None = None,
    rain_series: np.ndarray | None = None,
    is_texas: bool = False,
    is_illinois: bool = False,
) -> np.ndarray:
    csi = _compute_per_timestep_csi(
        pred_full[:, :, :, 0],
        true_full[:, :, :, 0],
        threshold,
        valid_mask=valid_mask,
    )

    time = np.arange(csi.shape[0], dtype=np.int32)
    title_fontsize = 22
    label_fontsize = 20
    tick_fontsize = 18
    csi_ylim = (0.6, 1.0) if is_illinois else (0.0, 1.0)
    if rain_series is not None:
        rain_series = np.asarray(rain_series, dtype=np.float32).reshape(-1)
        if rain_series.shape[0] != csi.shape[0]:
            raise ValueError(
                f"Rain series length {rain_series.shape[0]} does not match timestep count {csi.shape[0]}"
            )
        rain_ylabel = r"Rain ($\mathrm{mm/h}$)"
        if is_texas:
            rain_series = rain_series * 1000.0 * 3600.0
            rain_ylabel = r"Rain ($\mathrm{mm/h}$)"

        fig, (ax_csi, ax_rain) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax_csi.plot(time, csi, color="tab:green", linewidth=1.8)
        ax_csi.set_ylim(*csi_ylim)
        ax_csi.set_ylabel("CSI", fontsize=label_fontsize)
        ax_csi.set_title(
            f"Per-Timestep Flood CSI (h >= {threshold:.3f})",
            fontsize=title_fontsize,
        )
        ax_csi.tick_params(axis="both", labelsize=tick_fontsize)
        ax_csi.grid(True, alpha=0.3)

        ax_rain.plot(time, rain_series, color="tab:blue", linewidth=1.5)
        ax_rain.fill_between(time, 0.0, rain_series, color="tab:blue", alpha=0.2)
        ax_rain.set_xlabel("Timestep", fontsize=label_fontsize)
        ax_rain.set_ylabel(rain_ylabel, fontsize=label_fontsize)
        ax_rain.tick_params(axis="both", labelsize=tick_fontsize)
        ax_rain.grid(True, alpha=0.3)
    else:
        fig, ax_csi = plt.subplots(figsize=(10, 4))
        ax_csi.plot(time, csi, color="tab:green", linewidth=1.8)
        ax_csi.set_ylim(*csi_ylim)
        ax_csi.set_xlabel("Timestep", fontsize=label_fontsize)
        ax_csi.set_ylabel("CSI", fontsize=label_fontsize)
        ax_csi.set_title(
            f"Per-Timestep Flood CSI (h >= {threshold:.3f})",
            fontsize=title_fontsize,
        )
        ax_csi.tick_params(axis="both", labelsize=tick_fontsize)
        ax_csi.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return csi


def _infer_rec_output_dim(rec_ckpt: Path) -> int:
    state_dict = torch.load(rec_ckpt, map_location="cpu")
    candidates = []
    for key, value in state_dict.items():
        match = re.fullmatch(r"net\.(\d+)\.weight", key)
        if match is not None and torch.is_tensor(value) and value.ndim == 2:
            candidates.append((int(match.group(1)), int(value.shape[0])))
    if not candidates:
        raise ValueError(f"Could not infer reconstruction output dimension from {rec_ckpt}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _load_checkpoints(model, opt):
    dyn_ckpt, rec_ckpt, B_ckpt = _resolve_checkpoint_paths(opt)
    finetune_rec_ckpt, finetune_B_ckpt = _resolve_finetune_checkpoint_paths(opt)
    if finetune_rec_ckpt is not None:
        rec_ckpt = finetune_rec_ckpt
    if finetune_B_ckpt is not None:
        B_ckpt = finetune_B_ckpt

    if dyn_ckpt is None or rec_ckpt is None:
        print("No checkpoints provided; using random weights.")
        return

    print(f"Loading dyn checkpoint: {dyn_ckpt}")
    print(f"Loading rec checkpoint: {rec_ckpt}")
    model.dyn.load_state_dict(torch.load(dyn_ckpt, map_location="cpu"))
    model.rec.load_state_dict(torch.load(rec_ckpt, map_location="cpu"))

    # Load Fourier embedding B if available
    if B_ckpt is not None and B_ckpt.exists() and hasattr(model, 'B'):
        print(f"Loading B checkpoint: {B_ckpt}")
        model.B.load_state_dict(torch.load(B_ckpt, map_location="cpu"))
    elif hasattr(model, 'B'):
        print("WARNING: B checkpoint not found - Fourier embedding will use random weights!")


def _select_y_rows(h_dim: int):
    y_list_left = [1000, 2000, 3000, 4000]
    y_list_right = [1500, 2500, 3500, 4500]
    if max(y_list_left + y_list_right) < h_dim:
        return y_list_left, y_list_right

    left = (np.linspace(0.2, 0.8, 4) * (h_dim - 1)).astype(int).tolist()
    right = (np.linspace(0.25, 0.85, 4) * (h_dim - 1)).astype(int).tolist()
    return left, right


def _select_points(depth_truth: np.ndarray):
    t_len, h_dim, w_dim = depth_truth.shape
    y_list_left, y_list_right = _select_y_rows(h_dim)

    points_left = []
    for y in y_list_left:
        row_tw = depth_truth[:, y, :]
        row_tw_safe = np.where(np.isfinite(row_tw), row_tw, -np.inf)
        max_over_t = row_tw_safe.max(axis=0)
        if np.all(max_over_t == -np.inf):
            raise ValueError(f"Row y={y} is all-NaN across time; cannot select x")
        x = int(max_over_t.argmax())
        peak = float(max_over_t[x])
        points_left.append((y, x, peak))

    points_right = []
    for y in y_list_right:
        row_tw = depth_truth[:, y, :]
        finite = np.isfinite(row_tw)
        diffs = np.abs(np.diff(row_tw, axis=0))
        valid_pairs = finite[1:, :] & finite[:-1, :]
        diffs = np.where(valid_pairs, diffs, 0.0)
        tv = diffs.sum(axis=0)
        if not np.isfinite(tv).any() or np.all(tv == 0):
            raise ValueError(f"Row y={y} has no finite temporal variation; cannot select x")
        x = int(tv.argmax())
        points_right.append((y, x, float(tv[x])))

    return points_left, points_right


def plot_hydrographs(pred_full: np.ndarray, true_full: np.ndarray, output_path: Path):
    depth_pred = pred_full[:, :, :, 0]
    depth_true = true_full[:, :, :, 0]
    t_len = depth_true.shape[0]

    points_left, points_right = _select_points(depth_true)

    def _point_rel(y, x):
        true_ts = depth_true[:, y, x]
        pred_ts = depth_pred[:, y, x]
        denom = np.sum(true_ts ** 2)
        if denom == 0:
            return float("nan")
        return float(np.sqrt(np.sum((pred_ts - true_ts) ** 2) / denom))

    for i, (y, x, peak) in enumerate(points_left):
        print(f"Hydrograph LEFT {i}: y={y}, x={x}, peak={peak:.2f}, rel_l2={_point_rel(y, x):.4f}")
    for i, (y, x, tv) in enumerate(points_right):
        print(f"Hydrograph RIGHT {i}: y={y}, x={x}, tv={tv:.2f}, rel_l2={_point_rel(y, x):.4f}")

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 10), sharex=True)
    time = np.arange(t_len)
    for i, (y, x, peak) in enumerate(points_left):
        ax = axes[i, 0]
        ax.plot(time, depth_true[:, y, x], color="0.2", linewidth=1.5, label="truth")
        ax.plot(time, depth_pred[:, y, x], color="tab:blue", linewidth=1.5, label="pred")
        ax.set_title(f"Peak: y={y}, x={x} (peak {peak:.2f})")
        ax.grid(True, alpha=0.25)
        if i == 3:
            ax.set_xlabel("Hour")
        ax.set_ylabel("h (m)")

    for i, (y, x, tv) in enumerate(points_right):
        ax = axes[i, 1]
        ax.plot(time, depth_true[:, y, x], color="0.2", linewidth=1.5, label="truth")
        ax.plot(time, depth_pred[:, y, x], color="tab:orange", linewidth=1.5, label="pred")
        ax.set_title(f"TV: y={y}, x={x} (tv {tv:.2f})")
        ax.grid(True, alpha=0.25)
        if i == 3:
            ax.set_xlabel("Hour")
        ax.set_ylabel("h (m)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _estimate_limits(frames, percentile=99.9, is_signed=False):
    vals = []
    for frame in frames:
        finite = np.isfinite(frame)
        if np.any(finite):
            vals.append(frame[finite])
    if not vals:
        return (0.0, 1.0)
    vals = np.concatenate(vals, axis=0)
    if is_signed:
        vmax = float(np.nanpercentile(np.abs(vals), percentile))
        vmax = max(vmax, 0.1)
        return (-vmax, vmax)
    vmax = float(np.nanpercentile(vals, percentile))
    vmax = max(vmax, 0.1)
    return (0.0, vmax)


def create_depth_traces_animation(
    pred_full: np.ndarray,
    true_full: np.ndarray,
    output_path: Path,
    fps: int = 5,
    interval: int = 1,
    percentile: float = 99.9,
):
    depth_pred = pred_full[:, :, :, 0]
    depth_true = true_full[:, :, :, 0]
    t_len, h_dim, w_dim = depth_true.shape

    points_left, points_right = _select_points(depth_true)

    traces_left_true = []
    traces_left_pred = []
    for y, x, _peak in points_left:
        traces_left_true.append(depth_true[:, y, x].astype(np.float32))
        traces_left_pred.append(depth_pred[:, y, x].astype(np.float32))

    traces_right_true = []
    traces_right_pred = []
    for y, x, _tv in points_right:
        traces_right_true.append(depth_true[:, y, x].astype(np.float32))
        traces_right_pred.append(depth_pred[:, y, x].astype(np.float32))

    time_indices = list(range(0, t_len, interval))
    num_frames = len(time_indices)

    n_sample = min(10, len(time_indices))
    sample_ts = np.linspace(time_indices[0], time_indices[-1], n_sample, dtype=int).tolist()
    d_vmin, d_vmax = _estimate_limits([depth_true[t] for t in sample_ts], percentile=percentile)

    all_traces = traces_left_true + traces_left_pred + traces_right_true + traces_right_pred
    all_vals = np.concatenate([tr[np.isfinite(tr)] for tr in all_traces if np.isfinite(tr).any()])
    y_max = float(np.nanpercentile(all_vals, 99.5)) if all_vals.size else 1.0
    y_max = max(y_max, 0.1)
    y_lim = (0.0, 1.1 * y_max)

    aspect_ratio = h_dim / float(w_dim)
    fig_height = opt.height_multiplier * aspect_ratio / 1.5
    fig = plt.figure(figsize=(18, fig_height), constrained_layout=True)
    font_title = 16
    font_label = 13
    font_tick = 12
    line_truth = 2.5
    line_pred = 2.5
    gs = fig.add_gridspec(
        4,
        3,
        width_ratios=[1.35, max(2.8, opt.width_multiplier * 3.0), 1.35],
        wspace=0.02,
    )
    ax_ts_left = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    ax_map = fig.add_subplot(gs[:, 1])
    ax_ts_right = [fig.add_subplot(gs[i, 2]) for i in range(4)]

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color=(0, 0, 0, 0))
    depth_im = ax_map.imshow(depth_true[0], cmap=cmap, vmin=d_vmin, vmax=d_vmax, aspect="equal", alpha=0.9)
    cbar = fig.colorbar(
        depth_im, ax=ax_map, orientation="horizontal", label="Water Depth h (m)", pad=0.03, shrink=0.95
    )
    cbar.set_label("Water Depth h (m)", fontsize=font_label)
    cbar.ax.tick_params(labelsize=font_tick)

    ax_map.set_title("Water Depth (SynxFlow)", fontsize=font_title)
    ax_map.set_xlabel("X coordinate (30m cells)", fontsize=font_label)
    ax_map.set_ylabel("Y coordinate (30m cells)", fontsize=font_label)
    ax_map.tick_params(labelsize=font_tick)
    time_text = ax_map.text(
        0.02,
        0.98,
        "Hour 0",
        transform=ax_map.transAxes,
        va="top",
        ha="left",
        fontsize=font_title,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
    )

    left_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    right_colors = ["tab:purple", "tab:brown", "tab:pink", "tab:cyan"]

    ax_map.scatter(
        [x for (y, x, _p) in points_left],
        [y for (y, x, _p) in points_left],
        s=70,
        c=left_colors,
        marker="o",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )
    ax_map.scatter(
        [x for (y, x, _p) in points_right],
        [y for (y, x, _p) in points_right],
        s=70,
        c=right_colors,
        marker="s",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )

    prog_left_true = []
    prog_left_pred = []
    markers_left_true = []
    markers_left_pred = []
    vlines_left = []
    for i, ax in enumerate(ax_ts_left):
        y, x, peak = points_left[i]
        t = np.arange(t_len)
        ax.plot(t, traces_left_true[i], color="0.2", linewidth=line_truth, label="truth")
        ax.plot(t, traces_left_pred[i], color="tab:blue", linewidth=line_pred, linestyle="--", label="pred")
        (pline_true,) = ax.plot([0], [traces_left_true[i][0]], color="0.2", linewidth=line_prog)
        (pline_pred,) = ax.plot(
            [0], [traces_left_pred[i][0]], color=left_colors[i], linewidth=line_prog, linestyle="--"
        )
        (mk_true,) = ax.plot([0], [traces_left_true[i][0]], marker="o", color="0.2", markersize=5, linestyle="None")
        (mk_pred,) = ax.plot([0], [traces_left_pred[i][0]], marker="o", color=left_colors[i], markersize=5, linestyle="None")
        vl = ax.axvline(0, color=left_colors[i], linewidth=1.0, alpha=0.9)
        ax.set_xlim(0, t_len - 1)
        ax.set_ylim(*y_lim)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("h (m)", fontsize=font_label)
        ax.set_title(f"Peak Depth: y={y}, x={x} (peak {peak:.2f})", fontsize=font_label)
        ax.tick_params(labelsize=font_tick)
        if i == 3:
            ax.set_xlabel("Hour", fontsize=font_label)
        else:
            ax.set_xticklabels([])
        prog_left_true.append(pline_true)
        prog_left_pred.append(pline_pred)
        markers_left_true.append(mk_true)
        markers_left_pred.append(mk_pred)
        vlines_left.append(vl)

    prog_right_true = []
    prog_right_pred = []
    markers_right_true = []
    markers_right_pred = []
    vlines_right = []
    for i, ax in enumerate(ax_ts_right):
        y, x, tv = points_right[i]
        t = np.arange(t_len)
        ax.plot(t, traces_right_true[i], color="0.2", linewidth=line_truth, label="truth")
        ax.plot(t, traces_right_pred[i], color="tab:orange", linewidth=line_pred, linestyle="--", label="pred")
        (pline_true,) = ax.plot([0], [traces_right_true[i][0]], color="0.2", linewidth=line_prog)
        (pline_pred,) = ax.plot(
            [0], [traces_right_pred[i][0]], color=right_colors[i], linewidth=line_prog, linestyle="--"
        )
        (mk_true,) = ax.plot([0], [traces_right_true[i][0]], marker="s", color="0.2", markersize=5, linestyle="None")
        (mk_pred,) = ax.plot([0], [traces_right_pred[i][0]], marker="s", color=right_colors[i], markersize=5, linestyle="None")
        vl = ax.axvline(0, color=right_colors[i], linewidth=1.0, alpha=0.9)
        ax.set_xlim(0, t_len - 1)
        ax.set_ylim(*y_lim)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("h (m)", fontsize=font_label)
        ax.set_title(f"Peak Total Variation: y={y}, x={x} (tv {tv:.2f})", fontsize=font_label)
        ax.tick_params(labelsize=font_tick)
        if i == 3:
            ax.set_xlabel("Hour", fontsize=font_label)
        else:
            ax.set_xticklabels([])
        prog_right_true.append(pline_true)
        prog_right_pred.append(pline_pred)
        markers_right_true.append(mk_true)
        markers_right_pred.append(mk_pred)
        vlines_right.append(vl)

    legend_font = font_tick
    for ax in ax_ts_left + ax_ts_right:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper left", fontsize=legend_font, frameon=False)

    for i, (y, x, _p) in enumerate(points_left):
        con = ConnectionPatch(
            xyA=(x, y),
            coordsA=ax_map.transData,
            xyB=(1.0, 0.5),
            coordsB=ax_ts_left[i].transAxes,
            color=left_colors[i],
            linewidth=1.2,
            alpha=0.9,
        )
        fig.add_artist(con)
    for i, (y, x, _p) in enumerate(points_right):
        con = ConnectionPatch(
            xyA=(x, y),
            coordsA=ax_map.transData,
            xyB=(0.0, 0.5),
            coordsB=ax_ts_right[i].transAxes,
            color=right_colors[i],
            linewidth=1.2,
            alpha=0.9,
        )
        fig.add_artist(con)

    def update(frame_idx: int):
        t_idx = time_indices[frame_idx]
        depth_im.set_data(depth_true[t_idx])
        time_text.set_text(f"Hour {t_idx}")

        for i in range(4):
            prog_left_true[i].set_data(np.arange(t_idx + 1), traces_left_true[i][: t_idx + 1])
            prog_left_pred[i].set_data(np.arange(t_idx + 1), traces_left_pred[i][: t_idx + 1])
            markers_left_true[i].set_data([t_idx], [traces_left_true[i][t_idx]])
            markers_left_pred[i].set_data([t_idx], [traces_left_pred[i][t_idx]])
            vlines_left[i].set_xdata([t_idx, t_idx])

            prog_right_true[i].set_data(np.arange(t_idx + 1), traces_right_true[i][: t_idx + 1])
            prog_right_pred[i].set_data(np.arange(t_idx + 1), traces_right_pred[i][: t_idx + 1])
            markers_right_true[i].set_data([t_idx], [traces_right_true[i][t_idx]])
            markers_right_pred[i].set_data([t_idx], [traces_right_pred[i][t_idx]])
            vlines_right[i].set_xdata([t_idx, t_idx])

        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)


def save_dem_measurement_points_figure(
    dem_full: np.ndarray,
    true_full: np.ndarray,
    output_path: Path,
    dem_label: str = "Elevation (m)",
) -> None:
    dem_panel = np.asarray(dem_full, dtype=np.float32)
    dem_panel = np.ma.masked_invalid(dem_panel)

    depth_true = true_full[:, :, :, 0]
    points_left, points_right = _select_points(depth_true)
    all_points = points_left + points_right

    cmap = plt.get_cmap("terrain").copy()
    cmap.set_bad(color=(1.0, 1.0, 1.0, 1.0))

    if np.ma.isMaskedArray(dem_panel):
        finite_values = np.asarray(dem_panel.compressed(), dtype=np.float32)
    else:
        finite_values = dem_panel[np.isfinite(dem_panel)]
    if finite_values.size == 0:
        raise ValueError("DEM plot has no finite values to display.")
    dem_vmin = float(np.min(finite_values))
    dem_vmax = float(np.max(finite_values))

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    im = ax.imshow(dem_panel, cmap=cmap, vmin=dem_vmin, vmax=dem_vmax, interpolation="nearest", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.95)
    cbar.set_label(dem_label)

    ax.scatter(
        [x for (y, x, _p) in all_points],
        [y for (y, x, _p) in all_points],
        s=70,
        c="red",
        marker="^",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )

    ax.set_title("DEM with Measurement Points")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_area = 0
    best_bbox = None

    for y0, x0 in np.argwhere(mask):
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        visited[y0, x0] = True
        area = 0
        x_min = x_max = int(x0)
        y_min = y_max = int(y0)

        while stack:
            y, x = stack.pop()
            area += 1
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y

            if y > 0 and mask[y - 1, x] and not visited[y - 1, x]:
                visited[y - 1, x] = True
                stack.append((y - 1, x))
            if y + 1 < height and mask[y + 1, x] and not visited[y + 1, x]:
                visited[y + 1, x] = True
                stack.append((y + 1, x))
            if x > 0 and mask[y, x - 1] and not visited[y, x - 1]:
                visited[y, x - 1] = True
                stack.append((y, x - 1))
            if x + 1 < width and mask[y, x + 1] and not visited[y, x + 1]:
                visited[y, x + 1] = True
                stack.append((y, x + 1))

        if area > best_area:
            best_area = area
            best_bbox = (x_min, x_max, y_min, y_max)

    if best_bbox is None:
        raise ValueError("Could not detect DEM component in reference image.")
    return best_bbox


def _detect_reference_dem_bounds(reference_image: np.ndarray) -> tuple[int, int, int, int]:
    if reference_image.ndim != 3 or reference_image.shape[2] < 3:
        raise ValueError("Reference image must be an RGB/RGBA array.")

    rgb = reference_image[:, :, :3].astype(np.float32)
    if rgb.max() > 1.0:
        rgb /= 255.0
    color_span = rgb.max(axis=2) - rgb.min(axis=2)
    non_white = rgb.max(axis=2) < 0.98
    colored_mask = non_white & (color_span > 0.04)
    colored_mask[:80, :] = False
    return _largest_component_bbox(colored_mask)


def _estimate_grid_spacing(coord_grid: np.ndarray, axis: int) -> float:
    diffs = np.diff(np.asarray(coord_grid, dtype=np.float64), axis=axis)
    diffs = diffs[np.isfinite(diffs)]
    diffs = np.abs(diffs[np.abs(diffs) > 0])
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def save_depth_traces_reference_snapshot(
    pred_full: np.ndarray,
    true_full: np.ndarray,
    output_path: Path,
    reference_image_path: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    timestep: int | None = None,
    use_static_features: bool = False,
) -> int:
    depth_pred = pred_full[:, :, :, 0]
    depth_true = true_full[:, :, :, 0]
    t_len, h_dim, w_dim = depth_true.shape

    if x_grid.shape != (h_dim, w_dim) or y_grid.shape != (h_dim, w_dim):
        raise ValueError(
            f"Reference coordinate grids must match {(h_dim, w_dim)}; got {x_grid.shape} and {y_grid.shape}"
        )

    points_left, points_right = _select_points(depth_true)

    traces_left_true = [depth_true[:, y, x].astype(np.float32) for y, x, _peak in points_left]
    traces_left_pred = [depth_pred[:, y, x].astype(np.float32) for y, x, _peak in points_left]
    traces_right_true = [depth_true[:, y, x].astype(np.float32) for y, x, _tv in points_right]
    traces_right_pred = [depth_pred[:, y, x].astype(np.float32) for y, x, _tv in points_right]

    if timestep is None:
        timestep = int(np.nanargmax(np.nanmean(depth_true, axis=(1, 2))))
    timestep = int(np.clip(timestep, 0, t_len - 1))

    all_traces = traces_left_true + traces_left_pred + traces_right_true + traces_right_pred
    all_vals = np.concatenate([tr[np.isfinite(tr)] for tr in all_traces if np.isfinite(tr).any()])
    y_max = float(np.nanpercentile(all_vals, 99.5)) if all_vals.size else 1.0
    y_max = max(y_max, 0.1)
    y_lim = (0.0, 1.1 * y_max)

    aspect_ratio = h_dim / float(w_dim)
    fig_height = opt.height_multiplier * aspect_ratio / 1.5
    fig = plt.figure(figsize=(20, fig_height), constrained_layout=True)
    font_title = 18
    font_label = 16
    font_tick = 14
    legend_font = 14
    reference_label = "SynxFlow"
    model_label = "CLDNet" if use_static_features else "LDNet"
    line_truth = 2.5
    line_pred = 2.5
    gs = fig.add_gridspec(4, 3, width_ratios=[1.4, max(2.0, 2.0 * opt.width_multiplier), 1.4], wspace=0.04)
    ax_ts_left = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    ax_map = fig.add_subplot(gs[:, 1])
    ax_ts_right = [fig.add_subplot(gs[i, 2]) for i in range(4)]

    reference_image = plt.imread(reference_image_path)
    img_height, img_width = reference_image.shape[:2]
    x_left, x_right, y_top, y_bottom = _detect_reference_dem_bounds(reference_image)
    x_min = float(np.nanmin(x_grid))
    x_max = float(np.nanmax(x_grid))
    y_min = float(np.nanmin(y_grid))
    y_max_coord = float(np.nanmax(y_grid))
    dx = _estimate_grid_spacing(x_grid, axis=1)
    dy = _estimate_grid_spacing(y_grid, axis=0)
    x_extent_min = x_min - 0.5 * dx
    x_extent_max = x_max + 0.5 * dx
    y_extent_min = y_min - 0.5 * dy
    y_extent_max = y_max_coord + 0.5 * dy

    ax_map.imshow(reference_image, origin="upper", aspect="equal")
    ax_map.set_xlim(0, img_width)
    ax_map.set_ylim(img_height, 0)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)

    left_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    right_colors = ["tab:purple", "tab:brown", "tab:pink", "tab:cyan"]

    def _point_to_image_xy(y_idx: int, x_idx: int) -> tuple[float, float]:
        x_val = float(x_grid[y_idx, x_idx])
        y_val = float(y_grid[y_idx, x_idx])
        x_frac = 0.5 if x_extent_max <= x_extent_min else (x_val - x_extent_min) / (x_extent_max - x_extent_min)
        y_frac = 0.5 if y_extent_max <= y_extent_min else (y_val - y_extent_min) / (y_extent_max - y_extent_min)
        x_frac = float(np.clip(x_frac, 0.0, 1.0))
        y_frac = float(np.clip(y_frac, 0.0, 1.0))
        x_img = x_left + x_frac * (x_right - x_left)
        y_img = y_bottom - y_frac * (y_bottom - y_top)
        return x_img, y_img

    left_xy = [_point_to_image_xy(y, x) for y, x, _peak in points_left]
    right_xy = [_point_to_image_xy(y, x) for y, x, _tv in points_right]

    ax_map.scatter(
        [x for x, _y in left_xy],
        [y for _x, y in left_xy],
        s=70,
        c=left_colors,
        marker="o",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )
    ax_map.scatter(
        [x for x, _y in right_xy],
        [y for _x, y in right_xy],
        s=70,
        c=right_colors,
        marker="s",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )
    for i, ax in enumerate(ax_ts_left):
        y, x, peak = points_left[i]
        t = np.arange(t_len)
        ax.plot(t, traces_left_true[i], color="0.2", linewidth=line_truth, label=reference_label)
        ax.plot(t, traces_left_pred[i], color="tab:blue", linewidth=line_pred, linestyle="-", label=model_label)
        ax.set_xlim(0, t_len - 1)
        ax.set_ylim(*y_lim)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("h (m)", fontsize=font_label)
        ax.set_title(f"Peak Depth: y={y}, x={x} (peak {peak:.2f})", fontsize=font_label)
        ax.tick_params(labelsize=font_tick)
        if i == 3:
            ax.set_xlabel("Hour", fontsize=font_label)
        else:
            ax.set_xticklabels([])
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper left", fontsize=legend_font, frameon=False)

    for i, ax in enumerate(ax_ts_right):
        y, x, tv = points_right[i]
        t = np.arange(t_len)
        ax.plot(t, traces_right_true[i], color="0.2", linewidth=line_truth, label=reference_label)
        ax.plot(t, traces_right_pred[i], color="tab:orange", linewidth=line_pred, linestyle="-", label=model_label)
        ax.set_xlim(0, t_len - 1)
        ax.set_ylim(*y_lim)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("h (m)", fontsize=font_label)
        ax.set_title(f"Peak Total Variation: y={y}, x={x} (tv {tv:.2f})", fontsize=font_label)
        ax.tick_params(labelsize=font_tick)
        if i == 3:
            ax.set_xlabel("Hour", fontsize=font_label)
        else:
            ax.set_xticklabels([])
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper left", fontsize=legend_font, frameon=False)

    for i, (x_img, y_img) in enumerate(left_xy):
        con = ConnectionPatch(
            xyA=(x_img, y_img),
            coordsA=ax_map.transData,
            xyB=(1.0, 0.5),
            coordsB=ax_ts_left[i].transAxes,
            color=left_colors[i],
            linewidth=1.2,
            alpha=0.9,
        )
        fig.add_artist(con)
    for i, (x_img, y_img) in enumerate(right_xy):
        con = ConnectionPatch(
            xyA=(x_img, y_img),
            coordsA=ax_map.transData,
            xyB=(0.0, 0.5),
            coordsB=ax_ts_right[i].transAxes,
            color=right_colors[i],
            linewidth=1.2,
            alpha=0.9,
        )
        fig.add_artist(con)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return timestep


def main(opt):
    log = Logger(log_dir=_resolve_path(opt.base_path, opt.model_path) / opt.log_dir)
    log.info("=======================================================")
    log.info("                  Data Assimilation                     ")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    data_root = opt.data_root
    if not data_root.is_absolute():
        data_root = opt.base_path / data_root
    input_root = _resolve_path(opt.base_path, opt.input_root)

    flow, coords, rain = _load_reduced_data(
        data_root,
        opt.traj_id,
        opt.use_static_features,
        static_slope_xy=opt.static_slope_xy,
    )
    if opt.data_sanity:
        _print_data_sanity(rain, flow, coords)
        if opt.data_sanity_only:
            return
    t_len = flow.shape[1]
    x = np.repeat(coords[:, None, :, :], t_len, axis=1).astype(np.float32)
    requested_depth_only = not opt.all_vars
    dim_y_ckpt = None
    _dyn_ckpt, rec_ckpt, _B_ckpt = _resolve_checkpoint_paths(opt)
    finetune_rec_ckpt, _finetune_B_ckpt = _resolve_finetune_checkpoint_paths(opt)
    if finetune_rec_ckpt is not None:
        rec_ckpt = finetune_rec_ckpt
    if rec_ckpt is not None and rec_ckpt.exists():
        dim_y_ckpt = _infer_rec_output_dim(rec_ckpt)
        if dim_y_ckpt not in (1, 3):
            print(
                f"WARNING: rec checkpoint output dim is {dim_y_ckpt}; proceeding with first {dim_y_ckpt} channel(s)."
            )

    if dim_y_ckpt is not None:
        depth_only = dim_y_ckpt == 1
        if depth_only != requested_depth_only:
            requested = "depth-only (--all-vars off)" if requested_depth_only else "all-vars (--all-vars on)"
            inferred = "depth-only" if depth_only else "all-vars"
            print(
                f"WARNING: Requested {requested} but checkpoint rec head outputs {dim_y_ckpt} channel(s). "
                f"Using {inferred} to match checkpoint."
            )
    else:
        depth_only = requested_depth_only

    y_mean, y_std = _maybe_load_channel_normalization(
        opt.normalize,
        data_root / "mean.npy",
        data_root / "std.npy",
        "flow",
        log,
    )
    u_mean, u_std = _maybe_load_channel_normalization(
        opt.normalize_rain,
        data_root / "rain_mean.npy",
        data_root / "rain_std.npy",
        "rain",
        log,
    )

    if depth_only:
        y = flow[..., [0]].astype(np.float32)
    else:
        if dim_y_ckpt is None:
            y = flow.astype(np.float32)
        else:
            y = flow[..., :dim_y_ckpt].astype(np.float32)
    u = rain.astype(np.float32)
    if u_mean is not None and u_std is not None:
        if u_mean.shape != (u.shape[-1],) or u_std.shape != (u.shape[-1],):
            raise ValueError(
                f"Expected rain normalization shape {(u.shape[-1],)}, got {u_mean.shape} and {u_std.shape}"
            )
        u = (u - u_mean.reshape(1, 1, -1)) / u_std.reshape(1, 1, -1)

    dim_u = u.shape[-1]
    dim_x = x.shape[-1]
    dim_y = y.shape[-1]

    input_shape_d = opt.num_latent_states + dim_u
    input_shape_r = opt.num_latent_states + dim_x
    layer_sizes_dyn = [input_shape_d] + opt.NN_dyn_depth * [opt.NN_dyn_width] + [opt.num_latent_states]
    layer_sizes_rec = [input_shape_r] + opt.NN_rec_depth * [opt.NN_rec_width] + [dim_y]

    model = EfficientFourierLDNN(
        opt.fourier_mapping_size,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation=opt.activation,
        kernel_initializer=opt.kernel_initializer,
        chunk_size=opt.chunk_size,
    )
    _load_checkpoints(model, opt)
    model.to(opt.device)
    print("TOTAL PARAMETERS")
    print(sum(p.numel() for p in model.parameters()))
    model.eval()

    data = {
        "u": torch.from_numpy(u).to(opt.device),
        "x": torch.from_numpy(x).to(opt.device),
        "y": torch.from_numpy(y).to(opt.device),
        "dt": torch.tensor([dt], device=opt.device, dtype=torch.float32),
    }

    with torch.no_grad():
        if str(opt.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device=opt.device)
        inference_start = time.perf_counter()
        pred = model(data, opt.device, equilibrium=False, chunk_size=opt.chunk_size)
        if str(opt.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device=opt.device)
        inference_seconds = time.perf_counter() - inference_start
        pred_np = pred.cpu().numpy()
        true_np = y.astype(np.float32, copy=False)

    print(f"Inference time: {inference_seconds:.6f} s")

    if y_mean is not None and y_std is not None:
        y_mean_eff = y_mean[[0]] if depth_only else y_mean[:dim_y]
        y_std_eff = y_std[[0]] if depth_only else y_std[:dim_y]
        if y_mean_eff.shape != (dim_y,) or y_std_eff.shape != (dim_y,):
            raise ValueError(
                f"Expected flow normalization shape {(dim_y,)}, got {y_mean_eff.shape} and {y_std_eff.shape}"
            )
        pred_np = pred_np * y_std_eff.reshape(1, 1, 1, -1) + y_mean_eff.reshape(1, 1, 1, -1)

    if opt.metrics_only or opt.snapshot_metrics_only:
        if opt.metrics_only:
            for threshold in opt.flood_thresholds or [opt.flood_threshold]:
                _print_flood_inundation_confusion(
                    pred_np[0, :, :, 0],
                    true_np[0, :, :, 0],
                    threshold=threshold,
                )
        if opt.snapshot_metrics_only:
            reduced_truth = true_np[0, :, :, 0][:, None, :, None]
            reduced_prediction = pred_np[0, :, :, 0][:, None, :, None]
            stages = _select_flood_edge_timesteps(reduced_truth, threshold=opt.flood_threshold)
            print(
                "Snapshot stage timesteps: "
                f"rising={stages['rising_mid']}, peak={stages['peak']}, "
                f"recession={stages['falling_mid']}"
            )
            pairs, combined_truth, combined_prediction = _collect_snapshot_depth_pairs(
                reduced_prediction,
                reduced_truth,
                [
                    ("rising", stages["rising_mid"]),
                    ("peak", stages["peak"]),
                    ("recession", stages["falling_mid"]),
                ],
            )
            _print_snapshot_depth_metrics(pairs, combined_truth, combined_prediction)
        return

    error = pred_np - true_np
    rel = np.sqrt(np.sum(error ** 2) / np.sum(true_np ** 2))
    r2 = _r2_score(true_np, pred_np)
    rmse = _rmse_score(true_np, pred_np)
    print(f"Relative L2 error (reduced points): {rel}")
    print(f"R^2 (reduced points): {r2}")
    print(f"RMSE (reduced points): {rmse}")
    names = ["h"] if depth_only else ["h", "u", "v"][:dim_y]
    for ch, name in enumerate(names):
        err_ch = error[..., ch]
        true_ch = true_np[..., ch]
        denom = np.sum(true_ch ** 2)
        rel_ch = np.sqrt(np.sum(err_ch ** 2) / denom) if denom > 0 else float("nan")
        print(f"Relative L2 error {name}: {rel_ch}")
        r2_ch = _r2_score(true_ch, pred_np[..., ch])
        print(f"R^2 {name}: {r2_ch}")
        rmse_ch = _rmse_score(true_ch, pred_np[..., ch])
        print(f"RMSE {name}: {rmse_ch}")

    _print_flood_inundation_confusion(
        pred_np[0, :, :, 0],
        true_np[0, :, :, 0],
        threshold=opt.flood_threshold,
    )

    if opt.aggregate_mask_path is None:
        mask_path = data_root / "aggregate_mask.npy"
    else:
        mask_path = _resolve_path(opt.base_path, opt.aggregate_mask_path)

    valid, h_dim, w_dim = _load_aggregate_mask(mask_path)
    scatter_indices = _resolve_scatter_indices(valid, h_dim, w_dim, coords)
    if scatter_indices.size != pred_np.shape[2]:
        raise ValueError(f"Scatter mismatch: scatter={scatter_indices.size}, reduced={pred_np.shape[2]}")

    pred_full = np.zeros((t_len, h_dim * w_dim, dim_y), dtype=np.float32)
    true_full = np.zeros((t_len, h_dim * w_dim, dim_y), dtype=np.float32)
    pred_full[:, scatter_indices, :] = pred_np[0]
    true_full[:, scatter_indices, :] = true_np[0]

    pred_full = pred_full.reshape(t_len, h_dim, w_dim, dim_y)
    true_full = true_full.reshape(t_len, h_dim, w_dim, dim_y)

    dataset_root_str = str(data_root).lower()
    is_texas_dataset = "texas" in dataset_root_str
    is_chicago_dataset = not is_texas_dataset

    output_dir = opt.output_dir
    if output_dir is None:
        output_dir = _resolve_path(opt.base_path, opt.model_path)
    elif not output_dir.is_absolute():
        output_dir = opt.base_path / output_dir
    os.makedirs(output_dir, exist_ok=True)
    print("made directory")

    if is_chicago_dataset:
        dem_full, x_grid, y_grid, dem_label = _load_full_dem_georef(input_root)
        if dem_full.shape != (h_dim, w_dim):
            raise ValueError(f"DEM shape {dem_full.shape} does not match aggregate grid {(h_dim, w_dim)}")
        print("Using DEM from event_21")

        dem_points_path = output_dir / f"dem_measurement_points_traj{opt.traj_id}.png"
        save_dem_measurement_points_figure(
            dem_full,
            true_full,
            dem_points_path,
            dem_label=dem_label,
        )
        print(f"Saved DEM measurement-point figure to {dem_points_path}")

        reference_map_path = _resolve_path(opt.base_path, opt.reference_map_image)
        if reference_map_path.exists():
            depth_snapshot_path = output_dir / f"depth_traces_reference_snapshot_traj{opt.traj_id}.png"
            reference_snapshot_timestep = save_depth_traces_reference_snapshot(
                pred_full,
                true_full,
                depth_snapshot_path,
                reference_map_path,
                x_grid,
                y_grid,
                use_static_features=opt.use_static_features,
            )
            print(f"Saved depth-trace reference snapshot to {depth_snapshot_path}")

            reference_snapshot_pairs, reference_snapshot_ref, reference_snapshot_pred = _collect_snapshot_depth_pairs(
                pred_full,
                true_full,
                [("reference snapshot", reference_snapshot_timestep)],
                valid_mask=valid.reshape(h_dim, w_dim),
            )
            _print_snapshot_depth_metrics(
                reference_snapshot_pairs,
                reference_snapshot_ref,
                reference_snapshot_pred,
            )
        else:
            print(f"Skipping reference snapshot because image was not found at {reference_map_path}")
    else:
        print("Skipping DEM measurement-point figure because dataset is not Chicago.")

    flood_map_path = output_dir / f"flood_inundation_confusion_traj{opt.traj_id}.png"
    save_flood_inundation_map(
        pred_full,
        true_full,
        flood_map_path,
        threshold=opt.flood_threshold,
        valid_mask=valid.reshape(h_dim, w_dim),
    )
    print(f"Saved flood inundation map to {flood_map_path}")

    peak_inundation_path = output_dir / f"peak_inundation_qualitative_traj{opt.traj_id}.png"
    peak_r2, peak_rmse = save_peak_inundation_qualitative_figure(
        pred_full,
        true_full,
        peak_inundation_path,
        threshold=opt.flood_threshold,
        valid_mask=valid.reshape(h_dim, w_dim),
    )
    print(
        f"Saved peak inundation qualitative figure to {peak_inundation_path} "
        f"(R^2={peak_r2:.6f}, RMSE={peak_rmse:.6f})"
    )

    flood_stage_snapshot_path = output_dir / f"flood_stage_snapshots_traj{opt.traj_id}.png"
    stage_timesteps = save_flood_stage_snapshot_figure(
        pred_full,
        true_full,
        flood_stage_snapshot_path,
        threshold=opt.flood_threshold,
        valid_mask=valid.reshape(h_dim, w_dim),
        is_texas=is_texas_dataset,
    )
    print(
        "Saved flood stage snapshots to "
        f"{flood_stage_snapshot_path} "
        f"(rising={stage_timesteps['rising_mid']}, peak={stage_timesteps['peak']}, "
        f"falling={stage_timesteps['falling_mid']})"
    )

    snapshot_pairs, snapshot_ref, snapshot_pred = _collect_snapshot_depth_pairs(
        pred_full,
        true_full,
        [
            ("rising", stage_timesteps["rising_mid"]),
            ("peak", stage_timesteps["peak"]),
            ("recession", stage_timesteps["falling_mid"]),
        ],
        valid_mask=valid.reshape(h_dim, w_dim),
    )
    _print_snapshot_depth_metrics(snapshot_pairs, snapshot_ref, snapshot_pred)

    snapshot_scatter_path = output_dir / f"reference_vs_cldnet_depth_scatter_traj{opt.traj_id}.png"
    save_reference_vs_cldnet_depth_scatter(
        snapshot_pairs,
        snapshot_ref,
        snapshot_pred,
        snapshot_scatter_path,
    )
    print(f"Saved reference vs CLDNet depth scatter to {snapshot_scatter_path}")

    error_depth_path = output_dir / f"error_vs_depth_traj{opt.traj_id}.png"
    save_error_vs_depth_diagnostic(
        pred_full,
        true_full,
        error_depth_path,
        valid_mask=valid.reshape(h_dim, w_dim),
    )
    print(f"Saved error-vs-depth diagnostic to {error_depth_path}")

    error_depth_iqr_path = output_dir / f"error_vs_depth_iqr_traj{opt.traj_id}.png"
    save_error_vs_depth_iqr_diagnostic(
        pred_full,
        true_full,
        error_depth_iqr_path,
        valid_mask=valid.reshape(h_dim, w_dim),
    )
    print(f"Saved error-vs-depth median/IQR diagnostic to {error_depth_iqr_path}")

    error_depth_overlay_path = output_dir / f"error_vs_depth_overlay_traj{opt.traj_id}.png"
    save_error_vs_depth_overlay_diagnostic(
        pred_full,
        true_full,
        error_depth_overlay_path,
        valid_mask=valid.reshape(h_dim, w_dim),
    )
    print(f"Saved overlaid error-vs-depth diagnostic to {error_depth_overlay_path}")

    rain_series = None
    if rain.ndim == 3 and rain.shape[0] > 0:
        if rain.shape[-1] == 1:
            rain_series = rain[0, :, 0].astype(np.float32, copy=False)
        else:
            rain_series = rain[0].mean(axis=-1).astype(np.float32, copy=False)

    timestep_csi_path = output_dir / f"flood_inundation_csi_curve_traj{opt.traj_id}.png"
    timestep_csi = save_per_timestep_csi_curve(
        pred_full,
        true_full,
        timestep_csi_path,
        threshold=opt.flood_threshold,
        valid_mask=valid.reshape(h_dim, w_dim),
        rain_series=rain_series,
        is_texas=is_texas_dataset,
        is_illinois=is_chicago_dataset,
    )
    print(f"Saved per-timestep flood CSI curve to {timestep_csi_path}")
    valid_timestep_csi = timestep_csi[np.isfinite(timestep_csi)]
    if valid_timestep_csi.size > 0:
        print(f"Mean per-timestep CSI: {valid_timestep_csi.mean():.6f}")
    else:
        print("Mean per-timestep CSI: nan (no timestep had positive flood extent in truth or prediction)")

    # np.save(output_dir / f"predictions_traj{opt.traj_id}.npy", pred_full)
    # np.save(output_dir / f"truth_traj{opt.traj_id}.npy", true_full)
    print(f"Saved predictions/truth to {output_dir}")

    # plot_path = output_dir / f"hydrographs_traj{opt.traj_id}.png"
    # plot_hydrographs(pred_full, true_full, plot_path)
    # print(f"Saved hydrographs to {plot_path}")

    # anim_path = output_dir / f"depth_traces_truth_traj{opt.traj_id}.gif"
    # create_depth_traces_animation(pred_full, true_full, anim_path)
    # print(f"Saved depth traces animation to {anim_path}")


if __name__ == "__main__":
    # Keep float32 defaults even when DeepXDE isn't installed.
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    opt = create_options()
    main(opt)
