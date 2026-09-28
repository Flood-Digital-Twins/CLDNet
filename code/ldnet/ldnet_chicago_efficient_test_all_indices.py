"""
Efficient Fourier LDNN Test Script for Flood Surrogate Modeling.

Usage from the repository root:
    python code/ldnet/ldnet_chicago_efficient_test_all_indices.py --traj-id 109 \\
        --checkpoint-epoch 539 --model-path checkpoints/ldnet/illinois \\
        --input-root data/sims_30 --all-vars --fourier-mapping-size 32

    python code/ldnet/ldnet_chicago_efficient_test_all_indices.py --traj-id 109 \\
        --checkpoint-epoch 539 --model-path checkpoints/cldnet/illinois \\
        --input-root data/sims_30 --all-vars --fourier-mapping-size 32 \\
        --use-static-features

Full command arrays are in configs/{ldnet,cldnet}/illinois.json.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import ConnectionPatch

repo_path = Path(__file__).resolve().parent
sys.path.append(str(repo_path))

from src.logger import Logger
from efficient_fourier_ldnet import EfficientFourierLDNN

dt = 1


def create_options():
    parser = argparse.ArgumentParser()
    default_base_path = Path(__file__).resolve().parents[2]
    parser.add_argument("--base-path", type=Path, default=default_base_path)
    parser.add_argument("--log-dir", type=Path, default="log")
    parser.add_argument("--name", type=str, default="ldnet_chicago_efficient_test")
    parser.add_argument("--model-path", type=Path, default="checkpoints/ldnet")
    parser.add_argument("--data-root", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--use-static-features", action="store_true", default=False)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/sims_30"),
    )
    parser.add_argument("--traj-id", type=int, default=0)
    parser.add_argument("--burn-in-length", type=int, default=1)
    parser.add_argument("--depth-threshold", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--num-latent-states", type=int, default=200)
    parser.add_argument("--fourier-mapping-size", type=int, default=10)
    parser.add_argument("--NN-dyn-depth", type=int, default=8)
    parser.add_argument("--NN-dyn-width", type=int, default=50)
    parser.add_argument("--NN-rec-depth", type=int, default=10)
    parser.add_argument("--NN-rec-width", type=int, default=300)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--kernel-initializer", type=str, default="Glorot normal")
    parser.add_argument("--dyn-checkpoint", type=Path, default=None)
    parser.add_argument("--rec-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-epoch", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-sanity", action="store_true", default=False)
    parser.add_argument("--data-sanity-only", action="store_true", default=False)
    parser.add_argument("--all-vars", action="store_true", default=False)
    parser.add_argument("--skip-plots", action="store_true", default=False)
    return parser.parse_args()


def _load_reduced_data(data_root: Path, traj_id: int, use_static: bool):
    flow = np.load(data_root / f"flow_variables_traj{traj_id}.npy", mmap_mode="r")
    coords = np.load(data_root / f"coords_traj{traj_id}.npy", mmap_mode="r")
    rain = np.load(data_root / f"rain_source_traj{traj_id}.npy", mmap_mode="r")
    static = None
    if use_static:
        static = np.load(data_root / f"static_features_traj{traj_id}.npy", mmap_mode="r")
        coords = np.concatenate([coords, static], axis=-1)
    return flow, coords, rain


def _resolve_path(base_path: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_path / path


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


def _load_raw_flow(input_root: Path, traj_id: int, burn_in: int):
    dataset_directory = input_root / f"event_{traj_id}_filling_hours_0_filling_intensity_mm_hr_0"
    flow_path = dataset_directory / "flow_variables.npy"
    flow = np.load(flow_path).astype(np.float32)
    if flow.ndim == 4 and flow.shape[1] == 3:
        flow = flow[burn_in:]
        flow = flow.transpose(0, 2, 3, 1)  # (T, H, W, 3)
        h_dim, w_dim = flow.shape[1], flow.shape[2]
    elif flow.ndim == 4 and flow.shape[-1] == 3:
        flow = flow[burn_in:]
        h_dim, w_dim = flow.shape[1], flow.shape[2]
    else:
        raise ValueError(f"Unexpected flow_variables shape: {flow.shape}")
    return flow, h_dim, w_dim


def _build_coords(h_dim: int, w_dim: int) -> np.ndarray:
    x = 2.0 * (np.arange(w_dim, dtype=np.float32) / w_dim - 0.5)
    aspect = h_dim / float(w_dim)
    y = 2.0 * (np.arange(h_dim, dtype=np.float32) / h_dim - 0.5) * aspect
    x_coords = np.tile(x, h_dim)
    y_coords = np.repeat(y, w_dim)
    coords = np.stack([x_coords, y_coords], axis=1)[None, :, :]
    return coords.astype(np.float32)


def _load_rain_source(rain_path: Path, expect_t: int) -> np.ndarray:
    rain = np.load(rain_path).astype(np.float32)
    if rain.ndim == 2:
        rain = rain[np.newaxis, ...]
    elif rain.ndim == 3:
        rain = rain[np.newaxis, ...]
    elif rain.ndim == 4:
        rain = rain[np.newaxis, ...]
    elif rain.ndim == 5:
        pass
    else:
        raise ValueError(f"Unexpected rain_source shape: {rain.shape}")

    if rain.shape[1] > expect_t:
        rain = rain[:, :expect_t]
    elif rain.shape[1] < expect_t:
        raise ValueError(f"rain_source has fewer timesteps ({rain.shape[1]}) than expected ({expect_t})")

    rain = rain.reshape(rain.shape[0], rain.shape[1], -1)
    return rain


def _load_static_fields(dataset_directory: Path) -> np.ndarray:
    dem_path = dataset_directory / "DEM.npy"
    manning_path = dataset_directory / "input/field/manning.dat"

    if not dem_path.exists():
        raise FileNotFoundError(f"Missing DEM.npy at {dem_path}")
    dem = np.load(dem_path, mmap_mode="r")
    if dem.ndim == 3 and dem.shape[0] == 3:
        elev = np.asarray(dem[2], dtype=np.float32)
    elif dem.ndim == 2:
        elev = np.asarray(dem, dtype=np.float32)
    else:
        raise ValueError(f"Unexpected DEM shape: {dem.shape}")

    dem_valid = np.isfinite(elev)
    if not np.any(dem_valid):
        raise ValueError("DEM has no finite values")
    dem_mean = float(np.nanmean(elev))
    dem_std = float(np.nanstd(elev))
    if dem_std <= 0:
        raise ValueError("DEM std is zero; cannot z-score")
    dem_z = (elev - dem_mean) / dem_std

    elev_filled = np.nan_to_num(elev, nan=dem_mean)
    gy, gx = np.gradient(elev_filled)
    slope = np.sqrt(gx**2 + gy**2).astype(np.float32)
    slope[~dem_valid] = np.nan

    if not manning_path.exists():
        raise FileNotFoundError(f"Missing manning.dat at {manning_path}")

    manning_vals = []
    with open(manning_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("$Boundary"):
                break
            if line.startswith("$"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                manning_vals.append(float(parts[1]))
            except ValueError:
                continue
    manning_vals = np.asarray(manning_vals, dtype=np.float32)
    if manning_vals.size != int(dem_valid.sum()):
        raise ValueError(
            f"manning.dat count {manning_vals.size} does not match DEM valid count {int(dem_valid.sum())}"
        )
    manning_full = np.full(elev.shape, np.nan, dtype=np.float32)
    manning_full.ravel()[dem_valid.ravel()] = manning_vals

    manning_scaled = manning_full * 100.0
    static = np.stack([dem_z, slope, manning_scaled], axis=-1)
    return static


def _build_valid_mask(h: np.ndarray, depth_threshold: float) -> np.ndarray:
    h_flat = h.reshape(h.shape[0], -1)
    valid = ~np.isnan(h_flat).any(axis=0)
    return valid


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
    d_vmin, d_vmax = _estimate_limits([depth_pred[t] for t in sample_ts], percentile=percentile)

    all_traces = traces_left_true + traces_left_pred + traces_right_true + traces_right_pred
    all_vals = np.concatenate([tr[np.isfinite(tr)] for tr in all_traces if np.isfinite(tr).any()])
    y_max = float(np.nanpercentile(all_vals, 99.5)) if all_vals.size else 1.0
    y_max = max(y_max, 0.1)
    y_lim = (0.0, 1.1 * y_max)

    aspect_ratio = h_dim / float(w_dim)
    fig_height = 6 * aspect_ratio / 1.5
    fig = plt.figure(figsize=(18, fig_height), constrained_layout=True)
    font_title = 16
    font_label = 13
    font_tick = 12
    line_truth = 2.5
    line_pred = 2.5
    line_prog = 3.0
    gs = fig.add_gridspec(4, 3, width_ratios=[1.8, 1.0, 1.8], wspace=0.05)
    ax_ts_left = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    ax_map = fig.add_subplot(gs[:, 1])
    ax_ts_right = [fig.add_subplot(gs[i, 2]) for i in range(4)]

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color=(0, 0, 0, 0))
    depth_im = ax_map.imshow(depth_pred[0], cmap=cmap, vmin=d_vmin, vmax=d_vmax, aspect="equal", alpha=0.9)
    cbar = fig.colorbar(
        depth_im, ax=ax_map, orientation="horizontal", label="Water Depth h (m)", pad=0.03, shrink=0.95
    )
    cbar.set_label("Water Depth h (m)", fontsize=font_label)
    cbar.ax.tick_params(labelsize=font_tick)

    ax_map.set_title("Water Depth (pred)", fontsize=font_title)
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
        depth_im.set_data(depth_pred[t_idx])
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

    flow, coords, rain = _load_reduced_data(data_root, opt.traj_id, opt.use_static_features)
    if opt.data_sanity:
        _print_data_sanity(rain, flow, coords)
        if opt.data_sanity_only:
            return
    dataset_directory = input_root / f"event_{opt.traj_id}_filling_hours_0_filling_intensity_mm_hr_0"
    flow_raw, h_dim, w_dim = _load_raw_flow(input_root, opt.traj_id, opt.burn_in_length)
    t_len = flow_raw.shape[0]

    coords_full = _build_coords(h_dim, w_dim)
    if opt.use_static_features:
        static_full = _load_static_fields(dataset_directory)
        static_full = static_full.reshape(1, h_dim * w_dim, static_full.shape[-1]).astype(np.float32)
        coords_full = np.concatenate([coords_full, static_full], axis=-1)

    # Mask only NaN locations (no depth threshold).
    valid = ~np.isnan(flow_raw[..., 0]).any(axis=0).reshape(-1)
    coords_full = coords_full[:, valid, :]

    x = np.repeat(coords_full[:, None, :, :], t_len, axis=1).astype(np.float32)
    requested_depth_only = not opt.all_vars
    dim_y_ckpt = None
    _dyn_ckpt, rec_ckpt, _B_ckpt = _resolve_checkpoint_paths(opt)
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

    if depth_only:
        y = flow_raw[..., [0]].astype(np.float32)
    else:
        if dim_y_ckpt is None:
            y = flow_raw.astype(np.float32)
        else:
            y = flow_raw[..., :dim_y_ckpt].astype(np.float32)
    y = y.reshape(1, t_len, h_dim * w_dim, y.shape[-1])
    y = y[:, :, valid, :]

    rain_path = dataset_directory / "rain_source.npy"
    u = _load_rain_source(rain_path, expect_t=t_len).astype(np.float32)
    if u.shape[-1] == h_dim * w_dim:
        u = u[:, :, valid]

    dim_u = u.shape[-1]
    dim_x = x.shape[-1]
    dim_y = y.shape[-1]

    input_shape_d = opt.num_latent_states + dim_u
    input_shape_r = opt.num_latent_states + dim_x
    layer_sizes_dyn = [input_shape_d] + opt.NN_dyn_depth * [opt.NN_dyn_width] + [opt.num_latent_states]
    layer_sizes_rec = [input_shape_r] + opt.NN_rec_depth * [opt.NN_rec_width] + [dim_y]
    print(layer_sizes_rec)

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
    model.eval()

    data = {
        "u": torch.from_numpy(u).to(opt.device),
        "x": torch.from_numpy(x).to(opt.device),
        "y": torch.from_numpy(y).to(opt.device),
        "dt": torch.tensor([dt], device=opt.device, dtype=torch.float32),
    }

    with torch.no_grad():
        pred = model(data, opt.device, equilibrium=False, chunk_size=opt.chunk_size)
        pred_np = pred.cpu().numpy()
        true_np = y

    error = pred_np - true_np
    rel = np.sqrt(np.sum(error ** 2) / np.sum(true_np ** 2))
    print(f"Relative L2 error (reduced points): {rel}")
    names = ["h"] if depth_only else ["h", "hu", "hv"][:dim_y]
    for ch, name in enumerate(names):
        err_ch = error[..., ch]
        true_ch = true_np[..., ch]
        denom = np.sum(true_ch ** 2)
        rel_ch = np.sqrt(np.sum(err_ch ** 2) / denom) if denom > 0 else float("nan")
        print(f"Relative L2 error {name}: {rel_ch}")

    pred_full = np.full((t_len, h_dim * w_dim, dim_y), np.nan, dtype=np.float32)
    true_full = np.zeros((t_len, h_dim * w_dim, dim_y), dtype=np.float32)
    pred_full[:, valid, :] = pred_np[0]
    pred_full = pred_full.reshape(t_len, h_dim, w_dim, dim_y)

    true_full[:, valid, :] = true_np[0]
    true_full = true_full.reshape(t_len, h_dim, w_dim, dim_y)

    output_dir = opt.output_dir
    if output_dir is None:
        output_dir = _resolve_path(opt.base_path, opt.model_path)
    elif not output_dir.is_absolute():
        output_dir = opt.base_path / output_dir
    os.makedirs(output_dir, exist_ok=True)

    np.save(output_dir / f"predictions_allindices_traj{opt.traj_id}.npy", pred_full)
    np.save(output_dir / f"truth_traj{opt.traj_id}.npy", true_full)
    print(f"Saved predictions/truth to {output_dir}")

    if not opt.skip_plots:
        plot_path = output_dir / f"hydrographs_allindices_traj{opt.traj_id}.png"
        plot_hydrographs(pred_full, true_full, plot_path)
        print(f"Saved hydrographs to {plot_path}")

        anim_path = output_dir / f"depth_traces_pred_allindices_traj{opt.traj_id}.gif"
        create_depth_traces_animation(pred_full, true_full, anim_path)
        print(f"Saved depth traces animation to {anim_path}")


if __name__ == "__main__":
    # Keep float32 defaults even when DeepXDE isn't installed.
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    opt = create_options()
    main(opt)
