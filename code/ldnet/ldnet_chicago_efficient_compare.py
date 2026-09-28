#!/usr/bin/env python3
"""
Compare a static-feature CLDNet against a non-static LDNet on the same Chicago trajectory.

This script loads two separate checkpoints:
- one trained with static features appended to coordinates
- one trained without static features

It reconstructs both depth fields on the full grid and saves a combined
reference-snapshot figure with both model traces overlaid.

Example:
    python ldnet_chicago_efficient_compare.py \
        --static-model-path checkpoints/cldnet \
        --ldnet-model-path checkpoints/ldnet \
        --traj-id 0 \
        --checkpoint-epoch 539
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

repo_path = Path(__file__).resolve().parent
sys.path.append(str(repo_path))

import ldnet_chicago_efficient_test as base
from efficient_fourier_ldnet import EfficientFourierLDNN
from src.logger import Logger

dt = 1


class _NullLogger:
    def info(self, *_args, **_kwargs):
        return None


def create_options():
    parser = argparse.ArgumentParser()
    default_base_path = Path(__file__).resolve().parents[2]
    parser.add_argument("--base-path", type=Path, default=default_base_path)
    parser.add_argument("--log-dir", type=Path, default="log")
    parser.add_argument("--name", type=str, default="ldnet_chicago_efficient_compare")
    parser.add_argument("--static-model-path", type=Path, default="checkpoints/cldnet")
    parser.add_argument("--ldnet-model-path", type=Path, default="checkpoints/ldnet")
    parser.add_argument("--static-checkpoint-epoch", type=int, default=None)
    parser.add_argument("--ldnet-checkpoint-epoch", type=int, default=None)
    parser.add_argument("--checkpoint-epoch", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument(
        "--aggregate-mask-path",
        type=Path,
        default=Path("data/postprocessed/illinois/aggregate_mask.npy"),
    )
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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-map-image",
        type=Path,
        default=Path("plots/easting_northing.png"),
    )
    return parser.parse_args()


def _resolve_path(base_path: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_path / path


def _resolve_epoch(opt, model_epoch: int | None, fallback_epoch: int | None) -> int | None:
    if model_epoch is not None:
        return model_epoch
    return fallback_epoch


def _resolve_checkpoint_files(model_dir: Path, checkpoint_epoch: int | None) -> tuple[Path, Path, Path]:
    if checkpoint_epoch is None:
        checkpoint_epoch = base._latest_epoch_for_prefix(model_dir, "rec")
        if checkpoint_epoch is None:
            raise FileNotFoundError(f"Could not find any rec_*.ckpt files under {model_dir}")

    dyn_ckpt = model_dir / f"dyn_{checkpoint_epoch}.ckpt"
    rec_ckpt = model_dir / f"rec_{checkpoint_epoch}.ckpt"
    B_ckpt = model_dir / f"B_{checkpoint_epoch}.ckpt"

    if not dyn_ckpt.exists():
        raise FileNotFoundError(f"Missing dyn checkpoint at {dyn_ckpt}")
    if not rec_ckpt.exists():
        raise FileNotFoundError(f"Missing rec checkpoint at {rec_ckpt}")

    return dyn_ckpt, rec_ckpt, B_ckpt


def _predict_depth_field(
    opt,
    model_path: Path,
    use_static_features: bool,
    checkpoint_epoch: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_root = _resolve_path(opt.base_path, opt.data_root)
    flow, coords, rain = base._load_reduced_data(
        data_root,
        opt.traj_id,
        use_static_features,
        static_slope_xy=opt.static_slope_xy,
    )

    model_dir = _resolve_path(opt.base_path, model_path)
    dyn_ckpt, rec_ckpt, B_ckpt = _resolve_checkpoint_files(model_dir, checkpoint_epoch)

    dim_y_ckpt = base._infer_rec_output_dim(rec_ckpt)
    y_phys = flow[..., :dim_y_ckpt].astype(np.float32)

    y_mean, y_std = base._maybe_load_channel_normalization(
        opt.normalize,
        data_root / "mean.npy",
        data_root / "std.npy",
        "flow",
        _NullLogger(),
    )
    u_mean, u_std = base._maybe_load_channel_normalization(
        opt.normalize_rain,
        data_root / "rain_mean.npy",
        data_root / "rain_std.npy",
        "rain",
        _NullLogger(),
    )

    # Reuse the same logger-less normalization behavior as the main test script.
    if y_mean is not None and y_std is not None:
        y_mean_eff = y_mean[[0]] if dim_y_ckpt == 1 else y_mean[:dim_y_ckpt]
        y_std_eff = y_std[[0]] if dim_y_ckpt == 1 else y_std[:dim_y_ckpt]
        if y_mean_eff.shape != (dim_y_ckpt,) or y_std_eff.shape != (dim_y_ckpt,):
            raise ValueError(
                f"Expected flow normalization shape {(dim_y_ckpt,)}, got {y_mean_eff.shape} and {y_std_eff.shape}"
            )
        y = (y_phys - y_mean_eff.reshape(1, 1, -1)) / y_std_eff.reshape(1, 1, -1)
    else:
        y = y_phys

    u = rain.astype(np.float32)
    if u_mean is not None and u_std is not None:
        if u_mean.shape != (u.shape[-1],) or u_std.shape != (u.shape[-1],):
            raise ValueError(
                f"Expected rain normalization shape {(u.shape[-1],)}, got {u_mean.shape} and {u_std.shape}"
            )
        u = (u - u_mean.reshape(1, 1, -1)) / u_std.reshape(1, 1, -1)

    t_len = flow.shape[1]
    x = np.repeat(coords[:, None, :, :], t_len, axis=1).astype(np.float32)

    dim_u = u.shape[-1]
    dim_x = x.shape[-1]
    input_shape_d = opt.num_latent_states + dim_u
    input_shape_r = opt.num_latent_states + dim_x
    layer_sizes_dyn = [input_shape_d] + opt.NN_dyn_depth * [opt.NN_dyn_width] + [opt.num_latent_states]
    layer_sizes_rec = [input_shape_r] + opt.NN_rec_depth * [opt.NN_rec_width] + [dim_y_ckpt]

    model = EfficientFourierLDNN(
        opt.fourier_mapping_size,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation=opt.activation,
        kernel_initializer=opt.kernel_initializer,
        chunk_size=opt.chunk_size,
    )
    print(f"Loading dyn checkpoint: {dyn_ckpt}")
    print(f"Loading rec checkpoint: {rec_ckpt}")
    model.dyn.load_state_dict(torch.load(dyn_ckpt, map_location="cpu"))
    model.rec.load_state_dict(torch.load(rec_ckpt, map_location="cpu"))
    if B_ckpt.exists() and hasattr(model, "B"):
        print(f"Loading B checkpoint: {B_ckpt}")
        model.B.load_state_dict(torch.load(B_ckpt, map_location="cpu"))
    elif hasattr(model, "B"):
        print("WARNING: B checkpoint not found - Fourier embedding will use random weights!")
    model.to(opt.device)
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
        pred = model(data, opt.device, equilibrium=False, chunk_size=opt.chunk_size)
        if str(opt.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device=opt.device)

    pred_np = pred.cpu().numpy().astype(np.float32, copy=False)
    if y_mean is not None and y_std is not None:
        y_mean_eff = y_mean[[0]] if dim_y_ckpt == 1 else y_mean[:dim_y_ckpt]
        y_std_eff = y_std[[0]] if dim_y_ckpt == 1 else y_std[:dim_y_ckpt]
        pred_np = pred_np * y_std_eff.reshape(1, 1, 1, -1) + y_mean_eff.reshape(1, 1, 1, -1)

    true_np = y_phys.astype(np.float32, copy=False)
    return pred_np, true_np, coords


def _save_dual_depth_traces_reference_snapshot(
    pred_static_full: np.ndarray,
    pred_ldnet_full: np.ndarray,
    true_full: np.ndarray,
    output_path: Path,
    reference_image_path: Path,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    height_multiplier: float,
    width_multiplier: float,
    static_label: str = "CLDNet",
    ldnet_label: str = "LDNet",
    legend_mode: str = "figure",
) -> None:
    depth_true = true_full[:, :, :, 0]
    depth_static = pred_static_full[:, :, :, 0]
    depth_ldnet = pred_ldnet_full[:, :, :, 0]
    t_len, h_dim, w_dim = depth_true.shape

    if x_grid.shape != (h_dim, w_dim) or y_grid.shape != (h_dim, w_dim):
        raise ValueError(
            f"Reference coordinate grids must match {(h_dim, w_dim)}; got {x_grid.shape} and {y_grid.shape}"
        )

    points_left, points_right = base._select_points(depth_true)

    traces_left_true = [depth_true[:, y, x].astype(np.float32) for y, x, _peak in points_left]
    traces_left_static = [depth_static[:, y, x].astype(np.float32) for y, x, _peak in points_left]
    traces_left_ldnet = [depth_ldnet[:, y, x].astype(np.float32) for y, x, _peak in points_left]

    traces_right_true = [depth_true[:, y, x].astype(np.float32) for y, x, _tv in points_right]
    traces_right_static = [depth_static[:, y, x].astype(np.float32) for y, x, _tv in points_right]
    traces_right_ldnet = [depth_ldnet[:, y, x].astype(np.float32) for y, x, _tv in points_right]

    all_traces = (
        traces_left_true
        + traces_left_static
        + traces_left_ldnet
        + traces_right_true
        + traces_right_static
        + traces_right_ldnet
    )
    all_vals = np.concatenate([tr[np.isfinite(tr)] for tr in all_traces if np.isfinite(tr).any()])
    y_max = float(np.nanpercentile(all_vals, 99.5)) if all_vals.size else 1.0
    y_max = max(y_max, 0.1)
    y_lim = (0.0, 1.1 * y_max)

    aspect_ratio = h_dim / float(w_dim)
    fig_height = height_multiplier * aspect_ratio / 1.5
    fig = plt.figure(figsize=(20, fig_height), constrained_layout=True)
    font_title = 18
    font_label = 16
    font_tick = 14
    line_truth = 2.5
    line_static = 2.5
    line_ldnet = 2.5
    gs = fig.add_gridspec(4, 3, width_ratios=[1.4, max(2.0, 2.0 * width_multiplier), 1.4], wspace=0.04)
    ax_ts_left = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    ax_map = fig.add_subplot(gs[:, 1])
    ax_ts_right = [fig.add_subplot(gs[i, 2]) for i in range(4)]

    reference_image = plt.imread(reference_image_path)
    img_height, img_width = reference_image.shape[:2]
    x_left, x_right, y_top, y_bottom = base._detect_reference_dem_bounds(reference_image)
    x_min = float(np.nanmin(x_grid))
    x_max = float(np.nanmax(x_grid))
    y_min = float(np.nanmin(y_grid))
    y_max_coord = float(np.nanmax(y_grid))
    ax_map.imshow(reference_image, origin="upper", aspect="equal")
    ax_map.set_xlim(0, img_width)
    ax_map.set_ylim(img_height, 0)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    ax_map.tick_params(labelsize=font_tick)

    left_colors = ["tab:blue", "tab:blue", "tab:blue", "tab:blue"]
    right_colors = ["tab:orange", "tab:orange", "tab:orange", "tab:orange"]

    def _point_to_image_xy(y_idx: int, x_idx: int) -> tuple[float, float]:
        x_val = float(x_grid[y_idx, x_idx])
        y_val = float(y_grid[y_idx, x_idx])
        x_frac = 0.5 if x_max <= x_min else (x_val - x_min) / (x_max - x_min)
        y_frac = 0.5 if y_max_coord <= y_min else (y_val - y_min) / (y_max_coord - y_min)
        x_frac = float(np.clip(x_frac, 0.0, 1.0))
        y_frac = float(np.clip(y_frac, 0.0, 1.0))
        x_img = x_left + x_frac * (x_right - x_left)
        y_img = y_bottom - y_frac * (y_bottom - y_top)
        return x_img, y_img

    left_points_img = [_point_to_image_xy(y, x) for y, x, _peak in points_left]
    right_points_img = [_point_to_image_xy(y, x) for y, x, _tv in points_right]

    ax_map.scatter(
        [x for (x, _y) in left_points_img],
        [y for (_x, y) in left_points_img],
        s=70,
        c=left_colors,
        marker="o",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )
    ax_map.scatter(
        [x for (x, _y) in right_points_img],
        [y for (_x, y) in right_points_img],
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
        ax.plot(t, traces_left_true[i], color="0.2", linewidth=line_truth, label="truth")
        ax.plot(t, traces_left_static[i], color="tab:blue", linewidth=line_static, linestyle="-", label=static_label)
        ax.plot(t, traces_left_ldnet[i], color="tab:orange", linewidth=line_ldnet, linestyle="-", label=ldnet_label)
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
        if legend_mode == "panel":
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="upper left", fontsize=14, frameon=False)

    for i, ax in enumerate(ax_ts_right):
        y, x, tv = points_right[i]
        t = np.arange(t_len)
        ax.plot(t, traces_right_true[i], color="0.2", linewidth=line_truth, label="truth")
        ax.plot(t, traces_right_static[i], color="tab:blue", linewidth=line_static, linestyle="-", label=static_label)
        ax.plot(t, traces_right_ldnet[i], color="tab:orange", linewidth=line_ldnet, linestyle="-", label=ldnet_label)
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
        if legend_mode == "panel":
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="upper left", fontsize=14, frameon=False)

    for i, (y, x, _p) in enumerate(points_left):
        x_img, y_img = left_points_img[i]
        fig.add_artist(
            ConnectionPatch(
                xyA=(x_img, y_img),
                coordsA=ax_map.transData,
                xyB=(1.0, 0.5),
                coordsB=ax_ts_left[i].transAxes,
                color=left_colors[i],
                linewidth=1.2,
                alpha=0.9,
            )
        )
    for i, (y, x, _p) in enumerate(points_right):
        x_img, y_img = right_points_img[i]
        fig.add_artist(
            ConnectionPatch(
                xyA=(x_img, y_img),
                coordsA=ax_map.transData,
                xyB=(0.0, 0.5),
                coordsB=ax_ts_right[i].transAxes,
                color=right_colors[i],
                linewidth=1.2,
                alpha=0.9,
            )
        )

    if legend_mode == "figure":
        handles = [
            Line2D([0], [0], color="0.2", linewidth=line_truth, label="Reference"),
            Line2D([0], [0], color="tab:blue", linewidth=line_static, linestyle="-", label=static_label),
            Line2D([0], [0], color="tab:orange", linewidth=line_ldnet, linestyle="-", label=ldnet_label),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=18,
            handlelength=3.0,
            columnspacing=1.8,
        )
        fig.savefig(output_path, dpi=400, bbox_inches="tight")
    else:
        fig.savefig(output_path, dpi=400)
    plt.close(fig)


def main(opt):
    log = Logger(log_dir=_resolve_path(opt.base_path, opt.static_model_path) / opt.log_dir)
    log.info("=======================================================")
    log.info("        CLDNet vs LDNet Comparison Snapshot            ")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    data_root = _resolve_path(opt.base_path, opt.data_root)
    input_root = _resolve_path(opt.base_path, opt.input_root)
    flow_ldnet, coords_ldnet, _rain_ldnet = base._load_reduced_data(
        data_root,
        opt.traj_id,
        use_static=False,
        static_slope_xy=opt.static_slope_xy,
    )
    flow_static, coords_static, _rain_static = base._load_reduced_data(
        data_root,
        opt.traj_id,
        use_static=True,
        static_slope_xy=opt.static_slope_xy,
    )

    if flow_ldnet.shape[:2] != flow_static.shape[:2]:
        raise ValueError(
            f"Static and non-static trajectories do not match: {flow_static.shape} vs {flow_ldnet.shape}"
        )

    if not np.allclose(np.asarray(coords_ldnet[0, :, :2]), np.asarray(coords_static[0, :, :2])):
        raise ValueError("Static and non-static coordinate layouts differ; cannot compare on the same grid.")

    static_ckpt_epoch = _resolve_epoch(opt, opt.static_checkpoint_epoch, opt.checkpoint_epoch)
    ldnet_ckpt_epoch = _resolve_epoch(opt, opt.ldnet_checkpoint_epoch, opt.checkpoint_epoch)

    static_pred, static_true, _ = _predict_depth_field(
        opt,
        _resolve_path(opt.base_path, opt.static_model_path),
        use_static_features=True,
        checkpoint_epoch=static_ckpt_epoch,
    )
    ldnet_pred, ldnet_true, _ = _predict_depth_field(
        opt,
        _resolve_path(opt.base_path, opt.ldnet_model_path),
        use_static_features=False,
        checkpoint_epoch=ldnet_ckpt_epoch,
    )

    if static_true.shape[:3] != ldnet_true.shape[:3]:
        raise ValueError(f"Prediction grid shapes differ: {static_true.shape} vs {ldnet_true.shape}")

    t_len = static_true.shape[1]
    y = static_true.astype(np.float32, copy=False)

    if opt.aggregate_mask_path is None:
        mask_path = data_root / "aggregate_mask.npy"
    else:
        mask_path = _resolve_path(opt.base_path, opt.aggregate_mask_path)

    valid, h_dim, w_dim = base._load_aggregate_mask(mask_path)
    scatter_indices_static = base._resolve_scatter_indices(valid, h_dim, w_dim, coords_static)
    scatter_indices_ldnet = base._resolve_scatter_indices(valid, h_dim, w_dim, coords_ldnet)
    if not np.array_equal(scatter_indices_static, scatter_indices_ldnet):
        raise ValueError("Static and non-static scatter indices do not align.")

    pred_static_full = np.zeros((t_len, h_dim * w_dim, 1), dtype=np.float32)
    pred_ldnet_full = np.zeros((t_len, h_dim * w_dim, 1), dtype=np.float32)
    true_full = np.zeros((t_len, h_dim * w_dim, 1), dtype=np.float32)
    pred_static_full[:, scatter_indices_static, 0] = static_pred[0, :, :, 0]
    pred_ldnet_full[:, scatter_indices_ldnet, 0] = ldnet_pred[0, :, :, 0]
    true_full[:, scatter_indices_static, 0] = y[0, :, :, 0]

    pred_static_full = pred_static_full.reshape(t_len, h_dim, w_dim, 1)
    pred_ldnet_full = pred_ldnet_full.reshape(t_len, h_dim, w_dim, 1)
    true_full = true_full.reshape(t_len, h_dim, w_dim, 1)

    output_dir = opt.output_dir
    if output_dir is None:
        output_dir = _resolve_path(opt.base_path, opt.static_model_path)
    else:
        output_dir = _resolve_path(opt.base_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_map_path = _resolve_path(opt.base_path, opt.reference_map_image)
    if not reference_map_path.exists():
        print(f"Skipping reference snapshot because image was not found at {reference_map_path}")
        return

    dem_full, x_grid, y_grid, _dem_label = base._load_full_dem_georef(input_root)
    if dem_full.shape != (h_dim, w_dim):
        raise ValueError(f"DEM shape {dem_full.shape} does not match aggregate grid {(h_dim, w_dim)}")

    output_path = output_dir / f"depth_traces_reference_snapshot_cldnet_ldnet_traj{opt.traj_id}.png"
    _save_dual_depth_traces_reference_snapshot(
        pred_static_full,
        pred_ldnet_full,
        true_full,
        output_path,
        reference_map_path,
        x_grid,
        y_grid,
        height_multiplier=opt.height_multiplier,
        width_multiplier=opt.width_multiplier,
        static_label="CLDNet",
        ldnet_label="LDNet",
        legend_mode="figure",
    )
    print(f"Saved combined depth-trace reference snapshot to {output_path}")

    panel_legend_path = output_dir / f"depth_traces_reference_snapshot_cldnet_ldnet_panel_legend_traj{opt.traj_id}.png"
    _save_dual_depth_traces_reference_snapshot(
        pred_static_full,
        pred_ldnet_full,
        true_full,
        panel_legend_path,
        reference_map_path,
        x_grid,
        y_grid,
        height_multiplier=opt.height_multiplier,
        width_multiplier=opt.width_multiplier,
        static_label="CLDNet",
        ldnet_label="LDNet",
        legend_mode="panel",
    )
    print(f"Saved panel-legend depth-trace reference snapshot to {panel_legend_path}")

    static_rel = float(
        np.sqrt(
            np.sum((pred_static_full - true_full) ** 2) / np.sum(true_full**2)
        )
    )
    ldnet_rel = float(
        np.sqrt(
            np.sum((pred_ldnet_full - true_full) ** 2) / np.sum(true_full**2)
        )
    )
    print(f"Relative L2 error (CLDNet depth): {static_rel:.6f}")
    print(f"Relative L2 error (LDNet depth): {ldnet_rel:.6f}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    opt = create_options()
    main(opt)
