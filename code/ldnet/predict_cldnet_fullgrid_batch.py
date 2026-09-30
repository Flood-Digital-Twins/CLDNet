#!/usr/bin/env python3
"""Batch CLDNet full-grid prediction export for selected Chicago trajectories.

This script reuses the existing CLDNet inference path, reconstructs the
predictions onto the full 5075 x 1661 spatial grid with zeros outside the
CLDNet-predicted cells, and saves each result as a compressed NPZ file.

It also writes a simple smoke-test GIF for one chosen trajectory that animates
the predicted water depth field over time.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch

from efficient_fourier_ldnet import EfficientFourierLDNN
from ldnet_chicago_efficient_test import (
    _infer_rec_output_dim,
    _load_aggregate_mask,
    _load_reduced_data,
    _maybe_load_channel_normalization,
    _resolve_checkpoint_paths,
    _resolve_path,
    _resolve_scatter_indices,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = Path("data/sims_30")
DEFAULT_MODEL_PATH = Path("checkpoints/cldnet/illinois")
DEFAULT_DATA_ROOT = Path("data/postprocessed/illinois")
DEFAULT_CHECKPOINT_EPOCH = 539
DEFAULT_TRAJ_IDS = [107, 108, 109]
DEFAULT_ANIMATE_TRAJ_ID = 108


def create_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch CLDNet full-grid prediction export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-path", type=Path, default=DEFAULT_PYTHON_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--aggregate-mask-path", type=Path, default=Path("data/postprocessed/illinois/aggregate_mask.npy"))
    parser.add_argument("--checkpoint-epoch", type=int, default=DEFAULT_CHECKPOINT_EPOCH)
    parser.add_argument("--traj-ids", type=int, nargs="+", default=DEFAULT_TRAJ_IDS)
    parser.add_argument("--animate-traj-id", type=int, default=DEFAULT_ANIMATE_TRAJ_ID)
    parser.add_argument("--output-dir", type=Path, default=None)
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
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--normalize-rain", action="store_true", default=False)
    parser.add_argument("--use-static-features", action="store_true", default=True)
    parser.add_argument("--static-slope-xy", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--fps", type=int, default=5)
    return parser.parse_args()


def _build_model(opt: argparse.Namespace, dim_y: int, dim_u: int, dim_x: int) -> EfficientFourierLDNN:
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
    return model


def _resolve_output_dir(opt: argparse.Namespace) -> Path:
    if opt.output_dir is None:
        return opt.base_path / "outputs/fullgrid_predictions_traj107_109"
    if opt.output_dir.is_absolute():
        return opt.output_dir
    return opt.base_path / opt.output_dir


def _load_model_and_prepare(opt: argparse.Namespace):
    base_path = opt.base_path.resolve()
    data_root = _resolve_path(base_path, opt.data_root)
    model_path = _resolve_path(base_path, opt.model_path)
    if opt.aggregate_mask_path is None:
        mask_path = data_root / "aggregate_mask.npy"
    else:
        mask_path = _resolve_path(base_path, opt.aggregate_mask_path)

    valid, h_dim, w_dim = _load_aggregate_mask(mask_path)
    dyn_ckpt, rec_ckpt, _ = _resolve_checkpoint_paths(opt)
    dim_y = _infer_rec_output_dim(rec_ckpt)
    if dim_y not in (1, 3):
        raise ValueError(f"Unexpected reconstruction output dimension: {dim_y}")
    depth_only = dim_y == 1

    y_mean, y_std = _maybe_load_channel_normalization(
        opt.normalize,
        data_root / "mean.npy",
        data_root / "std.npy",
        "flow",
        logger=type("_NullLogger", (), {"info": staticmethod(print)})(),
    )
    u_mean, u_std = _maybe_load_channel_normalization(
        opt.normalize_rain,
        data_root / "rain_mean.npy",
        data_root / "rain_std.npy",
        "rain",
        logger=type("_NullLogger", (), {"info": staticmethod(print)})(),
    )

    model_cache = {
        "base_path": base_path,
        "data_root": data_root,
        "model_path": model_path,
        "mask_path": mask_path,
        "valid": valid,
        "h_dim": h_dim,
        "w_dim": w_dim,
        "dim_y": dim_y,
        "depth_only": depth_only,
        "y_mean": y_mean,
        "y_std": y_std,
        "u_mean": u_mean,
        "u_std": u_std,
        "dyn_ckpt": dyn_ckpt,
        "rec_ckpt": rec_ckpt,
    }
    return model_cache


def _predict_traj(opt: argparse.Namespace, cache: dict[str, object], traj_id: int) -> np.ndarray:
    base_path = cache["base_path"]
    data_root = cache["data_root"]
    valid = cache["valid"]
    h_dim = int(cache["h_dim"])
    w_dim = int(cache["w_dim"])
    dim_y = int(cache["dim_y"])
    depth_only = bool(cache["depth_only"])
    y_mean = cache["y_mean"]
    y_std = cache["y_std"]
    u_mean = cache["u_mean"]
    u_std = cache["u_std"]

    flow, coords, rain = _load_reduced_data(
        data_root,
        traj_id,
        opt.use_static_features,
        static_slope_xy=opt.static_slope_xy,
    )
    t_len = int(flow.shape[1])
    x = np.repeat(coords[:, None, :, :], t_len, axis=1).astype(np.float32)
    if depth_only:
        y = flow[..., [0]].astype(np.float32)
    else:
        y = flow.astype(np.float32)
        if dim_y == 1:
            y = y[..., :1]
        elif dim_y == 3:
            y = y[..., :3]
        else:
            raise ValueError(f"Unsupported reconstruction output dimension: {dim_y}")
    u = rain.astype(np.float32)

    if y_mean is not None and y_std is not None:
        y_mean_eff = y_mean[[0]] if depth_only else y_mean[:dim_y]
        y_std_eff = y_std[[0]] if depth_only else y_std[:dim_y]
        y = (y - y_mean_eff.reshape(1, 1, 1, -1)) / y_std_eff.reshape(1, 1, 1, -1)
    if u_mean is not None and u_std is not None:
        u = (u - u_mean.reshape(1, 1, -1)) / u_std.reshape(1, 1, -1)

    dim_u = int(u.shape[-1])
    dim_x = int(x.shape[-1])
    model = _build_model(opt, dim_y, dim_u, dim_x)
    dyn_ckpt = cache["dyn_ckpt"]
    rec_ckpt = cache["rec_ckpt"]
    print(f"Loading dyn checkpoint: {dyn_ckpt}")
    print(f"Loading rec checkpoint: {rec_ckpt}")
    model.dyn.load_state_dict(torch.load(dyn_ckpt, map_location="cpu"))
    model.rec.load_state_dict(torch.load(rec_ckpt, map_location="cpu"))
    if hasattr(model, "B"):
        B_ckpt = _resolve_path(base_path, opt.model_path) / f"B_{opt.checkpoint_epoch}.ckpt"
        if B_ckpt.exists():
            print(f"Loading B checkpoint: {B_ckpt}")
            model.B.load_state_dict(torch.load(B_ckpt, map_location="cpu"))
        else:
            print("WARNING: B checkpoint not found; Fourier embedding remains random.")

    device = torch.device(opt.device)
    model.to(device)
    model.eval()
    data = {
        "u": torch.from_numpy(u).to(device),
        "x": torch.from_numpy(x).to(device),
        "y": torch.from_numpy(y).to(device),
        "dt": torch.tensor([1.0], device=device, dtype=torch.float32),
    }

    with torch.no_grad():
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
        inference_start = time.perf_counter()
        pred = model(data, device, equilibrium=False, chunk_size=opt.chunk_size)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=device)
        inference_seconds = time.perf_counter() - inference_start

    pred_np = pred.cpu().numpy().astype(np.float32, copy=False)
    print(f"Trajectory {traj_id}: inference time {inference_seconds:.3f} s")
    if y_mean is not None and y_std is not None:
        y_mean_eff = y_mean[[0]] if depth_only else y_mean[:dim_y]
        y_std_eff = y_std[[0]] if depth_only else y_std[:dim_y]
        pred_np = pred_np * y_std_eff.reshape(1, 1, 1, -1) + y_mean_eff.reshape(1, 1, 1, -1)

    scatter_indices = _resolve_scatter_indices(valid, h_dim, w_dim, coords)
    if scatter_indices.size != pred_np.shape[2]:
        raise ValueError(
            f"Scatter mismatch for traj {traj_id}: scatter={scatter_indices.size}, reduced={pred_np.shape[2]}"
        )

    pred_full = np.zeros((t_len, h_dim * w_dim, dim_y), dtype=np.float32)
    pred_full[:, scatter_indices, :] = pred_np[0]
    pred_full = pred_full.reshape(t_len, h_dim, w_dim, dim_y)
    return pred_full


def _save_prediction_npz(pred_full: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, pred_full=pred_full.astype(np.float32, copy=False))


def _save_prediction_depth_animation(pred_full: np.ndarray, output_path: Path, fps: int = 5) -> None:
    depth = np.asarray(pred_full[:, :, :, 0], dtype=np.float32)
    positive = depth[np.isfinite(depth) & (depth > 0.0)]
    vmax = float(np.nanpercentile(positive, 99.5)) if positive.size else 1.0
    vmax = max(vmax, 0.05)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    depth_im = ax.imshow(
        depth[0],
        origin="upper",
        cmap="Blues",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    cbar = fig.colorbar(depth_im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.88)
    cbar.set_label("Water depth h (m)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title("CLDNet predicted water depth", fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    time_text = ax.text(
        0.02,
        0.98,
        "Hour 0",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
    )

    def update(frame_idx: int):
        depth_im.set_data(depth[frame_idx])
        time_text.set_text(f"Hour {frame_idx}")
        return depth_im, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=depth.shape[0],
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)


def main() -> None:
    opt = create_options()
    cache = _load_model_and_prepare(opt)
    output_dir = _resolve_output_dir(opt)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(
        f"Grid shape: {(int(cache['h_dim']), int(cache['w_dim']))} | "
        f"Trajectories: {opt.traj_ids}"
    )

    for traj_id in opt.traj_ids:
        out_path = output_dir / f"predictions_traj{traj_id}.npz"
        if out_path.exists() and not opt.overwrite:
            print(f"Skipping existing file: {out_path}")
            continue

        print(f"\n=== Predicting trajectory {traj_id} ===")
        pred_full = _predict_traj(opt, cache, traj_id)
        _save_prediction_npz(pred_full, out_path)
        print(f"Saved full-grid prediction to: {out_path}")

        if traj_id == opt.animate_traj_id:
            gif_path = output_dir / f"traj_{traj_id}_prediction_depth.gif"
            _save_prediction_depth_animation(pred_full, gif_path, fps=opt.fps)
            print(f"Saved smoke-test GIF to: {gif_path}")

        del pred_full
        if torch.cuda.is_available() and opt.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()
