#!/usr/bin/env python3
"""Batch LD-EnSF full-grid posterior export for selected trajectories.

This script reuses the existing LD-EnSF assimilation pipeline, decodes the
posterior latent history back onto the full 5075 x 1661 spatial grid, sets all
non-predicted cells to zero, and saves one compressed NPZ per trajectory.

It also writes a smoke-test GIF for one chosen trajectory that animates the
posterior water depth field over time.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path("checkpoints/cldnet")
DEFAULT_DATA_ROOT = Path("data/postprocessed/illinois")
DEFAULT_DATASET_PATH = Path("data/postprocessed/illinois/observation_ldensf_dataset_usgs_validation.pth")
DEFAULT_ENCODER_CHECKPOINT = Path("checkpoints/ldensf_lstm_usgs_validation/lstm_ldensf_usgs_validation.ckpt")
DEFAULT_CHECKPOINT_EPOCH = 539
DEFAULT_TRAJ_IDS = [116, 117, 118, 119, 120]
DEFAULT_ANIMATE_TRAJ_ID = 118

sys.path.append(str(REPO_ROOT))

from efficient_fourier_ldnet import EfficientFourierLDNN
from ldnet_chicago_efficient_test import _load_aggregate_mask, _resolve_scatter_indices
from test_ldensf_assimilation import (
    _denormalize,
    _ensemble_score_posterior,
    _infer_use_static_features,
    _list_trajectory_ids,
    _load_encoder,
    _load_ldnet_model,
    _load_reduced_trajectory,
    _normalize,
    _resolve_path,
    _select_background_traj_id,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "cuda" in device_str and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


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
    ax.set_title("LD-EnSF posterior water depth", fontsize=13)
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


def _decode_full_grid_posterior(
    *,
    model: EfficientFourierLDNN,
    assim_latent: np.ndarray,
    coords: np.ndarray,
    scatter_indices: np.ndarray,
    h_dim: int,
    w_dim: int,
    dim_y: int,
    device: torch.device,
) -> np.ndarray:
    t_len = int(assim_latent.shape[0])
    pred_full = np.zeros((t_len, h_dim * w_dim, dim_y), dtype=np.float32)
    coords_arr = np.asarray(coords, dtype=np.float32)
    if coords_arr.ndim == 3:
        coords_arr = coords_arr[0]
    if coords_arr.ndim != 2:
        raise ValueError(f"Expected reduced coordinates with shape (N, D) or (1, N, D); got {coords_arr.shape}")
    x_tensor = torch.from_numpy(coords_arr[None, :, :]).to(device)

    with torch.no_grad():
        for t in tqdm(range(t_len), desc="Decoding posterior", leave=False):
            latent_t = torch.from_numpy(np.asarray(assim_latent[t : t + 1], dtype=np.float32)).to(device)
            pred_t = model._decode_points(x_tensor, latent_t, chunk_size=model.chunk_size)
            pred_full[t, scatter_indices, :] = pred_t[0].detach().cpu().numpy().astype(np.float32, copy=False)

    return pred_full.reshape(t_len, h_dim, w_dim, dim_y)


def create_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch LD-EnSF full-grid posterior export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-path", type=Path, default=DEFAULT_PYTHON_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--encoder-checkpoint", type=Path, default=DEFAULT_ENCODER_CHECKPOINT)
    parser.add_argument("--aggregate-mask-path", type=Path, default=Path("data/postprocessed/illinois/aggregate_mask.npy"))
    parser.add_argument("--checkpoint-epoch", type=int, default=DEFAULT_CHECKPOINT_EPOCH)
    parser.add_argument("--dyn-checkpoint", type=Path, default=None)
    parser.add_argument("--rec-checkpoint", type=Path, default=None)
    parser.add_argument("--B-checkpoint", type=Path, default=None)
    parser.add_argument("--traj-ids", type=int, nargs="+", default=DEFAULT_TRAJ_IDS)
    parser.add_argument("--animate-traj-id", type=int, default=DEFAULT_ANIMATE_TRAJ_ID)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--da-seed", type=int, default=0)
    parser.add_argument("--obs-noise-std", type=float, default=0.05)
    parser.add_argument("--obs-sigma", type=float, default=0.1)
    parser.add_argument("--latent-scaling", type=float, default=1.0)
    parser.add_argument("--eps-alpha", type=float, default=0.05)
    parser.add_argument("--euler-steps", type=int, default=100)
    parser.add_argument("--ensemble-size", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--num-latent-states", type=int, default=200)
    parser.add_argument("--fourier-mapping-size", type=int, default=32)
    parser.add_argument("--NN-dyn-depth", type=int, default=8)
    parser.add_argument("--NN-dyn-width", type=int, default=50)
    parser.add_argument("--NN-rec-depth", type=int, default=10)
    parser.add_argument("--NN-rec-width", type=int, default=300)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--kernel-initializer", type=str, default="Glorot normal")
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--background-traj-id", type=int, default=None)
    static_group = parser.add_mutually_exclusive_group()
    static_group.add_argument("--use-static-features", dest="use_static_features", action="store_true")
    static_group.add_argument("--no-static-features", dest="use_static_features", action="store_false")
    parser.set_defaults(use_static_features=None)
    return parser.parse_args()


def _resolve_output_dir(opt: argparse.Namespace) -> Path:
    if opt.output_dir is None:
        return _resolve_path(opt.base_path, opt.model_path) / "fullgrid_ldensf_usgs_validation_traj116_120"
    if opt.output_dir.is_absolute():
        return opt.output_dir
    return opt.base_path / opt.output_dir


def _compute_assimilation_latent_history(
    *,
    model: EfficientFourierLDNN,
    encoder,
    enc_stats: dict[str, torch.Tensor | None],
    obs_noisy: np.ndarray,
    rain_tensor: torch.Tensor,
    t_len: int,
    opt: argparse.Namespace,
    device: torch.device,
    da_seed: int,
) -> np.ndarray:
    obs_tensor = torch.from_numpy(obs_noisy.reshape(1, t_len, -1)).to(device)
    obs_mean = enc_stats["obs_mean"].to(device) if enc_stats["obs_mean"] is not None else None
    obs_std = enc_stats["obs_std"].to(device) if enc_stats["obs_std"] is not None else None
    latent_mean = enc_stats["latent_mean"].to(device) if enc_stats["latent_mean"] is not None else None
    latent_std = enc_stats["latent_std"].to(device) if enc_stats["latent_std"] is not None else None

    obs_tensor = _normalize(obs_tensor, obs_mean, obs_std)
    with torch.no_grad():
        obs_latent = encoder(obs_tensor).squeeze(0)
    obs_latent = _denormalize(obs_latent, latent_mean, latent_std)

    device_rng = torch.Generator(device=device.type)
    device_rng.manual_seed(da_seed)

    state = torch.zeros(opt.ensemble_size, opt.num_latent_states, device=device)
    assim_latent_hist: list[torch.Tensor] = []
    for t in tqdm(range(t_len), desc="LD-EnSF", leave=False):
        u_t = rain_tensor[:, t, :].expand(opt.ensemble_size, -1)
        with torch.no_grad():
            state = state + model.dyn(torch.cat([u_t, state], dim=1))
        state = _ensemble_score_posterior(
            prior=state,
            obs_latent=obs_latent[t],
            time_steps=opt.euler_steps,
            obs_sigma=opt.obs_sigma,
            latent_scaling=opt.latent_scaling,
            eps_alpha=opt.eps_alpha,
            generator=device_rng,
        )
        assim_latent_hist.append(state.mean(dim=0).detach().cpu())

    return torch.stack(assim_latent_hist, dim=0).numpy()


def main(opt: argparse.Namespace) -> None:
    set_seed(opt.seed)
    device = resolve_device(opt.device)

    base_path = opt.base_path.resolve()
    data_root = _resolve_path(base_path, opt.data_root)
    dataset_path = _resolve_path(base_path, opt.dataset_path)
    model_path = _resolve_path(base_path, opt.model_path)
    encoder_path = _resolve_path(base_path, opt.encoder_checkpoint)
    output_dir = _resolve_output_dir(opt)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_static_features = _infer_use_static_features(model_path, opt.use_static_features)

    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    obs_idx = dataset["meta"]["selected_obs_idx"]
    if isinstance(obs_idx, torch.Tensor):
        obs_idx = obs_idx.cpu().numpy()
    else:
        obs_idx = np.asarray(obs_idx)

    available_traj_ids = _list_trajectory_ids(data_root)
    mask_path = _resolve_path(base_path, opt.aggregate_mask_path)
    domain_mask = np.load(mask_path)
    if domain_mask.ndim != 2:
        raise ValueError(f"Expected aggregate mask with shape (H, W); got {domain_mask.shape}")
    h_dim, w_dim = domain_mask.shape

    encoder, enc_stats, encoder_normalized = _load_encoder(encoder_path, device)

    print(f"Output directory: {output_dir}")
    print(f"Grid shape: {(h_dim, w_dim)} | Trajectories: {opt.traj_ids}")
    print(f"Using static features: {use_static_features}")
    print(f"Encoder normalized input: {encoder_normalized}")
    print(f"DA base seed: {opt.da_seed}")

    for traj_id in opt.traj_ids:
        print(f"\n=== Predicting trajectory {traj_id} ===")
        traj_seed = int(opt.da_seed) + int(traj_id)

        background_traj_id = opt.background_traj_id
        if background_traj_id is None:
            background_traj_id = _select_background_traj_id(
                available_traj_ids,
                target_traj_id=traj_id,
                requested_traj_id=None,
            )
        elif background_traj_id not in available_traj_ids:
            raise ValueError(f"Background trajectory id {background_traj_id} is not available")

        flow, coords, rain = _load_reduced_trajectory(data_root, traj_id, use_static_features)
        _, _, rain_bg = _load_reduced_trajectory(data_root, background_traj_id, use_static_features)
        t_len = int(flow.shape[1])
        dim_y = int(flow.shape[-1])
        dim_u = int(rain.shape[-1])
        dim_x = int(coords.shape[-1])

        # Rebuild the model with the correct feature dimensions for this dataset.
        model = _load_ldnet_model(
            opt=opt,
            device=device,
            dim_u=dim_u,
            dim_x=dim_x,
            dim_y=dim_y,
        )
        print(f"Loading dynamic/reconstruction checkpoints from: {model_path}")

        obs_truth = np.asarray(flow[0][:, obs_idx, :], dtype=np.float32)
        cpu_rng = torch.Generator(device="cpu")
        cpu_rng.manual_seed(traj_seed)
        obs_noise = opt.obs_noise_std * torch.randn(obs_truth.shape, generator=cpu_rng, dtype=torch.float32)
        obs_noisy = obs_truth + obs_noise.numpy().astype(np.float32)

        rain_tensor = torch.from_numpy(np.asarray(rain_bg, dtype=np.float32)).to(device)
        assim_latent = _compute_assimilation_latent_history(
            model=model,
            encoder=encoder,
            enc_stats=enc_stats,
            obs_noisy=obs_noisy,
            rain_tensor=rain_tensor,
            t_len=t_len,
            opt=opt,
            device=device,
            da_seed=traj_seed,
        )

        scatter_indices = _resolve_scatter_indices(domain_mask, h_dim, w_dim, np.asarray(coords))
        pred_full = _decode_full_grid_posterior(
            model=model,
            assim_latent=assim_latent,
            coords=np.asarray(coords, dtype=np.float32),
            scatter_indices=scatter_indices,
            h_dim=h_dim,
            w_dim=w_dim,
            dim_y=dim_y,
            device=device,
        )

        out_path = output_dir / f"predictions_traj{traj_id}.npz"
        if out_path.exists() and not opt.overwrite:
            print(f"Skipping existing file: {out_path}")
        else:
            np.savez_compressed(
                out_path,
                pred_full=pred_full.astype(np.float32, copy=False),
                traj_id=int(traj_id),
                background_traj_id=int(background_traj_id),
                da_seed=int(traj_seed),
                obs_idx=np.asarray(obs_idx, dtype=np.int64),
            )
            print(f"Saved full-grid posterior to: {out_path}")

        if traj_id == opt.animate_traj_id:
            gif_path = output_dir / f"traj_{traj_id}_ldensf_prediction_depth.gif"
            _save_prediction_depth_animation(pred_full, gif_path, fps=opt.fps)
            print(f"Saved smoke-test GIF to: {gif_path}")

        del pred_full
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main(create_options())
