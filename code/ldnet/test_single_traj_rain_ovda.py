#!/usr/bin/env python3
"""Single-trajectory OVDA-style test for hydro_surrogate_light.

This script compares three rain-driven trajectories for one example from
`data/postprocessed/illinois`:
1. Forecast with the original rainfall trajectory.
2. Forecast with rainfall from another trajectory.
3. OVDA/LEVDA-style assimilation using 200 noisy measurement points.

The GIF renders channel 0 as a domain map on a coarse version of the masked
grid, with noisy observation markers overlaid on the truth panel.

Optionally, a learned rain-propagation LSTM can be used to roll rainfall
forward instead of keeping the current rain fixed over the forecast horizon.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
from matplotlib import animation as mpl_animation
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from efficient_fourier_ldnet import EfficientFourierLDNN
from src.rain_lstm import RainPropagatorLSTM, load_checkpoint as load_rain_checkpoint, rollout_window as rollout_rain_window, rollout_sequence as rollout_rain_sequence


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def create_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-path", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/cldnet"))
    parser.add_argument("--output-dir", type=Path, default=Path("single_traj_da"))
    parser.add_argument("--traj-id", type=int, default=0)
    parser.add_argument(
        "--rain-background-traj-id",
        type=int,
        default=None,
        help=(
            "Trajectory id to use as the rain background. Defaults to the next available trajectory id; "
            "pass the same traj id if you want the true rain background."
        ),
    )
    parser.add_argument(
        "--use-true-rain-background",
        action="store_true",
        help="Force the background rain to come from the same trajectory as --traj-id.",
    )
    parser.add_argument(
        "--rain-propagator-checkpoint",
        type=Path,
        default=None,
        help="Optional LSTM checkpoint that autoregressively propagates rainfall forward in time.",
    )
    parser.add_argument(
        "--rain-seed-len",
        type=int,
        default=1,
        help="Number of initial rain steps to seed the rain propagator with.",
    )

    parser.add_argument("--num-latent-states", type=int, default=200)
    parser.add_argument("--fourier-mapping-size", type=int, default=32)
    parser.add_argument("--NN-dyn-depth", type=int, default=8)
    parser.add_argument("--NN-dyn-width", type=int, default=50)
    parser.add_argument("--NN-rec-depth", type=int, default=10)
    parser.add_argument("--NN-rec-width", type=int, default=300)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--kernel-initializer", type=str, default="Glorot normal")
    parser.add_argument("--chunk-size", type=int, default=10000)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--da-seed", type=int, default=None, help="Seed for DA noise and perturbations.")
    parser.add_argument("--checkpoint-epoch", type=int, default=539)
    parser.add_argument("--dyn-checkpoint", type=Path, default=None)
    parser.add_argument("--rec-checkpoint", type=Path, default=None)
    parser.add_argument("--B-checkpoint", type=Path, default=None)

    parser.add_argument("--obs-point-count", type=int, default=200)
    parser.add_argument(
        "--obs-water-depth-threshold",
        type=float,
        default=1.0,
        help="Only sample observation points whose channel-0 depth exceeds this value at least once over the trajectory.",
    )
    parser.add_argument("--obs-noise-std", type=float, default=0.05)
    parser.add_argument("--obs-sigma", type=float, default=0.05)
    parser.add_argument("--ensemble-size", type=int, default=20)
    parser.add_argument("--smoothing-steps", type=int, default=5)
    parser.add_argument("--lbfgs-max-iter", type=int, default=50)
    parser.add_argument("--u-init-noise-std", type=float, default=0.05)
    parser.add_argument("--state-reg", type=float, default=1.0)
    parser.add_argument("--u-reg", type=float, default=0.1)
    parser.add_argument("--plot-channel", type=int, default=0)
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--gif-stride", type=int, default=1)
    parser.add_argument("--map-stride", type=int, default=10, help="Spatial stride for the domain-map GIF.")
    parser.add_argument("--save-name", type=str, default=None)

    static_group = parser.add_mutually_exclusive_group()
    static_group.add_argument("--use-static-features", dest="use_static_features", action="store_true")
    static_group.add_argument("--no-static-features", dest="use_static_features", action="store_false")
    parser.set_defaults(use_static_features=None)

    return parser.parse_args()


def _normalize(arr: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if mean is None or std is None:
        return out
    reshape = (1,) * (out.ndim - 1) + (mean.shape[0],)
    return (out - mean.reshape(reshape)) / std.reshape(reshape)


def _denormalize(arr: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if mean is None or std is None:
        return out
    reshape = (1,) * (out.ndim - 1) + (mean.shape[0],)
    return out * std.reshape(reshape) + mean.reshape(reshape)


def _load_stats(search_dirs: list[Path], candidate_pairs: list[tuple[str, str]]) -> tuple[np.ndarray | None, np.ndarray | None, Path | None]:
    for directory in search_dirs:
        for mean_name, std_name in candidate_pairs:
            mean_path = directory / mean_name
            std_path = directory / std_name
            if mean_path.exists() and std_path.exists():
                mean = np.load(mean_path).astype(np.float32, copy=False)
                std = np.load(std_path).astype(np.float32, copy=False)
                if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
                    raise ValueError(
                        f"Normalization files must be matching 1D arrays; got {mean_path}={mean.shape} and {std_path}={std.shape}"
                    )
                if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)):
                    raise ValueError(f"Normalization files contain non-finite values: {mean_path}, {std_path}")
                if np.any(std <= 0):
                    raise ValueError(f"Normalization std contains non-positive values: {std_path}")
                return mean, std, directory
    return None, None, None


def _load_trajectory(data_root: Path, traj_id: int, use_static_features: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow = np.load(data_root / f"flow_variables_traj{traj_id}.npy", mmap_mode="r")
    coords = np.load(data_root / f"coords_traj{traj_id}.npy", mmap_mode="r")
    rain = np.load(data_root / f"rain_source_traj{traj_id}.npy", mmap_mode="r")
    if use_static_features:
        static = np.load(data_root / f"static_features_traj{traj_id}.npy", mmap_mode="r")
        coords = np.concatenate([coords, static], axis=-1)
    return flow, coords, rain


def _available_rain_traj_ids(data_root: Path) -> list[int]:
    ids: list[int] = []
    for path in data_root.glob("rain_source_traj*.npy"):
        match = re.search(r"traj(\d+)\.npy$", path.name)
        if match is not None:
            ids.append(int(match.group(1)))
    ids = sorted(set(ids))
    if not ids:
        raise FileNotFoundError(f"No rain_source_traj*.npy files found in {data_root}")
    return ids


def _resolve_rain_background_traj_id(data_root: Path, traj_id: int, requested: int | None) -> int:
    available_ids = _available_rain_traj_ids(data_root)
    if traj_id not in available_ids:
        raise ValueError(f"traj-id {traj_id} is not present in {data_root}")

    if requested is not None:
        if requested not in available_ids:
            raise ValueError(f"rain-background-traj-id {requested} is not present in {data_root}")
        return requested

    start_idx = available_ids.index(traj_id)
    for offset in range(1, len(available_ids) + 1):
        candidate = available_ids[(start_idx + offset) % len(available_ids)]
        if candidate != traj_id:
            return candidate

    raise ValueError("Could not find a distinct rain background trajectory")


def _build_time_broadcast_coords(coords: np.ndarray, t_len: int) -> np.ndarray:
    return np.repeat(coords[:, None, :, :], t_len, axis=1).astype(np.float32, copy=False)


def _relative_l2_error(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    truth_flat = truth.reshape(truth.shape[0], -1)
    pred_flat = pred.reshape(pred.shape[0], -1)
    valid = np.isfinite(truth_flat) & np.isfinite(pred_flat)
    denom = np.linalg.norm(np.where(valid, truth_flat, 0.0), axis=1) + 1e-12
    return np.linalg.norm(np.where(valid, pred_flat - truth_flat, 0.0), axis=1) / denom


def _build_coords(h_dim: int, w_dim: int) -> np.ndarray:
    x = 2.0 * (np.arange(w_dim, dtype=np.float32) / w_dim - 0.5)
    aspect = h_dim / float(w_dim)
    y = 2.0 * (np.arange(h_dim, dtype=np.float32) / h_dim - 0.5) * aspect
    x_coords = np.tile(x, h_dim)
    y_coords = np.repeat(y, w_dim)
    return np.stack([x_coords, y_coords], axis=1)[None, :, :].astype(np.float32)


def _resolve_scatter_indices(valid_mask: np.ndarray, h_dim: int, w_dim: int, coords: np.ndarray) -> np.ndarray:
    mask_indices = np.flatnonzero(valid_mask.reshape(-1).astype(bool))
    n_reduced = int(coords.shape[1])
    if mask_indices.size < n_reduced:
        raise ValueError(f"Mask points {mask_indices.size} < reduced points {n_reduced}")
    if mask_indices.size == n_reduced:
        return mask_indices

    full_coords = _build_coords(h_dim, w_dim).reshape(-1, 2).astype(np.float16)
    masked_coords = full_coords[mask_indices]
    reduced_xy = coords[0, :, :2].astype(np.float16, copy=False)

    coord_to_flat = {}
    for idx, xy in zip(mask_indices, masked_coords):
        coord_to_flat[(float(xy[0]), float(xy[1]))] = int(idx)

    scatter_indices = np.empty(n_reduced, dtype=np.int64)
    for j, xy in enumerate(reduced_xy):
        key = (float(xy[0]), float(xy[1]))
        flat_idx = coord_to_flat.get(key)
        if flat_idx is None:
            raise ValueError(f"Could not map reduced coord {key} at index {j}")
        scatter_indices[j] = flat_idx
    return scatter_indices


def _select_obs_indices(
    flow: np.ndarray,
    obs_point_count: int,
    seed: int,
    depth_threshold: float,
    depth_channel: int = 0,
) -> tuple[np.ndarray, int]:
    flow_phys = np.asarray(flow, dtype=np.float32)
    if flow_phys.ndim != 4:
        raise ValueError(f"Expected flow with shape (1, T, N, C); got {flow_phys.shape}")

    depth_series = flow_phys[0, :, :, depth_channel]
    eligible = np.where(np.max(depth_series, axis=0) > depth_threshold)[0]
    if eligible.size == 0:
        raise ValueError(
            f"No spatial points exceed depth threshold {depth_threshold} on channel {depth_channel}."
        )

    count = min(obs_point_count, eligible.size)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(eligible, size=count, replace=False)), int(eligible.size)


def _infer_use_static_features(opt: argparse.Namespace) -> bool:
    if opt.use_static_features is not None:
        return opt.use_static_features
    return opt.model_path.name.lower() == "cldnet" or "static" in opt.model_path.name.lower()


def _load_rain_propagator(
    opt: argparse.Namespace,
    device: torch.device,
) -> tuple[RainPropagatorLSTM | None, torch.Tensor | None, torch.Tensor | None, bool, Path | None]:
    if opt.rain_propagator_checkpoint is None:
        return None, None, None, False, None

    checkpoint_path = (
        opt.base_path / opt.rain_propagator_checkpoint
        if not opt.rain_propagator_checkpoint.is_absolute()
        else opt.rain_propagator_checkpoint
    )
    model, rain_mean, rain_std, metadata = load_rain_checkpoint(checkpoint_path, device)
    normalize = bool(metadata.get("normalize", False))
    return model, rain_mean, rain_std, normalize, checkpoint_path


def _rollout_rain_forcing(
    rain: torch.Tensor,
    rain_model: RainPropagatorLSTM | None,
    seed_len: int,
    *,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor:
    if rain_model is None:
        return rain
    seed_len = max(1, min(int(seed_len), int(rain.shape[1])))
    return rollout_rain_sequence(rain_model, rain[:, :seed_len, :], int(rain.shape[1]), mean=mean, std=std)


def _load_model(
    opt: argparse.Namespace,
    device: torch.device,
    dim_u: int,
    dim_x: int,
    dim_y: int,
) -> EfficientFourierLDNN:
    model = EfficientFourierLDNN(
        opt.fourier_mapping_size,
        [opt.num_latent_states + dim_u] + opt.NN_dyn_depth * [opt.NN_dyn_width] + [opt.num_latent_states],
        [opt.num_latent_states + dim_x] + opt.NN_rec_depth * [opt.NN_rec_width] + [dim_y],
        activation=opt.activation,
        kernel_initializer=opt.kernel_initializer,
        chunk_size=opt.chunk_size,
    )

    model_dir = opt.base_path / opt.model_path
    if opt.dyn_checkpoint is not None and opt.rec_checkpoint is not None:
        dyn_ckpt = opt.dyn_checkpoint
        rec_ckpt = opt.rec_checkpoint
        b_ckpt = opt.B_checkpoint
    else:
        dyn_ckpt = model_dir / f"dyn_{opt.checkpoint_epoch}.ckpt"
        rec_ckpt = model_dir / f"rec_{opt.checkpoint_epoch}.ckpt"
        b_ckpt = opt.B_checkpoint if opt.B_checkpoint is not None else model_dir / f"B_{opt.checkpoint_epoch}.ckpt"

    model.dyn.load_state_dict(torch.load(dyn_ckpt, map_location="cpu", weights_only=False))
    model.rec.load_state_dict(torch.load(rec_ckpt, map_location="cpu", weights_only=False))
    if b_ckpt is not None and Path(b_ckpt).exists():
        model.B.load_state_dict(torch.load(b_ckpt, map_location="cpu", weights_only=False))

    model.to(device)
    model.eval()
    return model


@contextmanager
def _freeze_modules(*modules: torch.nn.Module):
    param_states: list[tuple[torch.nn.Parameter, bool]] = []
    for module in modules:
        for param in module.parameters():
            param_states.append((param, param.requires_grad))
            param.requires_grad_(False)
    try:
        yield
    finally:
        for param, state in param_states:
            param.requires_grad_(state)


def _run_ovda_assimilation(
    model: EfficientFourierLDNN,
    rain_bg: torch.Tensor,
    x_obs: torch.Tensor,
    observations: torch.Tensor,
    dt: torch.Tensor,
    device: torch.device,
    rng: torch.Generator,
    ensemble_size: int,
    obs_sigma: float,
    smoothing_steps: int,
    lbfgs_max_iter: int,
    u_init_noise_std: float,
    state_reg: float,
    u_reg: float,
    rain_model: RainPropagatorLSTM | None = None,
    rain_mean: torch.Tensor | None = None,
    rain_std: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    t_len = rain_bg.shape[1]

    u = rain_bg.repeat(ensemble_size, 1, 1)
    u = u + u_init_noise_std * torch.randn(u.shape, device=device, dtype=u.dtype, generator=rng)
    state = torch.zeros(ensemble_size, model.num_latent_states, device=device)
    x_obs_ens = x_obs.repeat(ensemble_size, 1, 1, 1)

    pred_hist: list[torch.Tensor] = []
    state_hist: list[torch.Tensor] = []
    u_hist: list[torch.Tensor] = []

    with _freeze_modules(model.dyn, model.rec):
        for i in tqdm(range(t_len), desc="OVDA", leave=False):
            with torch.no_grad():
                dyn_input = torch.cat((u[:, i, :], state), dim=1)
                state = state + dt * model.dyn(dyn_input)

            if i < t_len - smoothing_steps:
                u_ti = u[:, i, :].detach()
                u_ti_perturb = torch.nn.Parameter(
                    0.0001 * torch.randn(u_ti.shape, device=device, dtype=u.dtype, generator=rng)
                )
                alpha = torch.nn.Parameter(
                    torch.eye(ensemble_size, device=device, dtype=u.dtype) * math.sqrt(ensemble_size - 1)
                )
                alpha_orig = alpha.detach().clone()
                state_mean = state.mean(dim=0, keepdim=True)
                perturb_basis = state - state_mean
                px = (perturb_basis / math.sqrt(ensemble_size - 1)).permute(1, 0)

                optimizer = torch.optim.LBFGS([alpha, u_ti_perturb], max_iter=lbfgs_max_iter, lr=0.001)

                def closure() -> torch.Tensor:
                    optimizer.zero_grad()
                    temp_state = state_mean + torch.matmul(px, alpha).permute(1, 0)
                    u_assim = u_ti + u_ti_perturb
                    u_prefix = torch.cat((u[:, :i, :].detach(), u_assim.unsqueeze(1)), dim=1)
                    with torch.no_grad():
                        u_window = rollout_rain_window(
                            rain_model,
                            u_prefix,
                            smoothing_steps,
                            mean=rain_mean,
                            std=rain_std,
                        )
                    pred_obs = model._decode_points(
                        x_obs_ens[:, i, :, :],
                        temp_state,
                        chunk_size=model.chunk_size,
                    )

                    loss = torch.mean((pred_obs - observations[:, i, :, :]) ** 2) / (obs_sigma**2 + 1e-12)
                    temp_states = [temp_state]
                    for j in range(smoothing_steps):
                        next_idx = i + j + 1
                        if next_idx >= t_len:
                            break
                        dyn_input = torch.cat((u_window[:, j, :], temp_states[j]), dim=1)
                        dyn_output = model.dyn(dyn_input)
                        temp_states.append(temp_states[j] + dt * dyn_output)
                        pred_obs = model._decode_points(
                            x_obs_ens[:, next_idx, :, :],
                            temp_states[-1],
                            chunk_size=model.chunk_size,
                        )
                        loss = loss + torch.mean((pred_obs - observations[:, next_idx, :, :]) ** 2) / (
                            obs_sigma**2 + 1e-12
                        )

                    loss = loss + state_reg * torch.mean((alpha - alpha_orig) ** 2)
                    loss = loss + u_reg * torch.mean(u_ti_perturb**2)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([alpha, u_ti_perturb], 1.0)
                    return loss

                optimizer.step(closure)
                with torch.no_grad():
                    current_u = (u_ti + u_ti_perturb).detach()
                    u_prefix = torch.cat((u[:, :i, :].detach(), current_u.unsqueeze(1)), dim=1)
                    # Carry the assimilated rain forward using the learned rain dynamics when available.
                    u[:, i:, :] = rollout_rain_window(
                        rain_model,
                        u_prefix,
                        t_len - i,
                        mean=rain_mean,
                        std=rain_std,
                    )
                    state = (state_mean + torch.matmul(px, alpha).permute(1, 0)).detach()

            state_hist.append(state.detach().cpu())
            u_hist.append(u[:, i, :].detach().cpu())

            with torch.no_grad():
                pred_obs = model._decode_points(
                    x_obs_ens[:, i, :, :],
                    state,
                    chunk_size=model.chunk_size,
                ).detach().cpu()
            pred_hist.append(pred_obs)

    pred_hist_np = torch.stack(pred_hist, dim=0).numpy()  # (T, E, n_obs, C)
    state_hist_np = torch.stack(state_hist, dim=0).numpy()  # (T, E, latent)
    u_hist_np = torch.stack(u_hist, dim=0).numpy()  # (T, E, dim_u)
    return pred_hist_np, state_hist_np, u_hist_np


def _make_plot(
    time: np.ndarray,
    truth_mean: np.ndarray,
    obs_mean: np.ndarray,
    obs_samples: np.ndarray,
    orig_mean: np.ndarray,
    bg_mean: np.ndarray,
    assim_mean: np.ndarray,
    err_orig: np.ndarray,
    err_bg: np.ndarray,
    err_assim: np.ndarray,
    plot_channel: int,
    bg_label: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True)

    left_titles = [
        "Data trajectory + noisy observations",
        "Original rainfall traj",
        bg_label,
        "Data assimilation traj",
    ]
    left_series = [truth_mean, orig_mean, bg_mean, assim_mean]
    left_colors = ["black", "tab:blue", "tab:red", "tab:green"]
    right_series = [np.abs(obs_mean - truth_mean), err_orig, err_bg, err_assim]
    right_colors = ["tab:orange", "tab:blue", "tab:red", "tab:green"]

    left_min = min(
        truth_mean.min(),
        obs_mean.min(),
        orig_mean.min(),
        bg_mean.min(),
        assim_mean.min(),
    )
    left_max = max(
        truth_mean.max(),
        obs_mean.max(),
        orig_mean.max(),
        bg_mean.max(),
        assim_mean.max(),
    )
    left_pad = 0.05 * (left_max - left_min + 1e-12)
    right_max = max(err_orig.max(), err_bg.max(), err_assim.max(), np.abs(obs_mean - truth_mean).max())

    for row in range(4):
        ax_left = axes[row, 0]
        ax_right = axes[row, 1]

        ax_left.set_title(left_titles[row], fontsize=11)
        if row == 0:
            ax_left.scatter(
                np.repeat(time, obs_samples.shape[1]),
                obs_samples.reshape(-1),
                s=6,
                alpha=0.10,
                color="tab:orange",
                label="noisy obs",
            )
            ax_left.plot(time, obs_mean, color="tab:orange", lw=1.5, label="obs mean")
            ax_left.plot(time, truth_mean, color="black", lw=2.0, label="data traj")
        else:
            ax_left.plot(time, truth_mean, color="black", lw=1.8, label="data traj")
            ax_left.plot(time, left_series[row], color=left_colors[row], lw=1.8, label=left_titles[row])
        ax_left.legend(frameon=False, fontsize=8, loc="best")
        ax_left.set_ylabel(f"mean channel {plot_channel}")
        ax_left.set_ylim(left_min - left_pad, left_max + left_pad)
        ax_left.grid(alpha=0.25)

        ax_right.plot(time, right_series[row], color=right_colors[row], lw=1.8)
        ax_right.set_ylabel("error")
        ax_right.grid(alpha=0.25)
        ax_right.set_ylim(0.0, 1.05 * right_max + 1e-12)

        if row == 3:
            ax_left.set_xlabel("time step")
            ax_right.set_xlabel("time step")

    axes[0, 1].set_title("Observation mismatch", fontsize=11)
    axes[1, 1].set_title("Trajectory errors", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _make_gif(
    time: np.ndarray,
    truth_mean: np.ndarray,
    obs_mean: np.ndarray,
    obs_samples: np.ndarray,
    orig_mean: np.ndarray,
    bg_mean: np.ndarray,
    assim_mean: np.ndarray,
    err_orig: np.ndarray,
    err_bg: np.ndarray,
    err_assim: np.ndarray,
    plot_channel: int,
    bg_label: str,
    out_path: Path,
    fps: int,
    stride: int,
) -> None:
    time = np.asarray(time)
    stride = max(1, int(stride))
    frame_indices = list(range(0, len(time), stride))
    if frame_indices[-1] != len(time) - 1:
        frame_indices.append(len(time) - 1)

    fig, axes = plt.subplots(4, 2, figsize=(18, 14), sharex=True)

    left_titles = [
        "Data trajectory + noisy observations",
        "Original rainfall traj",
        bg_label,
        "Data assimilation traj",
    ]
    left_series = [truth_mean, orig_mean, bg_mean, assim_mean]
    left_colors = ["black", "tab:blue", "tab:red", "tab:green"]
    right_series = [np.abs(obs_mean - truth_mean), err_orig, err_bg, err_assim]
    right_colors = ["tab:orange", "tab:blue", "tab:red", "tab:green"]

    left_min = min(
        truth_mean.min(),
        obs_mean.min(),
        orig_mean.min(),
        bg_mean.min(),
        assim_mean.min(),
    )
    left_max = max(
        truth_mean.max(),
        obs_mean.max(),
        orig_mean.max(),
        bg_mean.max(),
        assim_mean.max(),
    )
    left_pad = 0.05 * (left_max - left_min + 1e-12)
    right_max = max(err_orig.max(), err_bg.max(), err_assim.max(), np.abs(obs_mean - truth_mean).max())
    right_ylim = (0.0, 1.05 * right_max + 1e-12)

    ax_artists: list[dict[str, object]] = []

    for row in range(4):
        ax_left = axes[row, 0]
        ax_right = axes[row, 1]
        ax_left.set_title(left_titles[row], fontsize=11)

        if row == 0:
            ax_left.scatter(
                np.repeat(time, obs_samples.shape[1]),
                obs_samples.reshape(-1),
                s=6,
                alpha=0.08,
                color="tab:orange",
                label="noisy obs",
            )
            ax_left.plot(time, truth_mean, color="black", lw=1.8, alpha=0.22, label="data traj")
            ax_left.plot(time, obs_mean, color="tab:orange", lw=1.5, alpha=0.22, label="obs mean")
            (truth_prog,) = ax_left.plot([], [], color="black", lw=2.2)
            (obs_prog,) = ax_left.plot([], [], color="tab:orange", lw=2.0)
            truth_marker = ax_left.plot([], [], marker="o", color="black", markersize=4, linestyle="None")[0]
            obs_marker = ax_left.plot([], [], marker="o", color="tab:orange", markersize=4, linestyle="None")[0]
            ax_left.legend(frameon=False, fontsize=8, loc="best")

            ax_right.plot(time, right_series[row], color="tab:orange", lw=1.5, alpha=0.22)
            (err_prog,) = ax_right.plot([], [], color="tab:orange", lw=2.0)
            err_marker = ax_right.plot([], [], marker="o", color="tab:orange", markersize=4, linestyle="None")[0]
            ax_right.set_title("Observation mismatch", fontsize=11)
        else:
            ax_left.plot(time, truth_mean, color="black", lw=1.8, alpha=0.22, label="data traj")
            ax_left.plot(time, left_series[row], color=left_colors[row], lw=1.5, alpha=0.22, label=left_titles[row])
            (truth_prog,) = ax_left.plot([], [], color="black", lw=2.2)
            (obs_prog,) = ax_left.plot([], [], color=left_colors[row], lw=2.0)
            truth_marker = ax_left.plot([], [], marker="o", color="black", markersize=4, linestyle="None")[0]
            obs_marker = ax_left.plot([], [], marker="o", color=left_colors[row], markersize=4, linestyle="None")[0]
            ax_left.legend(frameon=False, fontsize=8, loc="best")

            ax_right.plot(time, right_series[row], color=right_colors[row], lw=1.5, alpha=0.22)
            (err_prog,) = ax_right.plot([], [], color=right_colors[row], lw=2.0)
            err_marker = ax_right.plot([], [], marker="o", color=right_colors[row], markersize=4, linestyle="None")[0]
            ax_right.set_title("Trajectory errors", fontsize=11)

        left_vline = ax_left.axvline(time[0], color="0.35", linestyle=":", lw=1.0, alpha=0.8)
        right_vline = ax_right.axvline(time[0], color="0.35", linestyle=":", lw=1.0, alpha=0.8)

        ax_left.set_ylabel(f"mean channel {plot_channel}")
        ax_left.set_ylim(left_min - left_pad, left_max + left_pad)
        ax_left.grid(alpha=0.25)

        ax_right.set_ylabel("error")
        ax_right.set_ylim(*right_ylim)
        ax_right.grid(alpha=0.25)

        if row == 3:
            ax_left.set_xlabel("time step")
            ax_right.set_xlabel("time step")

        ax_artists.append(
            {
                "truth_prog": truth_prog,
                "obs_prog": obs_prog,
                "truth_marker": truth_marker,
                "obs_marker": obs_marker,
                "err_prog": err_prog,
                "err_marker": err_marker,
                "left_vline": left_vline,
                "right_vline": right_vline,
            }
        )

    suptitle = fig.suptitle("Single-trajectory OVDA comparison", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    def update(frame_idx: int):
        idx = frame_indices[frame_idx]
        current_time = time[idx]
        suptitle.set_text(f"Single-trajectory OVDA comparison | frame {frame_idx + 1}/{len(frame_indices)} | time step {idx}")
        ax_artists[0]["truth_prog"].set_data(time[: idx + 1], truth_mean[: idx + 1])
        ax_artists[0]["obs_prog"].set_data(time[: idx + 1], obs_mean[: idx + 1])
        ax_artists[0]["truth_marker"].set_data([current_time], [truth_mean[idx]])
        ax_artists[0]["obs_marker"].set_data([current_time], [obs_mean[idx]])
        ax_artists[0]["err_prog"].set_data(time[: idx + 1], right_series[0][: idx + 1])
        ax_artists[0]["err_marker"].set_data([current_time], [right_series[0][idx]])
        ax_artists[0]["left_vline"].set_xdata([current_time, current_time])
        ax_artists[0]["right_vline"].set_xdata([current_time, current_time])

        for row in range(1, 4):
            ax_artists[row]["truth_prog"].set_data(time[: idx + 1], truth_mean[: idx + 1])
            ax_artists[row]["obs_prog"].set_data(time[: idx + 1], left_series[row][: idx + 1])
            ax_artists[row]["truth_marker"].set_data([current_time], [truth_mean[idx]])
            ax_artists[row]["obs_marker"].set_data([current_time], [left_series[row][idx]])
            ax_artists[row]["err_prog"].set_data(time[: idx + 1], right_series[row][: idx + 1])
            ax_artists[row]["err_marker"].set_data([current_time], [right_series[row][idx]])
            ax_artists[row]["left_vline"].set_xdata([current_time, current_time])
            ax_artists[row]["right_vline"].set_xdata([current_time, current_time])

        return []

    anim = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000.0 / max(1, fps),
        blit=False,
        repeat=True,
    )

    writer = None
    if mpl_animation.writers.is_available("pillow"):
        writer = mpl_animation.PillowWriter(fps=max(1, fps))
    elif mpl_animation.writers.is_available("imagemagick"):
        writer = mpl_animation.ImageMagickWriter(fps=max(1, fps))
    else:
        raise RuntimeError(
            "No GIF writer available. Install pillow or ImageMagick so matplotlib can save the animation."
        )

    anim.save(str(out_path), writer=writer, dpi=110)
    plt.close(fig)


def _make_map_gif(
    time: np.ndarray,
    truth_grid: np.ndarray,
    obs_coords: np.ndarray,
    obs_values: np.ndarray,
    obs_err_values: np.ndarray,
    orig_grid: np.ndarray,
    bg_grid: np.ndarray,
    assim_grid: np.ndarray,
    bg_label: str,
    left_vmin: float,
    left_vmax: float,
    err_vmax: float,
    extent: tuple[float, float, float, float],
    out_path: Path,
    fps: int,
    stride: int,
) -> None:
    time = np.asarray(time)
    stride = max(1, int(stride))
    frame_indices = list(range(0, len(time), stride))
    if frame_indices[-1] != len(time) - 1:
        frame_indices.append(len(time) - 1)
    truth_grid = np.asarray(truth_grid, dtype=np.float32)
    obs_coords = np.asarray(obs_coords, dtype=np.float32)
    obs_values = np.asarray(obs_values, dtype=np.float32)
    obs_err_values = np.asarray(obs_err_values, dtype=np.float32)
    orig_grid = np.asarray(orig_grid, dtype=np.float32)
    bg_grid = np.asarray(bg_grid, dtype=np.float32)
    assim_grid = np.asarray(assim_grid, dtype=np.float32)

    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=False, sharey=False)
    left_cmap = plt.get_cmap("Blues").copy()
    left_cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    err_cmap = plt.get_cmap("magma").copy()
    err_cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))

    left_titles = [
        "Data trajectory + noisy observations",
        "Original rainfall traj",
        bg_label,
        "Data assimilation traj",
    ]
    right_titles = [
        "Observation error",
        "Original rainfall error",
        "Perturbed rainfall error",
        "Data assimilation error",
    ]

    left_images: list[object] = []
    err_images: list[object] = []
    obs_scatter = None
    obs_err_scatter = None

    series_list = [orig_grid, bg_grid, assim_grid]

    for row in range(4):
        ax_left = axes[row, 0]
        ax_right = axes[row, 1]
        ax_left.set_title(left_titles[row], fontsize=11)
        ax_right.set_title(right_titles[row], fontsize=11)
        ax_left.set_xlim(extent[0], extent[1])
        ax_left.set_ylim(extent[2], extent[3])
        ax_right.set_xlim(extent[0], extent[1])
        ax_right.set_ylim(extent[2], extent[3])
        ax_left.set_aspect("equal")
        ax_right.set_aspect("equal")
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        ax_right.set_xticks([])
        ax_right.set_yticks([])

        if row == 0:
            left_im = ax_left.imshow(
                truth_grid[0],
                origin="lower",
                extent=extent,
                cmap=left_cmap,
                vmin=left_vmin,
                vmax=left_vmax,
                interpolation="nearest",
                aspect="equal",
            )
            obs_scatter = ax_left.scatter(
                obs_coords[:, 0],
                obs_coords[:, 1],
                c=obs_values[0],
                cmap=left_cmap,
                vmin=left_vmin,
                vmax=left_vmax,
                s=8,
                edgecolors="black",
                linewidths=0.15,
                alpha=0.95,
            )
            obs_err_scatter = ax_right.scatter(
                obs_coords[:, 0],
                obs_coords[:, 1],
                c=obs_err_values[0],
                cmap=err_cmap,
                vmin=0.0,
                vmax=err_vmax,
                s=8,
                edgecolors="black",
                linewidths=0.15,
                alpha=0.95,
            )
            err_im = obs_err_scatter
        else:
            series = series_list[row - 1]
            left_im = ax_left.imshow(
                series[0],
                origin="lower",
                extent=extent,
                cmap=left_cmap,
                vmin=left_vmin,
                vmax=left_vmax,
                interpolation="nearest",
                aspect="equal",
            )
            err_im = ax_right.imshow(
                np.abs(series[0] - truth_grid[0]),
                origin="lower",
                extent=extent,
                cmap=err_cmap,
                vmin=0.0,
                vmax=err_vmax,
                interpolation="nearest",
                aspect="equal",
            )

        left_images.append(left_im)
        err_images.append(err_im)

    axes[3, 0].set_xlabel("x")
    axes[3, 1].set_xlabel("x")
    for row in range(4):
        axes[row, 0].set_ylabel("y")

    suptitle = fig.suptitle("Single-trajectory OVDA comparison", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    def update(frame_idx: int):
        idx = frame_indices[frame_idx]
        suptitle.set_text(f"Single-trajectory OVDA comparison | time step {idx}")
        left_images[0].set_data(truth_grid[idx])
        if obs_scatter is not None:
            obs_scatter.set_array(obs_values[idx])
        if obs_err_scatter is not None:
            obs_err_scatter.set_array(obs_err_values[idx])

        for row, series in enumerate(series_list, start=1):
            left_images[row].set_data(series[idx])
            err_images[row].set_data(np.abs(series[idx] - truth_grid[idx]))

        artists = left_images + err_images
        if obs_scatter is not None:
            artists.append(obs_scatter)
        return artists

    anim = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000.0 / max(1, fps),
        blit=False,
        repeat=True,
    )

    writer = None
    if mpl_animation.writers.is_available("pillow"):
        writer = mpl_animation.PillowWriter(fps=max(1, fps))
    elif mpl_animation.writers.is_available("imagemagick"):
        writer = mpl_animation.ImageMagickWriter(fps=max(1, fps))
    else:
        raise RuntimeError(
            "No GIF writer available. Install pillow or ImageMagick so matplotlib can save the animation."
        )

    anim.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)


def main(opt: argparse.Namespace) -> None:
    set_seed(opt.seed)
    device = resolve_device(opt.device)
    da_seed = opt.seed if opt.da_seed is None else opt.da_seed
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(da_seed)
    device_rng = torch.Generator(device=device.type)
    device_rng.manual_seed(da_seed)

    base_path = opt.base_path
    data_root = base_path / opt.data_root if not opt.data_root.is_absolute() else opt.data_root
    model_dir = base_path / opt.model_path if not opt.model_path.is_absolute() else opt.model_path

    use_static_features = _infer_use_static_features(opt)
    flow, coords, rain = _load_trajectory(data_root, opt.traj_id, use_static_features)
    t_len = int(flow.shape[1])

    obs_idx, eligible_count = _select_obs_indices(
        flow=flow,
        obs_point_count=opt.obs_point_count,
        seed=opt.seed,
        depth_threshold=opt.obs_water_depth_threshold,
    )
    if eligible_count < opt.obs_point_count:
        print(
            f"Requested {opt.obs_point_count} observation points, but only {eligible_count} spatial points "
            f"ever exceeded depth {opt.obs_water_depth_threshold}; using all eligible points."
        )

    flow_mean, flow_std, flow_stats_dir = _load_stats(
        [data_root, model_dir, base_path],
        [("mean.npy", "std.npy"), ("flow_mean.npy", "flow_std.npy")],
    )
    rain_mean, rain_std, rain_stats_dir = _load_stats(
        [data_root, model_dir, base_path],
        [("rain_mean.npy", "rain_std.npy"), ("u_mean.npy", "u_std.npy")],
    )

    flow_norm = _normalize(np.asarray(flow, dtype=np.float32), flow_mean, flow_std)
    rain_norm = _normalize(np.asarray(rain, dtype=np.float32), rain_mean, rain_std)

    obs_truth_norm = flow_norm[:, :, obs_idx, :]
    obs_noise = opt.obs_noise_std * torch.randn(obs_truth_norm.shape, generator=cpu_rng, dtype=torch.float32)
    observations = obs_truth_norm + obs_noise.numpy().astype(np.float32)

    obs_coords = coords[:, obs_idx, :]
    x_obs = _build_time_broadcast_coords(obs_coords, t_len)

    mask_path = data_root / "aggregate_mask.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing aggregate mask: {mask_path}")
    domain_mask = np.load(mask_path)
    if domain_mask.ndim != 2:
        raise ValueError(f"Expected aggregate mask with shape (H, W); got {domain_mask.shape}")
    h_dim, w_dim = domain_mask.shape
    scatter_indices = _resolve_scatter_indices(domain_mask, h_dim, w_dim, coords)
    grid_to_reduced = -np.ones(h_dim * w_dim, dtype=np.int64)
    grid_to_reduced[scatter_indices] = np.arange(scatter_indices.size)

    map_stride = max(1, int(opt.map_stride))
    row_sel = np.arange(0, h_dim, map_stride, dtype=np.int64)
    col_sel = np.arange(0, w_dim, map_stride, dtype=np.int64)
    coarse_mask = domain_mask[np.ix_(row_sel, col_sel)].reshape(-1).astype(bool)
    coarse_flat = (row_sel[:, None] * w_dim + col_sel[None, :]).reshape(-1)
    coarse_reduced = grid_to_reduced[coarse_flat]
    coarse_reduced = coarse_reduced[coarse_mask]
    if coarse_reduced.size == 0:
        raise ValueError(
            f"Map stride {map_stride} did not leave any valid coarse domain cells; "
            "try a smaller --map-stride."
        )
    coarse_shape = (row_sel.size, col_sel.size)
    coarse_valid_flat = np.flatnonzero(coarse_mask)
    coarse_coords = coords[0, coarse_reduced, :].astype(np.float32, copy=False)
    x_map = np.broadcast_to(
        coarse_coords[None, None, :, :],
        (1, t_len, coarse_coords.shape[0], coarse_coords.shape[1]),
    ).copy()
    x_map_tensor = torch.from_numpy(x_map.astype(np.float32, copy=False)).to(device)
    x_vals = 2.0 * (col_sel.astype(np.float32) / float(w_dim) - 0.5)
    aspect = float(h_dim) / float(w_dim)
    y_vals = 2.0 * (row_sel.astype(np.float32) / float(h_dim) - 0.5) * aspect
    dx = float(np.diff(x_vals).mean()) if x_vals.size > 1 else 1.0
    dy = float(np.diff(y_vals).mean()) if y_vals.size > 1 else 1.0
    map_extent = (
        float(x_vals.min() - 0.5 * dx),
        float(x_vals.max() + 0.5 * dx),
        float(y_vals.min() - 0.5 * dy),
        float(y_vals.max() + 0.5 * dy),
    )

    dim_u = int(rain_norm.shape[-1])
    dim_x = int(obs_coords.shape[-1])
    dim_y = int(flow_norm.shape[-1])

    model = _load_model(opt, device, dim_u, dim_x, dim_y)
    rain_model, rain_prop_mean, rain_prop_std, rain_prop_normalized, rain_prop_checkpoint = _load_rain_propagator(
        opt,
        device,
    )
    if opt.use_true_rain_background:
        rain_bg_traj_id = opt.traj_id
    else:
        rain_bg_traj_id = _resolve_rain_background_traj_id(data_root, opt.traj_id, opt.rain_background_traj_id)
    _, _, rain_bg_raw = _load_trajectory(data_root, rain_bg_traj_id, use_static_features=False)

    rain_tensor = torch.from_numpy(rain_norm.astype(np.float32, copy=False)).to(device)
    x_tensor = torch.from_numpy(x_obs.astype(np.float32, copy=False)).to(device)
    obs_tensor = torch.from_numpy(observations.astype(np.float32, copy=False)).to(device)
    dt_tensor = torch.tensor(1.0, dtype=torch.float32, device=device)

    with torch.no_grad():
        orig_pred_norm = model({"u": rain_tensor, "x": x_tensor, "dt": dt_tensor}, device).detach().cpu().numpy()

    # Keep the background rainfall fixed on the selected trajectory instead of
    # rolling it forward with the learned rain propagator.
    rain_bg_phys = np.asarray(rain_bg_raw, dtype=np.float32)
    rain_bg_norm = _normalize(rain_bg_phys, rain_mean, rain_std)
    rain_bg_tensor = torch.from_numpy(rain_bg_norm.astype(np.float32, copy=False)).to(device)

    with torch.no_grad():
        bg_pred_norm = model({"u": rain_bg_tensor, "x": x_tensor, "dt": dt_tensor}, device).detach().cpu().numpy()

    assim_pred_norm, state_hist, u_hist = _run_ovda_assimilation(
        model=model,
        rain_bg=rain_bg_tensor,
        x_obs=x_tensor,
        observations=obs_tensor,
        dt=dt_tensor,
        device=device,
        rng=device_rng,
        ensemble_size=opt.ensemble_size,
        obs_sigma=opt.obs_sigma,
        smoothing_steps=opt.smoothing_steps,
        lbfgs_max_iter=opt.lbfgs_max_iter,
        u_init_noise_std=opt.u_init_noise_std,
        state_reg=opt.state_reg,
        u_reg=opt.u_reg,
        rain_model=rain_model,
        rain_mean=rain_prop_mean,
        rain_std=rain_prop_std,
    )

    with torch.no_grad():
        orig_map_norm = model({"u": rain_tensor, "x": x_map_tensor, "dt": dt_tensor}, device).detach().cpu().numpy()
        bg_map_norm = model({"u": rain_bg_tensor, "x": x_map_tensor, "dt": dt_tensor}, device).detach().cpu().numpy()

    state_hist_mean = state_hist.mean(axis=1).astype(np.float32, copy=False)
    assim_map_norm = np.empty((t_len, coarse_coords.shape[0], dim_y), dtype=np.float32)
    x_map_single = x_map_tensor[:, 0, :, :]
    with torch.no_grad():
        for t in range(t_len):
            state_t = torch.from_numpy(state_hist_mean[t : t + 1]).to(device)
            pred_t = model._decode_points(x_map_single, state_t, chunk_size=model.chunk_size)
            assim_map_norm[t] = pred_t[0].detach().cpu().numpy()

    truth_phys = _denormalize(flow_norm[:, :, obs_idx, :], flow_mean, flow_std)
    obs_phys = _denormalize(observations, flow_mean, flow_std)
    orig_phys = _denormalize(orig_pred_norm, flow_mean, flow_std)
    bg_phys = _denormalize(bg_pred_norm, flow_mean, flow_std)
    assim_phys = _denormalize(assim_pred_norm, flow_mean, flow_std)
    truth_map_phys = _denormalize(flow_norm[:, :, coarse_reduced, :], flow_mean, flow_std)
    orig_map_phys = _denormalize(orig_map_norm, flow_mean, flow_std)
    bg_map_phys = _denormalize(bg_map_norm, flow_mean, flow_std)
    assim_map_phys = _denormalize(assim_map_norm, flow_mean, flow_std)

    truth_mean = truth_phys[0, :, :, opt.plot_channel].mean(axis=1)
    obs_mean = obs_phys[0, :, :, opt.plot_channel].mean(axis=1)
    orig_mean = orig_phys[0, :, :, opt.plot_channel].mean(axis=1)
    bg_mean = bg_phys[0, :, :, opt.plot_channel].mean(axis=1)
    assim_mean = assim_phys.mean(axis=1)[:, :, opt.plot_channel].mean(axis=1)

    obs_coords_xy = obs_coords[0, :, :2].astype(np.float32, copy=False)
    obs_values = obs_phys[0, :, :, opt.plot_channel].astype(np.float32, copy=False)
    obs_truth_values = truth_phys[0, :, :, opt.plot_channel].astype(np.float32, copy=False)
    obs_err_values = np.abs(obs_values - obs_truth_values).astype(np.float32, copy=False)

    def _series_to_grid(series: np.ndarray) -> np.ndarray:
        grid = np.full((t_len, *coarse_shape), np.nan, dtype=np.float32)
        grid.reshape(t_len, -1)[:, coarse_valid_flat] = series.astype(np.float32, copy=False)
        return grid

    truth_grid = _series_to_grid(truth_map_phys[0, :, :, opt.plot_channel])
    orig_grid = _series_to_grid(orig_map_phys[0, :, :, opt.plot_channel])
    bg_grid = _series_to_grid(bg_map_phys[0, :, :, opt.plot_channel])
    assim_grid = _series_to_grid(assim_map_phys[:, :, opt.plot_channel])

    err_orig = _relative_l2_error(truth_grid, orig_grid)
    err_bg = _relative_l2_error(truth_grid, bg_grid)
    err_assim = _relative_l2_error(truth_grid, assim_grid)

    sample_idx = np.linspace(0, t_len - 1, min(t_len, 12), dtype=int)
    left_samples = np.concatenate(
        [
            truth_grid[sample_idx].ravel(),
            orig_grid[sample_idx].ravel(),
            bg_grid[sample_idx].ravel(),
            assim_grid[sample_idx].ravel(),
        ]
    )
    left_samples = left_samples[np.isfinite(left_samples)]
    if left_samples.size == 0:
        left_vmin, left_vmax = 0.0, 1.0
    else:
        left_vmin = min(0.0, float(np.nanpercentile(left_samples, 0.5)))
        left_vmax = float(np.nanpercentile(left_samples, 99.5))
        if not np.isfinite(left_vmax) or left_vmax <= left_vmin:
            left_vmin, left_vmax = 0.0, max(1.0, float(np.nanmax(left_samples)))

    err_samples = np.concatenate(
        [
            np.abs(orig_grid[sample_idx] - truth_grid[sample_idx]).ravel(),
            np.abs(bg_grid[sample_idx] - truth_grid[sample_idx]).ravel(),
            np.abs(assim_grid[sample_idx] - truth_grid[sample_idx]).ravel(),
            obs_err_values[sample_idx].ravel(),
        ]
    )
    err_samples = err_samples[np.isfinite(err_samples)]
    if err_samples.size == 0:
        err_vmax = 1.0
    else:
        err_vmax = float(np.nanpercentile(err_samples, 99.5))
        if not np.isfinite(err_vmax) or err_vmax <= 0:
            err_vmax = max(1.0, float(np.nanmax(err_samples)))

    print(f"Trajectory id: {opt.traj_id}")
    print(f"Rain background trajectory id: {rain_bg_traj_id}")
    print(f"Using true rain background: {rain_bg_traj_id == opt.traj_id}")
    print(f"Observation points: {len(obs_idx)} of {eligible_count} eligible wet points")
    print(f"Wet-point threshold (channel 0): {opt.obs_water_depth_threshold}")
    print(f"Model path: {model_dir}")
    print(f"Flow normalization source: {flow_stats_dir if flow_stats_dir is not None else 'raw'}")
    print(f"Rain normalization source: {rain_stats_dir if rain_stats_dir is not None else 'raw'}")
    print(f"Rain propagator checkpoint: {rain_prop_checkpoint if rain_prop_checkpoint is not None else 'none'}")
    print(f"Rain propagator normalized: {rain_prop_normalized}")
    print(f"Rain seed length: {opt.rain_seed_len}")
    print(f"DA seed: {da_seed}")
    print(f"Map stride: {map_stride}")
    print(f"Original rainfall mean/std: {np.mean(rain_norm):.6f}/{np.std(rain_norm):.6f}")
    print(f"Background rainfall mean/std: {np.mean(rain_bg_norm):.6f}/{np.std(rain_bg_norm):.6f}")
    print(f"Final relative L2 error - original: {err_orig[-1]:.4f}")
    print(f"Final relative L2 error - background: {err_bg[-1]:.4f}")
    print(f"Final relative L2 error - assimilation: {err_assim[-1]:.4f}")

    output_dir = model_dir / opt.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    default_save_name = f"traj_{opt.traj_id}_rainbg_{rain_bg_traj_id}_obs_{len(obs_idx)}_seed_{opt.seed}"
    if rain_model is not None:
        default_save_name = default_save_name + "_rainlstm"
    save_name = opt.save_name or default_save_name

    np.savez_compressed(
        output_dir / f"{save_name}.npz",
        truth_phys=truth_phys,
        obs_phys=obs_phys,
        orig_phys=orig_phys,
        bg_phys=bg_phys,
        assim_phys=assim_phys,
        err_orig=err_orig,
        err_bg=err_bg,
        err_assim=err_assim,
        obs_idx=obs_idx,
        obs_coords=obs_coords,
        obs_water_depth_threshold=opt.obs_water_depth_threshold,
        obs_eligible_count=eligible_count,
        rain_orig=rain_norm,
        rain_bg=rain_bg_norm,
        rain_bg_traj_id=rain_bg_traj_id,
        rain_propagator_checkpoint=str(rain_prop_checkpoint) if rain_prop_checkpoint is not None else "",
        rain_propagator_normalized=bool(rain_prop_normalized),
        rain_seed_len=int(opt.rain_seed_len),
        state_hist=state_hist,
        u_hist=u_hist,
    )

    figure_path = output_dir / f"{save_name}.png"
    bg_label = (
        "Background rainfall traj (true rain)"
        if rain_bg_traj_id == opt.traj_id
        else f"Background rainfall traj (raw from traj {rain_bg_traj_id})"
    )

    _make_plot(
        time=np.arange(t_len),
        truth_mean=truth_mean,
        obs_mean=obs_mean,
        obs_samples=obs_phys[0, :, :, opt.plot_channel],
        orig_mean=orig_mean,
        bg_mean=bg_mean,
        assim_mean=assim_mean,
        err_orig=err_orig,
        err_bg=err_bg,
        err_assim=err_assim,
        plot_channel=opt.plot_channel,
        bg_label=bg_label,
        out_path=figure_path,
    )

    gif_path = output_dir / f"{save_name}.gif"
    _make_map_gif(
        time=np.arange(t_len),
        truth_grid=truth_grid,
        obs_coords=obs_coords_xy,
        obs_values=obs_values,
        obs_err_values=obs_err_values,
        orig_grid=orig_grid,
        bg_grid=bg_grid,
        assim_grid=assim_grid,
        bg_label=bg_label,
        left_vmin=left_vmin,
        left_vmax=left_vmax,
        err_vmax=err_vmax,
        extent=map_extent,
        out_path=gif_path,
        fps=opt.gif_fps,
        stride=opt.gif_stride,
    )

    print(f"Saved arrays to: {output_dir / f'{save_name}.npz'}")
    print(f"Saved figure to: {figure_path}")
    print(f"Saved GIF to: {gif_path}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main(create_args())
