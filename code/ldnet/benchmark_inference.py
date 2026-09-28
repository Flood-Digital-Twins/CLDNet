#!/usr/bin/env python3
"""
Benchmark inference time for EfficientFourierLDNN on a single trajectory.

Runs two cases:
  - 10,000 spatial points (random subset)
  - all spatial points

Example:
  python benchmark_inference.py --traj-id 0 --checkpoint-epoch 199 --device cuda:0 \\
    --model-path checkpoints/ldnet --data-root data/postprocessed/illinois \\
    --log-file logs/inference_traj0.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

repo_path = Path(__file__).resolve().parent
sys.path.append(str(repo_path))

from efficient_fourier_ldnet import EfficientFourierLDNN

dt = 1


def create_options():
    parser = argparse.ArgumentParser()
    default_base_path = Path(__file__).resolve().parents[2]
    parser.add_argument("--base-path", type=Path, default=default_base_path)
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/ldnet"))
    parser.add_argument("--data-root", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--traj-id", type=int, default=0)
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
    parser.add_argument("--checkpoint-epoch", type=int, default=None)
    parser.add_argument("--dyn-checkpoint", type=Path, default=None)
    parser.add_argument("--rec-checkpoint", type=Path, default=None)
    parser.add_argument("--use-static-features", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sample-points", type=int, default=10000)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/sims_30"),
    )
    return parser.parse_args()


def _load_reduced_data(data_root: Path, traj_id: int, use_static: bool):
    flow = np.load(data_root / f"flow_variables_traj{traj_id}.npy", mmap_mode="r")
    coords = np.load(data_root / f"coords_traj{traj_id}.npy", mmap_mode="r")
    rain = np.load(data_root / f"rain_source_traj{traj_id}.npy", mmap_mode="r")
    if use_static:
        static = np.load(data_root / f"static_features_traj{traj_id}.npy", mmap_mode="r")
        coords = np.concatenate([coords, static], axis=-1)
    return flow, coords, rain


def _load_checkpoints(model, opt):
    if opt.checkpoint_epoch is not None:
        dyn_ckpt = opt.base_path / opt.model_path / f"dyn_{opt.checkpoint_epoch}.ckpt"
        rec_ckpt = opt.base_path / opt.model_path / f"rec_{opt.checkpoint_epoch}.ckpt"
        b_ckpt = opt.base_path / opt.model_path / f"B_{opt.checkpoint_epoch}.ckpt"
    else:
        dyn_ckpt = opt.dyn_checkpoint
        rec_ckpt = opt.rec_checkpoint
        b_ckpt = None

    if dyn_ckpt is None or rec_ckpt is None:
        print("No checkpoints provided; using random weights.")
        return

    print(f"Loading dyn checkpoint: {dyn_ckpt}")
    print(f"Loading rec checkpoint: {rec_ckpt}")
    model.dyn.load_state_dict(torch.load(dyn_ckpt, map_location="cpu"))
    model.rec.load_state_dict(torch.load(rec_ckpt, map_location="cpu"))
    if b_ckpt is not None and b_ckpt.exists() and hasattr(model, "B"):
        print(f"Loading B checkpoint: {b_ckpt}")
        try:
            model.B.load_state_dict(torch.load(b_ckpt, map_location="cpu"))
        except RuntimeError as exc:
            print(f"WARNING: B checkpoint load failed: {exc}")
            print("WARNING: Proceeding with randomly initialized Fourier embedding.")


def _peek_b_in_feats(opt: argparse.Namespace) -> int | None:
    if opt.checkpoint_epoch is not None:
        b_ckpt = opt.base_path / opt.model_path / f"B_{opt.checkpoint_epoch}.ckpt"
    else:
        b_ckpt = None
    if b_ckpt is None or not b_ckpt.exists():
        return None
    state = torch.load(b_ckpt, map_location="cpu")
    weight = state.get("encoding.weight")
    if weight is None:
        return None
    return int(weight.shape[1])


def _load_simulation_meta(input_root: Path, traj_id: int) -> tuple[float | None, float | None]:
    config_path = input_root / f"event_{traj_id}_filling_hours_0_filling_intensity_mm_hr_0" / "simulation_config.json"
    if not config_path.exists():
        return None, None
    try:
        import json

        data = json.loads(config_path.read_text())
        sim_hours = data.get("simulation_hours")
        sim_runtime = data.get("simulation_runtime_seconds")
        sim_hours_f = float(sim_hours) if sim_hours is not None else None
        sim_runtime_f = float(sim_runtime) if sim_runtime is not None else None
        return sim_hours_f, sim_runtime_f
    except Exception:
        return None, None


def _resolve_path(base_path: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_path / path


def _prepare_batch(u: np.ndarray, coords: np.ndarray, indices: np.ndarray | None):
    if indices is not None:
        coords = coords[:, indices, :]
    t_len = u.shape[1]
    x = np.repeat(coords[:, None, :, :], t_len, axis=1).astype(np.float32)
    data = {
        "u": torch.from_numpy(u.astype(np.float32, copy=False)),
        "x": torch.from_numpy(x),
        "dt": torch.tensor([dt], dtype=torch.float32),
    }
    return data, x.shape[2], t_len


def _time_forward(model, data, device, runs: int, warmup: int):
    model.eval()
    data = {k: v.to(device) for k, v in data.items()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            _ = model(data, device, equilibrium=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(data, device, equilibrium=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            times.append(end - start)
    return np.array(times, dtype=np.float64)


def main(opt):
    data_root = opt.data_root
    if not data_root.is_absolute():
        data_root = opt.base_path / data_root
    input_root = _resolve_path(opt.base_path, opt.input_root)

    flow, coords, rain = _load_reduced_data(data_root, opt.traj_id, use_static=False)
    expected_coords = _peek_b_in_feats(opt)
    use_static = opt.use_static_features
    if expected_coords is not None and expected_coords > coords.shape[-1]:
        use_static = True
    if use_static:
        static_path = data_root / f"static_features_traj{opt.traj_id}.npy"
        if not static_path.exists():
            raise FileNotFoundError(
                f"Static features required but missing: {static_path}. "
                "Re-run create_dataset.py with --include-static-features."
            )
        static = np.load(static_path, mmap_mode="r")
        coords = np.concatenate([coords, static], axis=-1)
    t_len = flow.shape[1]
    num_points = coords.shape[1]

    dim_u = rain.shape[-1]
    dim_x = coords.shape[-1]
    dim_y = 3
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

    device = torch.device(opt.device if torch.cuda.is_available() or "cuda" not in opt.device else "cpu")
    model.to(device)

    rng = np.random.default_rng(opt.seed)
    sample_n = min(opt.sample_points, num_points)
    sample_idx = rng.choice(num_points, size=sample_n, replace=False)

    data_10k, n_10k, t_10k = _prepare_batch(rain, coords, sample_idx)
    data_all, n_all, t_all = _prepare_batch(rain, coords, None)

    times_10k = _time_forward(model, data_10k, device, opt.runs, opt.warmup)
    times_all = _time_forward(model, data_all, device, opt.runs, opt.warmup)

    def report(label, times, n, t):
        mean = times.mean()
        std = times.std(ddof=0)
        print(f"{label}: points={n} T={t} runs={len(times)} mean={mean:.4f}s std={std:.4f}s")
        print(f"  per-step: {mean / t:.6f}s | per-point-step: {mean / (n * t):.3e}s")

    lines = []
    lines.append("Command: " + " ".join(sys.argv))
    lines.append(f"Trajectory {opt.traj_id}: total points={num_points}, T={t_len}")
    sim_hours, sim_runtime = _load_simulation_meta(input_root, opt.traj_id)
    if sim_hours is not None:
        lines.append(f"Simulation hours (raw metadata): {sim_hours}")
    if sim_runtime is not None:
        lines.append(f"Simulation runtime seconds (raw metadata): {sim_runtime}")
    lines.append(
        f"10k: points={n_10k} T={t_10k} runs={len(times_10k)} mean={times_10k.mean():.4f}s std={times_10k.std(ddof=0):.4f}s"
    )
    lines.append(
        f"  per-step: {times_10k.mean() / t_10k:.6f}s | per-point-step: {times_10k.mean() / (n_10k * t_10k):.3e}s"
    )
    lines.append(
        f"all: points={n_all} T={t_all} runs={len(times_all)} mean={times_all.mean():.4f}s std={times_all.std(ddof=0):.4f}s"
    )
    lines.append(
        f"  per-step: {times_all.mean() / t_all:.6f}s | per-point-step: {times_all.mean() / (n_all * t_all):.3e}s"
    )
    output = "\n".join(lines)
    print(output)
    if opt.log_file is not None:
        log_path = opt.log_file
        if not log_path.is_absolute():
            log_path = opt.base_path / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output + "\n")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    opt = create_options()
    main(opt)
