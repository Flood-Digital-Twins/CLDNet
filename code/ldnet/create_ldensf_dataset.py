#!/usr/bin/env python3
"""
Build an observation-to-latent dataset for hydro LD-ENSF experiments.

This script:
1. Finds trajectories in `data/postprocessed/illinois`.
2. Selects 200 spatial key points from cells that exceed a water-depth
   threshold of 3 at least once in any trajectory.
   Alternatively, it can select the cells nearest to a set of USGS gauge
   locations used in the validation notebook.
3. Runs the trained LDNet on the full truth rain forcing to obtain latent
   state targets.
4. Saves train/valid/test splits with observations, rainfall forcing, and
   latent-state labels.
"""

from __future__ import annotations

import csv
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from efficient_fourier_ldnet import EfficientFourierLDNN


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "cuda" in device_str and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def _resolve_path(base_path: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_path / path


def _infer_use_static_features(model_path: Path, user_choice: bool | None) -> bool:
    if user_choice is not None:
        return user_choice
    return model_path.name.lower() == "cldnet" or "static" in model_path.name.lower()


def _list_trajectory_ids(data_root: Path) -> list[int]:
    traj_ids: list[int] = []
    for path in data_root.glob("flow_variables_traj*.npy"):
        match = re.search(r"traj(\d+)\.npy$", path.name)
        if match is not None:
            traj_ids.append(int(match.group(1)))
    traj_ids = sorted(set(traj_ids))
    if not traj_ids:
        raise FileNotFoundError(f"No flow_variables_traj*.npy files found in {data_root}")
    return traj_ids


def _load_reduced_trajectory(
    data_root: Path,
    traj_id: int,
    use_static_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    flow = np.load(data_root / f"flow_variables_traj{traj_id}.npy", mmap_mode="r")
    coords = np.load(data_root / f"coords_traj{traj_id}.npy", mmap_mode="r")
    rain = np.load(data_root / f"rain_source_traj{traj_id}.npy", mmap_mode="r")
    static = None
    if use_static_features:
        static_path = data_root / f"static_features_traj{traj_id}.npy"
        if not static_path.exists():
            raise FileNotFoundError(f"Missing static features for traj {traj_id}: {static_path}")
        static = np.load(static_path, mmap_mode="r")
        coords = np.concatenate([coords, static], axis=-1)
    return flow, coords, rain, static


def _build_dummy_x(t_len: int, dim_x: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((1, t_len, 1, dim_x), dtype=torch.float32, device=device)


def _build_model(
    *,
    fourier_mapping_size: int,
    num_latent_states: int,
    dim_u: int,
    dim_x: int,
    dim_y: int,
    nn_dyn_depth: int,
    nn_dyn_width: int,
    nn_rec_depth: int,
    nn_rec_width: int,
    activation: str,
    kernel_initializer: str,
    chunk_size: int,
) -> EfficientFourierLDNN:
    return EfficientFourierLDNN(
        fourier_mapping_size,
        [num_latent_states + dim_u] + nn_dyn_depth * [nn_dyn_width] + [num_latent_states],
        [num_latent_states + dim_x] + nn_rec_depth * [nn_rec_width] + [dim_y],
        activation=activation,
        kernel_initializer=kernel_initializer,
        chunk_size=chunk_size,
    )


def _load_checkpoints(model: EfficientFourierLDNN, dyn_ckpt: Path, rec_ckpt: Path, b_ckpt: Path | None) -> None:
    print(f"Loading dyn checkpoint: {dyn_ckpt}")
    print(f"Loading rec checkpoint: {rec_ckpt}")
    model.dyn.load_state_dict(torch.load(dyn_ckpt, map_location="cpu", weights_only=False))
    model.rec.load_state_dict(torch.load(rec_ckpt, map_location="cpu", weights_only=False))
    if b_ckpt is not None and b_ckpt.exists():
        print(f"Loading B checkpoint: {b_ckpt}")
        model.B.load_state_dict(torch.load(b_ckpt, map_location="cpu", weights_only=False))


def _farthest_point_sample(coords_xy: np.ndarray, count: int, seed: int) -> np.ndarray:
    if coords_xy.shape[0] <= count:
        return np.arange(coords_xy.shape[0], dtype=np.int64)

    rng = np.random.default_rng(seed)
    first = int(rng.integers(coords_xy.shape[0]))
    selected_positions = np.empty(count, dtype=np.int64)
    selected_positions[0] = first

    min_dist = np.sum((coords_xy - coords_xy[first]) ** 2, axis=1)
    min_dist[first] = -np.inf

    for i in range(1, count):
        next_idx = int(np.argmax(min_dist))
        selected_positions[i] = next_idx
        dist = np.sum((coords_xy - coords_xy[next_idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected_positions[: i + 1]] = -np.inf

    order = np.lexsort((coords_xy[selected_positions, 1], coords_xy[selected_positions, 0]))
    return selected_positions[order]


def _load_usgs_gauge_points(mapping_path: Path, gauge_indices: list[int]) -> list[tuple[int, float, float]]:
    requested = [int(idx) for idx in gauge_indices]
    if not requested:
        raise ValueError("At least one USGS gauge index is required for gauge-nearest selection.")

    order = {idx: pos for pos, idx in enumerate(requested)}
    requested_set = set(requested)
    gauge_points: list[tuple[int, float, float]] = []

    with mapping_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["i"])
            if idx in requested_set:
                gauge_points.append((idx, float(row["x_proj"]), float(row["y_proj"])))

    found = {idx for idx, _, _ in gauge_points}
    missing = [idx for idx in requested if idx not in found]
    if missing:
        raise ValueError(f"Missing USGS gauge indices in {mapping_path}: {missing}")

    gauge_points.sort(key=lambda item: order[item[0]])
    return gauge_points


def _select_even_observation_points(
    traj_ids: list[int],
    data_root: Path,
    num_points: int,
    depth_threshold: float,
    seed: int,
    use_static_features: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    reference_coords = None
    eligible_union = None

    for traj_id in tqdm(traj_ids, desc="Selecting wet points"):
        flow, coords, _, _ = _load_reduced_trajectory(data_root, traj_id, use_static_features=False)
        if reference_coords is None:
            reference_coords = np.asarray(coords[0, :, :2], dtype=np.float32)
            eligible_union = np.zeros(reference_coords.shape[0], dtype=bool)
        elif coords.shape != (1, reference_coords.shape[0], coords.shape[-1]):
            raise ValueError(
                f"Coordinate shape mismatch for traj {traj_id}: got {coords.shape}, "
                f"expected (1, {reference_coords.shape[0]}, {coords.shape[-1]})"
            )

        depth_max = np.nanmax(flow[0, :, :, 0], axis=0)
        eligible_union |= depth_max > depth_threshold

    if reference_coords is None or eligible_union is None:
        raise RuntimeError("Could not build observation-point selection set.")

    eligible_idx = np.flatnonzero(eligible_union)
    if eligible_idx.size == 0:
        raise ValueError(f"No cells exceed the depth threshold {depth_threshold}")

    candidate_xy = reference_coords[eligible_idx]
    selected_pos = _farthest_point_sample(candidate_xy, min(num_points, candidate_xy.shape[0]), seed)
    selected_idx = eligible_idx[selected_pos]

    selected_xy = reference_coords[selected_idx]
    order = np.lexsort((selected_xy[:, 1], selected_xy[:, 0]))
    selected_idx = selected_idx[order]
    selected_xy = selected_xy[order]

    return selected_idx.astype(np.int64), selected_xy.astype(np.float32), int(eligible_idx.size)


def _select_usgs_gauge_observation_points(
    *,
    data_root: Path,
    reference_coords: np.ndarray,
    gauge_mapping_path: Path,
    dem_path: Path,
    gauge_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    mask_path = data_root / "aggregate_mask.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing aggregate mask: {mask_path}")

    valid_mask = np.asarray(np.load(mask_path), dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError(f"Expected a 2D aggregate mask, got shape {valid_mask.shape}")

    gauge_points = _load_usgs_gauge_points(gauge_mapping_path, gauge_indices)
    valid_rows, valid_cols = np.nonzero(valid_mask)

    with rasterio.open(dem_path) as dem:
        transform = dem.transform
        valid_x = (
            transform.c
            + (valid_cols.astype(np.float64) + 0.5) * transform.a
            + (valid_rows.astype(np.float64) + 0.5) * transform.b
        )
        valid_y = (
            transform.f
            + (valid_cols.astype(np.float64) + 0.5) * transform.d
            + (valid_rows.astype(np.float64) + 0.5) * transform.e
        )

    selected_positions: list[int] = []
    used_positions: set[int] = set()
    for gauge_idx, gauge_x, gauge_y in gauge_points:
        dist2 = (valid_x - gauge_x) ** 2 + (valid_y - gauge_y) ** 2
        order = np.argsort(dist2, kind="mergesort")
        best_pos = None
        for candidate_pos in order:
            candidate = int(candidate_pos)
            if candidate not in used_positions:
                best_pos = candidate
                break
        if best_pos is None:
            raise ValueError(
                f"Could not find a unique valid cell for USGS gauge {gauge_idx}; "
                "all candidate cells are already used."
            )
        selected_positions.append(best_pos)
        used_positions.add(best_pos)

    selected_positions = np.asarray(selected_positions, dtype=np.int64)
    selected_xy = np.asarray(reference_coords[selected_positions, :2], dtype=np.float32)
    return selected_positions, selected_xy, int(len(gauge_points))


def _split_ids_by_traj_id(
    traj_ids: list[int],
    train_max_traj_id: int,
    valid_max_traj_id: int,
    test_max_traj_id: int,
) -> tuple[list[int], list[int], list[int]]:
    if not (train_max_traj_id < valid_max_traj_id < test_max_traj_id):
        raise ValueError("Expected train_max_traj_id < valid_max_traj_id < test_max_traj_id")

    train_ids = [traj_id for traj_id in traj_ids if traj_id <= train_max_traj_id]
    valid_ids = [traj_id for traj_id in traj_ids if train_max_traj_id < traj_id <= valid_max_traj_id]
    test_ids = [traj_id for traj_id in traj_ids if valid_max_traj_id < traj_id <= test_max_traj_id]

    excluded = [traj_id for traj_id in traj_ids if traj_id > test_max_traj_id]
    if excluded:
        raise ValueError(
            f"Found trajectory ids beyond test_max_traj_id={test_max_traj_id}: {excluded}. "
            "Increase the split bounds if these should be included."
        )

    if not train_ids or not valid_ids or not test_ids:
        raise ValueError(
            "One of the requested trajectory-id splits is empty. "
            "Check that the selected bounds match the available data."
        )

    return train_ids, valid_ids, test_ids


def _compute_latent_targets(
    traj_ids: list[int],
    data_root: Path,
    model: EfficientFourierLDNN,
    device: torch.device,
    dim_x: int,
    selected_idx: np.ndarray,
    use_static_features: bool,
) -> dict[str, torch.Tensor]:
    observations = []
    rain_forcing = []
    latent_targets = []

    model.eval()
    for traj_id in tqdm(traj_ids, desc="Building latent targets"):
        flow, coords, rain, _ = _load_reduced_trajectory(data_root, traj_id, use_static_features)
        t_len = int(flow.shape[1])

        obs = np.asarray(flow[0][:, selected_idx, :], dtype=np.float32)
        rain_t = np.asarray(rain[0], dtype=np.float32)
        dummy_x = _build_dummy_x(t_len, dim_x, device)
        rain_tensor = torch.from_numpy(rain_t).unsqueeze(0).to(device)

        with torch.no_grad():
            latent = model({"u": rain_tensor, "x": dummy_x, "dt": torch.tensor([1.0], device=device)}, device, latent_state=True)

        observations.append(torch.from_numpy(obs))
        rain_forcing.append(torch.from_numpy(rain_t))
        latent_targets.append(latent.squeeze(0).detach().cpu())

    return {
        "observation": torch.stack(observations, dim=0).contiguous(),
        "u": torch.stack(rain_forcing, dim=0).contiguous(),
        "latent_states": torch.stack(latent_targets, dim=0).contiguous(),
    }


def create_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/cldnet"))
    parser.add_argument("--output-path", type=Path, default=Path("data/postprocessed/illinois/observation_ldensf_dataset.pth"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-observation-points", type=int, default=2000)
    parser.add_argument("--depth-threshold", type=float, default=3.0)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="wet-even",
        choices=["wet-even", "usgs-gauges"],
        help="Observation-point selection strategy.",
    )
    parser.add_argument(
        "--usgs-gauge-mapping-path",
        type=Path,
        default=Path("data/usgs_2013_validation/data_usgs/usgs_plot/GT_sample_data_folder/gage_index_mapping_with_lat_lon.csv"),
    )
    parser.add_argument(
        "--usgs-dem-path",
        type=Path,
        default=Path("data/usgs_2013_validation/data_usgs/usgs_plot/GT_sample_data_folder/dem_5070.tif"),
    )
    parser.add_argument(
        "--usgs-gauge-indices",
        type=int,
        nargs="*",
        default=[0, 7, 14, 19, 20, 25],
        help="USGS gauge indices used for gauge-nearest selection.",
    )
    parser.add_argument("--checkpoint-epoch", type=int, default=539)
    parser.add_argument("--dyn-checkpoint", type=Path, default=None)
    parser.add_argument("--rec-checkpoint", type=Path, default=None)
    parser.add_argument("--B-checkpoint", type=Path, default=None)
    parser.add_argument("--num-latent-states", type=int, default=200)
    parser.add_argument("--fourier-mapping-size", type=int, default=32)
    parser.add_argument("--NN-dyn-depth", type=int, default=8)
    parser.add_argument("--NN-dyn-width", type=int, default=50)
    parser.add_argument("--NN-rec-depth", type=int, default=10)
    parser.add_argument("--NN-rec-width", type=int, default=300)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--kernel-initializer", type=str, default="Glorot normal")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--train-max-traj-id", type=int, default=100)
    parser.add_argument("--valid-max-traj-id", type=int, default=117)
    parser.add_argument("--test-max-traj-id", type=int, default=120)
    static_group = parser.add_mutually_exclusive_group()
    static_group.add_argument("--use-static-features", dest="use_static_features", action="store_true")
    static_group.add_argument("--no-static-features", dest="use_static_features", action="store_false")
    parser.set_defaults(use_static_features=None)
    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    base_path = opt.base_path
    data_root = _resolve_path(base_path, opt.data_root)
    model_path = _resolve_path(base_path, opt.model_path)
    output_path = _resolve_path(base_path, opt.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    traj_ids = _list_trajectory_ids(data_root)
    use_static_features = _infer_use_static_features(model_path, opt.use_static_features)

    # Use the first available trajectory to infer coordinate feature dimensions.
    first_flow, first_coords, first_rain, first_static = _load_reduced_trajectory(
        data_root, traj_ids[0], use_static_features=use_static_features
    )
    dim_u = int(first_rain.shape[-1])
    dim_y = int(first_flow.shape[-1])
    dim_x = int(first_coords.shape[-1])
    if use_static_features:
        if first_static is None:
            raise RuntimeError("Static features were requested but not loaded.")
        dim_x = int(first_coords.shape[-1])

    first_coords_xy = np.asarray(first_coords[0, :, :2], dtype=np.float32)

    if opt.selection_mode == "wet-even":
        selected_idx, selected_xy, eligible_count = _select_even_observation_points(
            traj_ids=traj_ids,
            data_root=data_root,
            num_points=opt.num_observation_points,
            depth_threshold=opt.depth_threshold,
            seed=opt.selection_seed,
            use_static_features=use_static_features,
        )
        selection_summary = (
            f"Selected {selected_idx.size} key points from {eligible_count} eligible wet cells."
        )
        selection_detail = f"Observation point selection threshold: {opt.depth_threshold}"
    else:
        selected_idx, selected_xy, eligible_count = _select_usgs_gauge_observation_points(
            data_root=data_root,
            reference_coords=first_coords_xy,
            gauge_mapping_path=_resolve_path(base_path, opt.usgs_gauge_mapping_path),
            dem_path=_resolve_path(base_path, opt.usgs_dem_path),
            gauge_indices=opt.usgs_gauge_indices,
        )
        selection_summary = (
            f"Selected {selected_idx.size} observation locations nearest to "
            f"{eligible_count} USGS gauges."
        )
        selection_detail = f"USGS gauge indices: {list(opt.usgs_gauge_indices)}"

    model = _build_model(
        fourier_mapping_size=opt.fourier_mapping_size,
        num_latent_states=opt.num_latent_states,
        dim_u=dim_u,
        dim_x=dim_x,
        dim_y=dim_y,
        nn_dyn_depth=opt.NN_dyn_depth,
        nn_dyn_width=opt.NN_dyn_width,
        nn_rec_depth=opt.NN_rec_depth,
        nn_rec_width=opt.NN_rec_width,
        activation=opt.activation,
        kernel_initializer=opt.kernel_initializer,
        chunk_size=opt.chunk_size,
    )

    if opt.dyn_checkpoint is not None and opt.rec_checkpoint is not None:
        dyn_ckpt = _resolve_path(base_path, opt.dyn_checkpoint)
        rec_ckpt = _resolve_path(base_path, opt.rec_checkpoint)
        b_ckpt = _resolve_path(base_path, opt.B_checkpoint) if opt.B_checkpoint is not None else None
    else:
        dyn_ckpt = model_path / f"dyn_{opt.checkpoint_epoch}.ckpt"
        rec_ckpt = model_path / f"rec_{opt.checkpoint_epoch}.ckpt"
        b_ckpt = model_path / f"B_{opt.checkpoint_epoch}.ckpt"

    _load_checkpoints(model, dyn_ckpt, rec_ckpt, b_ckpt)

    device = resolve_device(opt.device)
    model.to(device)
    model.eval()

    train_ids, valid_ids, test_ids = _split_ids_by_traj_id(
        traj_ids,
        train_max_traj_id=opt.train_max_traj_id,
        valid_max_traj_id=opt.valid_max_traj_id,
        test_max_traj_id=opt.test_max_traj_id,
    )

    def build_split(split_ids: list[int]) -> dict[str, torch.Tensor]:
        split_data = _compute_latent_targets(
            split_ids,
            data_root=data_root,
            model=model,
            device=device,
            dim_x=dim_x,
            selected_idx=selected_idx,
            use_static_features=use_static_features,
        )
        split_data["traj_ids"] = torch.tensor(split_ids, dtype=torch.int64)
        return split_data

    print(selection_summary)
    print(selection_detail)
    print(f"Using static features: {use_static_features}")
    print(
        "Trajectory split ranges: "
        f"train<= {opt.train_max_traj_id}, "
        f"valid {opt.train_max_traj_id + 1}-{opt.valid_max_traj_id}, "
        f"test {opt.valid_max_traj_id + 1}-{opt.test_max_traj_id}"
    )
    print(f"Trajectory split sizes: train={len(train_ids)}, valid={len(valid_ids)}, test={len(test_ids)}")

    data = {
        "meta": {
            "selected_obs_idx": torch.from_numpy(selected_idx),
            "selected_obs_xy": torch.from_numpy(selected_xy),
            "eligible_count": int(eligible_count),
            "selection_mode": opt.selection_mode,
            "depth_threshold": float(opt.depth_threshold),
            "selection_seed": int(opt.selection_seed),
            "use_static_features": bool(use_static_features),
            "num_observation_points": int(selected_idx.size),
            "requested_observation_points": int(opt.num_observation_points),
            "obs_layout": "time_first",
            "split_mode": "traj_id_range",
            "train_max_traj_id": int(opt.train_max_traj_id),
            "valid_max_traj_id": int(opt.valid_max_traj_id),
            "test_max_traj_id": int(opt.test_max_traj_id),
            "usgs_gauge_indices": torch.tensor(opt.usgs_gauge_indices, dtype=torch.int64),
            "traj_ids": torch.tensor(traj_ids, dtype=torch.int64),
        },
        "data_train": build_split(train_ids),
        "data_valid": build_split(valid_ids),
        "data_test": build_split(test_ids),
    }

    torch.save(data, output_path)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main(create_options())
