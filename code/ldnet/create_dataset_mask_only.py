import argparse
import os
from pathlib import Path

import numpy as np

"""
Usage:
    # Build aggregate mask without static features
    python create_dataset_mask_only.py --num-trajectories 2 --output-dir data/postprocessed/illinois

    # Build aggregate mask with static features (DEM validity included)
    python create_dataset_mask_only.py --num-trajectories 2 --output-dir data/postprocessed/illinois --include-static-features
"""


def _load_flow_variables(flow_path: Path, burn_in: int) -> tuple[np.ndarray, int, int]:
    flow = np.load(flow_path).astype(np.float32)
    if flow.ndim != 4:
        raise ValueError(f"Unexpected flow_variables shape: {flow.shape}")
    if flow.shape[1] == 3:
        # (T, 3, H, W)
        flow = flow[np.newaxis, ...]
        flow = flow[:, burn_in:, :, :, :]
        flow = flow.transpose(0, 1, 3, 4, 2)  # (1, T, H, W, 3)
    elif flow.shape[-1] == 3:
        # (T, H, W, 3)
        flow = flow[np.newaxis, ...]
        flow = flow[:, burn_in:, :, :, :]
    else:
        raise ValueError(f"Unexpected flow_variables channel layout: {flow.shape}")

    _, t_len, h_dim, w_dim, _ = flow.shape
    flow = flow.reshape(1, t_len, h_dim * w_dim, 3)
    return flow, h_dim, w_dim


def _load_static_fields(dataset_directory: Path) -> tuple[np.ndarray, np.ndarray, dict]:
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

    scaling = {
        "dem_mean": dem_mean,
        "dem_std": dem_std,
        "slope_scale": 1.0,
        "has_manning": False,
    }
    static_fields = [dem_z, slope]

    if manning_path.exists():
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
        scaling["manning_scale"] = 100.0
        scaling["has_manning"] = True
        static_fields.append(manning_full * scaling["manning_scale"])
    else:
        print(f"Missing manning.dat at {manning_path}; using DEM-only static fields for validity.")

    static = np.stack(static_fields, axis=-1)
    return static, dem_valid, scaling


def main() -> None:
    parser = argparse.ArgumentParser()
    default_base_path = Path(__file__).resolve().parents[2]
    parser.add_argument("--base-path", type=Path, default=default_base_path)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/sims_30"),
    )
    parser.add_argument("--num-trajectories", type=int, default=120)
    parser.add_argument("--burn-in-length", type=int, default=1)
    parser.add_argument("--depth-threshold", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--mask-filename", type=str, default="aggregate_mask.npy")
    parser.add_argument("--include-static-features", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = args.base_path / args.output_dir
    input_root = args.input_root if args.input_root.is_absolute() else args.base_path / args.input_root
    os.makedirs(output_dir, exist_ok=True)

    available_trajectories = []
    global_valid_any = None
    global_h_dim = None
    global_w_dim = None
    global_dem_valid = None

    # Single pass: compute a shared valid mask across all trajectories.
    for i in range(args.num_trajectories):
        print(i)
        # dataset_directory = args.input_root / f"event_{i}_filling_hours_0_filling_intensity_mm_hr_0"
        dataset_directory = input_root / f"sample_{i:05d}"
        flow_path = dataset_directory / "flow_variables.npy"
        rain_path = dataset_directory / "rain_source.npy"

        if not flow_path.exists() or not rain_path.exists():
            print(f"Skipping {dataset_directory} (missing files)")
            continue

        available_trajectories.append((i, dataset_directory))

        flow_variables, h_dim, w_dim = _load_flow_variables(flow_path, args.burn_in_length)
        if global_h_dim is None:
            global_h_dim, global_w_dim = h_dim, w_dim
            
        elif (h_dim, w_dim) != (global_h_dim, global_w_dim):
            raise ValueError(
                f"Grid mismatch for traj {i}: got {h_dim}x{w_dim}, expected {global_h_dim}x{global_w_dim}"
            )

        h_vals = flow_variables[0, :, :, 0]
        traj_valid_no_nan = ~np.isnan(h_vals).any(axis=0)
        traj_h_max = np.max(np.where(np.isfinite(h_vals), h_vals, -np.inf), axis=0)
        traj_valid = traj_valid_no_nan & (traj_h_max >= args.depth_threshold)

        if global_valid_any is None:
            global_valid_any = traj_valid.copy()
        else:
            global_valid_any |= traj_valid

        if args.include_static_features:
            _, dem_valid, _ = _load_static_fields(dataset_directory)
            if dem_valid.shape != (h_dim, w_dim):
                raise ValueError(f"Static validity mask {dem_valid.shape} does not match grid {h_dim}x{w_dim}")
            if global_dem_valid is None:
                global_dem_valid = dem_valid.copy()
            else:
                global_dem_valid &= dem_valid

        global_valid_any_sum = int(global_valid_any.sum()) if global_valid_any is not None else 0
        global_dem_valid_sum = int(global_dem_valid.sum()) if global_dem_valid is not None else 0
        print(
            f"After traj {i}: sum(global_valid_any)={global_valid_any_sum}, "
            f"sum(global_dem_valid)={global_dem_valid_sum}"
        )

    if not available_trajectories:
        print("No trajectories found with both flow and rain files; nothing to write.")
        return

    global_valid = global_valid_any
    if global_dem_valid is not None:
        global_valid &= global_dem_valid.reshape(-1)

    n_global_valid = int(global_valid.sum())
    print(f"Global valid cells: {n_global_valid} / {global_valid.size}")

    mask_path = output_dir / args.mask_filename
    aggregate_mask = global_valid.reshape(global_h_dim, global_w_dim)
    np.save(mask_path, aggregate_mask)
    print(f"Wrote aggregate mask: {mask_path} with shape {aggregate_mask.shape}")


if __name__ == "__main__":
    main()
