import argparse
import os
from pathlib import Path

import numpy as np

"""
Usage:
    python create_dataset_static_from_mask.py \
        --num-trajectories 2 \
        --input-root data/sims_30 \
        --aggregate-mask-path data/postprocessed/illinois/aggregate_mask.npy \
        --output-dir data/postprocessed/illinois
"""


def _element_values_to_grid(values: np.ndarray, dem_valid: np.ndarray, legacy_order: bool = False) -> np.ndarray:
    """Place a per-element field of the simulator (input/field/*.dat) on the DEM grid.

    SynxFlow/HiPIMS number the valid cells row by row starting from the SOUTHERN (last) row of the raster,
    west to east within a row; z.dat read in that order reproduces DEM.npy to 5e-4 m.
    ``legacy_order=True`` instead fills the grid from the northern row downward. That is how the Des Plaines
    static features behind the released epoch-539 checkpoints were built; it mis-registers the Manning channel
    (wrong at 6.7 % of the evaluation cells) and is kept only to reproduce those checkpoints' inputs.
    """
    full = np.full(dem_valid.shape, np.nan, dtype=np.float32)
    if legacy_order:
        full.ravel()[dem_valid.ravel()] = values
    else:
        south_up = full[::-1]  # a view of `full` whose first row is the southern row
        south_up[dem_valid[::-1]] = values
    return full


def _load_static_fields(dataset_directory: Path, legacy_manning_order: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
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

        manning_full = _element_values_to_grid(manning_vals, dem_valid, legacy_manning_order)
        scaling["manning_scale"] = 100.0
        scaling["has_manning"] = True
        static_fields.append(manning_full * scaling["manning_scale"])
    else:
        print(f"Missing manning.dat at {manning_path}; writing static features without Manning.")

    static = np.stack(static_fields, axis=-1)
    return static, dem_valid, scaling


def _load_aggregate_mask(mask_path: Path) -> tuple[np.ndarray, tuple[int, int] | None]:
    mask_loaded = np.load(mask_path)
    if mask_loaded.ndim == 2:
        mask_flat = mask_loaded.reshape(-1)
        grid_shape = (int(mask_loaded.shape[0]), int(mask_loaded.shape[1]))
    elif mask_loaded.ndim == 1:
        mask_flat = mask_loaded
        grid_shape = None
    else:
        raise ValueError(f"Expected aggregate mask with 1D/2D shape, got {mask_loaded.shape}")

    if np.issubdtype(mask_flat.dtype, np.bool_):
        global_valid = mask_flat.astype(bool, copy=True)
    else:
        global_valid = np.where(np.isfinite(mask_flat), mask_flat, 0) > 0

    return global_valid, grid_shape


def _resolve_path(base_path: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_path / path


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
    parser.add_argument("--output-dir", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument(
        "--aggregate-mask-path",
        type=Path,
        default=Path("data/postprocessed/illinois/aggregate_mask.npy"),
    )
    parser.add_argument(
        "--legacy-manning-order",
        action="store_true",
        default=False,
        help="Map manning.dat as when the released Des Plaines epoch-539 checkpoints were trained (pasted from the "
        "northern row down, which mis-registers the channel). Needed to reproduce the paper with those "
        "checkpoints; leave off for new datasets and new models.",
    )
    args = parser.parse_args()

    output_dir = _resolve_path(args.base_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    mask_path = _resolve_path(args.base_path, args.aggregate_mask_path)
    global_valid, grid_shape = _load_aggregate_mask(mask_path)

    print(f"Using aggregate mask: {mask_path}")
    print(f"Global valid cells: {int(global_valid.sum())} / {global_valid.size}")

    scaling_written = False
    scaling_path = output_dir / "static_scaling.npz"

    expected_static_channels = None
    input_root = _resolve_path(args.base_path, args.input_root)
    for i in range(args.num_trajectories):
        print(i)
        dataset_directory = input_root / f"event_{i}_filling_hours_0_filling_intensity_mm_hr_0"
        dem_path = dataset_directory / "DEM.npy"
        manning_path = dataset_directory / "input/field/manning.dat"
        if not dem_path.exists() or not manning_path.exists():
            print(f"Skipping {dataset_directory} (missing DEM.npy or manning.dat)")
            continue

        static, _, scaling = _load_static_fields(dataset_directory, legacy_manning_order=args.legacy_manning_order)
        h_dim, w_dim = static.shape[:2]
        if expected_static_channels is None:
            expected_static_channels = int(static.shape[-1])
        elif static.shape[-1] != expected_static_channels:
            raise ValueError(
                "Static feature channel count is inconsistent across trajectories: "
                f"expected {expected_static_channels}, got {static.shape[-1]} for traj {i}. "
                "Use inputs with consistent Manning availability."
            )

        if grid_shape is not None and grid_shape != (h_dim, w_dim):
            raise ValueError(
                f"aggregate mask shape {grid_shape} does not match static grid {(h_dim, w_dim)} for traj {i}"
            )
        if global_valid.size != h_dim * w_dim:
            raise ValueError(
                f"aggregate mask length {global_valid.size} does not match static grid size {h_dim * w_dim} for traj {i}"
            )

        static = static.reshape(1, h_dim * w_dim, static.shape[-1])[:, global_valid, :]
        static = static.astype(np.float16)

        static_out = np.lib.format.open_memmap(
            output_dir / f"static_features_traj{i}.npy",
            mode="w+",
            dtype=static.dtype,
            shape=static.shape,
        )
        static_out[:] = static[:]
        static_out.flush()

        if not scaling_written:
            scaling_payload = {
                "dem_mean": np.float32(scaling["dem_mean"]),
                "dem_std": np.float32(scaling["dem_std"]),
                "slope_scale": np.float32(scaling["slope_scale"]),
                "num_static_channels": np.int32(static.shape[-1]),
                "has_manning": np.bool_(scaling.get("has_manning", False)),
            }
            if "manning_scale" in scaling:
                scaling_payload["manning_scale"] = np.float32(scaling["manning_scale"])
            np.savez(scaling_path, **scaling_payload)
            scaling_written = True

        # print(f"wrote traj {i}: static {static.shape}")


if __name__ == "__main__":
    main()
