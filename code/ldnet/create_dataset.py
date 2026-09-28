import argparse
import os
from pathlib import Path

import numpy as np

"""
Usage:
    # Build dataset without static features
    python create_dataset.py --num-trajectories 2 --output-dir data/postprocessed/illinois

    # Build dataset with static features (DEM z-score, slope raw, manning*100)
    python create_dataset.py --num-trajectories 2 --output-dir data/postprocessed/illinois --include-static-features

    # Build dataset with static features using slope x/y components
    python create_dataset.py --num-trajectories 2 --output-dir data/postprocessed/illinois --include-static-features --static-slope-xy
    python create_dataset.py --num-trajectories 101 --output-dir data/postprocessed/texas --aggregate-mask-path data/postprocessed/texas/aggregate_mask.npy --input-root data/texas/train_dataset --depth-threshold -1 --include-static-features --normalize --normalize-rain --static-slope-xy
"""


def _load_flow_variables(flow_path: Path, burn_in: int) -> tuple[np.ndarray, int, int]:
    flow = np.load(flow_path).astype(np.float32)[::4]
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


def _load_rain_source(rain_path: Path, expect_t: int) -> np.ndarray:
    rain = np.load(rain_path).astype(np.float32)
    if rain.ndim == 2:
        # (T, P)
        rain = rain[np.newaxis, ...]
    elif rain.ndim == 3:
        # (T, H, W) -> add batch dim
        rain = rain[np.newaxis, ...]
    elif rain.ndim == 4:
        # (T, 1, H, W) -> add batch dim
        rain = rain[np.newaxis, ...]
    elif rain.ndim == 5:
        # already (1, T, 1, H, W)
        pass
    else:
        raise ValueError(f"Unexpected rain_source shape: {rain.shape}")

    # ensure time length matches by trimming if needed
    if rain.shape[1] > expect_t:
        rain = rain[:, :expect_t]
    elif rain.shape[1] < expect_t:
        raise ValueError(f"rain_source has fewer timesteps ({rain.shape[1]}) than expected ({expect_t})")

    # flatten spatial dims
    rain = rain.reshape(rain.shape[0], rain.shape[1], -1)
    return rain


def _build_coords(h_dim: int, w_dim: int) -> np.ndarray:
    x = 2.0 * (np.arange(w_dim, dtype=np.float32) / w_dim - 0.5)
    aspect = h_dim / float(w_dim)
    y = 2.0 * (np.arange(h_dim, dtype=np.float32) / h_dim - 0.5) * aspect
    x_coords = np.tile(x, h_dim)
    y_coords = np.repeat(y, w_dim)
    coords = np.stack([x_coords, y_coords], axis=1)[None, :, :]
    return coords.astype(np.float32)


def _load_static_fields(dataset_directory: Path, static_slope_xy: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
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
    scaling = {
        "dem_mean": dem_mean,
        "dem_std": dem_std,
        "slope_scale": 1.0,
        "has_manning": False,
        "slope_mode": "xy" if static_slope_xy else "magnitude",
    }
    if static_slope_xy:
        slope_x = gx.astype(np.float32)
        slope_y = gy.astype(np.float32)
        slope_x[~dem_valid] = np.nan
        slope_y[~dem_valid] = np.nan
        static_fields = [dem_z, slope_x, slope_y]
    else:
        slope = np.sqrt(gx**2 + gy**2).astype(np.float32)
        slope[~dem_valid] = np.nan
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
        print(f"Missing manning.dat at {manning_path}; writing static features without Manning.")

    static = np.stack(static_fields, axis=-1)
    return static, dem_valid, scaling


def _accumulate_channel_stats(
    values: np.ndarray,
    count: np.ndarray,
    value_sum: np.ndarray,
    second_moment_sum: np.ndarray,
) -> None:
    flat = values.reshape(-1, values.shape[-1]).astype(np.float64, copy=False)
    finite = np.isfinite(flat)
    count += finite.sum(axis=0)
    safe_flat = np.where(finite, flat, 0.0)
    value_sum += safe_flat.sum(axis=0)
    second_moment_sum += np.square(safe_flat).sum(axis=0)


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
    parser.add_argument("--burn-in-length", type=int, default=1)
    parser.add_argument("--depth-threshold", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--aggregate-mask-path", type=Path, default=Path("data/postprocessed/illinois/aggregate_mask.npy"))
    parser.add_argument("--include-static-features", action="store_true", default=False)
    parser.add_argument("--static-slope-xy", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--normalize-rain", action="store_true", default=False)
    args = parser.parse_args()

    output_dir = _resolve_path(args.base_path, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    scaling_written = False
    scaling_path = output_dir / "static_scaling.npz"

    available_trajectories = []
    global_valid_any = None
    # global_h_dim = 5075
    # global_w_dim = 1661
    global_h_dim = 500
    global_w_dim = 500
    global_dem_valid = None
    global_valid = None
    input_root = _resolve_path(args.base_path, args.input_root)
    for i in range(args.num_trajectories):
        # dataset_directory = args.input_root / f"event_{i}_filling_hours_0_filling_intensity_mm_hr_0"
        dataset_directory = input_root / f"sample_{i:05d}"
        flow_path = dataset_directory / "flow_variables.npy"
        rain_path = dataset_directory / "rain_source.npy"

        if not flow_path.exists() or not rain_path.exists():
            print(f"Skipping {dataset_directory} (missing files)")
            continue

        available_trajectories.append((i, dataset_directory, flow_path, rain_path))

    if not available_trajectories:
        print("No trajectories found with both flow and rain files; nothing to write.")
        return

    if args.aggregate_mask_path is None:
        global_valid = global_valid_any
    else:
        mask_path = _resolve_path(args.base_path, args.aggregate_mask_path)
        mask_loaded = np.load(mask_path)
        if mask_loaded.ndim == 2:
            if mask_loaded.shape != (global_h_dim, global_w_dim):
                raise ValueError(
                    f"aggregate mask shape {mask_loaded.shape} does not match grid {global_h_dim}x{global_w_dim}"
                )
            mask_flat = mask_loaded.reshape(-1)
        elif mask_loaded.ndim == 1:
            if mask_loaded.size != global_h_dim * global_w_dim:
                raise ValueError(
                    f"aggregate mask length {mask_loaded.size} does not match grid size {global_h_dim * global_w_dim}"
                )
            mask_flat = mask_loaded
        else:
            raise ValueError(f"Expected aggregate mask with 1D/2D shape, got {mask_loaded.shape}")

        if np.issubdtype(mask_flat.dtype, np.bool_):
            global_valid = mask_flat.copy()
        else:
            global_valid = np.where(np.isfinite(mask_flat), mask_flat, 0) > 0
        print(f"Using aggregate mask from {mask_path}")

    if global_dem_valid is not None:
        global_valid &= global_dem_valid.reshape(-1)

    n_global_valid = int(global_valid.sum())
    print(f"Global valid cells: {n_global_valid} / {global_valid.size}")

    # Pass 2: write each trajectory using the shared mask.
    expected_static_channels = None
    flow_count = None
    flow_sum = None
    flow_second_moment_sum = None
    rain_count = None
    rain_sum = None
    rain_second_moment_sum = None
    for i, dataset_directory, flow_path, rain_path in available_trajectories:
        flow_variables, h_dim, w_dim = _load_flow_variables(flow_path, args.burn_in_length)
        t_len = flow_variables.shape[1]
        rain_source = _load_rain_source(rain_path, expect_t=t_len)[:, :, 1:]

        coords = _build_coords(h_dim, w_dim)

        static = None
        scaling = None
        if args.include_static_features:
            static, _, scaling = _load_static_fields(dataset_directory, static_slope_xy=args.static_slope_xy)
            if static.shape[:2] != (h_dim, w_dim):
                raise ValueError(f"Static feature shape {static.shape} does not match grid {h_dim}x{w_dim}")
            if expected_static_channels is None:
                expected_static_channels = int(static.shape[-1])
            elif static.shape[-1] != expected_static_channels:
                raise ValueError(
                    "Static feature channel count is inconsistent across trajectories: "
                    f"expected {expected_static_channels}, got {static.shape[-1]} for traj {i}. "
                    "Use inputs with consistent Manning availability."
                )

        flow_variables = flow_variables[:, :, global_valid, :]
        coords = coords[:, global_valid, :]
        if static is not None:
            static = static.reshape(1, h_dim * w_dim, static.shape[-1])[:, global_valid, :]

        if args.normalize:
            if flow_count is None:
                num_channels = flow_variables.shape[-1]
                flow_count = np.zeros(num_channels, dtype=np.int64)
                flow_sum = np.zeros(num_channels, dtype=np.float64)
                flow_second_moment_sum = np.zeros(num_channels, dtype=np.float64)
            _accumulate_channel_stats(
                flow_variables,
                flow_count,
                flow_sum,
                flow_second_moment_sum,
            )

        if args.normalize_rain:
            if rain_count is None:
                num_rain_channels = rain_source.shape[-1]
                rain_count = np.zeros(num_rain_channels, dtype=np.int64)
                rain_sum = np.zeros(num_rain_channels, dtype=np.float64)
                rain_second_moment_sum = np.zeros(num_rain_channels, dtype=np.float64)
            _accumulate_channel_stats(
                rain_source,
                rain_count,
                rain_sum,
                rain_second_moment_sum,
            )

        flow_variables = flow_variables.astype(np.float16)
        coords = coords.astype(np.float16)
        rain_source = rain_source.astype(np.float16)
        if static is not None:
            static = static.astype(np.float16)

        flow_out = np.lib.format.open_memmap(
            output_dir / f"flow_variables_traj{i}.npy",
            mode="w+",
            dtype=flow_variables.dtype,
            shape=flow_variables.shape,
        )
        flow_out[:] = flow_variables[:]
        flow_out.flush()

        coords_out = np.lib.format.open_memmap(
            output_dir / f"coords_traj{i}.npy",
            mode="w+",
            dtype=coords.dtype,
            shape=coords.shape,
        )
        coords_out[:] = coords[:]
        coords_out.flush()

        rain_out = np.lib.format.open_memmap(
            output_dir / f"rain_source_traj{i}.npy",
            mode="w+",
            dtype=rain_source.dtype,
            shape=rain_source.shape,
        )
        rain_out[:] = rain_source[:]
        rain_out.flush()

        if static is not None:
            static_filename = (
                f"static_features_xy_traj{i}.npy" if args.static_slope_xy else f"static_features_traj{i}.npy"
            )
            static_out = np.lib.format.open_memmap(
                output_dir / static_filename,
                mode="w+",
                dtype=static.dtype,
                shape=static.shape,
            )
            static_out[:] = static[:]
            static_out.flush()

            if scaling is not None and not scaling_written:
                scaling_payload = {
                    "dem_mean": np.float32(scaling["dem_mean"]),
                    "dem_std": np.float32(scaling["dem_std"]),
                    "slope_scale": np.float32(scaling["slope_scale"]),
                    "num_static_channels": np.int32(static.shape[-1]),
                    "has_manning": np.bool_(scaling.get("has_manning", False)),
                    "static_slope_xy": np.bool_(args.static_slope_xy),
                }
                if "manning_scale" in scaling:
                    scaling_payload["manning_scale"] = np.float32(scaling["manning_scale"])
                np.savez(scaling_path, **scaling_payload)
                scaling_written = True

        print(
            f"wrote traj {i}: flow {flow_variables.shape}, coords {coords.shape}, "
            f"rain {rain_source.shape}"
        )
        if static is not None:
            print(f"wrote traj {i}: static {static.shape}")

    if args.normalize:
        if flow_count is None or np.any(flow_count == 0):
            raise ValueError("Cannot compute normalization statistics because one or more flow channels had no finite values.")
        mean = flow_sum / flow_count
        var = flow_second_moment_sum / flow_count - np.square(mean)
        std = np.sqrt(np.maximum(var, 0.0))
        np.save(output_dir / "mean.npy", mean.astype(np.float32))
        np.save(output_dir / "std.npy", std.astype(np.float32))
        print(f"Saved normalization statistics to {output_dir / 'mean.npy'} and {output_dir / 'std.npy'}")

    if args.normalize_rain:
        if rain_count is None or np.any(rain_count == 0):
            raise ValueError("Cannot compute rain normalization statistics because one or more rain channels had no finite values.")
        rain_mean = rain_sum / rain_count
        rain_var = rain_second_moment_sum / rain_count - np.square(rain_mean)
        rain_std = np.sqrt(np.maximum(rain_var, 0.0))
        np.save(output_dir / "rain_mean.npy", rain_mean.astype(np.float32))
        np.save(output_dir / "rain_std.npy", rain_std.astype(np.float32))
        print(f"Saved rain normalization statistics to {output_dir / 'rain_mean.npy'} and {output_dir / 'rain_std.npy'}")


if __name__ == "__main__":
    main()
