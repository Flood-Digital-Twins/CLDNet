import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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


def _estimate_limits(frames: list[np.ndarray], percentile: float = 99.9) -> tuple[float, float]:
    vals = []
    for frame in frames:
        finite = np.isfinite(frame)
        if np.any(finite):
            vals.append(frame[finite])
    if not vals:
        return (0.0, 1.0)
    arr = np.concatenate(vals, axis=0)
    vmax = float(np.nanpercentile(arr, percentile))
    return (0.0, max(vmax, 0.1))


def _estimate_feature_limits(frame: np.ndarray, lo: float = 0.1, hi: float = 99.9) -> tuple[float, float]:
    finite = np.isfinite(frame)
    if not np.any(finite):
        return (0.0, 1.0)
    vals = frame[finite]
    vmin = float(np.nanpercentile(vals, lo))
    vmax = float(np.nanpercentile(vals, hi))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return (vmin if np.isfinite(vmin) else 0.0, (vmin + 1.0) if np.isfinite(vmin) else 1.0)
    return (vmin, vmax)


def _extract_step(arr: np.ndarray, step: int) -> np.ndarray:
    if arr.ndim == 3:
        if arr.shape[0] == 1:  # (1, T, R)
            return arr[0, step]
        if arr.shape[-1] == 1:  # (T, R, 1)
            return arr[step, :, 0]
        return arr[step]
    if arr.ndim == 2:
        return arr[step]
    if arr.ndim == 4 and arr.shape[0] == 1:  # (1, T, H, W)
        return arr[0, step]
    raise ValueError(f"Unsupported rain array shape: {arr.shape}")


def _to_rain_grid(frame: np.ndarray, rows: int = 39, cols: int = 13) -> np.ndarray:
    if frame.ndim == 2 and frame.shape == (rows, cols):
        return frame
    if frame.ndim == 1 and frame.size == rows * cols:
        return frame.reshape(rows, cols)
    raise ValueError(
        f"Expected rain step to be shape ({rows}, {cols}) or flat length {rows * cols}, got {frame.shape}"
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    mask_path = base_dir / "data/postprocessed/illinois" / "aggregate_mask.npy"

    mask = np.load(mask_path)
    flat = mask.reshape(-1)

    nan_count = int(np.isnan(flat).sum()) if np.issubdtype(flat.dtype, np.floating) else 0
    valid_count = int(np.sum(flat.astype(bool)))
    total_count = int(flat.size)

    print(f"Mask path: {mask_path}")
    print(f"Total points: {total_count}")
    print(f"NaN points: {nan_count}")
    print(f"Valid points: {valid_count}")
    static_check_path = base_dir / "data/postprocessed/illinois" / "static_features_traj99.npy"
    if static_check_path.exists():
        static_check = np.load(static_check_path, mmap_mode="r")
        print(f"Static features shape: {static_check.shape}")
    else:
        print(f"Static features file not found, skipping: {static_check_path}")

    # FLOW AND RAIN
    rain_path = base_dir / "data/postprocessed/illinois" / "rain_source_traj120.npy"
    rain = np.load(rain_path).reshape(1, 96, 39, 13)
    print(rain.shape)
    steps = [52, 53, 54]
    out_dir = base_dir / "plots" / "rain_steps_traj120"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for step in steps:
        frame = _extract_step(rain, step)
        frame_grid = np.clip(_to_rain_grid(frame, rows=39, cols=13), 0, 0.1)
        print(np.max(frame_grid))
        out_path = out_dir / f"rain_step_{step}.png"
    
        fig, ax = plt.subplots(figsize=(3.9, 1.3))
        ax.imshow(frame_grid, cmap="Blues", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)
    
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
        print(f"Saved borderless image: {out_path}")

    flow_path = base_dir / "data/postprocessed/illinois" / "flow_variables_traj120.npy"
    coords_path = base_dir / "data/postprocessed/illinois" / "coords_traj120.npy"
    flow = np.load(flow_path, mmap_mode="r")
    coords = np.load(coords_path, mmap_mode="r")
    print(f"Flow shape: {flow.shape}, dtype: {flow.dtype}")
    print(f"Coords shape: {coords.shape}, dtype: {coords.dtype}")
    
    steps = [52, 53, 54]
    out_dir = base_dir / "plots" / "flow_height_steps_traj120"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    h_dim, w_dim = mask.shape
    scatter_indices = _resolve_scatter_indices(mask, h_dim, w_dim, coords)
    if scatter_indices.size != flow.shape[2]:
        raise ValueError(f"Scatter mismatch: scatter={scatter_indices.size}, flow points={flow.shape[2]}")
    
    # Match ldnet_chicago_efficient_test.py visual style: Blues on white.
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="white")
    
    preview_frames = []
    for step in steps:
        h_valid = flow[0, step, :, 0].astype(np.float32, copy=False)
        full = np.full(h_dim * w_dim, np.nan, dtype=np.float32)
        full[scatter_indices] = h_valid
        preview_frames.append(full.reshape(h_dim, w_dim))
    vmin, vmax = _estimate_limits(preview_frames, percentile=99.9)

    
    for step in steps:
        # Height component is channel 0 in (h, u, v).
        h_valid = flow[0, step, :, 0].astype(np.float32, copy=False)
        full = np.full(h_dim * w_dim, np.nan, dtype=np.float32)
        full[scatter_indices] = h_valid
        frame_grid = full.reshape(h_dim, w_dim)
        vmax = vmax / 3.0  # Adjust vmax for better contrast in flow height visualization.
        out_path = out_dir / f"flow_height_step_{step}.png"
        fig_h = 6.0
        fig_w = max(2.5, fig_h * (w_dim / h_dim))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
        ax.set_facecolor("white")
        ax.imshow(frame_grid, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)
    
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02, transparent=False, facecolor="white")
        plt.close(fig)
        print(f"Saved flow height image: {out_path}")


    ### STATIC FEATURES###
    # coords_path = base_dir / "data/postprocessed/illinois" / "coords_traj120.npy"
    # static_path = base_dir / "data/postprocessed/illinois" / "static_features_traj120.npy"
    # coords = np.load(coords_path, mmap_mode="r")
    # static = np.load(static_path, mmap_mode="r")
    # print(f"Coords shape: {coords.shape}, dtype: {coords.dtype}")
    # print(f"Static shape: {static.shape}, dtype: {static.dtype}")

    # out_dir = base_dir / "plots" / "static_features_traj120"
    # out_dir.mkdir(parents=True, exist_ok=True)

    # h_dim, w_dim = mask.shape
    # scatter_indices = _resolve_scatter_indices(mask, h_dim, w_dim, coords)
    # if scatter_indices.size != static.shape[1]:
    #     raise ValueError(f"Scatter mismatch: scatter={scatter_indices.size}, static points={static.shape[1]}")

    # # Match ldnet_chicago_efficient_test.py visual style: Blues on white.
    # cmap = plt.get_cmap("Blues").copy()
    # cmap.set_bad(color="white")

    # feature_names = ["zscore", "slope", "manning"]
    # for feature_idx, feature_name in enumerate(feature_names):
    #     values = static[0, :, feature_idx].astype(np.float32, copy=False)
    #     full = np.full(h_dim * w_dim, np.nan, dtype=np.float32)
    #     full[scatter_indices] = values
    #     frame_grid = full.reshape(h_dim, w_dim)
    #     vmin, vmax = _estimate_feature_limits(frame_grid, lo=0.1, hi=99.9)

    #     out_path = out_dir / f"static_{feature_name}.png"
    #     fig_h = 6.0
    #     fig_w = max(2.5, fig_h * (w_dim / h_dim))
    #     fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    #     ax.set_facecolor("white")
    #     ax.imshow(frame_grid, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     for spine in ax.spines.values():
    #         spine.set_visible(True)
    #         spine.set_color("black")
    #         spine.set_linewidth(1.0)

    #     fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02, transparent=False, facecolor="white")
    #     plt.close(fig)
    #     print(f"Saved static feature image: {out_path}")


if __name__ == "__main__":
    main()
