#!/usr/bin/env python3
"""
Run a USGS-gauge LD-EnSF experiment with held-out validation gauges.

Pipeline:
1. Build an observation dataset using all USGS gauges except the six notebook
   validation gauges.
2. Train the LSTM encoder on those non-validation gauges.
3. Run LD-EnSF assimilation on trajectory 120.
4. Evaluate the six validation gauges with NSE, KGE, and peak relative error
   after the same mean-shift normalization used in the USGS notebook. The
   peak error uses the hydrograph amplitude for normalization, i.e.
   peak minus minimum.
5. Save a six-panel hydrograph figure at the held-out gauges using the DA
   forecast and posterior outputs.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.patches import ConnectionPatch

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path(sys.executable)
VAL_GAUGES = [0, 7, 14, 19, 20, 25]
START_DT = pd.Timestamp("2013-04-17 08:00:00")
END_DT = pd.Timestamp("2013-04-21 07:00:00")

sys.path.append(str(REPO_ROOT))

from efficient_fourier_ldnet import EfficientFourierLDNN
from create_ldensf_dataset import _load_reduced_trajectory
from test_ldensf_assimilation import _load_ldnet_model


def _resolve_path(base_path: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_path / path


def _run_step(label: str, cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def _load_usgs_height_series(usgs_height_root: Path, gage_idx: int) -> pd.DataFrame:
    matches = sorted(usgs_height_root.glob(f"{gage_idx}_*_height.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find height CSV for gauge index {gage_idx} in {usgs_height_root}")

    local_df = pd.read_csv(matches[0])
    df = local_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["Gage Height (ft)"] = pd.to_numeric(df["Gage Height (ft)"], errors="coerce")
    df = df.dropna(subset=["datetime", "Gage Height (ft)"]).sort_values("datetime")
    df = df[(df["datetime"] >= START_DT) & (df["datetime"] <= END_DT)]
    if df.empty:
        return df

    value_frame = df[["datetime", "Gage Height (ft)"]].drop_duplicates(subset="datetime", keep="last")
    value_frame = value_frame.set_index("datetime").resample("1h").nearest()
    value_frame = value_frame.loc[START_DT:END_DT]

    meta_cols = [c for c in df.columns if c not in {"datetime", "Gage Height (ft)"}]
    if meta_cols:
        meta_frame = df[["datetime"] + meta_cols].drop_duplicates(subset="datetime", keep="last")
        meta_frame = meta_frame.set_index("datetime").resample("1h").ffill().bfill()
        meta_frame = meta_frame.loc[START_DT:END_DT]
        value_frame = value_frame.join(meta_frame, how="left")

    if gage_idx in {14, 20}:
        alt_datum = (float(value_frame["alt_va"].iloc[0]) + 10.0) * 0.3048
    else:
        alt_datum = float(value_frame["alt_va"].iloc[0]) * 0.3048

    value_frame["stage"] = value_frame["Gage Height (ft)"] * 0.3048
    value_frame["wse"] = value_frame["stage"] + alt_datum
    return value_frame.reset_index()


def _load_usgs_site_no(usgs_height_root: Path, gage_idx: int) -> str:
    matches = sorted(usgs_height_root.glob(f"{gage_idx}_*_height.csv"))
    if not matches:
        return str(gage_idx)
    parts = matches[0].stem.split("_")
    return parts[1] if len(parts) >= 2 else str(gage_idx)


def _load_gauge_mapping(gauge_mapping_path: Path, gauge_indices: list[int]) -> pd.DataFrame:
    gauge_df = pd.read_csv(gauge_mapping_path)
    gauge_df["i"] = pd.to_numeric(gauge_df["i"], errors="coerce").astype("Int64")
    gauge_df["x_proj"] = pd.to_numeric(gauge_df["x_proj"], errors="coerce")
    gauge_df["y_proj"] = pd.to_numeric(gauge_df["y_proj"], errors="coerce")
    gauge_df = gauge_df.dropna(subset=["i", "staid", "x_proj", "y_proj"]).copy()
    gauge_df["i"] = gauge_df["i"].astype(int)
    requested = [int(idx) for idx in gauge_indices]
    gauge_df = gauge_df[gauge_df["i"].isin(requested)].copy()
    if gauge_df.empty:
        raise ValueError(f"No gauge locations found in {gauge_mapping_path} for indices {gauge_indices}")

    order = {idx: pos for pos, idx in enumerate(requested)}
    gauge_df["order"] = gauge_df["i"].map(order)
    gauge_df = gauge_df.sort_values("order").reset_index(drop=True)
    return gauge_df


def _nse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if len(obs) < 2 or np.isclose(denom, 0.0):
        return float("nan")
    return float(1.0 - np.sum((sim - obs) ** 2) / denom)


def _kge(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    if len(obs) < 2 or np.isclose(np.std(obs, ddof=0), 0.0) or np.isclose(np.mean(obs), 0.0):
        return float("nan")
    r = np.corrcoef(sim, obs)[0, 1]
    if np.isnan(r):
        return float("nan")
    alpha = np.std(sim, ddof=0) / np.std(obs, ddof=0)
    beta = np.mean(sim) / np.mean(obs)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def _peak_relative_error(obs: np.ndarray, sim: np.ndarray) -> float:
    # epsilon_h_peak = |h_peak^sim - h_peak^obs| / (h_peak^obs - h_min^obs)
    obs_peak = float(np.max(obs))
    obs_min = float(np.min(obs))
    sim_peak = float(np.max(sim))
    obs_range = obs_peak - obs_min
    if np.isclose(obs_range, 0.0):
        return float("nan")
    return float(abs((sim_peak - obs_peak) / obs_range))


def _align_sim_to_obs_mean(sim: np.ndarray, obs: np.ndarray) -> np.ndarray:
    sim = np.asarray(sim, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    return sim - float(np.nanmean(sim)) + float(np.nanmean(obs))


def _decode_latent_series(
    model: EfficientFourierLDNN,
    latent_hist: np.ndarray,
    x_coords: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    x_tensor = torch.from_numpy(np.asarray(x_coords, dtype=np.float32)[None, :, :]).to(device)
    series = np.empty((latent_hist.shape[0], x_coords.shape[0]), dtype=np.float32)
    with torch.no_grad():
        for t in range(latent_hist.shape[0]):
            latent_t = torch.from_numpy(np.asarray(latent_hist[t : t + 1], dtype=np.float32)).to(device)
            pred_t = model._decode_points(x_tensor, latent_t, chunk_size=model.chunk_size)
            series[t] = pred_t[0, :, 0].detach().cpu().numpy()
    return series


def _save_validation_hydrograph_figure(
    *,
    plot_entries: list[dict[str, object]],
    out_path: Path,
    background_traj_id: int,
) -> None:
    if not plot_entries:
        raise RuntimeError("No hydrograph entries were provided.")

    title_fontsize = 16
    label_fontsize = 15
    tick_fontsize = 13
    legend_fontsize = 16

    fig, axes = plt.subplots(2, 3, figsize=(20.5, 12.5), sharex=False, sharey=False)
    axes_flat = axes.ravel()

    obs_color = "black"
    base_color = "tab:green"
    forecast_color = "#d95f02"
    assim_color = "#1f77b4"
    base_label = "Base surrogate"

    legend_handles = None
    legend_labels = None

    for idx, entry in enumerate(plot_entries):
        ax = axes_flat[idx]
        times = pd.to_datetime(entry["datetime"])
        obs = np.asarray(entry["obs"], dtype=np.float32)
        base = np.asarray(entry["base"], dtype=np.float32) if "base" in entry else None
        forecast = np.asarray(entry["forecast"], dtype=np.float32)
        assim = np.asarray(entry["assim"], dtype=np.float32)
        site_no = str(entry["site_no"])
        gage_idx = int(entry["gage_idx"])

        ax.plot(times, obs, color=obs_color, lw=1.6, label="Reference", zorder=3)
        if base is not None:
            ax.plot(
                times,
                base,
                color=base_color,
                lw=1.4,
                ls=":",
                label=base_label,
                zorder=2.5,
            )
        ax.plot(
            times,
            forecast,
            color=forecast_color,
            lw=1.4,
            ls="--",
            label="Perturbed open-loop forecast",
            zorder=2,
        )
        ax.plot(
            times,
            assim,
            color=assim_color,
            lw=1.6,
            label="LD-EnSF posterior",
            zorder=4,
        )

        combined_parts = [obs.reshape(-1)]
        if base is not None:
            combined_parts.append(base.reshape(-1))
        combined_parts.extend([forecast.reshape(-1), assim.reshape(-1)])
        combined = np.concatenate(combined_parts)
        combined = combined[np.isfinite(combined)]
        if combined.size:
            y_min = float(np.min(combined))
            y_max = float(np.max(combined))
            # Keep a tight vertical window so the hydrographs stay easy to read.
            y_pad = max(0.02 * (y_max - y_min), 0.05)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.set_xlim(times.min(), times.max())
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.set_title(f"{site_no} (gauge {gage_idx})", fontsize=title_fontsize)
        ax.set_xlabel("Time", fontsize=label_fontsize)
        ax.set_ylabel("WSE (m)", fontsize=label_fontsize)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="both", labelsize=tick_fontsize, labelrotation=0)
        if idx < 3:
            ax.tick_params(labelbottom=False)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    for ax in axes_flat[len(plot_entries) :]:
        ax.axis("off")

    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=4,
            frameon=False,
            fontsize=legend_fontsize,
            bbox_to_anchor=(0.5, 0.95),
        )
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.12, top=0.84, wspace=0.28, hspace=0.42)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_validation_dem_pointer_figure(
    *,
    plot_entries: list[dict[str, object]],
    out_path: Path,
    background_traj_id: int,
    dem_path: Path,
    gauge_mapping_path: Path,
) -> None:
    if not plot_entries:
        raise RuntimeError("No hydrograph entries were provided.")

    entry_by_idx = {int(entry["gage_idx"]): entry for entry in plot_entries}
    ordered_gauge_indices = [int(entry["gage_idx"]) for entry in plot_entries]
    gauge_df = _load_gauge_mapping(gauge_mapping_path, ordered_gauge_indices)
    gauge_df = gauge_df[gauge_df["i"].isin(entry_by_idx)].copy().reset_index(drop=True)
    if gauge_df.empty:
        raise RuntimeError("Could not find gauge locations for the hydrograph entries.")

    gauge_df = gauge_df.sort_values("x_proj").reset_index(drop=True)
    half = len(gauge_df) // 2
    left_df = gauge_df.iloc[:half].sort_values("y_proj", ascending=False).reset_index(drop=True)
    right_df = gauge_df.iloc[half:].sort_values("y_proj", ascending=False).reset_index(drop=True)

    try:
        import rasterio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("rasterio is required to render the DEM pointer figure.") from exc

    with rasterio.open(dem_path) as ds:
        dem = ds.read(1, masked=True).astype(np.float32)
        bounds = ds.bounds
        dem_label = "Elevation (m)"

    dem_vals = np.asarray(dem.compressed() if np.ma.isMaskedArray(dem) else dem[np.isfinite(dem)], dtype=np.float32)
    if dem_vals.size == 0:
        raise RuntimeError(f"DEM at {dem_path} has no finite values.")
    dem_vmin = float(np.nanpercentile(dem_vals, 2.0))
    dem_vmax = float(np.nanpercentile(dem_vals, 98.0))
    if not np.isfinite(dem_vmin) or not np.isfinite(dem_vmax) or dem_vmax <= dem_vmin:
        dem_vmin = float(np.nanmin(dem_vals))
        dem_vmax = float(np.nanmax(dem_vals))

    panel_title_fontsize = 16
    panel_label_fontsize = 15
    panel_tick_fontsize = 13
    legend_fontsize = 16
    map_title_fontsize = 18
    map_label_fontsize = 16
    map_tick_fontsize = 14
    cbar_label_fontsize = 14
    cbar_tick_fontsize = 13

    fig = plt.figure(figsize=(21.5, 14.0))
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.18, 1.58, 1.18],
        wspace=0.04,
        hspace=0.30,
    )
    ax_left = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_map = fig.add_subplot(gs[:, 1])
    ax_right = [fig.add_subplot(gs[i, 2]) for i in range(3)]

    dem_cmap = plt.get_cmap("terrain").copy()
    dem_cmap.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    dem_im = ax_map.imshow(
        dem,
        origin="upper",
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        cmap=dem_cmap,
        vmin=dem_vmin,
        vmax=dem_vmax,
        interpolation="nearest",
        aspect="equal",
    )
    cbar = fig.colorbar(
        dem_im,
        ax=ax_map,
        orientation="horizontal",
        pad=0.060,
        shrink=0.58,
        fraction=0.038,
        aspect=28,
    )
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize, pad=1)
    cbar.set_label(dem_label, fontsize=cbar_label_fontsize, labelpad=2)

    ax_map.set_title("Digital elevation model with held-out gauges", fontsize=map_title_fontsize, pad=8)
    ax_map.set_xlabel("Projected Easting (m)", fontsize=map_label_fontsize)
    ax_map.set_ylabel("Projected Northing (m)", fontsize=map_label_fontsize)
    ax_map.tick_params(labelsize=map_tick_fontsize)
    ax_map.ticklabel_format(axis="x", style="sci", scilimits=(6, 6), useMathText=False)
    ax_map.xaxis.offset_text_position = "top"
    ax_map.xaxis.get_offset_text().set_fontsize(map_tick_fontsize + 1)

    left_colors = ["tab:blue", "tab:orange", "tab:green"]
    right_colors = ["tab:red", "tab:purple", "tab:brown"]
    obs_color = "black"
    base_color = "tab:green"
    forecast_color = "#d95f02"
    assim_color = "#1f77b4"
    base_label = "Base surrogate"

    def _plot_panel(ax: plt.Axes, entry: dict[str, object], color: str, show_xlabel: bool) -> None:
        times = pd.to_datetime(entry["datetime"])
        obs = np.asarray(entry["obs"], dtype=np.float32)
        base = np.asarray(entry["base"], dtype=np.float32) if "base" in entry else None
        forecast = np.asarray(entry["forecast"], dtype=np.float32)
        assim = np.asarray(entry["assim"], dtype=np.float32)
        site_no = str(entry["site_no"])
        gage_idx = int(entry["gage_idx"])

        ax.plot(times, obs, color=obs_color, lw=1.6, label="Reference", zorder=3)
        if base is not None:
            ax.plot(
                times,
                base,
                color=base_color,
                lw=1.4,
                ls=":",
                label=base_label,
                zorder=2.5,
            )
        ax.plot(
            times,
            forecast,
            color=forecast_color,
            lw=1.4,
            ls="--",
            label="Perturbed open-loop forecast",
            zorder=2,
        )
        ax.plot(times, assim, color=assim_color, lw=1.6, label="LD-EnSF posterior", zorder=4)

        combined_parts = [obs.reshape(-1)]
        if base is not None:
            combined_parts.append(base.reshape(-1))
        combined_parts.extend([forecast.reshape(-1), assim.reshape(-1)])
        combined = np.concatenate(combined_parts)
        combined = combined[np.isfinite(combined)]
        if combined.size:
            y_min = float(np.min(combined))
            y_max = float(np.max(combined))
            y_pad = max(0.02 * (y_max - y_min), 0.05)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.set_xlim(times.min(), times.max())
        ax.grid(True, alpha=0.25, linewidth=0.7)
        ax.set_title(f"{site_no} (gauge {gage_idx})", fontsize=panel_title_fontsize)
        ax.set_xlabel("Time", fontsize=panel_label_fontsize)
        ax.set_ylabel("WSE (m)", fontsize=panel_label_fontsize)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="both", labelrotation=0, labelsize=panel_tick_fontsize)
        if not show_xlabel:
            ax.tick_params(labelbottom=False)

    gauge_meta_by_idx = {int(row["i"]): row for _, row in gauge_df.iterrows()}
    left_entries = []
    for g in left_df["i"].tolist():
        idx = int(g)
        entry = dict(entry_by_idx[idx])
        meta = gauge_meta_by_idx[idx]
        entry["x_proj"] = float(meta["x_proj"])
        entry["y_proj"] = float(meta["y_proj"])
        left_entries.append(entry)

    right_entries = []
    for g in right_df["i"].tolist():
        idx = int(g)
        entry = dict(entry_by_idx[idx])
        meta = gauge_meta_by_idx[idx]
        entry["x_proj"] = float(meta["x_proj"])
        entry["y_proj"] = float(meta["y_proj"])
        right_entries.append(entry)

    for i, ax in enumerate(ax_left):
        if i >= len(left_entries):
            ax.axis("off")
            continue
        _plot_panel(ax, left_entries[i], left_colors[i], show_xlabel=(i == len(ax_left) - 1))

    for i, ax in enumerate(ax_right):
        if i >= len(right_entries):
            ax.axis("off")
            continue
        _plot_panel(ax, right_entries[i], right_colors[i], show_xlabel=(i == len(ax_right) - 1))

    for i, entry in enumerate(left_entries):
        x = float(entry["x_proj"])
        y = float(entry["y_proj"])
        color = left_colors[i]
        ax_map.scatter(x, y, s=75, c=color, marker="o", edgecolors="black", linewidths=0.8, zorder=5)
        ax_map.text(
            x + 350.0,
            y + 350.0,
            str(int(entry["gage_idx"])),
            color=color,
            fontsize=13,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5),
            zorder=6,
        )
    for i, entry in enumerate(right_entries):
        x = float(entry["x_proj"])
        y = float(entry["y_proj"])
        color = right_colors[i]
        ax_map.scatter(x, y, s=75, c=color, marker="s", edgecolors="black", linewidths=0.8, zorder=5)
        ax_map.text(
            x + 350.0,
            y + 350.0,
            str(int(entry["gage_idx"])),
            color=color,
            fontsize=13,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5),
            zorder=6,
        )

    for i, entry in enumerate(left_entries):
        con = ConnectionPatch(
            xyA=(float(entry["x_proj"]), float(entry["y_proj"])),
            coordsA=ax_map.transData,
            xyB=(1.0, 0.5),
            coordsB=ax_left[i].transAxes,
            color=left_colors[i],
            linewidth=1.1,
            alpha=0.9,
        )
        fig.add_artist(con)
    for i, entry in enumerate(right_entries):
        con = ConnectionPatch(
            xyA=(float(entry["x_proj"]), float(entry["y_proj"])),
            coordsA=ax_map.transData,
            xyB=(0.0, 0.5),
            coordsB=ax_right[i].transAxes,
            color=right_colors[i],
            linewidth=1.1,
            alpha=0.9,
        )
        fig.add_artist(con)

    legend_handles = None
    legend_labels = None
    for ax in [*ax_left, *ax_right]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend_handles, legend_labels = handles, labels
            break
    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            fontsize=legend_fontsize,
            bbox_to_anchor=(0.5, 0.975),
        )

    fig.subplots_adjust(left=0.03, right=0.998, bottom=0.08, top=0.89)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a USGS-gauge LD-EnSF experiment with held-out validation gauges.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-path", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/cldnet"))
    parser.add_argument("--checkpoint-epoch", type=int, default=539)
    parser.add_argument("--traj-id", type=int, default=120)
    parser.add_argument("--background-traj-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--da-seed", type=int, default=0)
    parser.add_argument("--obs-noise-std", type=float, default=0.05)
    parser.add_argument("--train-max-traj-id", type=int, default=100)
    parser.add_argument("--valid-max-traj-id", type=int, default=117)
    parser.add_argument("--test-max-traj-id", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    base_path = opt.base_path.resolve()
    python_executable = _resolve_path(base_path, opt.python_executable)
    data_root = _resolve_path(base_path, opt.data_root)
    model_path = _resolve_path(base_path, opt.model_path)
    if not python_executable.exists():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")

    usgs_root = base_path / "data/usgs_2013_validation/data_usgs/usgs_plot/GT_sample_data_folder/USGS_2013-04-15_2013-04-23/h"
    eval_dataset_path = data_root / "observation_ldensf_dataset_usgs_validation.pth"
    dataset_path = data_root / "observation_ldensf_dataset_usgs_assim_26.pth"
    save_dir = base_path / "checkpoints/ldensf_lstm_usgs_assim_26"
    save_name = "lstm_ldensf_usgs_assim_26.ckpt"
    checkpoint_path = save_dir / save_name
    result_dir = base_path / "ldensf_results_usgs_assim_26"
    result_npz = result_dir / f"traj_{opt.traj_id}_ldensf_results.npz"

    all_gauges = list(range(32))
    assim_gauge_indices = [idx for idx in all_gauges if idx not in VAL_GAUGES]

    print(f"Base path: {base_path}")
    print(f"Python executable: {python_executable}")
    print(f"Assimilation gauges used: {len(assim_gauge_indices)}")
    print(f"Validation gauges used for scoring: {len(VAL_GAUGES)}")
    print(f"Dataset path: {dataset_path}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Results path: {result_npz}")

    dataset_cmd = [
        str(python_executable),
        str(Path(__file__).resolve().with_name("create_ldensf_dataset.py")),
        "--base-path",
        str(base_path),
        "--data-root",
        str(data_root),
        "--model-path",
        str(model_path),
        "--output-path",
        str(dataset_path),
        "--selection-mode",
        "usgs-gauges",
        "--usgs-gauge-indices",
        *[str(idx) for idx in assim_gauge_indices],
        "--num-observation-points",
        str(len(assim_gauge_indices)),
        "--checkpoint-epoch",
        str(opt.checkpoint_epoch),
    ]
    train_cmd = [
        str(python_executable),
        str(Path(__file__).resolve().with_name("train_ldensf_lstm.py")),
        "--base-path",
        str(base_path),
        "--dataset-path",
        str(dataset_path),
        "--save-dir",
        str(save_dir),
        "--save-name",
        save_name,
        "--obs-noise-std",
        str(opt.obs_noise_std),
    ]
    if opt.normalize:
        train_cmd.append("--normalize")
    else:
        train_cmd.append("--no-normalize")

    test_cmd = [
        str(python_executable),
        str(Path(__file__).resolve().with_name("test_ldensf_assimilation.py")),
        "--base-path",
        str(base_path),
        "--data-root",
        str(data_root),
        "--dataset-path",
        str(dataset_path),
        "--model-path",
        str(model_path),
        "--checkpoint-epoch",
        str(opt.checkpoint_epoch),
        "--encoder-checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(result_dir),
        "--traj-id",
        str(opt.traj_id),
        "--background-traj-id",
        str(opt.background_traj_id),
        "--seed",
        str(opt.seed),
        "--da-seed",
        str(opt.da_seed),
        "--obs-noise-std",
        str(opt.obs_noise_std),
        "--use-static-features",
    ]

    if opt.overwrite or not dataset_path.exists():
        _run_step("Create USGS assimilation dataset", dataset_cmd, cwd=base_path, dry_run=opt.dry_run)
    else:
        print(f"Dataset already exists, skipping: {dataset_path}")

    if opt.overwrite or not checkpoint_path.exists():
        _run_step("Train USGS assimilation encoder", train_cmd, cwd=base_path, dry_run=opt.dry_run)
    else:
        print(f"Encoder checkpoint already exists, skipping training: {checkpoint_path}")

    _run_step("Run LD-ENSF assimilation", test_cmd, cwd=base_path, dry_run=opt.dry_run)
    if opt.dry_run:
        return

    if not result_npz.exists():
        raise FileNotFoundError(f"Missing assimilation results: {result_npz}")
    if not eval_dataset_path.exists():
        raise FileNotFoundError(f"Missing evaluation dataset: {eval_dataset_path}")

    eval_dataset = torch.load(eval_dataset_path, map_location="cpu", weights_only=False)
    eval_obs_idx = np.asarray(eval_dataset["meta"]["selected_obs_idx"], dtype=np.int64)
    use_static_features = bool(eval_dataset["meta"].get("use_static_features", True))

    result = np.load(result_npz)
    forecast_latent = np.asarray(result["forecast_latent"], dtype=np.float32)
    assim_latent = np.asarray(result["assim_latent"], dtype=np.float32)
    t_len = int(forecast_latent.shape[0])
    times = pd.date_range(START_DT, periods=t_len, freq="1h")

    first_flow, first_coords, first_rain, first_static = _load_reduced_trajectory(
        data_root, opt.traj_id, use_static_features=use_static_features
    )
    eval_x_coords = np.asarray(first_coords[0, eval_obs_idx, :], dtype=np.float32)
    dim_u = int(first_rain.shape[-1])
    dim_y = int(first_flow.shape[-1])
    dim_x = int(first_coords.shape[-1])
    if use_static_features:
        if first_static is None:
            raise RuntimeError("Static features were requested but not loaded.")
        dim_x = int(first_coords.shape[-1])

    model = _load_ldnet_model(
        opt=argparse.Namespace(
            base_path=base_path,
            model_path=model_path,
            checkpoint_epoch=opt.checkpoint_epoch,
            dyn_checkpoint=None,
            rec_checkpoint=None,
            B_checkpoint=None,
            fourier_mapping_size=32,
            num_latent_states=200,
            NN_dyn_depth=8,
            NN_dyn_width=50,
            NN_rec_depth=10,
            NN_rec_width=300,
            activation="relu",
            kernel_initializer="Glorot normal",
            chunk_size=10000,
        ),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        dim_u=dim_u,
        dim_x=dim_x,
        dim_y=dim_y,
    )

    device = next(model.parameters()).device
    forecast_series = _decode_latent_series(model, forecast_latent, eval_x_coords, device=device)
    assim_series = _decode_latent_series(model, assim_latent, eval_x_coords, device=device)
    true_rain_tensor = torch.from_numpy(np.asarray(first_rain, dtype=np.float32)).to(device)
    dummy_x = torch.zeros((1, t_len, 1, dim_x), dtype=torch.float32, device=device)
    with torch.no_grad():
        base_latent = model(
            {"u": true_rain_tensor, "x": dummy_x, "dt": torch.tensor([1.0], device=device)},
            device,
            latent_state=True,
        )
    base_series = _decode_latent_series(model, base_latent.squeeze(0).detach().cpu().numpy(), eval_x_coords, device=device)

    rows: list[dict[str, object]] = []
    plot_entries: list[dict[str, object]] = []
    for gauge_pos, gauge_idx in enumerate(VAL_GAUGES):
        usgs_df = _load_usgs_height_series(usgs_root, gauge_idx)
        if usgs_df.empty:
            print(f"Skipping gauge {gauge_idx}: no USGS data after alignment window.")
            continue

        sim_forecast = pd.DataFrame({
            "datetime": times,
            "sim_wse": forecast_series[:, gauge_pos],
        })
        sim_base = pd.DataFrame({
            "datetime": times,
            "sim_wse": base_series[:, gauge_pos],
        })
        sim_assim = pd.DataFrame({
            "datetime": times,
            "sim_wse": assim_series[:, gauge_pos],
        })

        merged_forecast = pd.merge(sim_forecast, usgs_df[["datetime", "wse"]], on="datetime", how="inner").dropna()
        merged_base = pd.merge(sim_base, usgs_df[["datetime", "wse"]], on="datetime", how="inner").dropna()
        merged_assim = pd.merge(sim_assim, usgs_df[["datetime", "wse"]], on="datetime", how="inner").dropna()

        if merged_forecast.empty or merged_base.empty or merged_assim.empty:
            print(f"Skipping gauge {gauge_idx}: no aligned points.")
            continue

        forecast_shifted = merged_forecast.copy()
        forecast_shifted["sim_wse"] = _align_sim_to_obs_mean(
            forecast_shifted["sim_wse"].to_numpy(), forecast_shifted["wse"].to_numpy()
        )
        base_shifted = merged_base.copy()
        base_shifted["sim_wse"] = _align_sim_to_obs_mean(
            base_shifted["sim_wse"].to_numpy(), base_shifted["wse"].to_numpy()
        )
        assim_shifted = merged_assim.copy()
        assim_shifted["sim_wse"] = _align_sim_to_obs_mean(
            assim_shifted["sim_wse"].to_numpy(), assim_shifted["wse"].to_numpy()
        )

        plot_merged = pd.merge(
            forecast_shifted[["datetime", "wse", "sim_wse"]].rename(columns={"sim_wse": "forecast_wse"}),
            base_shifted[["datetime", "sim_wse"]].rename(columns={"sim_wse": "base_wse"}),
            on="datetime",
            how="inner",
        )
        plot_merged = pd.merge(
            plot_merged,
            assim_shifted[["datetime", "sim_wse"]].rename(columns={"sim_wse": "assim_wse"}),
            on="datetime",
            how="inner",
        )
        if not plot_merged.empty:
            plot_entries.append(
                {
                    "gage_idx": gauge_idx,
                    "site_no": _load_usgs_site_no(usgs_root, gauge_idx),
                    "datetime": plot_merged["datetime"].to_numpy(),
                    "obs": plot_merged["wse"].to_numpy(),
                    "base": plot_merged["base_wse"].to_numpy(),
                    "forecast": plot_merged["forecast_wse"].to_numpy(),
                    "assim": plot_merged["assim_wse"].to_numpy(),
                }
            )

        rows.append(
            {
                "gage_index": gauge_idx,
                "model": "forecast",
                "n_points": int(len(merged_forecast)),
                "nse": _nse(forecast_shifted["wse"], forecast_shifted["sim_wse"]),
                "kge": _kge(forecast_shifted["wse"], forecast_shifted["sim_wse"]),
                "peak_relative_error": _peak_relative_error(
                    forecast_shifted["wse"].to_numpy(), forecast_shifted["sim_wse"].to_numpy()
                ),
            }
        )
        rows.append(
            {
                "gage_index": gauge_idx,
                "model": "assimilation",
                "n_points": int(len(merged_assim)),
                "nse": _nse(assim_shifted["wse"], assim_shifted["sim_wse"]),
                "kge": _kge(assim_shifted["wse"], assim_shifted["sim_wse"]),
                "peak_relative_error": _peak_relative_error(
                    assim_shifted["wse"].to_numpy(), assim_shifted["sim_wse"].to_numpy()
                ),
            }
        )

    if not rows:
        raise RuntimeError("No gauge metrics were computed.")

    metrics_df = pd.DataFrame(rows)
    metrics_path = result_dir / f"traj_{opt.traj_id}_usgs_validation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    hydrograph_path = result_dir / f"traj_{opt.traj_id}_usgs_validation_hydrographs.png"
    _save_validation_hydrograph_figure(
        plot_entries=plot_entries,
        out_path=hydrograph_path,
        background_traj_id=opt.background_traj_id,
    )
    dem_path = base_path / "data/usgs_2013_validation/data_usgs/usgs_plot/GT_sample_data_folder/dem_5070.tif"
    gauge_mapping_path = base_path / "data/usgs_2013_validation/data_usgs/usgs_plot/GT_sample_data_folder/gage_index_mapping_with_lat_lon.csv"
    dem_hydrograph_path = result_dir / f"traj_{opt.traj_id}_usgs_validation_hydrographs_dem.png"
    _save_validation_dem_pointer_figure(
        plot_entries=plot_entries,
        out_path=dem_hydrograph_path,
        background_traj_id=opt.background_traj_id,
        dem_path=dem_path,
        gauge_mapping_path=gauge_mapping_path,
    )

    print("\nUSGS validation gauge metrics")
    print(f"Validation gauges used: {len(VAL_GAUGES)}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Hydrograph figure saved to: {hydrograph_path}")
    print(f"DEM hydrograph figure saved to: {dem_hydrograph_path}")
    for model_name in ["forecast", "assimilation"]:
        sub = metrics_df[metrics_df["model"] == model_name]
        print(
            f"{model_name.capitalize():<12} "
            f"NSE={sub['nse'].mean():.4f} | "
            f"KGE={sub['kge'].mean():.4f} | "
            f"Peak rel. err.={sub['peak_relative_error'].mean():.4f}"
        )
        for _, row in sub.sort_values("gage_index").iterrows():
            print(
                f"  gauge {int(row['gage_index']):2d}: "
                f"NSE={row['nse']:.4f}, KGE={row['kge']:.4f}, "
                f"peak rel. err.={row['peak_relative_error']:.4f}, "
                f"n={int(row['n_points'])}"
            )


if __name__ == "__main__":
    main(create_options())
