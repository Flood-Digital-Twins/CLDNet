#!/usr/bin/env python3
"""Paper Fig. 4: event-severity histograms of the Des Plaines Stage IV storms.

For each storm, the basin-mean rate is the mean of `rain_source.npy` (97 hourly rows x 507 cells,
mm/h) over its 507 cells. Plotted per storm:
  total depth (mm)             = sum over hours of the basin-mean rate
  peak hourly intensity (mm/h) = max over hours of the basin-mean rate
  event duration (h)           = number of hours with basin-mean rate > 0

    python preprocessing/stage4_precip/plot_event_severity.py --out storm_event_rainfall_histograms.png

`--with-catalog-duplicates` adds back the 20 byte-identical catalog copies (85-89, 95-99, 110-119),
reproducing the 114-event figure of the original submission bar for bar.
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DUPLICATES = [25 + i for i in range(5)] + [35 + i for i in range(5)] + [50 + i for i in range(10)]


def storm_statistics(forcing_root, with_duplicates=False):
    rates = {}
    for folder in sorted(glob.glob(os.path.join(forcing_root, "event_*"))):
        rates[os.path.basename(folder)] = np.load(os.path.join(folder, "rain_source.npy")).mean(axis=1)
    names = list(rates) + ([f"event_{k}" for k in DUPLICATES] if with_duplicates else [])
    total = np.array([rates[n].sum() for n in names])
    peak = np.array([rates[n].max() for n in names])
    duration = np.array([(rates[n] > 0).sum() for n in names], dtype=float)
    return total, peak, duration


def plot(total, peak, duration, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = (
        (total, "Total depth (mm)", "#5e85b0"),
        (peak, "Peak hourly intensity (mm/h)", "#f6912f"),
        (duration, "Event duration (h)", "#65ab5d"),
    )
    for ax, (values, label, color) in zip(axes, panels):
        ax.hist(values, bins=16, color=color, edgecolor="white", linewidth=1.0)
        ax.set_xlabel(label, fontsize=14)
        ax.set_ylabel("Number of events", fontsize=14)
        ax.tick_params(labelsize=13)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.yaxis.get_major_locator().set_params(integer=True)
        summary = (f"min: {values.min():.1f}\nmedian: {np.median(values):.1f}\n"
                   f"mean: {values.mean():.1f}\nmax: {values.max():.1f}")
        ax.text(0.97, 0.97, summary, transform=ax.transAxes, ha="right", va="top", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.3"))
    fig.suptitle("Event severity statistics", fontsize=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"{len(total)} storms -> {out_path}")
    for name, v in (("total", total), ("peak", peak), ("duration", duration)):
        print(f"  {name:8s} min {v.min():.1f} median {np.median(v):.1f} mean {v.mean():.1f} max {v.max():.1f}"
              f"  counts {np.histogram(v, bins=16)[0].tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forcing-root", default=os.path.join(ROOT, "data", "forcings"))
    ap.add_argument("--out", default="storm_event_rainfall_histograms.png")
    ap.add_argument("--with-catalog-duplicates", action="store_true")
    args = ap.parse_args()
    plot(*storm_statistics(args.forcing_root, args.with_catalog_duplicates), args.out)
