#!/usr/bin/env python3
"""
Draw a schematic architecture diagram for EfficientFourierLDNN
in a style similar to the provided reference figure.

Outputs: architecture.png (by default)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, Rectangle, Circle


def _box(ax, xy, text, width=3.0, height=0.9, fc="#F2F2F2", ec="#2E2E2E", fontsize=12, weight="normal"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
    )
    return box


def _arrow(ax, start, end, color="#2E2E2E", lw=1.5):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle=ArrowStyle("->", head_length=6, head_width=3), color=color, lw=lw),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("architecture.png"))
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Palette
    c_dyn = "#3B73C1"
    c_rec = "#7A4CC2"
    c_box = "#FFFFFF"
    c_outline = "#2E2E2E"
    c_trace = "#2C6FB3"
    c_yellow = "#F1B500"

    # Left: time series sketch
    ax.plot([0.6, 2.6], [4.7, 4.7], color=c_outline, lw=1.5)
    ax.plot([0.6, 0.6], [4.1, 5.5], color=c_outline, lw=1.5)
    ax.text(0.45, 5.55, "u", fontsize=14)
    # sine-ish curves
    xs = [0.75, 1.1, 1.45, 1.8, 2.15, 2.5]
    ax.plot(xs, [4.55, 4.8, 4.6, 4.9, 4.65, 4.85], color=c_trace, lw=2.0)
    ax.plot(xs, [4.35, 4.6, 4.4, 4.7, 4.45, 4.65], color=c_trace, lw=2.0, alpha=0.8)
    ax.text(1.45, 3.95, "Time", fontsize=13, fontweight="bold")
    ax.add_patch(Circle((2.2, 4.65), 0.06, color=c_yellow, zorder=5))

    # Box: [u_{t-1}, s_{t-1}]
    _box(ax, (3.0, 4.1), "u$_{t-1}$\n s$_{t-1}$", width=1.4, height=1.2, fc=c_box, ec=c_outline, fontsize=12)
    ax.text(3.0, 5.5, "s$_{-1}$ = 0", fontsize=12)
    _arrow(ax, (2.35, 4.65), (3.0, 4.65), color=c_yellow, lw=2.0)

    # Dyn net
    _box(ax, (5.0, 4.0), "Dyn net", width=2.6, height=1.4, fc=c_dyn, ec=c_outline, fontsize=13, weight="bold")
    _arrow(ax, (4.4, 4.65), (5.0, 4.65))

    # sdot box
    _box(ax, (8.2, 4.3), r"$\dot{s}_{t-1}$", width=1.1, height=0.9, fc=c_box, ec=c_outline, fontsize=12)
    _arrow(ax, (7.6, 4.65), (8.2, 4.65))

    # update arrow and equation
    _arrow(ax, (8.75, 4.1), (8.75, 3.2))
    ax.text(5.2, 3.15, r"$s_t = s_{t-1} + \dot{s}_{t-1}\Delta t$", fontsize=12)

    # Lower: spatial domain and rec
    # domain axes
    ax.plot([0.6, 2.6], [1.4, 1.4], color=c_outline, lw=1.5)
    ax.plot([0.6, 0.6], [0.6, 2.2], color=c_outline, lw=1.5)
    ax.text(0.45, 2.2, r"$\xi_2$", fontsize=12)
    ax.text(2.45, 1.1, r"$\xi_1$", fontsize=12)
    # dashed domain
    ax.add_patch(Rectangle((1.1, 0.9), 1.0, 1.0, fill=False, ls="--", lw=1.2, ec="#9B9B9B"))
    ax.text(1.2, 1.95, r"$\Omega$", fontsize=12)
    ax.add_patch(Circle((1.6, 1.4), 0.06, color=c_yellow, zorder=5))

    # Box: [s_t, xi]
    _box(ax, (3.0, 1.0), "s$_t$\n$\\xi$", width=1.4, height=1.2, fc=c_box, ec=c_outline, fontsize=12)
    _arrow(ax, (2.1, 1.4), (3.0, 1.4), color=c_yellow, lw=2.0)

    # Rec net
    _box(ax, (5.0, 0.8), "Rec net", width=2.6, height=1.4, fc=c_rec, ec=c_outline, fontsize=13, weight="bold")
    _arrow(ax, (4.4, 1.4), (5.0, 1.4))

    # Output box
    _box(ax, (8.2, 1.05), r"$\tilde{x}(t,\xi)$", width=1.6, height=0.9, fc=c_box, ec=c_outline, fontsize=12)
    _arrow(ax, (7.6, 1.4), (8.2, 1.4))

    # Right: output field sketch
    ax.plot([10.2, 11.6], [1.0, 1.0], color=c_outline, lw=1.2)
    ax.plot([10.2, 10.2], [0.4, 2.4], color=c_outline, lw=1.2)
    ax.text(10.1, 2.45, r"$\xi_2$", fontsize=12)
    ax.text(11.5, 0.7, r"$\xi_1$", fontsize=12)
    # stacked heatmaps (rectangles)
    for i in range(4):
        ax.add_patch(Rectangle((10.6 + 0.1 * i, 0.8 + 0.1 * i), 0.9, 0.9, fc="#DCEAF7", ec="#9DB6D8"))
    ax.add_patch(Circle((11.0, 1.5), 0.06, color=c_yellow, zorder=5))
    _arrow(ax, (9.8, 1.2), (10.6, 1.25), color=c_yellow, lw=2.0)

    fig.tight_layout()
    out_path = args.output
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    fig.savefig(out_path, dpi=200)
    print(f"Saved architecture diagram to {out_path}")


if __name__ == "__main__":
    main()
