"""Paper Fig. 7b: per-storm aggregate rRMSE at the final epoch, training vs held-out storms (usage: python release/plot_fig7b.py OUT.png)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
import sys
r = json.load(open(HERE / "train_vs_heldout_epoch539.json"))
colors = {"cldnet": "tab:blue", "ldnet": "tab:orange"}
labels = {"cldnet": "CLDNet", "ldnet": "LDNet"}
groups = [("train", "Training storms\n($n=90$)"), ("test", "Held-out test\n($n=3$)"), ("2013", "2013 event\n(held out)")]

fig, ax = plt.subplots(figsize=(7.2, 4.0))
rng = np.random.default_rng(0)
for gi, (g, _) in enumerate(groups):
    for mi, m in enumerate(("cldnet", "ldnet")):
        vals = np.array([100 * e[m]["aggregate"] for e in r.values() if e["group"] == g])
        x0 = gi + (-0.17 if m == "cldnet" else 0.17)
        if g == "train":
            ax.boxplot(vals, positions=[x0], widths=0.26, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="none", edgecolor=colors[m], linewidth=1.5),
                       medianprops=dict(color=colors[m], linewidth=2), whiskerprops=dict(color=colors[m]),
                       capprops=dict(color=colors[m]))
            ax.scatter(x0 + rng.uniform(-0.09, 0.09, vals.size), vals, s=10, color=colors[m], alpha=0.45, zorder=3)
        else:
            ax.scatter(np.full(vals.size, x0), vals, s=45 if g == "test" else 170, color=colors[m], edgecolor="black", linewidth=0.6,
                       marker="o" if g == "test" else "*", zorder=4)
        if gi == 0:
            ax.scatter([], [], s=30, color=colors[m], label=labels[m])
ax.set_xticks(range(len(groups)), [lbl for _, lbl in groups], fontsize=12)
ax.set_ylabel("Aggregate rRMSE (%)", fontsize=13)
ax.set_title("Per-storm error at the final epoch", fontsize=15)
ax.set_ylim(0, None)
ax.grid(alpha=0.3)
ax.tick_params(axis="y", labelsize=12)
ax.legend(frameon=False, fontsize=12, loc="upper right")
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "train_vs_heldout_chicago.png"
fig.savefig(out, dpi=200)
print("saved", out)
