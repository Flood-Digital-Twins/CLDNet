#!/usr/bin/env python3
"""Run the released Des Plaines (Illinois) CLDNet, or the LDNet baseline, on any of the 94 storms, from the
repository inputs alone.

Predicts water depth h and unit discharges hu, hv for 96 hours on the 1,408,587 evaluation cells, using the storm's
Stage IV forcing (data/forcings) and the query grid (data/illinois_grid). No simulation data are needed.

    python scripts/predict_cldnet.py --storm 107                      # peak-depth map + summary
    python scripts/predict_cldnet.py --storm 2013-04-17_2013-04-21 --save-fields
    python scripts/predict_cldnet.py --storm 107 --model ldnet                        # unconditioned baseline
    python scripts/predict_cldnet.py --storm 107 --truth path/to/flow_variables_traj107.npy   # score a regenerated run

Outputs (in --out-dir): <model>_<storm>_peak_depth.png, <model>_<storm>.npz with the peak depth per cell and, with
--save-fields, the full (96, cells, 3) float16 prediction (~0.8 GB).

The released CLDNet is fed `static_features_as_trained`, the terrain features it was trained on, whose Manning
channel is mis-registered (see the README, "Correction: Manning channel"). Feeding it the corrected
`static_features` raises its error (held-out storms: 21.9 % -> 29.9 %).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ldnet"))
from efficient_fourier_ldnet import EfficientFourierLDNN  # noqa: E402


def build_model(name, device):
    cfg = json.loads((ROOT / "configs" / name / "illinois.json").read_text())
    a, ck = cfg["architecture"], cfg["checkpoint"]
    latent = a["num_latent_states"]
    model = EfficientFourierLDNN(
        a["fourier_mapping_size"],
        [latent + a["dim_u"]] + [a["NN_dyn_width"]] * a["NN_dyn_depth"] + [latent],
        [latent + a["dim_x"]] + [a["NN_rec_width"]] * a["NN_rec_depth"] + [a["dim_y"]],
        activation=a["activation"], kernel_initializer=a["kernel_initializer"])
    for part in ("dyn", "rec"):
        getattr(model, part).load_state_dict(torch.load(ROOT / ck[part], map_location="cpu"))
    model.B.load_state_dict(torch.load(ROOT / ck["fourier_B"], map_location="cpu"))
    return model.to(device).eval(), a["dim_x"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--storm", required=True, help="event id (e.g. 107) or 2013-04-17_2013-04-21")
    ap.add_argument("--model", choices=("cldnet", "ldnet"), default="cldnet")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk-size", type=int, default=200_000, help="query points decoded at once (lower if OOM)")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--save-fields", action="store_true", help="also save all 96 hourly fields (float16, ~0.8 GB)")
    ap.add_argument("--truth", help="model-ready flow_variables_traj<k>.npy from a regenerated simulation, to score")
    args = ap.parse_args()

    grid = np.load(ROOT / "data" / "illinois_grid" / "grid.npz")
    mask = grid["aggregate_mask"]
    rain = np.load(ROOT / "data" / "forcings" / f"event_{args.storm}" / "rain_source.npy")[:96]  # (96, 507) mm/h
    model, dim_x = build_model(args.model, args.device)
    # the released checkpoints were trained on the as-trained terrain features, not on the corrected `static_features`
    feats = np.concatenate([grid["coords"], grid["static_features_as_trained"]], axis=1)[:, :dim_x].astype(np.float32)

    x = torch.from_numpy(feats).to(args.device)
    data = {"u": torch.from_numpy(rain.astype(np.float32))[None].to(args.device),
            "x": x[None, None].expand(1, 96, *x.shape), "dt": torch.tensor([1.0], device=args.device)}
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(data, args.device, chunk_size=args.chunk_size)[0]          # (96, cells, 3)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    print(f"{args.model} storm {args.storm}: 96 h x {x.shape[0]:,} cells in {time.perf_counter() - t0:.1f} s")

    pred = pred.float().cpu().numpy()
    peak = pred[..., 0].max(axis=0)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{args.model}_{args.storm}"
    arrays = {"peak_depth": peak.astype(np.float32), "cell_index": np.flatnonzero(mask)}
    if args.save_fields:
        arrays["fields"] = pred.astype(np.float16)
    np.savez_compressed(f"{stem}.npz", **arrays)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = np.full(mask.size, np.nan, np.float32)
    img[arrays["cell_index"]] = np.where(peak >= 0.05, peak, np.nan)
    fig, ax = plt.subplots(figsize=(4, 9))
    im = ax.imshow(img.reshape(mask.shape), cmap="Blues", vmin=0, vmax=np.nanpercentile(img, 99.5),
                   interpolation="nearest")
    ax.set_axis_off(); ax.set_title(f"{args.model.upper()} peak depth, storm {args.storm}")
    fig.colorbar(im, ax=ax, shrink=0.6, label="water depth (m)")
    fig.savefig(f"{stem}_peak_depth.png", dpi=150, bbox_inches="tight")
    print(f"wrote {stem}.npz and {stem}_peak_depth.png")

    if args.truth:
        y = np.load(args.truth, mmap_mode="r")[0].astype(np.float32)             # (96, cells, 3)
        for name, sl in (("aggregate", slice(0, 3)), ("h", slice(0, 1))):
            e = pred[..., sl] - y[..., sl]
            print(f"  rRMSE {name}: {100 * np.sqrt((e ** 2).sum() / (y[..., sl] ** 2).sum()):.2f}%")


if __name__ == "__main__":
    main()
