"""Score the deposited Texas VAE-ConvLSTM on the held-out storms, from the raw simulations.

For each storm the first saved flow frame is encoded with the VAE, the residual ConvLSTM rolls the 25 x 25 x 8 latent
forward for the remaining 191 quarter-hour steps driven by the rainfall, every latent is decoded, and the 192 decoded
frames are compared with the simulation in physical units. As in the paper's Texas tables, rRMSE and RMSE are means
over storms of the per-storm values, R2 is pooled over all storms, and "aggregate" is the mean over h, hu, hv.
Paper (storms 101-120): aggregate 23.41 % / 0.08373 / 0.92113, depth 18.74 % / 0.06916 / 0.95310.

Writes per-storm, per-frame sufficient statistics (se = sum sq error, tt = sum truth^2, t = sum truth) so other
poolings can be computed afterwards.
Usage: python scripts/score_vae_convlstm.py [--samples 101 120]   (GPU; under a minute)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "vae_convlstm"))
from generate_latents import load_flow_stats, load_model  # noqa: E402
from ldm_ae.convlstm import ConvLSTM  # noqa: E402

CKPT = ROOT / "checkpoints" / "vae_convlstm"
OUT = Path(__file__).resolve().parent / "vae_convlstm_texas_latest.json"
NAMES = ("h", "hu", "hv")
CELLS = 500 * 500


def summarize(stats):
    """Paper convention: rRMSE and RMSE are means over storms of the per-storm values; R2 is pooled over all storms."""
    se = np.array([np.asarray(s["se"]).sum(0) for s in stats.values()])   # (storms, 3)
    tt = np.array([np.asarray(s["tt"]).sum(0) for s in stats.values()])
    t = np.array([np.asarray(s["t"]).sum(0) for s in stats.values()])
    n = CELLS * len(next(iter(stats.values()))["se"])                     # values per storm and variable
    rel, rmse = np.sqrt(se / tt).mean(0), np.sqrt(se / n).mean(0)
    r2 = 1 - se.sum(0) / (tt.sum(0) - t.sum(0) ** 2 / (n * len(stats)))
    out = {v: {"rRMSE_pct": round(100 * rel[i], 2), "RMSE": round(float(rmse[i]), 5), "R2": round(float(r2[i]), 5)}
           for i, v in enumerate(NAMES)}
    out["aggregate"] = {"rRMSE_pct": round(100 * rel.mean(), 2), "RMSE": round(float(rmse.mean()), 5),
                        "R2": round(float(r2.mean()), 5)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, nargs=2, default=(101, 120), metavar=("FIRST", "LAST"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=32, help="frames decoded at once")
    args = ap.parse_args()
    dev = torch.device(args.device)

    vae = load_model(CKPT / "checkpoint_vae.pth", dev)
    conv = ConvLSTM(input_dim=9, hidden_dim=[16, 16, 8], kernel_size=(5, 5), num_layers=3, batch_first=True,
                    bias=True, return_all_layers=True)
    conv.load_state_dict(torch.load(CKPT / "checkpoint_convlstm.pth", map_location="cpu",
                                    weights_only=False)["model_state_dict"])
    conv = conv.to(dev).eval()
    mean, std = (torch.from_numpy(a).to(dev) for a in load_flow_stats(CKPT / "mean_std.npz"))
    with np.load(CKPT / "mean_std_rain_source.npz") as s:
        rain_mean, rain_std = float(s["mean"].item()), float(s["std"].item())

    stats, seconds = {}, []
    for k in range(args.samples[0], args.samples[1] + 1):
        d = ROOT / "data" / "texas" / ("train_dataset" if k <= 100 else "test_dataset") / f"sample_{k:05d}"
        flow = np.load(d / "flow_variables.npy", mmap_mode="r")                     # (193, 3, 500, 500)
        rain = np.repeat(np.load(d / "rain_source.npy")[:-1, 1], 4).astype(np.float32)   # (192,) m/s
        r = torch.from_numpy((rain - rain_mean) / rain_std).to(dev)[None, :, None, None, None].expand(1, 192, 1, 25, 25)
        first = (torch.from_numpy(np.array(flow[1:2], dtype=np.float32)).to(dev) - mean) / std
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            z = [vae.encode(first)[:, :8][:, None]]                                 # posterior mean, (1, 1, 8, 25, 25)
            hidden = None
            for j in range(191):
                out, hidden = conv(torch.cat((z[-1], r[:, j:j + 1]), dim=2), hidden)
                z.append(z[-1] + out[-1])
            z = torch.cat(z, dim=1)[0]                                              # (192, 8, 25, 25)
            pred = torch.cat([vae.decode(z[i:i + args.batch_size]) for i in range(0, 192, args.batch_size)])
            pred = pred[:, :3] * std + mean                                         # physical units
        if dev.type == "cuda":
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - t0)
        truth = torch.from_numpy(np.array(flow[1:193], dtype=np.float32)).to(dev).double()
        e = pred.double() - truth
        stats[str(k)] = {"se": (e * e).sum((2, 3)).cpu().tolist(), "tt": (truth * truth).sum((2, 3)).cpu().tolist(),
                         "t": truth.sum((2, 3)).cpu().tolist()}
        one = summarize({str(k): stats[str(k)]})
        print(f"storm {k}: aggregate {one['aggregate']['rRMSE_pct']:.2f}%  h {one['h']['rRMSE_pct']:.2f}%  "
              f"({seconds[-1]:.1f} s)", flush=True)

    summary = summarize(stats)
    print(json.dumps(summary, indent=1))
    print(f"inference: {np.mean(seconds[1:] or seconds):.2f} s per storm on {torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu'}")
    OUT.write_text(json.dumps({"_summary": summary, "_seconds_per_storm": seconds, **stats}))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
