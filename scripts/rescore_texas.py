"""Score the released Texas CLDNet (epoch 489) on every model-ready Texas storm.

Writes per-storm sufficient statistics (se, tt, t, n) so metrics can be pooled or averaged afterwards.
Usage: python release/rescore_texas.py   (GPU; about 10 min)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ldnet"))
import ldnet_chicago_efficient_test as T  # noqa: E402
from efficient_fourier_ldnet import EfficientFourierLDNN  # noqa: E402

DATA = ROOT / "data/postprocessed/texas"
OUT = Path(__file__).resolve().parent / "rescore_texas_latest.json"


def opts(model, static):
    argv = ["x", "--model-path", f"checkpoints/{model}/texas", "--checkpoint-epoch", "489", "--all-vars",
            "--data-root", "data/postprocessed/texas", "--normalize-rain", "--fourier-mapping-size", "10",
            "--num-latent-states", "30", "--device", "cuda:0", "--chunk-size", "300000"]
    if static:
        argv.append("--use-static-features")
    saved, sys.argv = sys.argv, argv
    try:
        return T.create_options()
    finally:
        sys.argv = saved


def sums(pred, true):
    """Sufficient statistics so metrics can be pooled over storms afterwards."""
    out = {}
    for name, sl in (("aggregate", slice(0, 3)), ("h", slice(0, 1)), ("hu", slice(1, 2)), ("hv", slice(2, 3))):
        p = pred[..., sl].astype(np.float64).ravel()
        t = true[..., sl].astype(np.float64).ravel()
        e = p - t
        out[name] = {"se": float(e @ e), "tt": float(t @ t), "t": float(t.sum()), "n": int(t.size)}
    return out


rain_mean = np.load(DATA / "rain_mean.npy").astype(np.float32)
rain_std = np.load(DATA / "rain_std.npy").astype(np.float32)
ids = sorted(int(p.stem.split("traj")[1]) for p in DATA.glob("flow_variables_traj*.npy"))
res = {}
for model, static in [m for m in (("cldnet", True), ("ldnet", False)) if (ROOT / "checkpoints" / m[0] / "texas").exists()]:
    opt = opts(model, static)
    model_obj = None
    for k in ids:
        flow, coords, rain = T._load_reduced_data(DATA, k, static)
        x = np.repeat(coords[:, None], flow.shape[1], axis=1).astype(np.float32)
        y = np.asarray(flow).astype(np.float32)
        u = (np.asarray(rain).astype(np.float32) - rain_mean) / rain_std
        if model_obj is None:
            dyn = [30 + u.shape[-1]] + [50] * 8 + [30]
            rec = [30 + x.shape[-1]] + [300] * 10 + [3]
            model_obj = EfficientFourierLDNN(10, dyn, rec, activation="relu", kernel_initializer="Glorot normal",
                                             chunk_size=opt.chunk_size)
            T._load_checkpoints(model_obj, opt)
            model_obj.to("cuda:0").eval()
            print(model, sum(p.numel() for p in model_obj.parameters()), "parameters", flush=True)
        data = {"u": torch.from_numpy(u).cuda(), "x": torch.from_numpy(x).cuda(), "y": torch.from_numpy(y).cuda(),
                "dt": torch.tensor([T.dt], device="cuda:0", dtype=torch.float32)}
        with torch.no_grad():
            pred = model_obj(data, "cuda:0", equilibrium=False, chunk_size=opt.chunk_size).cpu().numpy()
        res[f"{model}_{k}"] = sums(pred, y)
    OUT.write_text(json.dumps(res))
print("wrote", OUT)
