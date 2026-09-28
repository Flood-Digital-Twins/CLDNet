"""Re-score the deposited Illinois LDNet/CLDNet epoch-539 checkpoints on the reduced (aggregate-mask) data.

Reuses the release's own loaders and model so the numbers are what a stranger would get.
Usage: python release/rescore_checkpoints.py [traj ids...]   (MODELS=cldnet,ldnet; default ids 107 108 109 120)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1] / "code" / "ldnet"
sys.path.insert(0, str(CODE))
import ldnet_chicago_efficient_test as T  # noqa: E402
from efficient_fourier_ldnet import EfficientFourierLDNN  # noqa: E402

OUT = Path(__file__).resolve().parent / os.environ.get("OUTNAME", "rescore_latest.json")
TRAJ = [int(a) for a in sys.argv[1:]] or [107, 108, 109, 120]


def build(model_name, static):
    argv = ["--model-path", f"checkpoints/{model_name}/illinois", "--checkpoint-epoch", "539", "--all-vars",
            "--fourier-mapping-size", "32",
            "--device", "cuda:0", "--chunk-size", "200000"] + (["--use-static-features"] if static else [])
    saved = sys.argv
    sys.argv = ["ldnet_chicago_efficient_test.py"] + argv
    try:
        return T.create_options()
    finally:
        sys.argv = saved


def metrics(pred, true):
    out = {}
    for name, sl in (("aggregate", slice(0, 3)), ("h", slice(0, 1)), ("hu", slice(1, 2)), ("hv", slice(2, 3))):
        p = pred[..., sl].astype(np.float64).ravel()
        t = true[..., sl].astype(np.float64).ravel()
        e = p - t
        out[name] = {
            "relative_rmse": float(np.sqrt(np.sum(e ** 2) / np.sum(t ** 2))),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
            "r2": float(1 - np.sum(e ** 2) / np.sum((t - t.mean()) ** 2)),
        }
    return out


def main():
    results = {}
    for model_name, static in [m for m in (("cldnet", True), ("ldnet", False)) if m[0] in os.environ.get("MODELS", "cldnet,ldnet") and (CODE.parents[1] / "checkpoints" / m[0] / "illinois").exists()]:
        opt = build(model_name, static)
        data_root = opt.base_path / opt.data_root
        model = None
        for k in TRAJ:
            flow, coords, rain = T._load_reduced_data(data_root, k, opt.use_static_features)
            t_len = flow.shape[1]
            x = np.repeat(coords[:, None, :, :], t_len, axis=1).astype(np.float32)
            y = np.asarray(flow).astype(np.float32)
            u = np.asarray(rain).astype(np.float32)
            if model is None:
                dim_u, dim_x, dim_y = u.shape[-1], x.shape[-1], y.shape[-1]
                dyn = [opt.num_latent_states + dim_u] + opt.NN_dyn_depth * [opt.NN_dyn_width] + [opt.num_latent_states]
                rec = [opt.num_latent_states + dim_x] + opt.NN_rec_depth * [opt.NN_rec_width] + [dim_y]
                model = EfficientFourierLDNN(opt.fourier_mapping_size, dyn, rec, activation=opt.activation,
                                             kernel_initializer=opt.kernel_initializer, chunk_size=opt.chunk_size)
                T._load_checkpoints(model, opt)
                model.to(opt.device).eval()
                nparam = sum(p.numel() for p in model.parameters())
                print(f"{model_name}: {nparam} parameters, dim_x={dim_x}", flush=True)
            data = {"u": torch.from_numpy(u).to(opt.device), "x": torch.from_numpy(x).to(opt.device),
                    "y": torch.from_numpy(y).to(opt.device),
                    "dt": torch.tensor([T.dt], device=opt.device, dtype=torch.float32)}
            with torch.no_grad():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred = model(data, opt.device, equilibrium=False, chunk_size=opt.chunk_size)
                torch.cuda.synchronize()
                sec = time.perf_counter() - t0
            m = metrics(pred.cpu().numpy(), y)
            m["inference_s"] = sec
            m["parameters"] = nparam
            results[f"{model_name}_{k}"] = m
            print(model_name, k, {v: round(m[v]["relative_rmse"] * 100, 2) for v in ("aggregate", "h", "hu", "hv")},
                  f"{sec:.1f}s", flush=True)
            del data, pred
            torch.cuda.empty_cache()
    OUT.write_text(json.dumps(results, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
