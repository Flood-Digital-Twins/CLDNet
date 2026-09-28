"""Paper Fig. 7b data: score the deposited Illinois LDNet and CLDNet (epoch 539) on all 94 storms, training vs held-out.

Usage: python release/score_train_vs_heldout.py   (GPU, about 75 min; resumes from the JSON if interrupted)

Each storm's data is loaded once and run through both models. Results are written after every storm.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ldnet"))
import ldnet_chicago_efficient_test as T  # noqa: E402
from efficient_fourier_ldnet import EfficientFourierLDNN  # noqa: E402

DATA = ROOT / "data/postprocessed/illinois"
OUT = Path(__file__).resolve().parent / "train_vs_heldout_epoch539.json"
split = json.load(open(ROOT / "splits/illinois_split.json"))


def build(model, static):
    argv = ["x", "--model-path", f"checkpoints/{model}/illinois", "--checkpoint-epoch", "539", "--all-vars",
            "--fourier-mapping-size", "32", "--num-latent-states", "200", "--device", "cuda:0",
            "--chunk-size", "200000"] + (["--use-static-features"] if static else [])
    saved, sys.argv = sys.argv, argv
    try:
        opt = T.create_options()
    finally:
        sys.argv = saved
    dim_x = 5 if static else 2
    net = EfficientFourierLDNN(32, [200 + 507] + [50] * 8 + [200], [200 + dim_x] + [300] * 10 + [3],
                               activation="relu", kernel_initializer="Glorot normal", chunk_size=200000)
    T._load_checkpoints(net, opt)
    return net.to("cuda:0").eval()


def rrmse(pred, true, sl):
    p, t = pred[..., sl].astype(np.float64), true[..., sl].astype(np.float64)
    return float(np.sqrt(np.sum((p - t) ** 2) / np.sum(t ** 2)))


models = {m: build(m, s) for m, s in (("cldnet", True), ("ldnet", False)) if (ROOT / "checkpoints" / m / "illinois").exists()}
ids = split["train"] + split["test"] + split["heldout_2013"]
res = json.load(open(OUT)) if OUT.exists() else {}
for k in ids:
    if str(k) in res:
        continue
    t0 = time.time()
    flow, coords, rain = T._load_reduced_data(DATA, k, True)
    y = np.asarray(flow).astype(np.float32)
    u = torch.from_numpy(np.asarray(rain).astype(np.float32)).cuda()
    entry = {"group": "train" if k in split["train"] else ("test" if k in split["test"] else "2013")}
    for name, static in [(m, m == "cldnet") for m in models]:
        c = np.asarray(coords)[..., : (5 if static else 2)]
        x = torch.from_numpy(np.repeat(c[:, None], y.shape[1], axis=1).astype(np.float32)).cuda()
        data = {"u": u, "x": x, "y": torch.from_numpy(y).cuda(),
                "dt": torch.tensor([T.dt], device="cuda:0", dtype=torch.float32)}
        with torch.no_grad():
            pred = models[name](data, "cuda:0", equilibrium=False, chunk_size=200000).cpu().numpy()
        entry[name] = {"aggregate": rrmse(pred, y, slice(0, 3)), "h": rrmse(pred, y, slice(0, 1))}
        del data, x, pred
        torch.cuda.empty_cache()
    res[str(k)] = entry
    OUT.write_text(json.dumps(res))
    print(k, entry["group"], {m: round(100 * entry[m]["aggregate"], 2) for m in models},
          f"{time.time() - t0:.0f}s", flush=True)
print("done", len(res))
