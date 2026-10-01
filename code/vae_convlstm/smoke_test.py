"""Run a short Texas VAE–ConvLSTM rollout and compare it with saved states.

Example: python code/vae_convlstm/smoke_test.py --sample-id 101 --start 64
"""

import argparse

import numpy as np
import torch

from data_layout import sample_ids_in_rain_order
from generate_latents import load_model
from ldm_ae.convlstm import ConvLSTM
from paths import TEXAS_DATA_ROOT, VAE_CHECKPOINT_DIR


def rmse(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((prediction.astype(np.float64) - truth.astype(np.float64)) ** 2,
                           axis=(0, 2, 3)))


def relative_rmse(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    reference_rms = np.sqrt(np.mean(truth.astype(np.float64) ** 2, axis=(0, 2, 3)))
    return rmse(prediction, truth) / reference_rms


def r2(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    truth = truth.astype(np.float64)
    prediction = prediction.astype(np.float64)
    residual = np.sum((prediction - truth) ** 2, axis=(0, 2, 3))
    variation = np.sum((truth - np.mean(truth, axis=(0, 2, 3), keepdims=True)) ** 2,
                       axis=(0, 2, 3))
    return 1 - residual / variation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", type=int, default=101)
    parser.add_argument("--start", type=int, default=64,
                        help="First normalized state frame used as the rollout initial condition")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.sample_id not in range(1, 121) or args.steps < 1 or not 0 <= args.start < 192 - args.steps:
        parser.error("sample ID, start, or steps is outside the saved arrays")

    split = "train" if args.sample_id <= 100 else "test"
    source_ids = list(range(1, 101)) if split == "train" else list(range(101, 121))
    data_dir = TEXAS_DATA_ROOT / "vae_and_latents_texas"
    row_order = sample_ids_in_rain_order(
        split, data_dir, source_ids, VAE_CHECKPOINT_DIR / "mean_std_rain_source.npz",
    )
    row = row_order.index(args.sample_id)
    latents = np.load(data_dir / f"latent_{split}.npy", mmap_mode="r")
    rain = np.load(data_dir / f"rain_{split}.npy", mmap_mode="r")
    states = np.load(data_dir / f"state_{split}.npy", mmap_mode="r")
    if latents.shape != (len(source_ids), 192, 16, 25, 25) or rain.shape != (len(source_ids), 192, 1):
        raise ValueError("Unexpected latent or rain shape")
    if states.shape != (len(source_ids), 192, 3, 500, 500):
        raise ValueError("Unexpected state shape")

    device = torch.device(args.device)
    vae = load_model(VAE_CHECKPOINT_DIR / "checkpoint_vae.pth", device)
    conv = ConvLSTM(input_dim=9, hidden_dim=[16, 16, 8], kernel_size=(5, 5),
                    num_layers=3, batch_first=True, bias=True, return_all_layers=True)
    checkpoint = torch.load(VAE_CHECKPOINT_DIR / "checkpoint_convlstm.pth",
                            map_location="cpu", weights_only=False)
    conv.load_state_dict(checkpoint["model_state_dict"], strict=True)
    conv = conv.to(device).eval()
    print(f"Loaded ConvLSTM epoch {checkpoint['epoch']}; {split} sample {args.sample_id}, row {row}")

    z = torch.from_numpy(np.asarray(latents[row, :, :8]).copy()).to(device)[None]
    r = torch.from_numpy(np.asarray(rain[row]).copy()).to(device)[None]
    r_grid = r[..., None, None].expand(-1, -1, -1, 25, 25)
    with torch.inference_mode():
        deltas, _ = conv(torch.cat((z[:, :-1], r_grid[:, :-1]), dim=2))
        one_step = z[:, :-1] + deltas[-1]
        one_step_mse = torch.mean((one_step - z[:, 1:]) ** 2).item()
        persistence_mse = torch.mean((z[:, :-1] - z[:, 1:]) ** 2).item()

        current = z[:, args.start:args.start + 1]
        hidden = None
        predictions = []
        for t in range(args.start, args.start + args.steps):
            outputs, hidden = conv(torch.cat((current, r_grid[:, t:t + 1]), dim=2), hidden)
            current = current + outputs[-1]
            predictions.append(current[:, 0])
        predicted_z = torch.stack(predictions, dim=1)
        target_z = z[:, args.start + 1:args.start + args.steps + 1]
        rollout_mse = torch.mean((predicted_z - target_z) ** 2).item()
        fixed_initial_mse = torch.mean((z[:, args.start:args.start + 1] - target_z) ** 2).item()

        picks = list(range(args.steps))
        predicted_frames = []
        encoded_truth_frames = []
        for index in picks:
            predicted_frames.append(vae.decode(predicted_z[:, index]).cpu().numpy()[0])
            encoded_truth_frames.append(vae.decode(target_z[:, index]).cpu().numpy()[0])

    with np.load(VAE_CHECKPOINT_DIR / "mean_std.npz") as stats:
        mean = stats["mean"].astype(np.float32).reshape(1, 3, 1, 1)
        std = stats["std"].astype(np.float32).reshape(1, 3, 1, 1)
    truth = np.stack([states[row, args.start + 1 + index] for index in picks])
    predicted = np.stack(predicted_frames)
    encoded_truth = np.stack(encoded_truth_frames)
    physical_truth = truth * std + mean
    physical_prediction = predicted * std + mean
    physical_reconstruction = encoded_truth * std + mean
    initial_physical = states[row, args.start][None] * std + mean
    channel_names = ("h", "hu", "hv")
    print(f"Decoded rollout frames {args.start + 1}–{args.start + args.steps}")
    print(f"Latent one-step MSE: {one_step_mse:.6g} (persistence {persistence_mse:.6g})")
    print(f"Latent {args.steps}-step rollout MSE: {rollout_mse:.6g} "
          f"(fixed-initial-state {fixed_initial_mse:.6g})")
    for label, values in (
        ("decoded rollout RMSE", rmse(physical_prediction, physical_truth)),
        ("VAE reconstruction RMSE", rmse(physical_reconstruction, physical_truth)),
        ("fixed-state RMSE", rmse(np.broadcast_to(initial_physical, physical_truth.shape),
                                  physical_truth)),
    ):
        print(label + ": " + ", ".join(f"{name}={value:.6g}" for name, value in zip(channel_names, values)))
    print("decoded rollout relative RMSE: " + ", ".join(
        f"{name}={100 * value:.2f}%" for name, value in
        zip(channel_names, relative_rmse(physical_prediction, physical_truth))))
    print("decoded rollout R2: " + ", ".join(
        f"{name}={value:.4f}" for name, value in
        zip(channel_names, r2(physical_prediction, physical_truth))))
    if not np.isfinite([one_step_mse, persistence_mse, rollout_mse, fixed_initial_mse]).all():
        raise ValueError("Non-finite latent loss")
    if not np.isfinite(physical_prediction).all() or not np.isfinite(physical_reconstruction).all():
        raise ValueError("Non-finite decoded prediction")


if __name__ == "__main__":
    main()
