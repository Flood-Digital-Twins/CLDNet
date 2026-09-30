"""Train the autoregressive FNO baseline on the fixed-DEM Texas dataset.

The model maps (DEM, slope, rainfall, flow state) at hour t to the flow state at
hour t+1, and is unrolled `--prediction-time-horizon` steps per optimizer update.

All paths default to locations inside this repository; nothing here depends on an
absolute path. See README.md for the exact command that produced the deposited
checkpoint.
"""

import argparse
import io
import json
from pathlib import Path

import imageio
import matplotlib
import numpy as np
import torch
from neuralop.models import FNO
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fno_utils import (
    BatchIndicesIterator,
    format_readable_memory_size,
    limited_gradient,
    load_latest_checkpoint,
    plot_prediction_truth_error,
    print_model_size,
    set_seed,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIRECTORY.parents[1]

DEFAULT_TRAIN_DATASET_DIRECTORY = REPO_ROOT / "data" / "texas" / "train_dataset"
DEFAULT_RESULTS_DIRECTORY = SCRIPT_DIRECTORY / "outputs" / "fno_autoregressive_texas_fixed_DEM_results"

ORIGINAL_TIME_STEP_HOURS = 0.25
MODEL_TIME_STEP_HOURS = 1.0
TEMPORAL_STRIDE = 4
EXPECTED_TIME_STATES_PER_TRAJECTORY = 49
RAIN_SOURCE_CHANNEL_INDEX = 1


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train autoregressive FNO on fixed Texas DEM data.")
    parser.add_argument(
        "--dataset-directory",
        default=str(DEFAULT_TRAIN_DATASET_DIRECTORY),
        help="Directory containing sample_XXXXX subdirectories.",
    )
    parser.add_argument("--sample-start-index", type=int, default=1)
    parser.add_argument("--sample-end-index", type=int, default=100)
    parser.add_argument("--burn-in-length", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--prediction-time-horizon", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-9)
    parser.add_argument("--cell-size", type=float, default=30.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--fourier-modes",
        type=int,
        nargs=2,
        default=[8, 8],
        metavar=("MODE_Y", "MODE_X"),
        help="Number of retained Fourier modes along each spatial dimension.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=32,
        help="Hidden channel width used by the FNO model.",
    )
    parser.add_argument(
        "--results-directory",
        default=str(DEFAULT_RESULTS_DIRECTORY),
        help="Base directory under which a config-specific run directory will be created.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_json_compatible(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {key: to_json_compatible(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    return value


def save_run_config(output_path: Path, config: dict):
    output_path.write_text(json.dumps(to_json_compatible(config), indent=2, sort_keys=True) + "\n")


def sanitize_config_token(value) -> str:
    return str(value).replace(":", "").replace("/", "_").replace(" ", "_")


def build_results_directory_name(args) -> str:
    fourier_modes_str = "x".join(str(mode) for mode in args.fourier_modes)
    return "_".join(
        [
            "device_" + sanitize_config_token(args.device),
            "modes_" + fourier_modes_str,
            "hidden_" + str(args.hidden_channels),
            "horizon_" + str(args.prediction_time_horizon),
            "batch_" + str(args.batch_size),
        ]
    )


def print_channel_statistics(name: str, tensor: torch.Tensor, channel_names, channel_axis: int):
    tensor = tensor.detach().cpu()
    print(f"{name} channel statistics:")
    for channel_idx, channel_name in enumerate(channel_names):
        channel_tensor = tensor.select(channel_axis, channel_idx)
        print(
            f"  {channel_name}: "
            f"mean={channel_tensor.mean().item():.6f}, "
            f"std={channel_tensor.std(unbiased=False).item():.6f}, "
            f"min={channel_tensor.min().item():.6f}, "
            f"max={channel_tensor.max().item():.6f}"
        )


def postprocess_prediction(prediction: torch.Tensor) -> torch.Tensor:
    water_depth = torch.relu(prediction[:, [0]])
    return torch.cat((water_depth, prediction[:, 1:]), dim=1)


def extract_batch_to_device(
    tensor: torch.Tensor,
    batch_indices,
    device: torch.device,
    *index_suffix,
) -> torch.Tensor:
    return tensor[batch_indices, *index_suffix].to(device)


def expand_static_tensor_batch_to_device(
    tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    return tensor.expand(batch_size, -1, -1, -1).to(device)


def downsample_to_hourly_states(
    flow_variables: np.ndarray,
    rain_source: np.ndarray,
    burn_in_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if burn_in_length < 0:
        raise ValueError(f"burn_in_length must be nonnegative, got {burn_in_length}")
    if burn_in_length % TEMPORAL_STRIDE != 0:
        raise ValueError(
            f"burn_in_length={burn_in_length} is not aligned with the hourly stride "
            f"{TEMPORAL_STRIDE}. Use a multiple of {TEMPORAL_STRIDE}."
        )

    original_num_flow_states = flow_variables.shape[1]
    rain_start_index = burn_in_length // TEMPORAL_STRIDE
    flow_variables = flow_variables[:, burn_in_length::TEMPORAL_STRIDE, :, :]
    rain_source = rain_source[:, rain_start_index:, :]

    if flow_variables.shape[1] != EXPECTED_TIME_STATES_PER_TRAJECTORY:
        raise ValueError(
            f"Expected {EXPECTED_TIME_STATES_PER_TRAJECTORY} hourly flow states after "
            f"downsampling with stride {TEMPORAL_STRIDE}, got {flow_variables.shape[1]}. "
            f"Original flow states before downsampling: {original_num_flow_states}, "
            f"burn_in_length: {burn_in_length}."
        )
    if rain_source.shape[1] != flow_variables.shape[1]:
        raise ValueError(
            f"Hourly rain_source has {rain_source.shape[1]} states, but downsampled "
            f"flow_variables has {flow_variables.shape[1]} states."
        )

    return flow_variables, rain_source


def load_dataset(dataset_directory: Path, sample_start_index: int, sample_end_index: int, burn_in_length: int):
    # The DEM is fixed across the dataset, so it is read once from the first sample.
    dem = np.load(dataset_directory / f"sample_{sample_start_index:05d}" / "DEM.npy").astype(
        np.float32,
        copy=False,
    )

    flow_variables = []
    rain_source = []
    for sample_idx in tqdm(range(sample_start_index, sample_end_index + 1), desc="Loading samples"):
        sample_directory = dataset_directory / f"sample_{sample_idx:05d}"
        flow_variables.append(
            np.load(sample_directory / "flow_variables.npy").astype(np.float32, copy=False)
        )
        rain_source.append(
            np.load(sample_directory / "rain_source.npy").astype(np.float32, copy=False)
        )

    flow_variables = np.array(flow_variables)
    rain_source = np.array(rain_source)
    flow_variables, rain_source = downsample_to_hourly_states(
        flow_variables=flow_variables,
        rain_source=rain_source,
        burn_in_length=burn_in_length,
    )
    return dem, flow_variables, rain_source


def prepare_tensors(dem: np.ndarray, flow_variables: np.ndarray, rain_source: np.ndarray):
    dem_tensor = torch.from_numpy(dem).to(torch.float32).unsqueeze(0)
    flow_variables_tensor = torch.from_numpy(flow_variables).to(torch.float32)

    # Linearly rescale coordinate channels to [0, 1] for more stable optimization.
    for channel_idx in (0, 1):
        channel_tensor = dem_tensor[:, channel_idx]
        channel_min = channel_tensor.min()
        channel_max = channel_tensor.max()
        channel_range = channel_max - channel_min
        if channel_range.item() >= 1e-10:
            dem_tensor[:, channel_idx] = (channel_tensor - channel_min) / channel_range
        else:
            dem_tensor[:, channel_idx] = channel_tensor - channel_min

    rain_source_tensor = torch.from_numpy(rain_source).to(torch.float32)[:, :, RAIN_SOURCE_CHANNEL_INDEX]
    rain_source_mean = rain_source_tensor.mean()
    rain_source_std = rain_source_tensor.std(unbiased=False)
    normalization_stats = {
        "rain_source_mean": rain_source_mean.item(),
        "rain_source_std": rain_source_std.item(),
    }
    if rain_source_std.item() >= 1e-10:
        rain_source_tensor = (rain_source_tensor - rain_source_mean) / rain_source_std
    else:
        rain_source_tensor = rain_source_tensor - rain_source_mean

    height, width = dem_tensor.shape[-2:]
    rain_source_image_tensor = rain_source_tensor[:, :, None, None, None].expand(-1, -1, 1, height, width)

    water_depth_tensor = flow_variables_tensor[:, :, 0, :, :]
    discharge_x_tensor = flow_variables_tensor[:, :, 1, :, :]
    discharge_y_tensor = flow_variables_tensor[:, :, 2, :, :]
    flow_variables_tensor = torch.stack([water_depth_tensor, discharge_x_tensor, discharge_y_tensor], dim=2)

    if rain_source_tensor.shape[1] != flow_variables_tensor.shape[1]:
        raise ValueError(
            f"rain_source_tensor has {rain_source_tensor.shape[1]} timesteps, "
            f"but flow_variables_tensor has {flow_variables_tensor.shape[1]}"
        )

    print("DEM_tensor shape:", dem_tensor.shape)
    print("flow_variables_tensor shape:", flow_variables_tensor.shape)
    print("rain_source_tensor shape:", rain_source_tensor.shape)
    print("rain_source_image_tensor shape:", rain_source_image_tensor.shape)
    print("rain_source mean/std:", rain_source_mean.item(), rain_source_std.item())
    print_channel_statistics(
        name="DEM",
        tensor=dem_tensor,
        channel_names=["x", "y", "elevation"],
        channel_axis=1,
    )
    print(
        "Rain source channel statistics:\n"
        f"  precipitation: mean={rain_source_tensor.mean().item():.6f}, "
        f"std={rain_source_tensor.std(unbiased=False).item():.6f}, "
        f"min={rain_source_tensor.min().item():.6f}, "
        f"max={rain_source_tensor.max().item():.6f}"
    )
    print_channel_statistics(
        name="Flow variables",
        tensor=flow_variables_tensor,
        channel_names=["water_depth", "discharge_x", "discharge_y"],
        channel_axis=2,
    )

    return dem_tensor, flow_variables_tensor, rain_source_image_tensor, normalization_stats


def build_slope_tensor(dem_tensor: torch.Tensor, cell_size: float) -> torch.Tensor:
    slope_list = []
    for i in range(len(dem_tensor)):
        elevation = dem_tensor[i, 2].detach().cpu().numpy()
        gradient = limited_gradient(elevation, cell_size).transpose(2, 0, 1)
        slope_list.append(gradient)
    slope_tensor = torch.from_numpy(np.array(slope_list)).to(torch.float32)
    print("slope_tensor shape:", slope_tensor.shape)
    print_channel_statistics(
        name="Slope",
        tensor=slope_tensor,
        channel_names=["slope_x", "slope_y"],
        channel_axis=1,
    )
    return slope_tensor


def build_train_time_dict(num_time_instants: int, prediction_time_horizon: int):
    start_time_instant = 0
    end_time_instant = num_time_instants - 1 - prediction_time_horizon
    num_time_intervals = end_time_instant - start_time_instant
    return {
        "start": start_time_instant,
        "milestone_1": start_time_instant + int(0.25 * num_time_intervals),
        "milestone_2": start_time_instant + int(0.5 * num_time_intervals),
        "milestone_3": start_time_instant + int(0.75 * num_time_intervals),
        "milestone_4": end_time_instant,
        "end": end_time_instant,
    }


def build_test_time_dict(num_time_instants: int):
    start_time_instant = 0
    end_time_instant = num_time_instants - 2
    num_time_intervals = end_time_instant - start_time_instant
    return {
        "start": start_time_instant,
        "milestone_1": start_time_instant + int(0.25 * num_time_intervals),
        "milestone_2": start_time_instant + int(0.5 * num_time_intervals),
        "milestone_3": start_time_instant + int(0.75 * num_time_intervals),
        "milestone_4": end_time_instant,
        "end": end_time_instant,
    }


def make_prediction_gif(output_path: Path, prediction: torch.Tensor, truth: torch.Tensor, channel: int, title_prefix: str, vmin=None, vmax=None):
    frames = []
    for time_instant in range(prediction.shape[1] - 1):
        pred_time_instant = time_instant + 1
        fig = plot_prediction_truth_error(
            prediction[0, pred_time_instant, channel],
            truth[0, pred_time_instant, channel],
            title=f"{title_prefix} at time instant {pred_time_instant}",
            vmin=vmin,
            vmax=vmax,
        )
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        frames.append(imageio.v2.imread(buf))
        buf.close()
        plt.close(fig)
    imageio.mimsave(output_path, frames, duration=1.0, loop=0)


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_directory = Path(args.dataset_directory)
    results_base_directory = Path(args.results_directory)
    results_directory = results_base_directory / build_results_directory_name(args)
    checkpoint_directory = results_directory / "checkpoints"
    ensure_dir(results_directory)
    ensure_dir(checkpoint_directory)

    device = torch.device(args.device)
    print("repo_root:", REPO_ROOT)
    print("dataset_directory:", dataset_directory)
    print("results_directory:", results_directory)
    print("device:", device)

    dem, flow_variables, rain_source = load_dataset(
        dataset_directory=dataset_directory,
        sample_start_index=args.sample_start_index,
        sample_end_index=args.sample_end_index,
        burn_in_length=args.burn_in_length,
    )

    dem_tensor, flow_variables_tensor, rain_source_image_tensor, normalization_stats = prepare_tensors(
        dem=dem,
        flow_variables=flow_variables,
        rain_source=rain_source,
    )

    num_trajectories, num_time_instants, num_flow_variables, _, _ = flow_variables_tensor.shape
    print("num_trajectories:", num_trajectories)
    print("num_flow_variables:", num_flow_variables)
    print("num_time_instants:", num_time_instants)

    model_train_time_instants_dict = build_train_time_dict(
        num_time_instants=num_time_instants,
        prediction_time_horizon=args.prediction_time_horizon,
    )
    model_test_time_instants_dict = build_test_time_dict(num_time_instants=num_time_instants)
    print("model_train_time_instants_dict:", model_train_time_instants_dict)
    print("model_test_time_instants_dict:", model_test_time_instants_dict)

    slope_tensor = build_slope_tensor(dem_tensor=dem_tensor, cell_size=args.cell_size)

    fno_config = {
        "n_modes": tuple(args.fourier_modes),
        "hidden_channels": args.hidden_channels,
        # DEM (x, y, elevation) + slope (x, y) + rainfall + flow state (h, hU_x, hU_y)
        "in_channels": 3 + 2 + 1 + 3,
        "out_channels": 3,
        "n_layers": 4,
    }
    model = FNO(**fno_config).to(device)
    total_parameters, total_memory_bytes = print_model_size(model)
    print(
        f"Model size summary: parameters={total_parameters:,}, "
        f"parameter_memory={format_readable_memory_size(total_memory_bytes)}"
    )

    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    optimizer_config = {
        "name": "AdamW",
        **optimizer_kwargs,
    }
    loss_config = {
        "name": "MSELoss",
        "reduction": "mean",
    }
    loss_function = torch.nn.MSELoss(reduction="mean")
    trajectory_batch_indices_iterator = BatchIndicesIterator(
        start=0,
        end=num_trajectories,
        batch_size=args.batch_size,
        shuffle=False,
    )
    run_config = {
        "script": Path(__file__).name,
        "dataset": {
            "dataset_directory": dataset_directory,
            "sample_start_index": args.sample_start_index,
            "sample_end_index": args.sample_end_index,
            "burn_in_length": args.burn_in_length,
            "temporal": {
                "original_time_step_hours": ORIGINAL_TIME_STEP_HOURS,
                "model_time_step_hours": MODEL_TIME_STEP_HOURS,
                "temporal_stride": TEMPORAL_STRIDE,
                "expected_time_states_per_trajectory": EXPECTED_TIME_STATES_PER_TRAJECTORY,
            },
            "cell_size": args.cell_size,
            "dem_shape": list(dem_tensor.shape),
            "flow_variables_shape": list(flow_variables_tensor.shape),
            "rain_source_image_shape": list(rain_source_image_tensor.shape),
            "slope_shape": list(slope_tensor.shape),
            "normalization_stats": normalization_stats,
        },
        "training": {
            "device": device,
            "seed": args.seed,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "prediction_time_horizon": args.prediction_time_horizon,
            "save_every": args.save_every,
            "loss": loss_config,
            "optimizer": optimizer_config,
            "train_time_instants": model_train_time_instants_dict,
            "test_time_instants": model_test_time_instants_dict,
            "num_trajectories": num_trajectories,
            "num_time_instants": num_time_instants,
            "num_flow_variables": num_flow_variables,
        },
        "model": {
            "name": "FNO",
            "config": fno_config,
            "total_parameters": total_parameters,
            "parameter_memory_bytes": total_memory_bytes,
        },
        "results": {
            "results_base_directory": results_base_directory,
            "results_directory": results_directory,
            "checkpoint_directory": checkpoint_directory,
        },
        "cli_args": vars(args),
    }
    save_run_config(results_directory / "run_config.json", run_config)

    try:
        checkpoint = load_latest_checkpoint(str(checkpoint_directory))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        epoch_loss_list = checkpoint["epoch_loss_list"]
        frames = checkpoint.get("frames", [])
        print(f"Resuming from epoch {start_epoch}.")
    except FileNotFoundError:
        print("No checkpoint found. Initializing model from scratch.")
        start_epoch = 1
        epoch_loss_list = []
        frames = []
        torch.save(
            {
                "epoch": start_epoch - 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch_loss_list": epoch_loss_list,
                "frames": frames,
                "normalization_stats": normalization_stats,
            },
            checkpoint_directory / f"checkpoint_epoch_{start_epoch - 1}.pth",
        )

    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        new_epoch = True
        trajectory_batch_indices_iterator.reset()

        for trajectory_batch_indices in trajectory_batch_indices_iterator:
            batch_size = len(trajectory_batch_indices)
            batch_dem_tensor = expand_static_tensor_batch_to_device(dem_tensor, batch_size, device)
            batch_slope_tensor = expand_static_tensor_batch_to_device(slope_tensor, batch_size, device)

            for time_instant in range(
                model_train_time_instants_dict["start"],
                model_train_time_instants_dict["end"] + 1,
            ):
                temp = []
                multi_step_loss = 0.0

                for step_idx in range(args.prediction_time_horizon):
                    batch_rain_source_tensor = extract_batch_to_device(
                        rain_source_image_tensor,
                        trajectory_batch_indices,
                        device,
                        time_instant + step_idx,
                    )

                    if step_idx == 0:
                        batch_flow_state = extract_batch_to_device(
                            flow_variables_tensor,
                            trajectory_batch_indices,
                            device,
                            time_instant,
                        )
                    else:
                        batch_flow_state = temp[-1]

                    batch_inputs = torch.cat(
                        (
                            batch_dem_tensor,
                            batch_slope_tensor,
                            batch_rain_source_tensor,
                            batch_flow_state,
                        ),
                        dim=1,
                    )

                    batch_outputs = postprocess_prediction(model(batch_inputs))
                    batch_labels = extract_batch_to_device(
                        flow_variables_tensor,
                        trajectory_batch_indices,
                        device,
                        time_instant + 1 + step_idx,
                    )
                    one_step_loss = loss_function(batch_outputs, batch_labels)
                    multi_step_loss += one_step_loss
                    temp.append(batch_outputs)

                multi_step_loss /= args.prediction_time_horizon
                optimizer.zero_grad()
                multi_step_loss.backward()
                optimizer.step()

                print(
                    f"Epoch {epoch}, Time instant {time_instant}, "
                    f"Multi-step Loss: {multi_step_loss.item():.8f}"
                )
                epoch_loss += multi_step_loss.item()

                if new_epoch and time_instant == model_train_time_instants_dict["milestone_1"]:
                    fig = plot_prediction_truth_error(
                        batch_outputs[0, 0],
                        batch_labels[0, 0],
                        title=f"Water depth at time instant {time_instant} | Epoch {epoch}",
                    )
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    frames.append(imageio.v2.imread(buf))
                    buf.close()
                    plt.close(fig)

            new_epoch = False

        print(f"Epoch {epoch}/{args.num_epochs} completed. Loss: {epoch_loss:.8f}")
        epoch_loss_list.append(epoch_loss)

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch_loss_list": epoch_loss_list,
                    "frames": frames,
                    "normalization_stats": normalization_stats,
                },
                checkpoint_directory / f"checkpoint_epoch_{epoch}.pth",
            )

    final_epoch = args.num_epochs if args.num_epochs >= start_epoch else start_epoch - 1
    torch.save(
        {
            "epoch": final_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch_loss_list": epoch_loss_list,
            "frames": frames,
            "normalization_stats": normalization_stats,
        },
        checkpoint_directory / f"checkpoint_epoch_{final_epoch}.pth",
    )
    torch.save(model.state_dict(), results_directory / "fno_params.pth")
    torch.save(
        normalization_stats,
        results_directory / "rain_source_normalization_stats.pth",
    )
    np.save(results_directory / "fno_epoch_loss.npy", np.array(epoch_loss_list).reshape(-1, 1))
    imageio.mimsave(results_directory / "fno_train_water_depth.gif", frames, duration=1.0, loop=0)

    all_inference_outputs = []
    with torch.no_grad():
        model.eval()
        trajectory_batch_indices_iterator.reset()
        for trajectory_batch_indices in trajectory_batch_indices_iterator:
            batch_size = len(trajectory_batch_indices)
            batch_dem_tensor = expand_static_tensor_batch_to_device(dem_tensor, batch_size, device)
            batch_slope_tensor = expand_static_tensor_batch_to_device(slope_tensor, batch_size, device)
            batch_inference_outputs = []

            for time_instant in range(
                model_test_time_instants_dict["start"],
                model_test_time_instants_dict["end"] + 1,
            ):
                batch_rain_source_tensor = extract_batch_to_device(
                    rain_source_image_tensor,
                    trajectory_batch_indices,
                    device,
                    time_instant,
                )

                if time_instant == model_test_time_instants_dict["start"]:
                    batch_flow_state = extract_batch_to_device(
                        flow_variables_tensor,
                        trajectory_batch_indices,
                        device,
                        time_instant,
                    )
                else:
                    batch_flow_state = batch_inference_outputs[-1]

                batch_inputs = torch.cat(
                    (
                        batch_dem_tensor,
                        batch_slope_tensor,
                        batch_rain_source_tensor,
                        batch_flow_state,
                    ),
                    dim=1,
                )
                batch_outputs = postprocess_prediction(model(batch_inputs))
                batch_inference_outputs.append(batch_outputs)

            all_inference_outputs.append(torch.stack(batch_inference_outputs, dim=1).cpu())

    all_inference_outputs = torch.concatenate(all_inference_outputs, dim=0)
    all_inference_outputs = torch.concatenate((flow_variables_tensor[:, [0]], all_inference_outputs), dim=1)

    colorbar_vmin = {
        "water_depth": 0.0,
        "discharge_x": -20.0,
        "discharge_y": -20.0,
    }
    colorbar_vmax = {
        "water_depth": 10.0,
        "discharge_x": 20.0,
        "discharge_y": 20.0,
    }

    make_prediction_gif(
        output_path=results_directory / "fno_train_autoregressive_water_depth.gif",
        prediction=all_inference_outputs,
        truth=flow_variables_tensor,
        channel=0,
        title_prefix="Water depth",
        vmin=colorbar_vmin["water_depth"],
        vmax=colorbar_vmax["water_depth"],
    )
    make_prediction_gif(
        output_path=results_directory / "fno_train_autoregressive_discharge_x.gif",
        prediction=all_inference_outputs,
        truth=flow_variables_tensor,
        channel=1,
        title_prefix="discharge-x",
        vmin=colorbar_vmin["discharge_x"],
        vmax=colorbar_vmax["discharge_x"],
    )
    make_prediction_gif(
        output_path=results_directory / "fno_train_autoregressive_discharge_y.gif",
        prediction=all_inference_outputs,
        truth=flow_variables_tensor,
        channel=2,
        title_prefix="discharge-y",
        vmin=colorbar_vmin["discharge_y"],
        vmax=colorbar_vmax["discharge_y"],
    )

    print("Finished. Outputs saved to:", results_directory)


if __name__ == "__main__":
    main()
