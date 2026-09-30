"""Autoregressive rollout and scoring for the FNO baseline on the fixed-DEM Texas dataset.

Rolls the trained model forward over a full trajectory from the initial state only,
then reports relative RMSE, absolute RMSE and R^2 per flow variable on both the
training and the held-out samples.

Example (paths are relative to this file; run from anywhere):

    python fno_autoregressive_inference_texas_fixed_DEM.py \
      --checkpoint-path ../../checkpoints/fno/checkpoint_epoch_50.pth

The model architecture is read from a run_config.json; with no --run-config given
the script looks next to the checkpoint first, then falls back to configs/fno/.
"""

import argparse
import io
import json
import re
import sys
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
    limited_gradient,
    plot_prediction_truth_error,
    print_model_size,
    set_seed,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIRECTORY.parents[1]

DEFAULT_TRAIN_DATASET_DIRECTORY = REPO_ROOT / "data" / "texas" / "train_dataset"
DEFAULT_TEST_DATASET_DIRECTORY = REPO_ROOT / "data" / "texas" / "test_dataset"
DEFAULT_RUN_CONFIG_PATH = REPO_ROOT / "configs" / "fno" / "run_config.json"
DEFAULT_RESULTS_ROOT = SCRIPT_DIRECTORY / "outputs" / "fno_autoregressive_inference_results"

ORIGINAL_TIME_STEP_HOURS = 0.25
MODEL_TIME_STEP_HOURS = 1.0
TEMPORAL_STRIDE = 4
EXPECTED_TIME_STATES_PER_TRAJECTORY = 49
RAIN_SOURCE_CHANNEL_INDEX = 1


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def progress(iterable=None, **kwargs):
    return tqdm(
        iterable,
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stdout,
        **kwargs,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run autoregressive FNO inference on fixed Texas DEM train and test datasets."
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to a checkpoint file or a checkpoint directory.",
    )
    parser.add_argument(
        "--run-config",
        default=None,
        help=(
            "Path to the run_config.json holding the model configuration. Defaults to the one "
            "next to the checkpoint, then to configs/fno/run_config.json."
        ),
    )
    parser.add_argument(
        "--train-dataset-directory",
        default=str(DEFAULT_TRAIN_DATASET_DIRECTORY),
        help="Directory containing training sample_XXXXX subdirectories.",
    )
    parser.add_argument(
        "--test-dataset-directory",
        default=str(DEFAULT_TEST_DATASET_DIRECTORY),
        help="Directory containing test sample_XXXXX subdirectories.",
    )
    parser.add_argument("--train-sample-start-index", type=int, default=1)
    parser.add_argument("--train-sample-end-index", type=int, default=100)
    parser.add_argument("--test-sample-start-index", type=int, default=101)
    parser.add_argument("--test-sample-end-index", type=int, default=120)
    parser.add_argument("--burn-in-length", type=int, default=0)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--cell-size", type=float, default=30.0)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--water-depth-vmin", type=float, default=0.0)
    parser.add_argument("--water-depth-vmax", type=float, default=10.0)
    parser.add_argument("--discharge-vmin", type=float, default=-20.0)
    parser.add_argument("--discharge-vmax", type=float, default=20.0)
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory used to store inference outputs.",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_torch_compatible(value):
    if isinstance(value, list):
        return tuple(value)
    return value


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


def discover_sample_indices(dataset_directory: Path):
    sample_indices = sorted(
        int(path.name.split("_")[1])
        for path in dataset_directory.glob("sample_*")
        if path.is_dir()
    )
    if not sample_indices:
        raise FileNotFoundError(f"No sample directories found under {dataset_directory}")
    return sample_indices


def resolve_sample_indices(dataset_directory: Path, sample_start_index: int, sample_end_index: int):
    available_indices = discover_sample_indices(dataset_directory)
    selected_indices = [
        sample_idx
        for sample_idx in available_indices
        if sample_start_index <= sample_idx <= sample_end_index
    ]
    if not selected_indices:
        raise ValueError(
            f"No samples selected in {dataset_directory} for range "
            f"[{sample_start_index}, {sample_end_index}]"
        )
    return selected_indices


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


def load_dataset(dataset_directory: Path, sample_indices, burn_in_length: int):
    dem = np.load(dataset_directory / f"sample_{sample_indices[0]:05d}" / "DEM.npy").astype(
        np.float32,
        copy=False,
    )

    # Peek at the first sample to learn the per-sample shapes, then preallocate the
    # stacked arrays so the loop below fills them in place. This lets the progress bar
    # reflect the full loading cost (read + copy) instead of leaving a large, silent
    # np.array() stack at the end.
    first_sample_directory = dataset_directory / f"sample_{sample_indices[0]:05d}"
    first_flow_variables = np.load(first_sample_directory / "flow_variables.npy").astype(
        np.float32, copy=False
    )
    first_rain_source = np.load(first_sample_directory / "rain_source.npy").astype(
        np.float32, copy=False
    )

    num_samples = len(sample_indices)
    flow_variables = np.empty((num_samples, *first_flow_variables.shape), dtype=np.float32)
    rain_source = np.empty((num_samples, *first_rain_source.shape), dtype=np.float32)

    for position, sample_idx in enumerate(
        progress(sample_indices, desc=f"Loading {dataset_directory.name} samples")
    ):
        if position == 0:
            flow_variables[position] = first_flow_variables
            rain_source[position] = first_rain_source
            continue
        sample_directory = dataset_directory / f"sample_{sample_idx:05d}"
        flow_variables[position] = np.load(sample_directory / "flow_variables.npy")
        rain_source[position] = np.load(sample_directory / "rain_source.npy")
    flow_variables, rain_source = downsample_to_hourly_states(
        flow_variables=flow_variables,
        rain_source=rain_source,
        burn_in_length=burn_in_length,
    )
    return dem, flow_variables, rain_source


def prepare_tensors(
    dem: np.ndarray,
    flow_variables: np.ndarray,
    rain_source: np.ndarray,
    normalization_stats=None,
):
    dem_tensor = torch.from_numpy(dem).to(torch.float32).unsqueeze(0)
    flow_variables_tensor = torch.from_numpy(flow_variables).to(torch.float32)

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

    computed_mean = rain_source_tensor.mean()
    computed_std = rain_source_tensor.std(unbiased=False)
    if normalization_stats is None:
        rain_source_mean = computed_mean
        rain_source_std = computed_std
        normalization_stats = {
            "rain_source_mean": rain_source_mean.item(),
            "rain_source_std": rain_source_std.item(),
        }
        normalization_source = "computed from current dataset"
    else:
        rain_source_mean = torch.tensor(normalization_stats["rain_source_mean"], dtype=torch.float32)
        rain_source_std = torch.tensor(normalization_stats["rain_source_std"], dtype=torch.float32)
        normalization_source = "loaded from checkpoint"

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
    print("rain_source normalization source:", normalization_source)
    print("rain_source mean/std used:", rain_source_mean.item(), rain_source_std.item())
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
    for i in progress(range(len(dem_tensor)), desc="Building slope tensor"):
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


def compute_relative_rmse_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = prediction.detach().cpu()
    ground_truth = ground_truth.detach().cpu()

    componentwise_numerator = torch.sqrt(torch.sum((prediction - ground_truth) ** 2, dim=(0, 1, 3, 4)))
    componentwise_denominator = torch.sqrt(torch.sum(ground_truth**2, dim=(0, 1, 3, 4)))
    componentwise_relative_rmse = componentwise_numerator / componentwise_denominator.clamp_min(1e-10)

    whole_numerator = torch.sqrt(torch.sum((prediction - ground_truth) ** 2, dim=(0, 1, 2, 3, 4)))
    whole_denominator = torch.sqrt(torch.sum(ground_truth**2, dim=(0, 1, 2, 3, 4)))
    whole_relative_rmse = whole_numerator / whole_denominator.clamp_min(1e-10)

    return componentwise_relative_rmse, whole_relative_rmse


def compute_rmse_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute (non-relative) RMSE reduced per flow-variable channel and over everything.

    Mirrors ``compute_relative_rmse_metrics`` but divides the squared error by the
    element count instead of the ground-truth energy, so the result carries the
    physical units of each channel. The channel axis (dim=2) is kept; the
    trajectory, time and spatial axes (dims 0, 1, 3, 4) are reduced.
    """
    prediction = prediction.detach().cpu()
    ground_truth = ground_truth.detach().cpu()

    squared_error = (prediction - ground_truth) ** 2
    componentwise_rmse = torch.sqrt(torch.mean(squared_error, dim=(0, 1, 3, 4)))
    whole_rmse = torch.sqrt(torch.mean(squared_error))

    return componentwise_rmse, whole_rmse


def _r2_from_flat(truth_flat: torch.Tensor, pred_flat: torch.Tensor) -> float:
    """Coefficient of determination for a pair of flattened tensors.

    Masks non-finite entries, uses ss_res = sum((pred - truth)^2),
    ss_tot = sum((truth - mean(truth))^2), and returns 1 - ss_res / ss_tot,
    with NaN when ss_tot is non-positive.
    """
    truth_flat = truth_flat.reshape(-1).to(torch.float64)
    pred_flat = pred_flat.reshape(-1).to(torch.float64)
    valid = torch.isfinite(truth_flat) & torch.isfinite(pred_flat)
    if not torch.any(valid):
        return float("nan")

    truth_valid = truth_flat[valid]
    pred_valid = pred_flat[valid]
    ss_res = torch.sum((pred_valid - truth_valid) ** 2).item()
    ss_tot = torch.sum((truth_valid - truth_valid.mean()) ** 2).item()
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_r2_metrics(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """R^2 per flow-variable channel and over all channels combined.

    Each channel (dim=2) is flattened across trajectories, time and space and
    scored with ``_r2_from_flat``.
    """
    prediction = prediction.detach().cpu()
    ground_truth = ground_truth.detach().cpu()

    num_channels = ground_truth.shape[2]
    componentwise_r2 = torch.tensor(
        [
            _r2_from_flat(ground_truth[:, :, channel_idx], prediction[:, :, channel_idx])
            for channel_idx in range(num_channels)
        ],
        dtype=torch.float32,
    )
    whole_r2 = torch.tensor(_r2_from_flat(ground_truth, prediction), dtype=torch.float32)

    return componentwise_r2, whole_r2


def resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_dir():
        checkpoint_search_directories = [checkpoint_path]
        if (checkpoint_path / "checkpoints").is_dir():
            checkpoint_search_directories.insert(0, checkpoint_path / "checkpoints")

        checkpoint_files = []
        for checkpoint_directory in checkpoint_search_directories:
            checkpoint_files.extend(checkpoint_directory.glob("checkpoint_epoch_*.pth"))

        if not checkpoint_files:
            fallback_model_path = checkpoint_path / "fno_params.pth"
            if fallback_model_path.exists():
                return fallback_model_path
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")

        def extract_epoch(path: Path) -> int:
            match = re.search(r"checkpoint_epoch_(\d+)\.pth$", path.name)
            return int(match.group(1)) if match else -1

        return max(checkpoint_files, key=extract_epoch)
    return checkpoint_path


def load_checkpoint_payload(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_info = checkpoint
    else:
        state_dict = checkpoint
        checkpoint_info = {}
    return checkpoint_info, state_dict


def resolve_run_directory(path: Path) -> Path | None:
    if path.is_file():
        if path.parent.name == "checkpoints":
            return path.parent.parent
        return path.parent

    if path.is_dir():
        if (path / "run_config.json").exists():
            return path
        if (path / "checkpoints").is_dir():
            return path
    return None


def load_run_config(run_directory: Path | None):
    if run_directory is None:
        return None
    run_config_path = run_directory / "run_config.json"
    if not run_config_path.exists():
        return None
    return json.loads(run_config_path.read_text())


def resolve_run_config(explicit_run_config: str | None, run_directory: Path | None):
    """Model configuration, from --run-config, then the run directory, then configs/fno/."""
    if explicit_run_config is not None:
        run_config_path = Path(explicit_run_config)
        if not run_config_path.exists():
            raise FileNotFoundError(f"run_config.json not found: {run_config_path}")
        return json.loads(run_config_path.read_text()), run_config_path

    run_config = load_run_config(run_directory)
    if run_config is not None:
        return run_config, run_directory / "run_config.json"

    if DEFAULT_RUN_CONFIG_PATH.exists():
        return json.loads(DEFAULT_RUN_CONFIG_PATH.read_text()), DEFAULT_RUN_CONFIG_PATH

    return None, None


def build_model_from_run_config(run_config):
    if run_config is None:
        return None

    model_config = run_config.get("model", {}).get("config")
    if not model_config:
        return None

    normalized_model_config = {
        key: to_torch_compatible(value) for key, value in model_config.items()
    }
    return FNO(**normalized_model_config)


def load_normalization_stats(run_directory: Path | None, checkpoint_info: dict, run_config):
    checkpoint_normalization_stats = checkpoint_info.get("normalization_stats")
    if checkpoint_normalization_stats is not None:
        return checkpoint_normalization_stats, "checkpoint"

    dataset_stats = (
        run_config.get("dataset", {}).get("normalization_stats")
        if run_config is not None
        else None
    )
    if dataset_stats is not None:
        return dataset_stats, "run_config"

    if run_directory is not None:
        normalization_stats_path = run_directory / "rain_source_normalization_stats.pth"
        if normalization_stats_path.exists():
            normalization_stats = torch.load(
                normalization_stats_path,
                map_location="cpu",
                weights_only=False,
            )
            return normalization_stats, str(normalization_stats_path)

    return None, "dataset fallback"


def make_prediction_gif(
    output_path: Path,
    prediction: torch.Tensor,
    truth: torch.Tensor,
    channel: int,
    title_prefix: str,
    trajectory_index: int,
    vmin=None,
    vmax=None,
):
    frames = []
    for time_instant in progress(
        range(prediction.shape[1] - 1),
        desc=f"Rendering {output_path.stem}",
    ):
        pred_time_instant = time_instant + 1
        fig = plot_prediction_truth_error(
            prediction[trajectory_index, pred_time_instant, channel],
            truth[trajectory_index, pred_time_instant, channel],
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


def save_relative_rmse_figure(
    output_path: Path,
    componentwise_relative_rmse: torch.Tensor,
    whole_relative_rmse: torch.Tensor,
    dataset_name: str,
):
    labels = ["Water depth", "Discharge-x", "Discharge-y", "Whole"]
    values = np.concatenate([componentwise_relative_rmse.numpy(), np.array([whole_relative_rmse.item()])])
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylabel("Relative RMSE")
    plt.title(f"Relative RMSE on {dataset_name} set")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_comparison_figure(
    train_componentwise_relative_rmse: torch.Tensor,
    train_whole_relative_rmse: torch.Tensor,
    test_componentwise_relative_rmse: torch.Tensor,
    test_whole_relative_rmse: torch.Tensor,
    output_path: Path,
):
    labels = ["Water depth", "Discharge-x", "Discharge-y", "Whole"]
    train_values = np.concatenate(
        [train_componentwise_relative_rmse.numpy(), np.array([train_whole_relative_rmse.item()])]
    )
    test_values = np.concatenate(
        [test_componentwise_relative_rmse.numpy(), np.array([test_whole_relative_rmse.item()])]
    )

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, train_values, width, label="Train")
    ax.bar(x + width / 2, test_values, width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Relative RMSE")
    ax.set_title("Train vs Test Relative RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_autoregressive_inference(
    model,
    flow_variables_tensor: torch.Tensor,
    rain_source_image_tensor: torch.Tensor,
    dem_tensor: torch.Tensor,
    slope_tensor: torch.Tensor,
    batch_size: int,
    device: torch.device,
):
    num_trajectories, num_time_instants, _, _, _ = flow_variables_tensor.shape
    time_dict = build_test_time_dict(num_time_instants)
    num_time_steps = time_dict["end"] - time_dict["start"] + 1
    iterator = BatchIndicesIterator(
        start=0,
        end=num_trajectories,
        batch_size=batch_size,
        shuffle=False,
    )

    all_inference_outputs = []
    with torch.no_grad():
        model.eval()
        iterator.reset()
        total_batches = len(iterator)
        progress_bar = progress(
            total=total_batches * num_time_steps,
            desc="Autoregressive inference",
        )
        for trajectory_batch_indices in iterator:
            current_batch_size = len(trajectory_batch_indices)
            batch_dem_tensor = dem_tensor.expand(current_batch_size, -1, -1, -1).to(device)
            batch_slope_tensor = slope_tensor.expand(current_batch_size, -1, -1, -1).to(device)
            batch_inference_outputs = []

            for time_instant in range(time_dict["start"], time_dict["end"] + 1):
                batch_rain_source_tensor = rain_source_image_tensor[trajectory_batch_indices, time_instant].to(device)

                if time_instant == time_dict["start"]:
                    batch_flow_state = flow_variables_tensor[trajectory_batch_indices, time_instant].to(device)
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
                progress_bar.update(1)

            all_inference_outputs.append(torch.stack(batch_inference_outputs, dim=1).cpu())
        progress_bar.close()

    all_inference_outputs = torch.concatenate(all_inference_outputs, dim=0)
    all_inference_outputs = torch.concatenate((flow_variables_tensor[:, [0]], all_inference_outputs), dim=1)
    return all_inference_outputs


def evaluate_dataset(
    dataset_name: str,
    dataset_directory: Path,
    sample_start_index: int,
    sample_end_index: int,
    burn_in_length: int,
    checkpoint_normalization_stats,
    model,
    device: torch.device,
    batch_size: int,
    cell_size: float,
    trajectory_index: int,
    output_root: Path,
    colorbar_limits: dict,
):
    sample_indices = resolve_sample_indices(
        dataset_directory=dataset_directory,
        sample_start_index=sample_start_index,
        sample_end_index=sample_end_index,
    )
    print(
        f"Evaluating {dataset_name} dataset with samples "
        f"{sample_indices[0]} to {sample_indices[-1]} ({len(sample_indices)} trajectories)."
    )
    if not 0 <= trajectory_index < len(sample_indices):
        raise ValueError(
            f"trajectory_index={trajectory_index} is out of range for {dataset_name} "
            f"dataset with {len(sample_indices)} trajectories."
        )

    print(f"[{dataset_name}] Loading dataset")
    dem, flow_variables, rain_source = load_dataset(
        dataset_directory=dataset_directory,
        sample_indices=sample_indices,
        burn_in_length=burn_in_length,
    )
    print(f"[{dataset_name}] Preparing tensors")
    dem_tensor, flow_variables_tensor, rain_source_image_tensor, used_normalization_stats = prepare_tensors(
        dem=dem,
        flow_variables=flow_variables,
        rain_source=rain_source,
        normalization_stats=checkpoint_normalization_stats,
    )
    print(f"[{dataset_name}] Computing slope tensor")
    slope_tensor = build_slope_tensor(dem_tensor=dem_tensor, cell_size=cell_size)

    print(f"[{dataset_name}] Running autoregressive rollout")
    predictions = run_autoregressive_inference(
        model=model,
        flow_variables_tensor=flow_variables_tensor,
        rain_source_image_tensor=rain_source_image_tensor,
        dem_tensor=dem_tensor,
        slope_tensor=slope_tensor,
        batch_size=batch_size,
        device=device,
    )

    componentwise_relative_rmse, whole_relative_rmse = compute_relative_rmse_metrics(
        predictions,
        flow_variables_tensor,
    )
    componentwise_rmse, whole_rmse = compute_rmse_metrics(
        predictions,
        flow_variables_tensor,
    )
    componentwise_r2, whole_r2 = compute_r2_metrics(
        predictions,
        flow_variables_tensor,
    )

    dataset_output_directory = output_root / dataset_name
    ensure_dir(dataset_output_directory)

    np.save(
        dataset_output_directory / "autoregressive_predictions.npy",
        predictions.numpy(),
    )
    np.save(
        dataset_output_directory / "componentwise_relative_rmse.npy",
        componentwise_relative_rmse.numpy(),
    )
    np.save(
        dataset_output_directory / "whole_relative_rmse.npy",
        np.array([whole_relative_rmse.item()]),
    )
    np.save(
        dataset_output_directory / "componentwise_rmse.npy",
        componentwise_rmse.numpy(),
    )
    np.save(
        dataset_output_directory / "whole_rmse.npy",
        np.array([whole_rmse.item()]),
    )
    np.save(
        dataset_output_directory / "componentwise_r2.npy",
        componentwise_r2.numpy(),
    )
    np.save(
        dataset_output_directory / "whole_r2.npy",
        np.array([whole_r2.item()]),
    )

    print(f"[{dataset_name}] Saving GIFs and figures")
    make_prediction_gif(
        output_path=dataset_output_directory / f"{dataset_name}_autoregressive_water_depth.gif",
        prediction=predictions,
        truth=flow_variables_tensor,
        channel=0,
        title_prefix=f"{dataset_name} water depth",
        trajectory_index=trajectory_index,
        vmin=colorbar_limits["water_depth"][0],
        vmax=colorbar_limits["water_depth"][1],
    )
    make_prediction_gif(
        output_path=dataset_output_directory / f"{dataset_name}_autoregressive_discharge_x.gif",
        prediction=predictions,
        truth=flow_variables_tensor,
        channel=1,
        title_prefix=f"{dataset_name} discharge-x",
        trajectory_index=trajectory_index,
        vmin=colorbar_limits["discharge_x"][0],
        vmax=colorbar_limits["discharge_x"][1],
    )
    make_prediction_gif(
        output_path=dataset_output_directory / f"{dataset_name}_autoregressive_discharge_y.gif",
        prediction=predictions,
        truth=flow_variables_tensor,
        channel=2,
        title_prefix=f"{dataset_name} discharge-y",
        trajectory_index=trajectory_index,
        vmin=colorbar_limits["discharge_y"][0],
        vmax=colorbar_limits["discharge_y"][1],
    )
    save_relative_rmse_figure(
        output_path=dataset_output_directory / f"{dataset_name}_relative_rmse.png",
        componentwise_relative_rmse=componentwise_relative_rmse,
        whole_relative_rmse=whole_relative_rmse,
        dataset_name=dataset_name,
    )

    print(
        f"[{dataset_name}] Relative RMSE: "
        f"water_depth={componentwise_relative_rmse[0].item():.6f}, "
        f"discharge_x={componentwise_relative_rmse[1].item():.6f}, "
        f"discharge_y={componentwise_relative_rmse[2].item():.6f}, "
        f"whole={whole_relative_rmse.item():.6f}"
    )
    print(
        f"[{dataset_name}] RMSE: "
        f"water_depth={componentwise_rmse[0].item():.6f}, "
        f"discharge_x={componentwise_rmse[1].item():.6f}, "
        f"discharge_y={componentwise_rmse[2].item():.6f}, "
        f"whole={whole_rmse.item():.6f}"
    )
    print(
        f"[{dataset_name}] R^2: "
        f"water_depth={componentwise_r2[0].item():.6f}, "
        f"discharge_x={componentwise_r2[1].item():.6f}, "
        f"discharge_y={componentwise_r2[2].item():.6f}, "
        f"whole={whole_r2.item():.6f}"
    )

    print(f"Saved {dataset_name} outputs to {dataset_output_directory}")
    return {
        "componentwise_relative_rmse": componentwise_relative_rmse,
        "whole_relative_rmse": whole_relative_rmse,
        "componentwise_rmse": componentwise_rmse,
        "whole_rmse": whole_rmse,
        "componentwise_r2": componentwise_r2,
        "whole_r2": whole_r2,
        "output_directory": dataset_output_directory,
        "normalization_stats": used_normalization_stats,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    checkpoint_input_path = Path(args.checkpoint_path)
    checkpoint_path = resolve_checkpoint_path(checkpoint_input_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_directory = resolve_run_directory(checkpoint_input_path) or resolve_run_directory(checkpoint_path)
    run_config, run_config_path = resolve_run_config(args.run_config, run_directory)
    run_name = run_directory.name if run_directory is not None else checkpoint_path.parent.name
    checkpoint_stem = checkpoint_path.stem
    results_root = Path(args.results_root)
    results_directory = results_root / run_name / checkpoint_stem
    ensure_dir(results_directory)

    device = torch.device(args.device)
    colorbar_limits = {
        "water_depth": (args.water_depth_vmin, args.water_depth_vmax),
        "discharge_x": (args.discharge_vmin, args.discharge_vmax),
        "discharge_y": (args.discharge_vmin, args.discharge_vmax),
    }

    print("repo_root:", REPO_ROOT)
    print("checkpoint_input_path:", checkpoint_input_path)
    print("checkpoint_path:", checkpoint_path)
    print("run_directory:", run_directory)
    print("run_config_path:", run_config_path)
    print("results_directory:", results_directory)
    print("device:", device)

    model = build_model_from_run_config(run_config)
    if model is None:
        print("No run_config.json with a model config found; falling back to legacy FNO config.")
        model = FNO(
            n_modes=(16, 16),
            hidden_channels=32,
            in_channels=3 + 2 + 1 + 3,
            out_channels=3,
            n_layers=4,
        )
    print_model_size(model)

    checkpoint_info, model_state_dict = load_checkpoint_payload(checkpoint_path=checkpoint_path)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    checkpoint_normalization_stats, normalization_stats_source = load_normalization_stats(
        run_directory=run_directory,
        checkpoint_info=checkpoint_info,
        run_config=run_config,
    )
    print("checkpoint keys:", sorted(checkpoint_info.keys()) if checkpoint_info else "raw state_dict")
    print("run_config model config:", None if run_config is None else run_config.get("model", {}).get("config"))
    print(
        "checkpoint normalization stats:",
        checkpoint_normalization_stats,
        f"(source: {normalization_stats_source})",
    )

    train_results = evaluate_dataset(
        dataset_name="train",
        dataset_directory=Path(args.train_dataset_directory),
        sample_start_index=args.train_sample_start_index,
        sample_end_index=args.train_sample_end_index,
        burn_in_length=args.burn_in_length,
        checkpoint_normalization_stats=checkpoint_normalization_stats,
        model=model,
        device=device,
        batch_size=args.batch_size,
        cell_size=args.cell_size,
        trajectory_index=args.trajectory_index,
        output_root=results_directory,
        colorbar_limits=colorbar_limits,
    )
    test_results = evaluate_dataset(
        dataset_name="test",
        dataset_directory=Path(args.test_dataset_directory),
        sample_start_index=args.test_sample_start_index,
        sample_end_index=args.test_sample_end_index,
        burn_in_length=args.burn_in_length,
        checkpoint_normalization_stats=checkpoint_normalization_stats,
        model=model,
        device=device,
        batch_size=args.batch_size,
        cell_size=args.cell_size,
        trajectory_index=args.trajectory_index,
        output_root=results_directory,
        colorbar_limits=colorbar_limits,
    )

    comparison_figure_path = results_directory / "train_vs_test_relative_rmse.png"
    save_comparison_figure(
        train_componentwise_relative_rmse=train_results["componentwise_relative_rmse"],
        train_whole_relative_rmse=train_results["whole_relative_rmse"],
        test_componentwise_relative_rmse=test_results["componentwise_relative_rmse"],
        test_whole_relative_rmse=test_results["whole_relative_rmse"],
        output_path=comparison_figure_path,
    )

    print("Finished. Outputs saved to:", results_directory)
    print("Train outputs:", train_results["output_directory"])
    print("Test outputs:", test_results["output_directory"])
    print("Comparison figure:", comparison_figure_path)


if __name__ == "__main__":
    main()
