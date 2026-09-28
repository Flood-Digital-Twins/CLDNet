"""
Efficient Fourier LDNN Training Script for Flood Surrogate Modeling.

Usage:
    # From the repository root, train with 90 events and validate on 107-109.
    torchrun --nproc_per_node=8 code/ldnet/ldnet_chicago_efficient.py --ddp \\
        --num-trajectories 110 --num-train 90 --num-valid 3 \\
        --split-file splits/illinois_split.json --sample-indices 100000 \\
        --all-vars --fourier-mapping-size 32 \\
        --model-path outputs/training/ldnet_illinois_h200

    # Append Illinois static features for CLDNet.
    # Add --use-static-features --static-feature-dim 3 and choose a separate model path.
"""
import argparse
import json
import os
import re
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from efficient_fourier_ldnet import EfficientFourierLDNN
from src.logger import Logger
from src.train import Trainer_Sample

dt = 1


class WetWeightedMSE(nn.Module):
    """MSE loss with optional weighting by water depth."""

    def __init__(self, alpha: float = 0.0, max_weight: float | None = None):
        super().__init__()
        self.alpha = alpha
        self.max_weight = max_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.alpha <= 0:
            return torch.mean((pred - target) ** 2)
        depth = target[..., 0:1] if target.shape[-1] > 1 else target
        depth = torch.clamp(depth, min=0.0)
        weight = 1.0 + self.alpha * depth
        if self.max_weight is not None:
            weight = torch.clamp(weight, max=self.max_weight)
        return torch.mean(((pred - target) ** 2) * weight)


def create_training_options():
    parser = argparse.ArgumentParser()
    default_base_path = Path(__file__).resolve().parents[2]
    parser.add_argument("--base-path", type=Path, default=default_base_path)
    parser.add_argument("--log-dir", type=Path, default="log")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="hydro_surrogate_chiago")
    parser.add_argument("--name", type=str, default="ldnet_light")
    parser.add_argument("--model-path", type=Path, default="checkpoints/ldnet")
    parser.add_argument("--data-root", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--use-static-features", action="store_true", default=False)
    parser.add_argument("--static-feature-dim", type=int, default=3)
    parser.add_argument("--static-slope-xy", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--normalize-rain", action="store_true", default=False)

    # Model parameters
    parser.add_argument("--num-latent-states", type=int, default=200)
    parser.add_argument("--dim-u", type=int, default=39*13)
    parser.add_argument("--fourier-mapping-size", type=int, default=10)
    parser.add_argument("--NN-dyn-depth", type=int, default=8)
    parser.add_argument("--NN-dyn-width", type=int, default=50)
    parser.add_argument("--NN-rec-depth", type=int, default=10)
    parser.add_argument("--NN-rec-width", type=int, default=300)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--kernel-initializer", type=str, default="Glorot normal")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=10000)

    # Training parameters
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ddp", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-epochs", type=int, default=10000)
    parser.add_argument("--smoke-test", action="store_true", default=False)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--lr-step-size", type=int, default=50)
    parser.add_argument("--num-trajectories", type=int, default=2)
    parser.add_argument("--sample-indices", type=int, default=20000)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-valid", type=int, default=None)
    parser.add_argument("--split-file", type=Path, default=None,
                        help="JSON file with train/validation/test/heldout_2013 trajectory IDs.")
    parser.add_argument("--validation-split", choices=("file", "none", "test"), default="file",
                        help="With --split-file, use validation IDs by default; 'test' evaluates held-out test IDs.")
    parser.add_argument("--all-vars", action="store_true", default=False)
    parser.add_argument("--wet-weight-alpha", type=float, default=0.0)
    parser.add_argument("--wet-weight-max", type=float, default=5.0)
    parser.add_argument("--data-sanity", action="store_true", default=False)
    parser.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine", "plateau"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping max norm (0=disabled)")
    parser.add_argument("--resume-latest", dest="resume_latest", action="store_true", default=True)
    parser.add_argument("--no-resume-latest", dest="resume_latest", action="store_false")
    parser.add_argument("--rec-finetune", action="store_true", default=False)
    parser.add_argument("--finetune-base-epoch", type=int, default=539)

    return parser.parse_args()


def _extract_ckpt_epoch(path: Path, prefix: str) -> int | None:
    match = re.fullmatch(rf"{prefix}_(\d+)\.ckpt", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _latest_complete_epoch(save_path: Path) -> int | None:
    dyn_epochs = {
        epoch
        for ckpt in save_path.glob("dyn_*.ckpt")
        if (epoch := _extract_ckpt_epoch(ckpt, "dyn")) is not None
    }
    rec_epochs = {
        epoch
        for ckpt in save_path.glob("rec_*.ckpt")
        if (epoch := _extract_ckpt_epoch(ckpt, "rec")) is not None
    }
    complete = sorted(dyn_epochs.intersection(rec_epochs))
    return complete[-1] if complete else None


def _latest_complete_finetune_epoch(save_path: Path) -> int | None:
    rec_epochs = {
        epoch
        for ckpt in save_path.glob("rec_finetune_*.ckpt")
        if (epoch := _extract_ckpt_epoch(ckpt, "rec_finetune")) is not None
    }
    b_epochs = {
        epoch
        for ckpt in save_path.glob("B_finetune_*.ckpt")
        if (epoch := _extract_ckpt_epoch(ckpt, "B_finetune")) is not None
    }
    complete = sorted(rec_epochs.intersection(b_epochs))
    return complete[-1] if complete else None


def _latest_epoch_for_prefix(save_path: Path, prefix: str) -> int | None:
    epochs = {
        epoch
        for ckpt in save_path.glob(f"{prefix}_*.ckpt")
        if (epoch := _extract_ckpt_epoch(ckpt, prefix)) is not None
    }
    return max(epochs) if epochs else None


def _broadcast_epoch(epoch: int, rank: int) -> int:
    if dist.is_available() and dist.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device())
        epoch_tensor = torch.tensor([epoch if rank == 0 else -1], dtype=torch.int64, device=device)
        dist.broadcast(epoch_tensor, src=0)
        return int(epoch_tensor.item())
    return epoch


def _load_state_or_fail(module, ckpt_path: Path, logger, rank: int, label: str) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {label} checkpoint: {ckpt_path}")
    module.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    if rank == 0:
        logger.info(f"Loaded {label} checkpoint: {ckpt_path}")


def _resume_from_latest(
    model,
    optimizer,
    save_path: Path,
    rank: int,
    world_size: int,
    use_fourier_B: bool,
    logger,
) -> int:
    latest_epoch = _latest_complete_epoch(save_path) if rank == 0 else -1
    if latest_epoch is None:
        latest_epoch = -1
    latest_epoch = _broadcast_epoch(latest_epoch, rank)

    if latest_epoch < 0:
        if rank == 0:
            logger.info("No existing checkpoints found. Starting from scratch.")
        return 0

    dyn_ckpt = save_path / f"dyn_{latest_epoch}.ckpt"
    rec_ckpt = save_path / f"rec_{latest_epoch}.ckpt"
    B_ckpt = save_path / f"B_{latest_epoch}.ckpt"
    optimizer_ckpt = save_path / f"optimizer_{latest_epoch}.ckpt"

    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.dyn.load_state_dict(torch.load(dyn_ckpt, map_location="cpu"))
    model_to_load.rec.load_state_dict(torch.load(rec_ckpt, map_location="cpu"))

    if use_fourier_B and hasattr(model_to_load, "B"):
        if B_ckpt.exists():
            model_to_load.B.load_state_dict(torch.load(B_ckpt, map_location="cpu"))
            if rank == 0:
                logger.info(f"Loaded B checkpoint: {B_ckpt}")
        elif rank == 0:
            logger.info(f"B checkpoint missing for epoch {latest_epoch}; keeping current B weights.")

    if optimizer_ckpt.exists():
        optimizer.load_state_dict(torch.load(optimizer_ckpt, map_location="cpu"))
        if rank == 0:
            logger.info(f"Loaded optimizer checkpoint: {optimizer_ckpt}")
    elif rank == 0:
        logger.info(f"Optimizer checkpoint missing for epoch {latest_epoch}; optimizer starts fresh.")

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()

    if rank == 0:
        logger.info(
            f"Resumed from epoch index {latest_epoch}; continuing at epoch index {latest_epoch + 1}."
        )
    return latest_epoch + 1


def _prepare_rec_finetune(
    model,
    optimizer,
    save_path: Path,
    rank: int,
    world_size: int,
    logger,
    base_epoch: int,
) -> int:
    model_to_load = model.module if hasattr(model, "module") else model
    latest_finetune_epoch = _latest_complete_finetune_epoch(save_path) if rank == 0 else -1
    if latest_finetune_epoch is None:
        latest_finetune_epoch = -1
    latest_finetune_epoch = _broadcast_epoch(latest_finetune_epoch, rank)

    dyn_ckpt = save_path / f"dyn_{base_epoch}.ckpt"
    rec_base_ckpt = save_path / f"rec_{base_epoch}.ckpt"
    B_ckpt = save_path / f"B_{base_epoch}.ckpt"
    _load_state_or_fail(model_to_load.dyn, dyn_ckpt, logger, rank, "base dyn")
    _load_state_or_fail(model_to_load.B, B_ckpt, logger, rank, "base B")

    if latest_finetune_epoch >= 0:
        rec_ckpt = save_path / f"rec_finetune_{latest_finetune_epoch}.ckpt"
        B_finetune_ckpt = save_path / f"B_finetune_{latest_finetune_epoch}.ckpt"
        _load_state_or_fail(model_to_load.rec, rec_ckpt, logger, rank, "finetune rec")
        _load_state_or_fail(model_to_load.B, B_finetune_ckpt, logger, rank, "finetune B")

        optimizer_ckpt = save_path / f"optimizer_rec_finetune_{latest_finetune_epoch}.ckpt"
        if optimizer_ckpt.exists():
            optimizer.load_state_dict(torch.load(optimizer_ckpt, map_location="cpu"))
            if rank == 0:
                logger.info(f"Loaded finetune optimizer checkpoint: {optimizer_ckpt}")
        elif rank == 0:
            logger.info(
                f"Finetune optimizer checkpoint missing for epoch {latest_finetune_epoch}; optimizer starts fresh."
            )

        if dist.is_available() and dist.is_initialized() and world_size > 1:
            dist.barrier()

        if rank == 0:
            logger.info(
                "Resumed rec finetune from epoch index "
                f"{latest_finetune_epoch}; continuing at epoch index {latest_finetune_epoch + 1}."
            )
        return latest_finetune_epoch + 1

    _load_state_or_fail(model_to_load.rec, rec_base_ckpt, logger, rank, "base rec")
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        dist.barrier()
    if rank == 0:
        logger.info(
            f"No rec_finetune checkpoints found. Starting finetune from base epoch {base_epoch} at epoch index 0."
        )
    return 0


def _init_or_resume_wandb(opt, save_path: Path, start_epoch: int, rank: int, logger):
    if not opt.use_wandb or rank != 0:
        return

    run_id_path = save_path / "wandb_run_id.txt"
    run_id = None
    resume_mode = None

    if start_epoch > 0:
        if run_id_path.exists():
            run_id = run_id_path.read_text(encoding="utf-8").strip()
            if run_id:
                resume_mode = "allow"
                logger.info(f"Resuming W&B run id: {run_id}")
            else:
                run_id = None
        if run_id is None:
            logger.info("Checkpoint found but W&B run id missing; starting a new W&B run.")
            run_id = wandb.util.generate_id()
            run_id_path.write_text(run_id + "\n", encoding="utf-8")
    else:
        run_id = wandb.util.generate_id()
        run_id_path.write_text(run_id + "\n", encoding="utf-8")
        logger.info(f"Starting new W&B run id: {run_id}")

    init_kwargs = {
        "project": opt.wandb_project,
        "entity": opt.wandb_entity,
        "name": opt.name,
        "config": vars(opt),
        "id": run_id,
    }
    if resume_mode is not None:
        init_kwargs["resume"] = resume_mode

    wandb.init(**init_kwargs)


def setup_ddp(opt):
    if not opt.ddp:
        device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device, None

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise ValueError("DDP requested but RANK/WORLD_SIZE not set. Use torchrun for DDP.")

    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=7200))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if not torch.cuda.is_available():
        raise RuntimeError("DDP with NCCL requested but CUDA is unavailable.")
    visible_cuda = torch.cuda.device_count()
    if local_rank >= visible_cuda:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {visible_cuda} CUDA device(s) are visible "
            f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})."
        )
    device_index = local_rank
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    dist.new_group(backend="gloo", timeout=timedelta(seconds=7200))
    pin_info = (
        f"LOCAL_RANK={local_rank} -> cuda:{device_index}; "
        f"visible_cuda={visible_cuda}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
    )
    return rank, world_size, local_rank, device, pin_info


def build_dataset_lists(opt):
    dataset_directory = opt.data_root
    if not dataset_directory.is_absolute():
        dataset_directory = opt.base_path / dataset_directory
    rain_source, flow_variables, coords, static = [], [], [], []
    static_prefix = "static_features_xy" if opt.static_slope_xy else "static_features"
    for i in tqdm(range(opt.num_trajectories), desc="Scanning trajectories"):
        rain_path = dataset_directory / f"rain_source_traj{i}.npy"
        flow_path = dataset_directory / f"flow_variables_traj{i}.npy"
        coord_path = dataset_directory / f"coords_traj{i}.npy"
        static_path = dataset_directory / f"{static_prefix}_traj{i}.npy"
        has_static = static_path.exists()
        if rain_path.exists() and flow_path.exists() and coord_path.exists():
            rain_source.append(str(rain_path))
            flow_variables.append(str(flow_path))
            coords.append(str(coord_path))
            if opt.use_static_features:
                if not has_static:
                    raise FileNotFoundError(f"Missing static features: {static_path}")
                static.append(str(static_path))
    return rain_source, flow_variables, coords, static, dataset_directory


def select_explicit_split(opt, rain_source, flow_variables, coords, static):
    """Select trajectories by event ID, preserving test and 2013 holdouts."""
    split_path = opt.split_file if opt.split_file.is_absolute() else opt.base_path / opt.split_file
    with split_path.open() as handle:
        split = json.load(handle)
    required = ("train", "validation", "test", "heldout_2013")
    if any(not isinstance(split.get(key), list) for key in required):
        raise ValueError(f"{split_path} must contain train, validation, test, and heldout_2013 lists")
    for key in required:
        ids = split[key]
        if any(not isinstance(i, int) or isinstance(i, bool) for i in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{split_path}: {key} must contain unique integer IDs")
    train_ids, file_valid_ids, test_ids, heldout_ids = (split[key] for key in required)
    if set(file_valid_ids) != set(test_ids):
        raise ValueError(f"{split_path}: validation and test must contain the same IDs")
    for index, left in enumerate(required):
        for right in required[index + 1:]:
            if (left, right) == ("validation", "test"):
                continue
            if set(split[left]) & set(split[right]):
                raise ValueError(f"{split_path}: {left} and {right} must be disjoint")
    valid_ids = (file_valid_ids if opt.validation_split == "file" else
                 test_ids if opt.validation_split == "test" else [])
    if opt.num_train is not None and opt.num_train != len(train_ids):
        raise ValueError(f"--num-train={opt.num_train} differs from {len(train_ids)} train IDs in {split_path}")
    if opt.num_valid is not None and opt.num_valid != len(valid_ids):
        raise ValueError(f"--num-valid={opt.num_valid} differs from {len(valid_ids)} validation IDs")

    available = {}
    for position, rain_path in enumerate(rain_source):
        match = re.fullmatch(r"rain_source_traj(\d+)\.npy", Path(rain_path).name)
        if match is None:
            raise ValueError(f"Could not extract trajectory ID from {rain_path}")
        trajectory_id = int(match.group(1))
        available[trajectory_id] = position
    missing = (set(train_ids) | set(file_valid_ids) | set(test_ids)) - set(available)
    if missing:
        raise FileNotFoundError(f"Trajectories in {split_path} missing from scanned data: {sorted(missing)}")

    def select(ids):
        if not ids:
            return None
        positions = [available[i] for i in ids]
        selected = {
            "u": [rain_source[j] for j in positions],
            "x": [coords[j] for j in positions],
            "y": [flow_variables[j] for j in positions],
            "dt": np.array([dt]),
        }
        if opt.use_static_features:
            selected["s"] = [static[j] for j in positions]
        return selected

    return select(train_ids), select(valid_ids), train_ids, test_ids, heldout_ids


def _print_data_sanity(rain_path, flow_path, coord_path, static_path=None, sample_points=1000):
    rain = np.load(rain_path, mmap_mode="r")
    flow = np.load(flow_path, mmap_mode="r")
    coords = np.load(coord_path, mmap_mode="r")
    static = np.load(static_path, mmap_mode="r") if static_path is not None else None

    print(f"Data sanity: rain {rain.shape} {rain.dtype}")
    print(f"Data sanity: flow {flow.shape} {flow.dtype}")
    print(f"Data sanity: coords {coords.shape} {coords.dtype}")
    if static is not None:
        print(f"Data sanity: static {static.shape} {static.dtype}")

    max_points = min(flow.shape[2], sample_points)
    rng = np.random.default_rng(0)
    idx = rng.choice(flow.shape[2], size=max_points, replace=False)
    rain_slice = np.asarray(rain[:, :1, :min(rain.shape[2], sample_points)], dtype=np.float32)
    flow_slice = np.asarray(flow[:, :1, idx, :], dtype=np.float32)
    coords_slice = np.asarray(coords[:, idx, :], dtype=np.float32)
    static_slice = np.asarray(static[:, idx, :], dtype=np.float32) if static is not None else None

    print(f"Rain sample min/max: {rain_slice.min():.6f}/{rain_slice.max():.6f}")
    print(f"Flow sample min/max: {flow_slice.min():.6f}/{flow_slice.max():.6f}")
    print(f"Coords sample min/max: {coords_slice.min():.6f}/{coords_slice.max():.6f}")
    if static_slice is not None:
        print(f"Static sample min/max: {static_slice.min():.6f}/{static_slice.max():.6f}")


def _maybe_load_channel_normalization(
    enabled: bool,
    mean_path: Path,
    std_path: Path,
    label: str,
    logger,
    rank: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not enabled:
        return None, None

    if not mean_path.exists() or not std_path.exists():
        if rank == 0:
            logger.info(
                f"{label} normalization requested, but normalization files were not both found at "
                f"{mean_path} and {std_path}. Proceeding without {label} normalization."
            )
        return None, None

    mean = np.load(mean_path).astype(np.float32, copy=False)
    std = np.load(std_path).astype(np.float32, copy=False)
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise ValueError(
            f"Normalization files must be matching 1D arrays; got mean {mean.shape} and std {std.shape}"
        )
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)):
        raise ValueError("Normalization files contain non-finite values.")
    if np.any(std <= 0):
        raise ValueError(f"{label} normalization std contains non-positive values.")

    if rank == 0:
        logger.info(f"Applying {label} normalization from {mean_path} and {std_path}")
    return mean, std


def main(opt):
    if opt.smoke_test:
        opt.num_epochs = min(opt.num_epochs, 2)
        opt.sample_indices = min(opt.sample_indices, 2000)
        opt.num_trajectories = min(opt.num_trajectories, 1)

    rank, world_size, local_rank, device, pin_info = setup_ddp(opt)
    opt.device = device
    log = Logger(log_dir=opt.base_path / opt.model_path / opt.log_dir)
    log.info("=======================================================")
    log.info("           Efficient Fourier LDNN Training             ")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")
    if pin_info is not None:
        log.info(f"DDP GPU pinning: {pin_info}; torch device={device}")

    rain_source, flow_variables, coords, static, data_root = build_dataset_lists(opt)
    if rank == 0:
        log.info(f"Data root: {data_root}")
        log.info(f"Trajectories found: {len(rain_source)}")
        if rain_source:
            log.info(f"Example rain file: {rain_source[0]}")
            log.info(f"Example flow file: {flow_variables[0]}")
            log.info(f"Example coords file: {coords[0]}")
            if opt.use_static_features:
                log.info(f"Example static file: {static[0]}")
        if opt.data_sanity and len(rain_source) > 0:
            _print_data_sanity(
                rain_source[0],
                flow_variables[0],
                coords[0],
                static[0] if opt.use_static_features else None,
            )
    if len(rain_source) == 0:
        raise FileNotFoundError(f"No trajectories found under {opt.data_root}")

    y_mean, y_std = _maybe_load_channel_normalization(
        opt.normalize,
        data_root / "mean.npy",
        data_root / "std.npy",
        "flow",
        log,
        rank,
    )
    u_mean, u_std = _maybe_load_channel_normalization(
        opt.normalize_rain,
        data_root / "rain_mean.npy",
        data_root / "rain_std.npy",
        "rain",
        log,
        rank,
    )

    if opt.split_file is not None:
        data_train, data_valid, train_ids, test_ids, heldout_ids = select_explicit_split(
            opt, rain_source, flow_variables, coords, static
        )
        if rank == 0:
            log.info(
                f"Explicit split: {len(train_ids)} train, "
                f"{0 if data_valid is None else len(data_valid['u'])} validation/test, "
                f"{len(heldout_ids)} held-out 2013; "
                f"validation={opt.validation_split}"
            )
    else:
        if opt.validation_split != "file":
            raise ValueError("--validation-split requires --split-file")
        total = len(rain_source)
        num_train = opt.num_train
        num_valid = opt.num_valid
        if num_train is None and num_valid is None:
            num_train = total // 2
            num_valid = total - num_train
        else:
            if num_train is None:
                num_train = total - num_valid
            if num_valid is None:
                num_valid = total - num_train
            if num_train < 0 or num_valid < 0 or num_train + num_valid > total:
                raise ValueError(f"Invalid split: num_train={num_train}, num_valid={num_valid}, total={total}")
        # Preserve the legacy positional split for existing training commands.
        data_train = {
            "u": rain_source[num_valid:num_train + num_valid],
            "x": coords[num_valid:num_train + num_valid],
            "y": flow_variables[num_valid:num_train + num_valid],
            "dt": np.array([dt]),
        }
        data_valid = {
            "u": rain_source[:num_valid],
            "x": coords[:num_valid],
            "y": flow_variables[:num_valid],
            "dt": np.array([dt]),
        }
        if opt.use_static_features:
            data_train["s"] = static[num_valid:num_train + num_valid]
            data_valid["s"] = static[:num_valid]
    if y_mean is not None and y_std is not None:
        data_train["y_mean"] = y_mean
        data_train["y_std"] = y_std
        if data_valid is not None:
            data_valid["y_mean"] = y_mean
            data_valid["y_std"] = y_std
    if u_mean is not None and u_std is not None:
        data_train["u_mean"] = u_mean
        data_train["u_std"] = u_std
        if data_valid is not None:
            data_valid["u_mean"] = u_mean
            data_valid["u_std"] = u_std

    depth_only = not opt.all_vars
    if depth_only:
        data_train["y_idx"] = [0]
        if data_valid is not None:
            data_valid["y_idx"] = [0]
        if rank == 0:
            log.info("Training depth-only (channel 0)")
    if len(data_train["u"]) == 0 or (data_valid is not None and len(data_valid["u"]) == 0):
        raise ValueError("The training or validation split is empty")
    if data_valid is None and opt.scheduler == "plateau":
        raise ValueError("The plateau scheduler requires validation data")

    # Model dimensions
    dim_u = opt.dim_u#39 * 13
    if opt.use_static_features:
        inferred_static_dim = int(np.load(static[0], mmap_mode="r").shape[-1])
        if opt.static_feature_dim != inferred_static_dim and rank == 0:
            log.info(
                f"Overriding static_feature_dim from {opt.static_feature_dim} to inferred value {inferred_static_dim} "
                f"based on {static[0]}"
            )
        opt.static_feature_dim = inferred_static_dim
    dim_x = 2 + (opt.static_feature_dim if opt.use_static_features else 0)
    dim_y = 1 if depth_only else 3

    layer_sizes_dyn = [opt.num_latent_states + dim_u] + opt.NN_dyn_depth * [opt.NN_dyn_width] + [opt.num_latent_states]
    layer_sizes_rec = [opt.num_latent_states + dim_x] + opt.NN_rec_depth * [opt.NN_rec_width] + [dim_y]

    model = EfficientFourierLDNN(
        opt.fourier_mapping_size,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation=opt.activation,
        kernel_initializer=opt.kernel_initializer,
        dropout=opt.dropout,
        chunk_size=opt.chunk_size,
    )
    if opt.rec_finetune:
        for param in model.dyn.parameters():
            param.requires_grad = False
        if rank == 0:
            log.info(
                f"Rec finetune mode enabled. Freezing dyn and training rec+B from base epoch {opt.finetune_base_epoch}."
            )

    model.to(opt.device)
    if world_size > 1:
        ddp_device = opt.device.index if opt.device.type == "cuda" else None
        model = DDP(model, device_ids=[ddp_device], output_device=ddp_device)

    criterion = WetWeightedMSE(alpha=opt.wet_weight_alpha, max_weight=opt.wet_weight_max)
    if rank == 0:
        if opt.wet_weight_alpha > 0:
            log.info(f"Wet-weighted MSE: alpha={opt.wet_weight_alpha}, max={opt.wet_weight_max}")
        else:
            log.info("Using standard MSE loss")

    model_to_optimize = model.module if hasattr(model, "module") else model
    if opt.rec_finetune:
        trainable_parameters = list(model_to_optimize.rec.parameters()) + list(model_to_optimize.B.parameters())
    else:
        trainable_parameters = list(model.parameters())

    optimizer = optim.Adam(trainable_parameters, lr=opt.learning_rate, weight_decay=opt.weight_decay)

    if opt.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_gamma)
    elif opt.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs, eta_min=1e-6)
    elif opt.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_gamma)

    trainer = Trainer_Sample(
        model,
        optimizer,
        criterion,
        batch_size=opt.batch_size,
        device=opt.device,
        lr_scheduler=scheduler,
        grad_clip=opt.grad_clip,
    )

    save_path = opt.base_path / opt.model_path
    save_path.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if opt.rec_finetune:
        start_epoch = _prepare_rec_finetune(
            model=model,
            optimizer=optimizer,
            save_path=save_path,
            rank=rank,
            world_size=world_size,
            logger=log,
            base_epoch=opt.finetune_base_epoch,
        )
    elif opt.resume_latest:
        start_epoch = _resume_from_latest(
            model=model,
            optimizer=optimizer,
            save_path=save_path,
            rank=rank,
            world_size=world_size,
            use_fourier_B=True,
            logger=log,
        )
        if start_epoch >= opt.num_epochs and rank == 0:
            log.info(
                f"Latest checkpoint already reached requested num_epochs={opt.num_epochs}. Nothing to train."
            )
            return

    _init_or_resume_wandb(opt, save_path, start_epoch, rank, log)

    log.info("Starting training...")
    trainer.train(
        data_train,
        data_valid,
        num_epochs=opt.num_epochs,
        start_epoch=start_epoch,
        eval_interval=opt.eval_interval,
        save_path=save_path,
        sample_indices=opt.sample_indices,
        rank=rank,
        rec_checkpoint_prefix="rec_finetune" if opt.rec_finetune else "rec",
        b_checkpoint_prefix="B_finetune" if opt.rec_finetune else "B",
        optimizer_checkpoint_prefix="optimizer_rec_finetune" if opt.rec_finetune else "optimizer",
        save_dyn=not opt.rec_finetune,
    )
    if opt.use_wandb and rank == 0:
        wandb.finish()

    log.info("Training complete.")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    opt = create_training_options()
    main(opt)
