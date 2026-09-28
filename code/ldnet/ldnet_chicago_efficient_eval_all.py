"""
Aggregate evaluation script for Efficient Fourier LDNN flood surrogate models.

This script runs inference across all available trajectories in the reduced dataset,
using the same train/test split convention as training, and prints aggregate relative
L2 and R^2 metrics without generating any plots.

Example:
    python ldnet_chicago_efficient_eval_all.py --checkpoint-epoch 539 \
        --device cuda:0 --all-vars --model-path checkpoints/cldnet/ \
        --data-root data/postprocessed/illinois --num-trajectories 120 --num-train 100 --use-static-features \
        --normalize-rain --fourier-mapping-size 10 --num-latent-states 30

    python ldnet_chicago_efficient_eval_all.py --checkpoint-epoch 319         --device cuda:0 --all-vars --model-path checkpoints/texas_normalizedrain_first10test/         --data-root data/postprocessed/texas --num-trajectories 101 --num-train 91 --validation-first-n 10 --normalize-rain --fourier-mapping-size 10 --num-latent-states 30

    python ldnet_chicago_efficient_eval_all.py --checkpoint-epoch 489 \
        --device cuda:0 --all-vars --model-path checkpoints/texas_static_normalizedrain/ \
        --data-root data/postprocessed/texas --num-trajectories 120 --num-train 100 --validation-only \
        --normalize-rain --fourier-mapping-size 10 --num-latent-states 30 --use-static-features
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

repo_path = Path(__file__).resolve().parent
sys.path.append(str(repo_path))

from efficient_fourier_ldnet import EfficientFourierLDNN
from src.logger import Logger

dt = 1


def create_options():
    parser = argparse.ArgumentParser()
    default_base_path = Path(__file__).resolve().parents[2]
    parser.add_argument("--base-path", type=Path, default=default_base_path)
    parser.add_argument("--log-dir", type=Path, default="log")
    parser.add_argument("--name", type=str, default="ldnet_chicago_efficient_eval_all")
    parser.add_argument("--model-path", type=Path, default="checkpoints/ldnet")
    parser.add_argument("--data-root", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--num-trajectories", type=int, default=2)
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-test", type=int, default=None)
    parser.add_argument(
        "--validation-first-n",
        type=int,
        default=None,
        help="Use the first N available trajectories as validation and the remaining ones as train.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        default=False,
        help="Only evaluate the held-out split and skip train evaluation.",
    )
    parser.add_argument("--use-static-features", action="store_true", default=False)
    parser.add_argument("--static-slope-xy", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--normalize-rain", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--num-latent-states", type=int, default=200)
    parser.add_argument("--fourier-mapping-size", type=int, default=32)
    parser.add_argument("--NN-dyn-depth", type=int, default=8)
    parser.add_argument("--NN-dyn-width", type=int, default=50)
    parser.add_argument("--NN-rec-depth", type=int, default=10)
    parser.add_argument("--NN-rec-width", type=int, default=300)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--kernel-initializer", type=str, default="Glorot normal")
    parser.add_argument("--dyn-checkpoint", type=Path, default=None)
    parser.add_argument("--rec-checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-epoch", type=int, default=None)
    parser.add_argument(
        "--rec-finetune",
        action="store_true",
        default=False,
        help="Load the latest rec_finetune/B_finetune checkpoints for rec and B.",
    )
    parser.add_argument(
        "--finetune-path",
        type=Path,
        default=None,
        help="Directory containing rec_finetune_*.ckpt and B_finetune_*.ckpt files. Defaults to --model-path.",
    )
    parser.add_argument("--all-vars", action="store_true", default=False)
    return parser.parse_args()


def _extract_ckpt_epoch(path: Path, prefix: str) -> int | None:
    match = re.fullmatch(rf"{prefix}_(\d+)\.ckpt", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _latest_epoch_for_prefix(save_path: Path, prefix: str) -> int | None:
    epochs = {
        epoch
        for ckpt in save_path.glob(f"{prefix}_*.ckpt")
        if (epoch := _extract_ckpt_epoch(ckpt, prefix)) is not None
    }
    return max(epochs) if epochs else None


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


def _resolve_latest_finetune_dir(opt) -> Path:
    finetune_dir = opt.finetune_path
    if finetune_dir is None:
        finetune_dir = opt.model_path
    if not finetune_dir.is_absolute():
        finetune_dir = opt.base_path / finetune_dir
    return finetune_dir


def _resolve_checkpoint_paths(opt):
    if opt.checkpoint_epoch is not None:
        dyn_ckpt = opt.base_path / opt.model_path / f"dyn_{opt.checkpoint_epoch}.ckpt"
        rec_ckpt = opt.base_path / opt.model_path / f"rec_{opt.checkpoint_epoch}.ckpt"
        B_ckpt = opt.base_path / opt.model_path / f"B_{opt.checkpoint_epoch}.ckpt"
    else:
        dyn_ckpt = opt.dyn_checkpoint
        rec_ckpt = opt.rec_checkpoint
        B_ckpt = None

    if opt.rec_finetune:
        finetune_dir = _resolve_latest_finetune_dir(opt)
        latest_finetune_epoch = _latest_complete_finetune_epoch(finetune_dir)
        if latest_finetune_epoch is None:
            raise FileNotFoundError(
                f"No matching rec_finetune/B_finetune checkpoint pairs found under {finetune_dir}"
            )
        rec_ckpt = finetune_dir / f"rec_finetune_{latest_finetune_epoch}.ckpt"
        B_ckpt = finetune_dir / f"B_finetune_{latest_finetune_epoch}.ckpt"

    return dyn_ckpt, rec_ckpt, B_ckpt


def _infer_rec_output_dim(rec_ckpt: Path) -> int:
    state_dict = torch.load(rec_ckpt, map_location="cpu")
    candidates = []
    for key, value in state_dict.items():
        match = re.fullmatch(r"net\.(\d+)\.weight", key)
        if match is not None and torch.is_tensor(value) and value.ndim == 2:
            candidates.append((int(match.group(1)), int(value.shape[0])))
    if not candidates:
        raise ValueError(f"Could not infer reconstruction output dimension from {rec_ckpt}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _load_checkpoints(model, opt, logger):
    dyn_ckpt, rec_ckpt, B_ckpt = _resolve_checkpoint_paths(opt)
    if dyn_ckpt is None or rec_ckpt is None:
        raise ValueError("Checkpoint paths are required for aggregate evaluation.")

    logger.info(f"Loading dyn checkpoint: {dyn_ckpt}")
    logger.info(f"Loading rec checkpoint: {rec_ckpt}")
    model.dyn.load_state_dict(torch.load(dyn_ckpt, map_location="cpu"))
    model.rec.load_state_dict(torch.load(rec_ckpt, map_location="cpu"))

    if B_ckpt is not None and B_ckpt.exists() and hasattr(model, "B"):
        logger.info(f"Loading B checkpoint: {B_ckpt}")
        model.B.load_state_dict(torch.load(B_ckpt, map_location="cpu"))
    elif hasattr(model, "B"):
        logger.info("B checkpoint missing; Fourier embedding will use current weights.")


def _maybe_load_channel_normalization(
    enabled: bool,
    mean_path: Path,
    std_path: Path,
    label: str,
    logger,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not enabled:
        return None, None

    if not mean_path.exists() or not std_path.exists():
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

    logger.info(f"Applying {label} normalization from {mean_path} and {std_path}")
    return mean, std


def _r2_from_sums(sum_squared_error: float, sum_truth: float, sum_truth_sq: float, count: float) -> float:
    if count <= 0.0:
        return float("nan")
    mean_truth = sum_truth / count
    sum_squared_true = sum_truth_sq - count * mean_truth * mean_truth
    if sum_squared_true <= 0.0:
        return float("nan")
    return float(1.0 - sum_squared_error / sum_squared_true)


def _rmse_from_sums(sum_squared_error: float, count: float) -> float:
    if count <= 0.0:
        return float("nan")
    return float(np.sqrt(sum_squared_error / count))


def _load_reduced_data(data_root: Path, traj_id: int, use_static: bool, static_slope_xy: bool = False):
    flow = np.load(data_root / f"flow_variables_traj{traj_id}.npy", mmap_mode="r")
    coords = np.load(data_root / f"coords_traj{traj_id}.npy", mmap_mode="r")
    rain = np.load(data_root / f"rain_source_traj{traj_id}.npy", mmap_mode="r")
    if use_static:
        static_prefix = "static_features_xy" if static_slope_xy else "static_features"
        static = np.load(data_root / f"{static_prefix}_traj{traj_id}.npy", mmap_mode="r")
        coords = np.concatenate([coords, static], axis=-1)
    return flow, coords, rain


def _list_available_trajectories(data_root: Path, num_trajectories: int, use_static: bool, static_slope_xy: bool) -> list[int]:
    static_prefix = "static_features_xy" if static_slope_xy else "static_features"
    available = []
    for i in range(num_trajectories):
        flow_path = data_root / f"flow_variables_traj{i}.npy"
        coord_path = data_root / f"coords_traj{i}.npy"
        rain_path = data_root / f"rain_source_traj{i}.npy"
        static_ok = True
        if use_static:
            static_ok = (data_root / f"{static_prefix}_traj{i}.npy").exists()
        if flow_path.exists() and coord_path.exists() and rain_path.exists() and static_ok:
            available.append(i)
    return available


def _resolve_split(total: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    if num_train is None and num_test is None:
        num_train = total // 2
        num_test = total - num_train
    else:
        if num_train is None:
            num_train = total - num_test
        if num_test is None:
            num_test = total - num_train
        if num_train < 0 or num_test < 0 or num_train + num_test > total:
            raise ValueError(f"Invalid split: num_train={num_train}, num_test={num_test}, total={total}")
    return num_train, num_test


def _resolve_train_validation_ids(
    available_ids: list[int],
    num_train: int | None,
    num_test: int | None,
    validation_first_n: int | None,
) -> tuple[list[int], list[int], str]:
    if validation_first_n is not None:
        if validation_first_n <= 0:
            raise ValueError(f"validation_first_n must be positive, got {validation_first_n}")
        if validation_first_n >= len(available_ids):
            raise ValueError(
                "validation_first_n must leave at least one training trajectory; "
                f"got validation_first_n={validation_first_n}, total={len(available_ids)}"
            )
        validation_ids = available_ids[:validation_first_n]
        train_ids = available_ids[validation_first_n:]
        return train_ids, validation_ids, "validation"

    num_train, num_test = _resolve_split(len(available_ids), num_train, num_test)
    train_ids = available_ids[:num_train]
    test_ids = available_ids[num_train : num_train + num_test]
    return train_ids, test_ids, "test"


def _evaluate_split(
    model,
    traj_ids: list[int],
    data_root: Path,
    opt,
    depth_only: bool,
    y_mean: np.ndarray | None,
    y_std: np.ndarray | None,
    u_mean: np.ndarray | None,
    u_std: np.ndarray | None,
    split_name: str,
) -> tuple[float, dict[str, float], float, dict[str, float], float, dict[str, float]]:
    sum_squared_error = 0.0
    sum_squared_true = 0.0
    channel_names = ["h"] if depth_only else ["h", "u", "v"]
    channel_error = np.zeros(len(channel_names), dtype=np.float64)
    channel_true = np.zeros(len(channel_names), dtype=np.float64)
    r2_sum_squared_error = 0.0
    r2_sum_truth = 0.0
    r2_sum_truth_sq = 0.0
    r2_count = 0.0
    channel_r2_sum_squared_error = np.zeros(len(channel_names), dtype=np.float64)
    channel_r2_sum_truth = np.zeros(len(channel_names), dtype=np.float64)
    channel_r2_sum_truth_sq = np.zeros(len(channel_names), dtype=np.float64)
    channel_r2_count = np.zeros(len(channel_names), dtype=np.float64)
    rmse_sum_squared_error = 0.0
    rmse_count = 0.0
    channel_rmse_sum_squared_error = np.zeros(len(channel_names), dtype=np.float64)
    channel_rmse_count = np.zeros(len(channel_names), dtype=np.float64)

    with torch.no_grad():
        for traj_id in tqdm(traj_ids, desc=f"Evaluating {split_name}", leave=False):
            flow, coords, rain = _load_reduced_data(
                data_root,
                traj_id,
                opt.use_static_features,
                static_slope_xy=opt.static_slope_xy,
            )
            t_len = flow.shape[1]
            x = np.repeat(coords[:, None, :, :], t_len, axis=1).astype(np.float32, copy=False)

            if depth_only:
                y = flow[..., [0]].astype(np.float32, copy=False)
            else:
                y = flow.astype(np.float32, copy=False)

            u = rain.astype(np.float32, copy=False)
            if u_mean is not None and u_std is not None:
                if u_mean.shape != (u.shape[-1],) or u_std.shape != (u.shape[-1],):
                    raise ValueError(
                        f"Expected rain normalization shape {(u.shape[-1],)}, got {u_mean.shape} and {u_std.shape}"
                    )
                u = (u - u_mean.reshape(1, 1, -1)) / u_std.reshape(1, 1, -1)

            data = {
                "u": torch.from_numpy(u).to(opt.device),
                "x": torch.from_numpy(x).to(opt.device),
                "y": torch.from_numpy(y).to(opt.device),
                "dt": torch.tensor([dt], device=opt.device, dtype=torch.float32),
            }

            pred_np = model(data, opt.device, equilibrium=False, chunk_size=opt.chunk_size).cpu().numpy()
            true_np = y

            if y_mean is not None and y_std is not None:
                dim_y = true_np.shape[-1]
                y_mean_eff = y_mean[[0]] if depth_only else y_mean[:dim_y]
                y_std_eff = y_std[[0]] if depth_only else y_std[:dim_y]
                if y_mean_eff.shape != (dim_y,) or y_std_eff.shape != (dim_y,):
                    raise ValueError(
                        f"Expected flow normalization shape {(dim_y,)}, got {y_mean_eff.shape} and {y_std_eff.shape}"
                    )
                pred_np = pred_np * y_std_eff.reshape(1, 1, 1, -1) + y_mean_eff.reshape(1, 1, 1, -1)

            error = pred_np - true_np
            sum_squared_error += np.sqrt(float(np.sum(error ** 2)))
            sum_squared_true += np.sqrt(float(np.sum(true_np ** 2)))
            for ch in range(len(channel_names)):
                channel_error[ch] += np.sqrt(float(np.sum(error[..., ch] ** 2)))
                channel_true[ch] += np.sqrt(float(np.sum(true_np[..., ch] ** 2)))

            valid = np.isfinite(pred_np) & np.isfinite(true_np)
            if np.any(valid):
                truth_valid = np.where(valid, true_np, 0.0)
                error_valid = np.where(valid, error, 0.0)
                r2_sum_squared_error += float(np.sum(error_valid ** 2))
                r2_sum_truth += float(np.sum(truth_valid))
                r2_sum_truth_sq += float(np.sum(truth_valid ** 2))
                r2_count += float(np.sum(valid))
                rmse_sum_squared_error += float(np.sum(error_valid ** 2))
                rmse_count += float(np.sum(valid))
                for ch in range(len(channel_names)):
                    ch_valid = valid[..., ch]
                    if np.any(ch_valid):
                        ch_truth_valid = np.where(ch_valid, true_np[..., ch], 0.0)
                        ch_error_valid = np.where(ch_valid, error[..., ch], 0.0)
                        channel_r2_sum_squared_error[ch] += float(np.sum(ch_error_valid ** 2))
                        channel_r2_sum_truth[ch] += float(np.sum(ch_truth_valid))
                        channel_r2_sum_truth_sq[ch] += float(np.sum(ch_truth_valid ** 2))
                        channel_r2_count[ch] += float(np.sum(ch_valid))
                        channel_rmse_sum_squared_error[ch] += float(np.sum(ch_error_valid ** 2))
                        channel_rmse_count[ch] += float(np.sum(ch_valid))

    rel = float((sum_squared_error / sum_squared_true)) if sum_squared_true > 0 else float("nan")
    rel_by_channel = {
        channel_names[ch]: (
            float((channel_error[ch] / channel_true[ch])) if channel_true[ch] > 0 else float("nan")
        )
        for ch in range(len(channel_names))
    }
    r2 = _r2_from_sums(r2_sum_squared_error, r2_sum_truth, r2_sum_truth_sq, r2_count)
    r2_by_channel = {
        channel_names[ch]: _r2_from_sums(
            channel_r2_sum_squared_error[ch],
            channel_r2_sum_truth[ch],
            channel_r2_sum_truth_sq[ch],
            channel_r2_count[ch],
        )
        for ch in range(len(channel_names))
    }
    rmse = _rmse_from_sums(rmse_sum_squared_error, rmse_count)
    rmse_by_channel = {
        channel_names[ch]: _rmse_from_sums(
            channel_rmse_sum_squared_error[ch],
            channel_rmse_count[ch],
        )
        for ch in range(len(channel_names))
    }
    return rel, rel_by_channel, r2, r2_by_channel, rmse, rmse_by_channel


def main(opt):
    opt.device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    log = Logger(log_dir=opt.base_path / opt.model_path / opt.log_dir)
    log.info("=======================================================")
    log.info("             Aggregate LDNN Evaluation                 ")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    data_root = opt.data_root
    if not data_root.is_absolute():
        data_root = opt.base_path / data_root

    available_ids = _list_available_trajectories(
        data_root,
        opt.num_trajectories,
        opt.use_static_features,
        opt.static_slope_xy,
    )
    if not available_ids:
        raise FileNotFoundError(f"No trajectories found under {data_root}")
    train_ids, eval_ids, eval_split_name = _resolve_train_validation_ids(
        available_ids,
        opt.num_train,
        opt.num_test,
        opt.validation_first_n,
    )
    if not eval_ids:
        raise ValueError(f"Empty split: {eval_split_name}={len(eval_ids)}")
    if not opt.validation_only and not train_ids:
        raise ValueError(f"Empty split: train={len(train_ids)}, {eval_split_name}={len(eval_ids)}")

    log.info(f"Data root: {data_root}")
    log.info(f"Trajectories found: {len(available_ids)}")
    if opt.validation_only:
        log.info("Train evaluation: skipped (--validation-only)")
    else:
        log.info(f"Train trajectories: {len(train_ids)}")
    log.info(f"{eval_split_name.capitalize()} trajectories: {len(eval_ids)}")
    if opt.validation_only:
        log.info(f"Example trajectory id: {eval_split_name}={eval_ids[0]}")
    else:
        log.info(f"Example trajectory ids: train={train_ids[0]}, {eval_split_name}={eval_ids[0]}")

    y_mean, y_std = _maybe_load_channel_normalization(
        opt.normalize,
        data_root / "mean.npy",
        data_root / "std.npy",
        "flow",
        log,
    )
    u_mean, u_std = _maybe_load_channel_normalization(
        opt.normalize_rain,
        data_root / "rain_mean.npy",
        data_root / "rain_std.npy",
        "rain",
        log,
    )

    _dyn_ckpt, rec_ckpt, _B_ckpt = _resolve_checkpoint_paths(opt)
    if rec_ckpt is None or not rec_ckpt.exists():
        raise FileNotFoundError(f"Missing rec checkpoint: {rec_ckpt}")
    dim_y_ckpt = _infer_rec_output_dim(rec_ckpt)
    if dim_y_ckpt not in (1, 3):
        raise ValueError(f"Unexpected checkpoint output dimension {dim_y_ckpt}; expected 1 or 3.")
    depth_only = dim_y_ckpt == 1
    requested_depth_only = not opt.all_vars
    if depth_only != requested_depth_only:
        requested = "depth-only (--all-vars off)" if requested_depth_only else "all-vars (--all-vars on)"
        inferred = "depth-only" if depth_only else "all-vars"
        log.info(
            f"Requested {requested} but checkpoint rec head outputs {dim_y_ckpt} channel(s). "
            f"Using {inferred} to match checkpoint."
        )

    flow0, coords0, rain0 = _load_reduced_data(
        data_root,
        available_ids[0],
        opt.use_static_features,
        static_slope_xy=opt.static_slope_xy,
    )
    dim_u = int(rain0.shape[-1])
    dim_x = int(coords0.shape[-1])
    dim_y = 1 if depth_only else min(int(flow0.shape[-1]), dim_y_ckpt)

    layer_sizes_dyn = [opt.num_latent_states + dim_u] + opt.NN_dyn_depth * [opt.NN_dyn_width] + [opt.num_latent_states]
    layer_sizes_rec = [opt.num_latent_states + dim_x] + opt.NN_rec_depth * [opt.NN_rec_width] + [dim_y]
    model = EfficientFourierLDNN(
        opt.fourier_mapping_size,
        layer_sizes_dyn,
        layer_sizes_rec,
        activation=opt.activation,
        kernel_initializer=opt.kernel_initializer,
        chunk_size=opt.chunk_size,
    )
    _load_checkpoints(model, opt, log)
    model.to(opt.device)
    model.eval()

    eval_rel, eval_rel_by_channel, eval_r2, eval_r2_by_channel, eval_rmse, eval_rmse_by_channel = _evaluate_split(
        model, eval_ids, data_root, opt, depth_only, y_mean, y_std, u_mean, u_std, eval_split_name
    )

    if not opt.validation_only:
        train_rel, train_rel_by_channel, train_r2, train_r2_by_channel, train_rmse, train_rmse_by_channel = _evaluate_split(
            model, train_ids, data_root, opt, depth_only, y_mean, y_std, u_mean, u_std, "train"
        )
        print(f"Train aggregate relative L2: {train_rel:.6f}")
        for name, value in train_rel_by_channel.items():
            print(f"Train aggregate relative L2 {name}: {value:.6f}")
        print(f"Train aggregate R^2: {train_r2:.6f}")
        for name, value in train_r2_by_channel.items():
            print(f"Train aggregate R^2 {name}: {value:.6f}")
        print(f"Train aggregate RMSE: {train_rmse:.6f}")
        for name, value in train_rmse_by_channel.items():
            print(f"Train aggregate RMSE {name}: {value:.6f}")
    print(f"{eval_split_name.capitalize()} aggregate relative L2: {eval_rel:.6f}")
    for name, value in eval_rel_by_channel.items():
        print(f"{eval_split_name.capitalize()} aggregate relative L2 {name}: {value:.6f}")
    print(f"{eval_split_name.capitalize()} aggregate R^2: {eval_r2:.6f}")
    for name, value in eval_r2_by_channel.items():
        print(f"{eval_split_name.capitalize()} aggregate R^2 {name}: {value:.6f}")
    print(f"{eval_split_name.capitalize()} aggregate RMSE: {eval_rmse:.6f}")
    for name, value in eval_rmse_by_channel.items():
        print(f"{eval_split_name.capitalize()} aggregate RMSE {name}: {value:.6f}")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    opt = create_options()
    main(opt)
