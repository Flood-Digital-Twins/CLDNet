#!/usr/bin/env python3
"""
Bundle the LD-ENSF comparison pipeline into one command.

By default this will:
1. Reuse existing datasets/checkpoints when they already exist.
2. Create and train any missing observation-count variants.
3. Rerun the LD-ENSF test for every requested observation count.

The default sweep covers:
    2000, 500, 200, 50, 10, 5, 2 observation points

The 2000-point run reuses the existing unsuffixed dataset/checkpoint paths.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PYTHON = Path(sys.executable)


@dataclass(frozen=True)
class SweepPaths:
    count: int
    dataset_path: Path
    save_dir: Path
    save_name: str
    checkpoint_path: Path
    output_dir: Path


def _resolve_path(base_path: Path, path: Path) -> Path:
    return path if path.is_absolute() else base_path / path


def _paths_for_count(base_path: Path, count: int) -> SweepPaths:
    if count == 2000:
        dataset_path = base_path / "data/postprocessed/illinois/observation_ldensf_dataset.pth"
        save_dir = base_path / "checkpoints/ldensf_lstm"
        save_name = "lstm_ldensf.ckpt"
        output_dir = base_path / "ldensf_results_t120_2000"
    else:
        dataset_path = base_path / f"data/postprocessed/illinois/observation_ldensf_dataset_{count}.pth"
        save_dir = base_path / f"checkpoints/ldensf_lstm_{count}"
        save_name = f"lstm_ldensf_{count}.ckpt"
        output_dir = base_path / f"ldensf_results_t120_{count}"

    return SweepPaths(
        count=count,
        dataset_path=dataset_path,
        save_dir=save_dir,
        save_name=save_name,
        checkpoint_path=save_dir / save_name,
        output_dir=output_dir,
    )


def _flag_args(enabled: bool, positive_flag: str, negative_flag: str | None = None) -> list[str]:
    if enabled:
        return [positive_flag]
    if negative_flag is not None:
        return [negative_flag]
    return []


def _run_step(label: str, cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def create_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dataset creation, encoder training, and LD-ENSF testing for multiple observation counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-path", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=Path("data/postprocessed/illinois"))
    parser.add_argument("--model-path", type=Path, default=Path("checkpoints/cldnet"))
    parser.add_argument("--checkpoint-epoch", type=int, default=539)
    parser.add_argument("--traj-id", type=int, default=120)
    parser.add_argument("--background-traj-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--da-seed", type=int, default=0)
    parser.add_argument("--obs-noise-std", type=float, default=0.05)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--depth-threshold", type=float, default=3.0)
    parser.add_argument("--train-max-traj-id", type=int, default=100)
    parser.add_argument("--valid-max-traj-id", type=int, default=117)
    parser.add_argument("--test-max-traj-id", type=int, default=120)
    parser.add_argument(
        "--counts",
        type=int,
        nargs="*",
        default=[2000, 500, 200, 50, 10, 5, 2],
        help="Observation-point counts to run in order.",
    )
    static_group = parser.add_mutually_exclusive_group()
    static_group.add_argument("--use-static-features", dest="use_static_features", action="store_true")
    static_group.add_argument("--no-static-features", dest="use_static_features", action="store_false")
    parser.set_defaults(use_static_features=True)
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument("--normalize", dest="normalize", action="store_true")
    norm_group.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Recreate datasets and retrain encoders even if the target files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the commands without executing them.",
    )
    return parser.parse_args()


def main(opt: argparse.Namespace) -> None:
    base_path = opt.base_path.resolve()
    python_executable = _resolve_path(base_path, opt.python_executable)
    data_root = _resolve_path(base_path, opt.data_root)
    model_path = _resolve_path(base_path, opt.model_path)

    if not python_executable.exists():
        raise FileNotFoundError(f"Python executable not found: {python_executable}")

    counts = list(dict.fromkeys(opt.counts))
    if not counts:
        raise ValueError("At least one observation count is required.")

    print(f"Base path: {base_path}")
    print(f"Python executable: {python_executable}")
    print(f"Model path: {model_path}")
    print(f"Data root: {data_root}")
    print(f"Counts: {counts}")
    print(f"Overwrite missing artifacts: {opt.overwrite}")
    print(f"Dry run: {opt.dry_run}")

    for count in counts:
        paths = _paths_for_count(base_path, count)
        print(f"\n######################################################")
        print(f"# Observation points: {count}")
        print(f"######################################################")

        if opt.overwrite or not paths.dataset_path.exists():
            dataset_cmd = [
                str(python_executable),
                str(Path(__file__).resolve().with_name("create_ldensf_dataset.py")),
                "--base-path",
                str(base_path),
                "--data-root",
                str(data_root),
                "--model-path",
                str(model_path),
                "--checkpoint-epoch",
                str(opt.checkpoint_epoch),
                "--output-path",
                str(paths.dataset_path),
                "--num-observation-points",
                str(count),
                "--depth-threshold",
                str(opt.depth_threshold),
                "--selection-seed",
                str(opt.selection_seed),
                "--train-max-traj-id",
                str(opt.train_max_traj_id),
                "--valid-max-traj-id",
                str(opt.valid_max_traj_id),
                "--test-max-traj-id",
                str(opt.test_max_traj_id),
            ]
            dataset_cmd += _flag_args(opt.use_static_features, "--use-static-features", "--no-static-features")
            _run_step(f"Create dataset ({count} points)", dataset_cmd, cwd=base_path, dry_run=opt.dry_run)
        else:
            print(f"Dataset already exists, skipping: {paths.dataset_path}")

        if opt.overwrite or not paths.checkpoint_path.exists():
            train_cmd = [
                str(python_executable),
                str(Path(__file__).resolve().with_name("train_ldensf_lstm.py")),
                "--base-path",
                str(base_path),
                "--dataset-path",
                str(paths.dataset_path),
                "--save-dir",
                str(paths.save_dir),
                "--save-name",
                paths.save_name,
                "--obs-noise-std",
                str(opt.obs_noise_std),
            ]
            train_cmd += _flag_args(opt.normalize, "--normalize", "--no-normalize")
            _run_step(f"Train encoder ({count} points)", train_cmd, cwd=base_path, dry_run=opt.dry_run)
        else:
            print(f"Checkpoint already exists, skipping training: {paths.checkpoint_path}")

        test_cmd = [
            str(python_executable),
            str(Path(__file__).resolve().with_name("test_ldensf_assimilation.py")),
            "--base-path",
            str(base_path),
            "--data-root",
            str(data_root),
            "--dataset-path",
            str(paths.dataset_path),
            "--model-path",
            str(model_path),
            "--checkpoint-epoch",
            str(opt.checkpoint_epoch),
            "--encoder-checkpoint",
            str(paths.checkpoint_path),
            "--output-dir",
            str(paths.output_dir),
            "--traj-id",
            str(opt.traj_id),
            "--background-traj-id",
            str(opt.background_traj_id),
            "--seed",
            str(opt.seed),
            "--da-seed",
            str(opt.da_seed),
            "--obs-noise-std",
            str(opt.obs_noise_std),
        ]
        test_cmd += _flag_args(opt.use_static_features, "--use-static-features", "--no-static-features")
        _run_step(f"Test LD-ENSF ({count} points)", test_cmd, cwd=base_path, dry_run=opt.dry_run)

        print(f"Finished count {count}:")
        print(f"  dataset: {paths.dataset_path}")
        print(f"  checkpoint: {paths.checkpoint_path}")
        print(f"  results: {paths.output_dir}")


if __name__ == "__main__":
    main(create_options())
