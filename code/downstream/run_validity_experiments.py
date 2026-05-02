"""Orchestrate the downstream frozen-transition validity experiment."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from code.downstream.families import resolve_source_families
from code.downstream.config import (
    DATA_DIR,
    DOMAINS,
    PROJECT_ROOT,
    RETRAINED_HEAD_DIR,
    SOURCE_MODEL_DIR,
    SPLITS_EVAL,
    VALIDITY_DATASET_DIR,
)


DEFAULT_SPLITS = ["train", *SPLITS_EVAL]
DEFAULT_SEEDS = [13, 23, 37]


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().isoformat() + "Z"
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n[{stamp}] {' '.join(cmd)}\n")
        print(" ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=log,
            text=True,
        )
        log.write(f"[exit_code] {result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run downstream frozen-transition validity experiments."
    )
    parser.add_argument("--run_root", default=str(SOURCE_MODEL_DIR))
    parser.add_argument("--source_data_dir", default=str(DATA_DIR))
    parser.add_argument(
        "--output_dir",
        default=str(VALIDITY_DATASET_DIR),
    )
    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--source_families",
        nargs="+",
        default=["weighted_best"],
    )
    parser.add_argument("--negative_ratio", type=int, default=4)
    parser.add_argument("--candidate_seed", type=int, default=2026)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--val_path", default=None)
    parser.add_argument(
        "--labeler",
        choices=["val", "internal"],
        default="val",
        help="'internal' is for local smoke tests when VAL is unavailable.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="cpu")
    parser.add_argument("--xgb_n_jobs", type=int, default=1)
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument(
        "--rebuild_candidates",
        action="store_true",
        help=(
            "Regenerate candidates from raw state trajectories. The included "
            "workflow uses the checked-in candidate files."
        ),
    )
    parser.add_argument("--overwrite_features", action="store_true")
    parser.add_argument("--mlp_epochs", type=int, default=100)
    parser.add_argument("--mlp_batch_size", type=int, default=64)
    parser.add_argument("--mlp_device", choices=["auto", "cuda", "mps", "cpu"], default="cpu")
    args = parser.parse_args()

    repo_root = PROJECT_ROOT
    args.run_root = str(Path(args.run_root).resolve())
    args.source_data_dir = str(Path(args.source_data_dir).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    output_dir = Path(args.output_dir)
    log_path = output_dir / "logs" / "commands.log"
    families = resolve_source_families(args.source_families)

    if not args.skip_build:
        build_cmd = [
            sys.executable,
            "-m",
            "code.downstream.build_validity_dataset",
            "--run_root",
            args.run_root,
            "--source_data_dir",
            args.source_data_dir,
            "--output_dir",
            args.output_dir,
            "--domains",
            *args.domains,
            "--splits",
            *args.splits,
            "--seeds",
            *[str(seed) for seed in args.seeds],
            "--source_families",
            *[family.family_id for family in families],
            "--negative_ratio",
            str(args.negative_ratio),
            "--candidate_seed",
            str(args.candidate_seed),
            "--labeler",
            args.labeler,
            "--device",
            args.device,
            "--xgb_n_jobs",
            str(args.xgb_n_jobs),
        ]
        if args.max_problems is not None:
            build_cmd.extend(["--max_problems", str(args.max_problems)])
        if args.val_path:
            build_cmd.extend(["--val_path", args.val_path])
        if not args.rebuild_candidates:
            build_cmd.append("--skip_candidates")
        if args.overwrite_features:
            build_cmd.append("--overwrite_features")
        run_command(build_cmd, repo_root, log_path)

    for family in families:
        for seed in args.seeds:
            train_cmd = [
                sys.executable,
                "-m",
                "code.downstream.train_validity",
                "--dataset_dir",
                args.output_dir,
                "--family",
                family.family_id,
                "--source_seed",
                str(seed),
                "--seed",
                str(seed),
                "--epochs",
                str(args.mlp_epochs),
                "--batch_size",
                str(args.mlp_batch_size),
                "--device",
                args.mlp_device,
                "--output_dir",
                str(RETRAINED_HEAD_DIR),
            ]
            run_command(train_cmd, repo_root, log_path)

    print(f"Downstream validity experiment complete: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
