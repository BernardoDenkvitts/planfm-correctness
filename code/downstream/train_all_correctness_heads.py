"""Orchestrate the downstream plan correctness experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from code.downstream.config import (
    PROJECT_ROOT,
    RETRAINED_HEAD_DIR,
    CORRECTNESS_HEAD_DIR,
    CORRECTNESS_DATASET_DIR,
)

DEFAULT_FAMILIES = [
    "ad_lstm_wl_delta",
    "ad_xgb_wl_delta",
    "dd_lstm_shortest_path_delta",
    "dd_xgb_wl_delta",
]
DEFAULT_SEEDS = [13, 23, 37]


def run_command(cmd: list[str], cwd: Path) -> None:
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run downstream plan correctness experiments."
    )
    parser.add_argument(
        "--best_params_path", 
        default=str(PROJECT_ROOT / "code" / "downstream" / "experiment_config" / "baselines" / "best_params.json")
    )
    parser.add_argument("--dataset_dir", default=str(CORRECTNESS_DATASET_DIR))
    parser.add_argument("--output_dir", default=str(RETRAINED_HEAD_DIR))
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="cpu")
    parser.add_argument("--experiment", type=str, help="Name of the experiment directory")
    parser.add_argument("--skip_build", action="store_true", help="skip if already trained")
    parser.add_argument(
        "--final",
        action="store_true",
        help=(
            "Save heads to models/correctness_heads/ (tracked by git) instead of "
            "outputs/regression_correctness_heads/ (gitignored). "
            "Use only when experiments are complete and results are final."
        ),
    )
    args = parser.parse_args()

    # --final overrides --output_dir with the final heads directory
    if args.final and args.output_dir == str(RETRAINED_HEAD_DIR):
        args.output_dir = str(CORRECTNESS_HEAD_DIR)
        print("[--final] Saving to final heads directory:", args.output_dir)

    repo_root = PROJECT_ROOT
    args.best_params_path = str(Path(args.best_params_path).resolve())
    args.dataset_dir = str(Path(args.dataset_dir).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    output_dir = Path(args.output_dir)
    best_params_path = Path(args.best_params_path)

    if not best_params_path.exists():
        raise FileNotFoundError(f"Best params file not found: {best_params_path}")
    with open(best_params_path, "r", encoding="utf-8") as f:
        all_best_params = json.load(f)

    for family in args.families:
        if family not in all_best_params:
            print(f"Skipping {family}: not found in best_params.json")
            continue
        best_mlp = all_best_params[family]

        for seed in args.seeds:

            head_dir = output_dir / family / f"source_seed_{seed}" / f"head_seed_{seed}"
            if args.skip_build and (head_dir / "correctness_mlp.pt").exists():
                print(f"Skipping {family} seed {seed}: already trained at {head_dir}")
                continue

            # Build a YAML config for this family × seed and pass it to train_correctness
            cfg = {
                "family":       family,
                "seed":         seed,
                "hidden_dims":  best_mlp["hidden_dims"],
                "dropout":      best_mlp["dropout"],
                "activation":   best_mlp.get("activation", "GELU"),
                "lr":           best_mlp["lr"],
                "weight_decay": best_mlp["weight_decay"],
                "epochs":       args.epochs,
                "batch_size":   args.batch_size,
                "patience":     args.patience,
                "device":       args.device,
                "experiment":   args.experiment,
                "output_dir":   args.output_dir,
                "dataset_dir":  args.dataset_dir,
                "description":  f"Main run — {family} seed {seed}",
            }
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as tmp:
                yaml.dump(cfg, tmp, default_flow_style=False)
                tmp_path = tmp.name

            cmd = [sys.executable, "-m", "code.downstream.train_correctness", tmp_path]
            run_command(cmd, repo_root)

        # Aggregate metrics across seeds
        experiment_name = args.experiment
        experiment_dir = output_dir / family / experiment_name
        all_metrics = []
        for seed in args.seeds:
            metrics_path = experiment_dir / f"seed_{seed}" / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    all_metrics.append(json.load(f))
        
        if all_metrics:
            splits = all_metrics[0].keys()
            mean_metrics = {}
            for split in splits:
                metric_keys = [k for k in all_metrics[0][split]
                               if k not in ("split", "group", "group_value", "num_examples")]
                values = {
                    k: [m[split][k] for m in all_metrics if k in m[split]]
                    for k in metric_keys
                }
                mean_metrics[split] = {
                    k: {
                        "mean": float(np.nanmean(values[k])),
                        "std":  float(np.nanstd(values[k])),
                    }
                    for k in metric_keys if values[k]
                }
            
            with open(experiment_dir / "mean_metrics.json", "w", encoding="utf-8") as f:
                json.dump(mean_metrics, f, indent=2)
            print(f"Aggregated mean_metrics.json for {family} across {len(all_metrics)} seeds.")

    print(f"Downstream correctness experiment complete: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
