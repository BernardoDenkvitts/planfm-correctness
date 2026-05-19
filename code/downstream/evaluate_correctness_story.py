"""Post-hoc analyses for the downstream plan correctness story."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from code.downstream.config import ANALYSIS_DIR, CORRECTNESS_HEAD_DIR, CORRECTNESS_DATASET_DIR
from code.downstream.features import load_feature_matrix
from code.downstream.baselines import evaluate_baselines


DEFAULT_FAMILIES = [
    "ad_lstm_wl_delta",
    "ad_xgb_wl_delta",
    "dd_lstm_shortest_path_delta",
    "dd_xgb_wl_delta",
]
DEFAULT_SEEDS = [13, 23, 37]
DEFAULT_SPLITS = ["train", "validation", "test-interpolation", "test-extrapolation"]


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if len(y_true) == 0:
        return {
            "num_examples": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
        }
    return {
        "num_examples": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def read_predictions(path: Path) -> dict[str, dict[str, list]]:
    by_split: dict[str, dict[str, list]] = defaultdict(lambda: {"y": [], "preds": []})
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            by_split[split]["y"].append(float(row["true_score"]))
            by_split[split]["preds"].append(float(row["pred_score"]))
    return by_split


def analyze_mlp_results(args) -> list[dict]:
    rows: list[dict] = []
    mlp_root = Path(args.results_dir) if args.results_dir else REGRESSION_HEAD_DIR
    for family in args.families:
        for seed in args.seeds:
            if getattr(args, "experiment", None):
                pred_path = (
                    mlp_root
                    / family
                    / args.experiment
                    / f"seed_{seed}"
                    / "predictions.csv"
                )
            else:
                pred_path = (
                    mlp_root
                    / family
                    / f"source_seed_{seed}"
                    / f"head_seed_{seed}"
                    / "predictions.csv"
                )
            if not pred_path.exists():
                continue
            preds = read_predictions(pred_path)
            for split, values in preds.items():
                metrics = compute_metrics(values["y"], values["preds"])
                metrics.update(
                    {
                        "method": "mlp",
                        "family": family,
                        "seed": seed,
                        "split": split,
                    }
                )
                rows.append(metrics)
    return rows


def load_family_data(family: str, feature_dir: Path, seed: int) -> dict:
    data = {}
    for split in DEFAULT_SPLITS:
        path = feature_dir / family / f"seed_{seed}" / f"{split}.npz"
        if not path.exists():
            continue
        npz = load_feature_matrix(path)
        y_val = npz.get("correctness_scores", npz.get("y"))
        data[split] = {
            "X": np.asarray(npz["X"], dtype=np.float32),
            "y": np.asarray(y_val, dtype=np.float32),
            "feature_names": list(npz["feature_names"]),
        }
    return data


def evaluate_regression_baselines(args) -> list[dict]:
    rows: list[dict] = []
    feature_dir = Path(args.dataset_dir) / "features"
    for seed in args.seeds:
        family_data = load_family_data(args.baseline_reference_family, feature_dir, seed)
        if "train" not in family_data:
            continue
        baseline_rows = evaluate_baselines(family_data, family="baseline", splits=DEFAULT_SPLITS)
        for r in baseline_rows:
            r["seed"] = seed
            r["method"] = r["model"]
            r["num_examples"] = len(family_data.get(r["split"], {}).get("y", []))
            del r["model"]
            rows.append(r)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "family",
        "seed",
        "split",
        "num_examples",
        "MAE",
        "RMSE",
        "R2",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def mean_std(values):
    vals = [float(value) for value in values if not math.isnan(float(value))]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    var = sum((value - mean) ** 2 for value in vals) / len(vals)
    return mean, math.sqrt(var)


def write_summary(path: Path, rows: list[dict]) -> None:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["family"], row["split"])].append(row)

    fields = [
        "method",
        "family",
        "split",
        "num_runs",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
        "R2_mean",
        "R2_std",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            method, family, split = key
            group = grouped[key]
            out = {
                "method": method,
                "family": family,
                "split": split,
                "num_runs": len(group),
            }
            for metric in ["MAE", "RMSE", "R2"]:
                mean, std = mean_std([row[metric] for row in group if metric in row])
                out[f"{metric}_mean"] = mean
                out[f"{metric}_std"] = std
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze downstream correctness experiments.")
    parser.add_argument("--dataset_dir", default=str(CORRECTNESS_DATASET_DIR))
    parser.add_argument("--results_dir", default=str(CORRECTNESS_HEAD_DIR))
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--baseline_reference_family", default="dd_lstm_shortest_path_delta")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--experiment", type=str, help="Experiment folder name to evaluate")
    args = parser.parse_args()

    args.dataset_dir = str(Path(args.dataset_dir).resolve())
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ANALYSIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    mlp_rows = analyze_mlp_results(args)
    baseline_rows = [] if args.skip_baselines else evaluate_regression_baselines(args)
    all_rows = mlp_rows + baseline_rows

    write_rows(output_dir / "correctness_metrics.csv", mlp_rows)
    write_summary(output_dir / "correctness_summary.csv", mlp_rows)
    write_rows(output_dir / "baseline_metrics.csv", baseline_rows)
    write_summary(output_dir / "baseline_summary.csv", baseline_rows)
    write_rows(output_dir / "story_metrics.csv", all_rows)
    write_summary(output_dir / "story_summary.csv", all_rows)
    print(f"Wrote downstream correctness analysis to {output_dir}")


if __name__ == "__main__":
    main()
