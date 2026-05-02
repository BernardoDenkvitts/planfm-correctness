"""Post-hoc analyses for the downstream plan-validity story."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from code.downstream.config import ANALYSIS_DIR, PRETRAINED_HEAD_DIR, VALIDITY_DATASET_DIR
from code.downstream.features import load_feature_matrix


DEFAULT_FAMILIES = [
    "ad_lstm_wl_delta",
    "ad_xgb_wl_delta",
    "dd_lstm_shortest_path_delta",
    "dd_xgb_wl_delta",
]
DEFAULT_SEEDS = [13, 23, 37]
DEFAULT_SPLITS = ["train", "validation", "test-interpolation", "test-extrapolation"]


def compute_metrics(y_true, probs, threshold=0.5) -> dict:
    labels = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(probs, dtype=np.float64)
    preds = (scores >= threshold).astype(np.int64)
    unique = np.unique(labels)
    auroc = float("nan")
    auprc = float("nan")
    bal_acc = float("nan")
    if len(unique) == 2:
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
        bal_acc = float(balanced_accuracy_score(labels, preds))
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "num_examples": int(len(labels)),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)) if len(labels) else float("nan"),
        "balanced_accuracy": bal_acc,
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_threshold(y_true, probs, objective: str) -> float:
    """Choose a threshold on validation predictions."""
    scores = np.asarray(probs, dtype=np.float64)
    candidates = sorted(set([0.0, 0.5, 1.0] + scores.tolist()))
    best_threshold = 0.5
    best_score = -float("inf")
    for threshold in candidates:
        metrics = compute_metrics(y_true, probs, threshold)
        score = float(metrics[objective])
        if math.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def read_predictions(path: Path) -> dict[str, dict[str, list]]:
    by_split: dict[str, dict[str, list]] = defaultdict(lambda: {"y": [], "probs": []})
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            by_split[split]["y"].append(int(row["label_valid"]))
            by_split[split]["probs"].append(float(row["prob_valid"]))
    return by_split


def analyze_mlp_thresholds(args) -> list[dict]:
    rows: list[dict] = []
    mlp_root = Path(args.mlp_results_dir) if args.mlp_results_dir else PRETRAINED_HEAD_DIR
    for family in args.families:
        for seed in args.seeds:
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
            if "validation" not in preds:
                continue
            threshold = tune_threshold(
                preds["validation"]["y"],
                preds["validation"]["probs"],
                args.threshold_objective,
            )
            for split, values in preds.items():
                metrics = compute_metrics(values["y"], values["probs"], threshold)
                metrics.update(
                    {
                        "method": f"mlp_threshold_tuned_{args.threshold_objective}",
                        "family": family,
                        "seed": seed,
                        "split": split,
                    }
                )
                rows.append(metrics)
    return rows


def evaluate_baselines(args) -> list[dict]:
    rows: list[dict] = []
    for seed in args.seeds:
        split_data = {
            split: load_feature_matrix(
                Path(args.dataset_dir)
                / "features"
                / args.baseline_reference_family
                / f"seed_{seed}"
                / f"{split}.npz"
            )
            for split in DEFAULT_SPLITS
        }
        rows.extend(evaluate_majority(seed, split_data))
        rows.extend(evaluate_plan_length_logreg(seed, split_data))
        rows.extend(evaluate_corruption_diagnostic(seed, split_data))
    return rows


def evaluate_majority(seed: int, split_data: dict) -> list[dict]:
    train_y = np.asarray(split_data["train"]["y"], dtype=np.int64)
    positive_rate = float(train_y.mean())
    majority_prob = 1.0 if positive_rate >= 0.5 else 0.0
    rows = []
    for split, data in split_data.items():
        y = np.asarray(data["y"], dtype=np.int64)
        probs = np.full(len(y), majority_prob, dtype=np.float64)
        metrics = compute_metrics(y, probs, threshold=0.5)
        metrics.update(
            {
                "method": "majority",
                "family": "baseline",
                "seed": seed,
                "split": split,
            }
        )
        rows.append(metrics)
    return rows


def evaluate_plan_length_logreg(seed: int, split_data: dict) -> list[dict]:
    feature_names = [str(name) for name in split_data["train"]["feature_names"]]
    cols = [
        idx
        for idx, name in enumerate(feature_names)
        if name in {"plan_len", "log_plan_len", "plan_to_budget_ratio"}
    ]
    if not cols:
        raise RuntimeError("Plan-length feature columns were not found.")

    train_X = np.asarray(split_data["train"]["X"], dtype=np.float32)[:, cols]
    train_y = np.asarray(split_data["train"]["y"], dtype=np.int64)
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    model.fit(train_X, train_y)

    rows = []
    for split, data in split_data.items():
        X = scaler.transform(np.asarray(data["X"], dtype=np.float32)[:, cols])
        y = np.asarray(data["y"], dtype=np.int64)
        probs = model.predict_proba(X)[:, 1]
        metrics = compute_metrics(y, probs, threshold=0.5)
        metrics.update(
            {
                "method": "plan_length_logreg",
                "family": "baseline",
                "seed": seed,
                "split": split,
            }
        )
        rows.append(metrics)
    return rows


def evaluate_corruption_diagnostic(seed: int, split_data: dict) -> list[dict]:
    train_types = sorted(set(str(item) for item in split_data["train"]["corruption_types"]))

    def one_hot(data):
        values = [str(item) for item in data["corruption_types"]]
        arr = np.zeros((len(values), len(train_types)), dtype=np.float32)
        index = {name: idx for idx, name in enumerate(train_types)}
        for row_idx, value in enumerate(values):
            if value in index:
                arr[row_idx, index[value]] = 1.0
        return arr

    train_X = one_hot(split_data["train"])
    train_y = np.asarray(split_data["train"]["y"], dtype=np.int64)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    model.fit(train_X, train_y)

    rows = []
    for split, data in split_data.items():
        y = np.asarray(data["y"], dtype=np.int64)
        probs = model.predict_proba(one_hot(data))[:, 1]
        metrics = compute_metrics(y, probs, threshold=0.5)
        metrics.update(
            {
                "method": "corruption_type_diagnostic",
                "family": "diagnostic",
                "seed": seed,
                "split": split,
            }
        )
        rows.append(metrics)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "family",
        "seed",
        "split",
        "num_examples",
        "positive_rate",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "auroc",
        "auprc",
        "f1",
        "precision",
        "recall",
        "tn",
        "fp",
        "fn",
        "tp",
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
        "accuracy_mean",
        "accuracy_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "auroc_mean",
        "auroc_std",
        "auprc_mean",
        "auprc_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
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
            for metric in ["accuracy", "balanced_accuracy", "auroc", "auprc", "f1", "precision", "recall"]:
                mean, std = mean_std([row[metric] for row in group])
                out[f"{metric}_mean"] = mean
                out[f"{metric}_std"] = std
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze downstream validity experiments.")
    parser.add_argument("--dataset_dir", default=str(VALIDITY_DATASET_DIR))
    parser.add_argument(
        "--mlp_results_dir",
        default=None,
        help="Optional alternate MLP results directory, e.g. an ablation folder.",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--threshold_objective", choices=["balanced_accuracy", "f1"], default="balanced_accuracy")
    parser.add_argument("--baseline_reference_family", default="ad_lstm_wl_delta")
    parser.add_argument("--skip_mlp_thresholds", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    args = parser.parse_args()

    args.dataset_dir = str(Path(args.dataset_dir).resolve())
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ANALYSIS_DIR

    threshold_rows = [] if args.skip_mlp_thresholds else analyze_mlp_thresholds(args)
    baseline_rows = [] if args.skip_baselines else evaluate_baselines(args)
    all_rows = threshold_rows + baseline_rows

    write_rows(output_dir / "tuned_threshold_metrics.csv", threshold_rows)
    write_summary(output_dir / "tuned_threshold_summary.csv", threshold_rows)
    write_rows(output_dir / "baseline_metrics.csv", baseline_rows)
    write_summary(output_dir / "baseline_summary.csv", baseline_rows)
    write_rows(output_dir / "story_metrics.csv", all_rows)
    write_summary(output_dir / "story_summary.csv", all_rows)
    print(f"Wrote downstream story analysis to {output_dir}")


if __name__ == "__main__":
    main()
