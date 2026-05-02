"""Train lightweight MLP heads for downstream plan-validity classification."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
from torch.utils.data import DataLoader, TensorDataset

from code.downstream.config import RETRAINED_HEAD_DIR, VALIDITY_DATASET_DIR
from code.downstream.features import load_feature_matrix


DEFAULT_EVAL_SPLITS = ["validation", "test-interpolation", "test-extrapolation"]


class ValidityMLP(nn.Module):
    """Small binary classifier over frozen transition-model summaries."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device_arg == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def load_split(dataset_dir: str | Path, family: str, seed: int, split: str) -> dict:
    path = Path(dataset_dir) / "features" / family / f"seed_{seed}" / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature split: {path}")
    return load_feature_matrix(path)


def standardize(train_X: np.ndarray, *arrays: np.ndarray):
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return [(arr - mean) / std for arr in arrays], mean, std


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train(args) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_data = load_split(args.dataset_dir, args.family, args.source_seed, "train")
    val_data = load_split(args.dataset_dir, args.family, args.source_seed, "validation")
    eval_data = {
        split: load_split(args.dataset_dir, args.family, args.source_seed, split)
        for split in args.eval_splits
    }

    feature_names = [str(name) for name in train_data["feature_names"]]
    keep_mask = build_feature_keep_mask(feature_names, args.exclude_feature_patterns)
    filtered_feature_names = [name for name, keep in zip(feature_names, keep_mask) if keep]

    train_X = np.asarray(train_data["X"], dtype=np.float32)[:, keep_mask]
    train_y = np.asarray(train_data["y"], dtype=np.int64)
    val_X = np.asarray(val_data["X"], dtype=np.float32)[:, keep_mask]
    val_y = np.asarray(val_data["y"], dtype=np.int64)

    if train_X.shape[0] == 0:
        raise RuntimeError("Training feature matrix is empty.")
    if len(np.unique(train_y)) < 2:
        raise RuntimeError("Training labels contain only one class.")

    arrays = [train_X, val_X] + [
        np.asarray(data["X"], dtype=np.float32)[:, keep_mask]
        for data in eval_data.values()
    ]
    standardized, mean, std = standardize(train_X, *arrays)
    train_X = standardized[0]
    val_X = standardized[1]
    eval_X_by_split = {
        split: standardized[idx + 2]
        for idx, split in enumerate(eval_data)
    }

    model = ValidityMLP(
        input_dim=train_X.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    positives = float(train_y.sum())
    negatives = float(len(train_y) - train_y.sum())
    pos_weight_value = negatives / positives if positives > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_loader(train_X, train_y, args.batch_size, shuffle=True)
    val_loader = make_loader(val_X, val_y, args.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience_left = args.patience
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if args.verbose:
            print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    split_predictions = {}
    split_metrics = {}
    all_eval = {
        "train": (train_X, train_y, train_data),
        "validation": (val_X, val_y, val_data),
    }
    for split, data in eval_data.items():
        all_eval[split] = (
            eval_X_by_split[split],
            np.asarray(data["y"], dtype=np.int64),
            data,
        )

    for split, (X, y, data) in all_eval.items():
        probs = predict_probs(model, X, args.batch_size, device)
        metrics = compute_metrics(y, probs)
        metrics.update({"split": split, "group": "overall", "group_value": "overall"})
        split_metrics[split] = metrics
        split_predictions[split] = (probs, data)

    output_dir = Path(args.output_dir) / args.family / f"source_seed_{args.source_seed}" / f"head_seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(
        output_dir=output_dir,
        model=model,
        mean=mean,
        std=std,
        args=args,
        feature_names=[str(name) for name in train_data["feature_names"]],
        kept_feature_names=filtered_feature_names,
        excluded_feature_patterns=args.exclude_feature_patterns,
        history=history,
        split_metrics=split_metrics,
        split_predictions=split_predictions,
    )
    print(f"Wrote downstream validity outputs to {output_dir}")
    return split_metrics


def build_feature_keep_mask(
    feature_names: list[str],
    exclude_patterns: list[str] | None,
) -> np.ndarray:
    """Return a boolean mask excluding feature names containing any pattern."""
    patterns = [pattern.lower() for pattern in (exclude_patterns or []) if pattern]
    if not patterns:
        return np.ones(len(feature_names), dtype=bool)
    keep = []
    for name in feature_names:
        lowered = name.lower()
        keep.append(not any(pattern in lowered for pattern in patterns))
    mask = np.asarray(keep, dtype=bool)
    if not mask.any():
        raise ValueError("Feature exclusion removed every feature.")
    return mask


def evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            losses.append(float(criterion(model(xb), yb).item()))
    return float(np.mean(losses)) if losses else 0.0


def predict_probs(model, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=batch_size)
    probs = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            probs.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.asarray([], dtype=np.float32)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs >= 0.5).astype(np.int64)
    labels = np.asarray(y_true, dtype=np.int64)
    unique = np.unique(labels)
    roc_auc = float("nan")
    auprc = float("nan")
    if len(unique) == 2:
        roc_auc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "num_examples": int(len(labels)),
        "positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "accuracy": float(accuracy_score(labels, preds)) if len(labels) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds))
        if len(unique) == 2
        else float("nan"),
        "auroc": roc_auc,
        "auprc": auprc,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def write_outputs(
    *,
    output_dir: Path,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    args,
    feature_names: list[str],
    kept_feature_names: list[str],
    excluded_feature_patterns: list[str],
    history: list[dict],
    split_metrics: dict,
    split_predictions: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "original_feature_names": feature_names,
            "feature_names": kept_feature_names,
            "excluded_feature_patterns": excluded_feature_patterns,
            "args": vars(args),
        },
        output_dir / "validity_mlp.pt",
    )

    with open(output_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    metric_fields = [
        "split",
        "group",
        "group_value",
        "num_examples",
        "positive_rate",
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
    rows = list(split_metrics.values())
    rows.extend(group_metric_rows(split_predictions))
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in metric_fields})

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(split_metrics, f, indent=2, allow_nan=True)

    prediction_fields = [
        "split",
        "candidate_id",
        "domain",
        "problem",
        "corruption_type",
        "label_valid",
        "prob_valid",
        "pred_valid",
    ]
    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=prediction_fields)
        writer.writeheader()
        for split, (probs, data) in split_predictions.items():
            labels = np.asarray(data["y"], dtype=np.int64)
            for idx, prob in enumerate(probs):
                writer.writerow(
                    {
                        "split": split,
                        "candidate_id": str(data["candidate_ids"][idx]),
                        "domain": str(data["domains"][idx]),
                        "problem": str(data["problems"][idx]),
                        "corruption_type": str(data["corruption_types"][idx]),
                        "label_valid": int(labels[idx]),
                        "prob_valid": float(prob),
                        "pred_valid": int(prob >= 0.5),
                    }
                )


def group_metric_rows(split_predictions: dict) -> list[dict]:
    rows: list[dict] = []
    for split, (probs, data) in split_predictions.items():
        y = np.asarray(data["y"], dtype=np.int64)
        for group_name, values in [
            ("domain", data["domains"]),
            ("corruption_type", data["corruption_types"]),
        ]:
            for value in sorted(set(str(item) for item in values)):
                idxs = np.asarray([str(item) == value for item in values], dtype=bool)
                if not idxs.any():
                    continue
                metrics = compute_metrics(y[idxs], probs[idxs])
                metrics.update(
                    {
                        "split": split,
                        "group": group_name,
                        "group_value": value,
                    }
                )
                rows.append(metrics)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train downstream validity MLP.")
    parser.add_argument("--dataset_dir", default=str(VALIDITY_DATASET_DIR))
    parser.add_argument("--family", required=True)
    parser.add_argument("--source_seed", type=int, default=13)
    parser.add_argument("--seed", type=int, default=13, help="MLP head seed")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--eval_splits", nargs="+", default=DEFAULT_EVAL_SPLITS)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="cpu")
    parser.add_argument(
        "--exclude_feature_patterns",
        nargs="*",
        default=[],
        help="Exclude features whose names contain any of these substrings.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.dataset_dir = str(Path(args.dataset_dir).resolve())
    if args.output_dir is None:
        args.output_dir = str(RETRAINED_HEAD_DIR)
    else:
        args.output_dir = str(Path(args.output_dir).resolve())
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
