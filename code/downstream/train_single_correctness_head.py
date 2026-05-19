"""Train lightweight MLP heads for downstream plan correctness regression.

Usage:
    python -m code.downstream.train_correctness path/to/config.yaml
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys
import types
from pathlib import Path

import yaml

import matplotlib
matplotlib.use("Agg")  # headless no display needed
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from code.downstream.config import RETRAINED_HEAD_DIR, CORRECTNESS_DATASET_DIR
from code.downstream.features import load_feature_matrix


DEFAULT_EVAL_SPLITS = ["validation", "test-interpolation", "test-extrapolation"]

# Supported activation functions
ACTIVATION_MAP = {
    "ReLU": "nn.ReLU",
    "GELU": "nn.GELU",
    "SiLU": "nn.SiLU",
}

REQUIRED_KEYS = [
    "family",
    "experiment",
    "seeds",
    "hidden_dims",
    "dropout",
    "activation",
    "lr",
    "weight_decay",
    "batch_size",
    "epochs",
    "patience",
]

# Defaults for purely runtime/optional settings
OPTIONAL_DEFAULTS: dict = {
    "features_to_drop":    [],
    "features_to_keep":    [],
    "features_added":      [],
    "eval_splits":         DEFAULT_EVAL_SPLITS,
    "device":              "auto",
    "verbose":             False,
    "description":         "",
    "dataset_dir":         None,
    "output_dir":          None,
}


class CorrectnessMLP(nn.Module):
    """MLP for regression with configurable activation and sigmoid output."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, activation: str = "ReLU"):
        super().__init__()
        act_cls = getattr(nn, activation, None)
        if act_cls is None:
            raise ValueError(f"Unknown activation '{activation}'. Choose from: {list(ACTIVATION_MAP)}")
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), act_cls(), nn.Dropout(dropout)])
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        return torch.sigmoid(self.head(self.trunk(x)).squeeze(-1))


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
    
    npz = load_feature_matrix(path)
    
    # Handle the fact that 'correctness_scores' might be named 'y' due to different dataset versions
    if "correctness_scores" in npz:
        y_val = npz["correctness_scores"]
    elif "y" in npz:
        y_val = npz["y"]
    else:
        raise KeyError("Neither 'correctness_scores' nor 'y' found in npz file")
        
    npz["correctness_scores"] = y_val
    return npz


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# def add_ratio_features(data: dict, features_added: list[str]) -> None:
#     if not features_added:
#         return

#     X = data["X"]
#     feature_names = [str(name) for name in data["feature_names"]]

#     def get_col(name: str) -> np.ndarray:
#         if name not in feature_names:
#             raise ValueError(f"Feature {name} not found in dataset for computing ratios.")
#         return X[:, feature_names.index(name)]

#     epsilon = 1e-8  # to prevent division by zero

#     new_features = []
#     new_names = []

#     if "final_goal_ratio" in features_added:
#         new_features.append(get_col("final_goal_l2") / (get_col("initial_goal_l2") + epsilon))
#         new_names.append("final_goal_ratio")

#     if "goal_reduction_ratio" in features_added:
#         col = (get_col("initial_goal_l2") - get_col("final_goal_l2")) / (get_col("initial_goal_l2") + epsilon)
#         new_features.append(col)
#         new_names.append("goal_reduction_ratio")

#     if "pred_to_current_goal_final_ratio" in features_added:
#         new_features.append(get_col("pred_goal_l2_final") / (get_col("current_goal_l2_final") + epsilon))
#         new_names.append("pred_to_current_goal_final_ratio")

#     if "pred_to_current_goal_mean_ratio" in features_added:
#         new_features.append(get_col("pred_goal_l2_mean") / (get_col("current_goal_l2_mean") + epsilon))
#         new_names.append("pred_to_current_goal_mean_ratio")

#     if "current_goal_final_over_max" in features_added:
#         new_features.append(get_col("current_goal_l2_final") / (get_col("current_goal_l2_max") + epsilon))
#         new_names.append("current_goal_final_over_max")

#     if "pred_goal_final_over_max" in features_added:
#         new_features.append(get_col("pred_goal_l2_final") / (get_col("pred_goal_l2_max") + epsilon))
#         new_names.append("pred_goal_final_over_max")

#     if new_features:
#         data["X"] = np.column_stack([X] + new_features)
#         data["feature_names"] = np.array(feature_names + new_names)


def train(args) -> dict:
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_data = load_split(args.dataset_dir, args.family, args.seed, "train")
    val_data = load_split(args.dataset_dir, args.family, args.seed, "validation")
    eval_data = {
        split: load_split(args.dataset_dir, args.family, args.seed, split)
        for split in args.eval_splits
    }

    features_added = getattr(args, "features_added", [])
    if features_added:
        add_ratio_features(train_data, features_added)
        add_ratio_features(val_data, features_added)
        for split_data in eval_data.values():
            add_ratio_features(split_data, features_added)

    feature_names = [str(name) for name in train_data["feature_names"]]
    keep_mask = build_feature_keep_mask(
        feature_names,
        exclude_patterns=getattr(args, "features_to_drop", []),
        keep_only=getattr(args, "features_to_keep", []),
    )
    filtered_feature_names = [name for name, keep in zip(feature_names, keep_mask) if keep]

    train_X = np.asarray(train_data["X"], dtype=np.float32)[:, keep_mask]
    train_y = np.asarray(train_data["correctness_scores"], dtype=np.float32)
    val_X = np.asarray(val_data["X"], dtype=np.float32)[:, keep_mask]
    val_y = np.asarray(val_data["correctness_scores"], dtype=np.float32)

    if train_X.shape[0] == 0:
        raise RuntimeError("Training feature matrix is empty.")

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X).astype(np.float32)
    val_X = scaler.transform(val_X).astype(np.float32)
    eval_X_by_split = {
        split: scaler.transform(np.asarray(data["X"], dtype=np.float32)[:, keep_mask]).astype(np.float32)
        for split, data in eval_data.items()
    }
    mean = scaler.mean_
    std = scaler.scale_

    model = CorrectnessMLP(
        input_dim=train_X.shape[1],
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        activation=getattr(args, "activation", "ReLU"),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

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
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)
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
            np.asarray(data["correctness_scores"], dtype=np.float32),
            data,
        )

    for split, (X, y, data) in all_eval.items():
        preds = predict(model, X, args.batch_size, device)
        metrics = compute_metrics(y, preds)
        metrics.update({"split": split, "group": "overall", "group_value": "overall"})
        split_metrics[split] = metrics
        split_predictions[split] = (preds, data)

    output_dir = Path(args.output_dir) / args.family / args.experiment / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(
        output_dir=output_dir,
        model=model,
        mean=mean,
        std=std,
        args=args,
        feature_names=[str(name) for name in train_data["feature_names"]],
        kept_feature_names=filtered_feature_names,
        excluded_feature_patterns=getattr(args, "features_to_drop", []),
        history=history,
        split_metrics=split_metrics,
        split_predictions=split_predictions,
    )
    generate_plots(output_dir, history, split_predictions)
    print(f"Wrote downstream correctness outputs to {output_dir}")
    return split_metrics


def build_feature_keep_mask(
    feature_names: list[str],
    exclude_patterns: list[str] | None = None,
    keep_only: list[str] | None = None,
) -> np.ndarray:
    """Return a boolean mask for feature selection.

    If *keep_only* is non-empty, use only those exact feature names.
    Otherwise, exclude features whose names contain any of *exclude_patterns*.
    """
    if keep_only:
        keep_set = set(keep_only)
        missing = keep_set - set(feature_names)
        if missing:
            raise ValueError(f"features_to_keep contains unknown names: {missing}")
        return np.array([name in keep_set for name in feature_names], dtype=bool)
    patterns = [p.lower() for p in (exclude_patterns or []) if p]
    if not patterns:
        return np.ones(len(feature_names), dtype=bool)
    mask = np.array([not any(p in name.lower() for p in patterns) for name in feature_names], dtype=bool)
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


def predict(model, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=batch_size)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.asarray([], dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {
            "num_examples": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "R2": float("nan"),
        }

    # R² requires at least 2 samples; return nan for singleton groups
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    return {
        "num_examples": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2,
    }


def generate_plots(output_dir: Path, history: list[dict], split_predictions: dict) -> None:
    """Save learning curve and true-vs-predicted / residual plots to *output_dir*."""

    # --- Learning curve ---
    epochs     = [row["epoch"]      for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss   = [row["val_loss"]   for row in history]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss,   label="Val Loss",   linewidth=2)
    ax.set_title("Learning Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curve.png", dpi=120)
    plt.close(fig)

    # --- True vs Predicted + Residual histogram (interp & extrap) ---
    plot_splits = [
        ("test-interpolation", "Interpolation", "green"),
        ("test-extrapolation", "Extrapolation", "purple"),
    ]
    available = [(split_name, title, color) for split_name, title, color in plot_splits if split_name in split_predictions]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 2, figsize=(11, 4 * len(available)))
    if len(available) == 1:
        axes = [axes]  # make iterable

    for (split, title, color), (ax_scatter, ax_hist) in zip(available, axes):
        preds, data = split_predictions[split]
        targets = np.asarray(data["correctness_scores"], dtype=np.float32)
        residuals = targets - preds

        # Scatter
        ax_scatter.scatter(targets, preds, alpha=0.3, color=color, s=10)
        lim = [targets.min(), targets.max()]
        ax_scatter.plot(lim, lim, "r--", linewidth=1)
        ax_scatter.set_title(f"True vs Predicted ({title})")
        ax_scatter.set_xlabel("True Correctness")
        ax_scatter.set_ylabel("Predicted Correctness")

        # Residual histogram
        sns.histplot(residuals, kde=True, ax=ax_hist, color=color)
        ax_hist.set_title(f"Residuals ({title})")
        ax_hist.set_xlabel("Residual (true - pred)")
        ax_hist.set_xlim(-1, 1)
        ax_hist.set_xticks(np.arange(-0.6, 0.61, 0.2))

    fig.tight_layout()
    fig.savefig(output_dir / "visualization.png", dpi=120)
    plt.close(fig)


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
        output_dir / "correctness_mlp.pt",
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
        "MAE",
        "RMSE",
        "R2",
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
        "true_score",
        "pred_score",
        "residual",
    ]
    with open(output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=prediction_fields)
        writer.writeheader()
        for split, (preds, data) in split_predictions.items():
            labels = np.asarray(data["correctness_scores"], dtype=np.float32)
            for idx, pred in enumerate(preds):
                writer.writerow(
                    {
                        "split": split,
                        "candidate_id": str(data["candidate_ids"][idx]),
                        "domain": str(data["domains"][idx]),
                        "problem": str(data["problems"][idx]),
                        "corruption_type": str(data["corruption_types"][idx]),
                        "true_score": float(labels[idx]),
                        "pred_score": float(pred),
                        "residual": float(labels[idx] - pred),
                    }
                )


def group_metric_rows(split_predictions: dict) -> list[dict]:
    rows: list[dict] = []
    for split, (preds, data) in split_predictions.items():
        y = np.asarray(data["correctness_scores"], dtype=np.float32)
        for group_name, values in [
            ("domain", data["domains"]),
            ("corruption_type", data["corruption_types"]),
        ]:
            for value in sorted(set(str(item) for item in values)):
                idxs = np.asarray([str(item) == value for item in values], dtype=bool)
                if not idxs.any():
                    continue
                metrics = compute_metrics(y[idxs], preds[idxs])
                metrics.update(
                    {
                        "split": split,
                        "group": group_name,
                        "group_value": value,
                    }
                )
                rows.append(metrics)
    return rows


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file, apply defaults, and resolve paths."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if "family" not in cfg:
        raise ValueError(f"Config '{config_path}' must specify 'family'.")

    # Normalise: 'seed' (int) is an alias for 'seeds' (list of one)
    if "seed" in cfg and "seeds" not in cfg:
        cfg["seeds"] = [cfg.pop("seed")]
    elif "seed" in cfg:
        cfg.pop("seed")  # seeds takes precedence

    # Enforce required keys
    missing_keys = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing_keys:
        raise ValueError(
            f"Config '{config_path}' is missing required fields for reproducibility: {missing_keys}"
        )

    # Apply defaults for optional runtime keys
    for key, default in OPTIONAL_DEFAULTS.items():
        cfg.setdefault(key, default)

    # Resolve paths
    cfg["dataset_dir"] = str(Path(cfg["dataset_dir"]).resolve()) if cfg["dataset_dir"] else str(CORRECTNESS_DATASET_DIR)
    cfg["output_dir"]  = str(Path(cfg["output_dir"]).resolve())  if cfg["output_dir"]  else str(RETRAINED_HEAD_DIR)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Store original config path for traceability
    cfg["config_path"] = str(Path(config_path).resolve())

    return cfg


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m code.downstream.train_correctness path/to/config.yaml")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)
    seeds = cfg.pop("seeds")

    # Create the experiment directory and save a copy of the config for traceability
    experiment_dir = Path(cfg["output_dir"]) / cfg["family"] / cfg["experiment"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, experiment_dir / "config.yaml")

    all_metrics: list[dict] = []
    for seed in seeds:
        cfg["seed"] = seed
        args = types.SimpleNamespace(**cfg)
        metrics = train(args)
        all_metrics.append(metrics)
        print(f"  seed={seed} done")

    experiment_dir = Path(cfg["output_dir"]) / cfg["family"] / cfg["experiment"]
    splits = all_metrics[0].keys()
    mean_metrics: dict = {}

    print(f"\n=== Mean across {len(seeds)} seeds ===")
    for split in splits:
        metric_keys = [k for k in all_metrics[0][split]
                        if k not in ("split", "group", "group_value", "num_examples")]
        values = {
            k: [m[split][k] for m in all_metrics]
            for k in metric_keys
        }
        mean_metrics[split] = {
            k: {
                "mean": float(np.nanmean(values[k])),
                "std":  float(np.nanstd(values[k])),
            }
            for k in metric_keys
        }
        row = "  ".join(
            f"{k}={mean_metrics[split][k]['mean']:.4f}±{mean_metrics[split][k]['std']:.4f}"
            for k in metric_keys
        )
        print(f"  {split}: {row}")

    with open(experiment_dir / "mean_metrics.json", "w", encoding="utf-8") as f:
        json.dump(mean_metrics, f, indent=2)
    print(f"Saved mean_metrics.json to {experiment_dir}")


if __name__ == "__main__":
    main()
