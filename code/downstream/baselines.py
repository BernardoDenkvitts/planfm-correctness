"""Regression baselines for plan correctness estimation.

Two baselines are provided:

1. MeanPredictor  — always predicts the training-set mean.
   Lower bound: if your model does not beat this, something is wrong.

2. PlanLengthRegressor — Ridge regression trained on three plan-length
   features only (plan_len, log_plan_len, plan_to_budget_ratio).
   Tests whether the frozen transition features add anything beyond
   what plan length alone can explain.

Both baselines share the same interface as the models in model_comparison.py:
they are fit on the training split and evaluated on any split, returning
metric rows in the same dict format {"MAE", "RMSE", "R2", "family",
"model", "split"}.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Feature names used by the plan-length baseline.
PLAN_LENGTH_FEATURES = ["plan_len", "log_plan_len", "plan_to_budget_ratio"]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, and R² for a set of predictions."""
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def _find_feature_indices(feature_names: list[str], targets: list[str]) -> np.ndarray:
    """Return the column indices for each name in *targets*.

    Raises ValueError if any target name is not found.
    """
    index = {name: idx for idx, name in enumerate(feature_names)}
    missing = [t for t in targets if t not in index]
    if missing:
        raise ValueError(
            f"Plan-length features not found in feature matrix: {missing}. "
            f"Available: {feature_names}"
        )
    return np.array([index[t] for t in targets], dtype=int)


# ---------------------------------------------------------------------------
# Baseline 1 — Mean predictor
# ---------------------------------------------------------------------------

class MeanPredictor:
    """Predicts the training-set mean for every example.

    This is the simplest possible baseline. Any useful model must produce
    lower MAE / RMSE and higher R² than this predictor.
    """

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, y_train: np.ndarray) -> "MeanPredictor":
        self._mean = float(y_train.mean())
        return self

    def predict(self, n: int) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("Call fit() before predict().")
        return np.full(n, self._mean, dtype=np.float32)

    @property
    def train_mean(self) -> float:
        if self._mean is None:
            raise RuntimeError("Call fit() before accessing train_mean.")
        return self._mean


# ---------------------------------------------------------------------------
# Baseline 2 — Plan-length linear regression
# ---------------------------------------------------------------------------

class PlanLengthRegressor:
    """Ridge regression trained on plan-length features only.

    Uses plan_len, log_plan_len, and plan_to_budget_ratio — the three
    features that capture plan size but carry no transition-model information.

    If the MLP is not clearly better than this baseline, the frozen
    transition features are not contributing useful signal.
    """

    def __init__(self, alpha: float = 1.0, seed: int = 13) -> None:
        self._alpha = alpha
        self._seed = seed
        self._col_indices: np.ndarray | None = None
        self._scaler: StandardScaler | None = None
        self._model: Ridge | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str],
    ) -> "PlanLengthRegressor":
        self._col_indices = _find_feature_indices(feature_names, PLAN_LENGTH_FEATURES)
        X_len = X_train[:, self._col_indices]

        self._scaler = StandardScaler().fit(X_len)
        X_scaled = self._scaler.transform(X_len)

        self._model = Ridge(alpha=self._alpha, random_state=self._seed)
        self._model.fit(X_scaled, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._scaler is None or self._col_indices is None:
            raise RuntimeError("Call fit() before predict().")
        X_len = X[:, self._col_indices]
        X_scaled = self._scaler.transform(X_len)
        preds = self._model.predict(X_scaled).astype(np.float32)
        return np.clip(preds, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Evaluation helper — produces metric rows in model_comparison.py format
# ---------------------------------------------------------------------------

def evaluate_baselines(
    family_data: dict,
    family: str,
    splits: list[str] | None = None,
) -> list[dict]:
    """Fit both baselines on the training split and evaluate on all splits.

    Parameters
    ----------
    family_data:
        Dict returned by ``load_family_data`` in model_comparison.py.
        Keys are split names; values are dicts with "X", "y", "feature_names".
    family:
        Family name string, used to populate the "family" field in output rows.
    splits:
        Which splits to evaluate. Defaults to all splits present in family_data.

    Returns
    -------
    List of metric dicts with keys:
        MAE, RMSE, R2, family, model, split
    Matches the format written by model_comparison.py so rows can be
    concatenated with the main results DataFrame directly.
    """
    if splits is None:
        splits = list(family_data.keys())

    X_train = family_data["train"]["X"]
    y_train = family_data["train"]["y"]
    feature_names = family_data["train"]["feature_names"]

    # --- Fit baselines on training data ---
    mean_pred = MeanPredictor().fit(y_train)
    len_reg = PlanLengthRegressor().fit(X_train, y_train, feature_names)

    rows: list[dict] = []

    for split in splits:
        X_split = family_data[split]["X"]
        y_split = family_data[split]["y"]

        # Mean predictor
        preds_mean = mean_pred.predict(len(y_split))
        metrics = _compute_metrics(y_split, preds_mean)
        metrics.update({"family": family, "model": "MeanBaseline", "split": split})
        rows.append(metrics)

        # Plan-length regressor
        preds_len = len_reg.predict(X_split)
        metrics = _compute_metrics(y_split, preds_len)
        metrics.update({"family": family, "model": "PlanLengthBaseline", "split": split})
        rows.append(metrics)

    return rows
