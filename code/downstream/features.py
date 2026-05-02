"""Frozen transition-model feature extraction for candidate plans."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from code.downstream.families import SourceFamily
from code.downstream.plan_utils import ProblemContext, load_problem_context, rollout_plan
from code.tokenization.factory import create_tokenizer
from code.tokenization.multidomain import MultiDomainUnionTokenizer, load_tokenizer_from_manifest


SERIES_NAMES = [
    "residual_l2",
    "residual_l1_mean",
    "cosine_distance",
    "pred_norm",
    "pred_delta_norm",
    "candidate_delta_norm",
    "current_goal_l2",
    "pred_goal_l2",
    "goal_progress_l2",
]
SUMMARY_STATS = ["mean", "std", "min", "max", "final"]
SCALAR_FEATURES = [
    "plan_len",
    "log_plan_len",
    "plan_to_budget_ratio",
    "initial_goal_l2",
    "final_goal_l2",
    "final_minus_initial_goal_l2",
    "hidden_h_final_norm",
    "hidden_c_final_norm",
    "hidden_h_abs_mean",
    "hidden_c_abs_mean",
]
FEATURE_NAMES = [
    f"{series}_{stat}"
    for series in SERIES_NAMES
    for stat in SUMMARY_STATS
] + SCALAR_FEATURES


@dataclass
class LoadedSource:
    """Loaded frozen model/tokenizer artifacts for one family/domain/seed."""

    family: SourceFamily
    domain: str
    model: object
    tokenizer: object
    mode: str
    model_kind: str
    input_dim: int
    device: object | None = None


class FrozenTransitionFeatureExtractor:
    """Extract fixed-size consistency summaries from frozen transition models."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        source_data_dir: str | Path,
        family: SourceFamily,
        seed: int,
        device: str = "cpu",
        xgb_n_jobs: int = 1,
    ) -> None:
        self.run_root = Path(run_root)
        self.source_data_dir = Path(source_data_dir)
        self.family = family
        self.seed = seed
        self.device_arg = device
        self.xgb_n_jobs = xgb_n_jobs
        self._source_cache: dict[str, LoadedSource] = {}
        self._problem_cache: dict[tuple[str, str, str], ProblemContext] = {}
        self._wl_problem_cache: dict[tuple[str, str, str], object] = {}

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    def extract(self, candidate: dict) -> np.ndarray:
        """Extract one feature vector for a candidate-plan record."""
        domain = candidate["domain"]
        split = candidate["split"]
        problem = candidate["problem"]
        plan = [str(action) for action in candidate.get("plan", [])]

        source = self._load_source(domain)
        problem_context = self._load_problem_context(domain, split, problem)
        states = rollout_plan(problem_context, plan)

        goal_vec = self._embed_goal(source, problem_context)
        initial_vec = self._embed_state(source, problem_context, states[0])
        final_vec = self._embed_state(source, problem_context, states[-1])
        initial_goal_l2 = _l2(initial_vec - goal_vec)
        final_goal_l2 = _l2(final_vec - goal_vec)

        series_values = {name: [] for name in SERIES_NAMES}
        hidden_summary = {
            "hidden_h_final_norm": 0.0,
            "hidden_c_final_norm": 0.0,
            "hidden_h_abs_mean": 0.0,
            "hidden_c_abs_mean": 0.0,
        }

        hidden = None
        for current_atoms, next_atoms in zip(states, states[1:]):
            current_vec = self._embed_state(source, problem_context, current_atoms)
            next_vec = self._embed_state(source, problem_context, next_atoms)
            pred_next_vec, pred_delta_vec, hidden = self._predict_next(
                source,
                current_vec,
                goal_vec,
                hidden,
            )

            candidate_delta = next_vec - current_vec
            residual = pred_next_vec - next_vec
            current_goal_l2 = _l2(current_vec - goal_vec)
            next_goal_l2 = _l2(next_vec - goal_vec)
            pred_goal_l2 = _l2(pred_next_vec - goal_vec)

            series_values["residual_l2"].append(_l2(residual))
            series_values["residual_l1_mean"].append(float(np.mean(np.abs(residual))))
            series_values["cosine_distance"].append(_cosine_distance(pred_next_vec, next_vec))
            series_values["pred_norm"].append(_l2(pred_next_vec))
            series_values["pred_delta_norm"].append(_l2(pred_delta_vec))
            series_values["candidate_delta_norm"].append(_l2(candidate_delta))
            series_values["current_goal_l2"].append(current_goal_l2)
            series_values["pred_goal_l2"].append(pred_goal_l2)
            series_values["goal_progress_l2"].append(current_goal_l2 - next_goal_l2)

        if source.model_kind == "lstm" and hidden is not None:
            hidden_summary.update(_summarize_lstm_hidden(hidden))

        values: list[float] = []
        for series in SERIES_NAMES:
            values.extend(_summarize_series(series_values[series]))

        plan_len = float(len(plan))
        transition_budget = float(max(100, 10 * max(1, len(problem_context.objects))))
        values.extend(
            [
                plan_len,
                math.log1p(plan_len),
                plan_len / transition_budget,
                initial_goal_l2,
                final_goal_l2,
                final_goal_l2 - initial_goal_l2,
                hidden_summary["hidden_h_final_norm"],
                hidden_summary["hidden_c_final_norm"],
                hidden_summary["hidden_h_abs_mean"],
                hidden_summary["hidden_c_abs_mean"],
            ]
        )

        return np.nan_to_num(np.asarray(values, dtype=np.float32))

    def _load_problem_context(self, domain: str, split: str, problem: str) -> ProblemContext:
        key = (domain, split, problem)
        cached = self._problem_cache.get(key)
        if cached is None:
            cached = load_problem_context(
                data_dir=self.source_data_dir,
                domain=domain,
                split=split,
                problem=problem,
            )
            self._problem_cache[key] = cached
        return cached

    def _load_source(self, domain: str) -> LoadedSource:
        cached = self._source_cache.get(domain)
        if cached is not None:
            return cached

        if self.family.model == "xgboost":
            source = self._load_xgb_source(domain)
        elif self.family.model == "lstm":
            source = self._load_lstm_source(domain)
        else:
            raise ValueError(f"Unsupported source model: {self.family.model}")

        self._source_cache[domain] = source
        return source

    def _load_xgb_source(self, domain: str) -> LoadedSource:
        import xgboost as xgb

        checkpoint_dir = self._checkpoint_dir(domain)
        model_name = domain if self.family.is_domain_dependent else "all_domains"
        model_path = checkpoint_dir / f"{model_name}_xgb.json"
        meta_path = checkpoint_dir / f"{model_name}_xgb_meta.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing XGBoost source model: {model_path}")

        model = xgb.XGBRegressor(device="cpu", n_jobs=self.xgb_n_jobs)
        model.load_model(str(model_path))
        meta = _read_json(meta_path)
        tokenizer = self._load_tokenizer(domain)
        input_dim = int(meta.get("output_dim", tokenizer.get_embedding_dim()))

        return LoadedSource(
            family=self.family,
            domain=domain,
            model=model,
            tokenizer=tokenizer,
            mode=str(meta.get("delta", True) and "delta" or "state"),
            model_kind="xgboost",
            input_dim=input_dim,
        )

    def _load_lstm_source(self, domain: str) -> LoadedSource:
        import torch

        from code.modeling.models import StateCentricLSTM, StateCentricLSTM_Delta

        checkpoint_dir = self._checkpoint_dir(domain)
        model_name = domain if self.family.is_domain_dependent else "all_domains"
        checkpoint_path = checkpoint_dir / f"{model_name}_lstm_best.pt"
        meta_path = checkpoint_dir / f"{model_name}_lstm_meta.json"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing LSTM source model: {checkpoint_path}")

        meta = _read_json(meta_path)
        tokenizer = self._load_tokenizer(domain)
        input_dim = int(meta.get("input_dim", tokenizer.get_embedding_dim()))
        hidden_dim = int(meta.get("hidden_dim", 256))
        use_projection = not bool(meta.get("no_projection", False))
        mode = str(meta.get("mode", self.family.mode))

        device = _resolve_torch_device(self.device_arg)
        if mode == "delta":
            model = StateCentricLSTM_Delta(
                input_dim,
                hidden_dim=hidden_dim,
                use_projection=use_projection,
            ).to(device)
        else:
            model = StateCentricLSTM(
                input_dim,
                hidden_dim=hidden_dim,
                use_projection=use_projection,
            ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        return LoadedSource(
            family=self.family,
            domain=domain,
            model=model,
            tokenizer=tokenizer,
            mode=mode,
            model_kind="lstm",
            input_dim=input_dim,
            device=device,
        )

    def _checkpoint_dir(self, domain: str) -> Path:
        root = self.run_root / self.family.regime / "checkpoints" / f"seed_{self.seed}"
        root = root / self.family.tokenizer / self.family.model_mode_dir
        if self.family.is_domain_dependent:
            root = root / domain
        return root

    def _load_tokenizer(self, domain: str):
        domain_pddl = self.source_data_dir / "pddl" / domain / "domain.pddl"
        if self.family.is_all_domains:
            manifest = (
                self.run_root
                / "all_domains"
                / "tokenizers"
                / self.family.tokenizer
                / f"all_domains_{self.family.tokenizer}.json"
            )
            tokenizer = load_tokenizer_from_manifest(str(manifest))
            if isinstance(tokenizer, MultiDomainUnionTokenizer):
                tokenizer.set_active_domain(domain, str(domain_pddl))
            elif hasattr(tokenizer, "set_domain"):
                tokenizer.set_domain(str(domain_pddl))
            return tokenizer

        tokenizer = create_tokenizer(self.family.tokenizer)
        vocab_path = self._domain_vocab_path(domain)
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing tokenizer vocabulary: {vocab_path}")
        tokenizer.load_vocabulary(str(vocab_path))
        if hasattr(tokenizer, "set_domain"):
            tokenizer.set_domain(str(domain_pddl))
        return tokenizer

    def _domain_vocab_path(self, domain: str) -> Path:
        model_dir = (
            self.run_root
            / "domain_dependent"
            / "data"
            / "encodings"
            / "models"
        )
        if self.family.tokenizer == "wl":
            return model_dir / f"{domain}_wl_tok.json"
        return model_dir / f"{domain}_{self.family.tokenizer}.json"

    def _embed_state(
        self,
        source: LoadedSource,
        context: ProblemContext,
        atoms: frozenset[str],
    ) -> np.ndarray:
        tokenizer = source.tokenizer
        wl_prob = self._wl_problem(source, context)
        state_atoms = list(atoms)
        try:
            vec = tokenizer.transform_state(
                state_atoms,
                list(context.goal_atoms),
                list(context.objects),
                problem_pddl_path=str(context.problem_path),
                _wl_prob=wl_prob,
            )
        except TypeError:
            try:
                vec = tokenizer.transform_state(
                    state_atoms,
                    list(context.goal_atoms),
                    list(context.objects),
                    problem_pddl_path=str(context.problem_path),
                )
            except TypeError:
                vec = tokenizer.transform_state(
                    state_atoms,
                    list(context.goal_atoms),
                    list(context.objects),
                )
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    def _embed_goal(self, source: LoadedSource, context: ProblemContext) -> np.ndarray:
        tokenizer = source.tokenizer
        wl_prob = self._wl_problem(source, context)
        try:
            vec = tokenizer.transform_goal(
                list(context.goal_atoms),
                list(context.objects),
                problem_pddl_path=str(context.problem_path),
                _wl_prob=wl_prob,
            )
        except TypeError:
            try:
                vec = tokenizer.transform_goal(
                    list(context.goal_atoms),
                    list(context.objects),
                    problem_pddl_path=str(context.problem_path),
                )
            except TypeError:
                vec = tokenizer.transform_goal(list(context.goal_atoms), list(context.objects))
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    def _predict_next(
        self,
        source: LoadedSource,
        current_vec: np.ndarray,
        goal_vec: np.ndarray,
        hidden,
    ) -> tuple[np.ndarray, np.ndarray, object | None]:
        if source.model_kind == "xgboost":
            x = np.hstack([current_vec.reshape(1, -1), goal_vec.reshape(1, -1)])
            pred = np.asarray(source.model.predict(x), dtype=np.float32).reshape(-1)
            if source.mode == "delta":
                return current_vec + pred, pred, hidden
            return pred, pred - current_vec, hidden

        return self._predict_lstm(source, current_vec, goal_vec, hidden)

    def _predict_lstm(
        self,
        source: LoadedSource,
        current_vec: np.ndarray,
        goal_vec: np.ndarray,
        hidden,
    ) -> tuple[np.ndarray, np.ndarray, object | None]:
        import torch

        device = source.device
        state_tensor = (
            torch.tensor(current_vec, dtype=torch.float32, device=device)
            .reshape(1, 1, -1)
        )
        goal_tensor = torch.tensor(goal_vec, dtype=torch.float32, device=device).reshape(1, -1)
        with torch.inference_mode():
            pred, next_hidden = source.model(state_tensor, goal_tensor, hidden=hidden)
        pred_vec = pred.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if source.mode == "delta":
            return current_vec + pred_vec, pred_vec, next_hidden
        return pred_vec, pred_vec - current_vec, next_hidden

    def _wl_problem(self, source: LoadedSource, context: ProblemContext):
        if source.family.tokenizer != "wl" and not isinstance(source.tokenizer, MultiDomainUnionTokenizer):
            return None
        key = (context.domain, context.split, context.problem)
        if key not in self._wl_problem_cache:
            try:
                from wlplan.planning import parse_problem as wl_parse_problem

                self._wl_problem_cache[key] = wl_parse_problem(
                    str(context.domain_path),
                    str(context.problem_path),
                )
            except Exception:
                self._wl_problem_cache[key] = None
        return self._wl_problem_cache[key]


def save_feature_matrix(
    *,
    path: str | Path,
    candidates: list[dict],
    features: np.ndarray,
    feature_names: list[str],
) -> None:
    """Save feature arrays and row metadata in one compressed artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=features.astype(np.float32),
        y=np.asarray([int(row["label_valid"]) for row in candidates], dtype=np.int64),
        candidate_ids=np.asarray([row["candidate_id"] for row in candidates], dtype=object),
        domains=np.asarray([row["domain"] for row in candidates], dtype=object),
        splits=np.asarray([row["split"] for row in candidates], dtype=object),
        problems=np.asarray([row["problem"] for row in candidates], dtype=object),
        corruption_types=np.asarray(
            [row["corruption_type"] for row in candidates],
            dtype=object,
        ),
        feature_names=np.asarray(feature_names, dtype=object),
    )


def load_feature_matrix(path: str | Path) -> dict:
    """Load a feature matrix artifact with object metadata enabled."""
    return dict(np.load(path, allow_pickle=True))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_torch_device(device_arg: str):
    import torch

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


def _summarize_series(values: list[float]) -> list[float]:
    if not values:
        return [0.0 for _ in SUMMARY_STATS]
    arr = np.asarray(values, dtype=np.float32)
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
        float(arr[-1]),
    ]


def _summarize_lstm_hidden(hidden) -> dict[str, float]:
    h, c = hidden
    h_cpu = h.detach().float().cpu()
    c_cpu = c.detach().float().cpu()
    return {
        "hidden_h_final_norm": float(h_cpu[-1].norm().item()),
        "hidden_c_final_norm": float(c_cpu[-1].norm().item()),
        "hidden_h_abs_mean": float(h_cpu.abs().mean().item()),
        "hidden_c_abs_mean": float(c_cpu.abs().mean().item()),
    }


def _l2(vec: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vec, dtype=np.float32).reshape(-1)))


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    u = np.asarray(left, dtype=np.float32).reshape(-1)
    v = np.asarray(right, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom == 0.0:
        return 1.0
    return float(1.0 - np.dot(u, v) / denom)
