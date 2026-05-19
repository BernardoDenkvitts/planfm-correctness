"""Registry for frozen source models used by downstream correctness heads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFamily:
    """A frozen transition-model family selected from the tokenizer study."""

    family_id: str
    regime: str
    model: str
    tokenizer: str
    encoding_dir: str
    mode: str
    description: str

    @property
    def model_mode_dir(self) -> str:
        return f"{self.model}_{self.mode}"

    @property
    def is_domain_dependent(self) -> bool:
        return self.regime == "domain_dependent"

    @property
    def is_all_domains(self) -> bool:
        return self.regime == "all_domains"


WEIGHTED_BEST_FAMILIES: dict[str, SourceFamily] = {
    "dd_xgb_wl_delta": SourceFamily(
        family_id="dd_xgb_wl_delta",
        regime="domain_dependent",
        model="xgboost",
        tokenizer="wl",
        encoding_dir="graphs",
        mode="delta",
        description="Domain-dependent XGBoost, WL tokenizer, delta target",
    ),
    "dd_lstm_shortest_path_delta": SourceFamily(
        family_id="dd_lstm_shortest_path_delta",
        regime="domain_dependent",
        model="lstm",
        tokenizer="shortest_path",
        encoding_dir="shortest_path",
        mode="delta",
        description="Domain-dependent LSTM, shortest-path tokenizer, delta target",
    ),
    "ad_xgb_wl_delta": SourceFamily(
        family_id="ad_xgb_wl_delta",
        regime="all_domains",
        model="xgboost",
        tokenizer="wl",
        encoding_dir="wl",
        mode="delta",
        description="All-domains XGBoost, WL tokenizer, delta target",
    ),
    "ad_lstm_wl_delta": SourceFamily(
        family_id="ad_lstm_wl_delta",
        regime="all_domains",
        model="lstm",
        tokenizer="wl",
        encoding_dir="wl",
        mode="delta",
        description="All-domains LSTM, WL tokenizer, delta target",
    ),
}


def get_source_family(family_id: str) -> SourceFamily:
    """Return a source-family spec by id."""
    try:
        return WEIGHTED_BEST_FAMILIES[family_id]
    except KeyError as exc:
        valid = ", ".join(sorted(WEIGHTED_BEST_FAMILIES))
        raise ValueError(f"Unknown source family '{family_id}'. Valid ids: {valid}") from exc


def resolve_source_families(requested: list[str] | None) -> list[SourceFamily]:
    """Resolve CLI family ids, with 'weighted_best' as a convenient alias."""
    if not requested or requested == ["weighted_best"] or "weighted_best" in requested:
        return [WEIGHTED_BEST_FAMILIES[key] for key in sorted(WEIGHTED_BEST_FAMILIES)]
    return [get_source_family(family_id) for family_id in requested]

