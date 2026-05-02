"""Loader for all-domain tokenizer manifests used by validity features."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from code.tokenization.base import TokenizationStrategy
from code.tokenization.factory import create_tokenizer


class MultiDomainUnionTokenizer(TokenizationStrategy):
    """
    Block-union tokenizer over one fitted tokenizer per planning domain.

    The all-domain WL source models use this representation. Each domain has
    its own WL vocabulary, and the global embedding is a concatenation of the
    domain-specific vector spaces. At transform time only the active domain's
    block is populated.
    """

    def __init__(self, base_tokenizer_name: str, tokenizer_kwargs: dict | None = None):
        super().__init__(name=f"MultiDomainUnion[{base_tokenizer_name}]")
        self.base_tokenizer_name = base_tokenizer_name
        self.tokenizer_kwargs = dict(tokenizer_kwargs or {})
        self.domain_tokenizers: dict[str, TokenizationStrategy] = {}
        self.domain_dimensions: dict[str, int] = {}
        self.domain_offsets: dict[str, int] = {}
        self.domain_order: list[str] = []
        self.active_domain: str | None = None

    def fit(self, domain_pddl_path: str, train_states_dir: str, train_pddl_dir: str) -> None:
        _ = (domain_pddl_path, train_states_dir, train_pddl_dir)
        raise NotImplementedError("This extracted project only loads fitted tokenizers.")

    def set_active_domain(self, domain_name: str, domain_pddl_path: str | None = None) -> None:
        self._check_fitted()
        if domain_name not in self.domain_tokenizers:
            raise KeyError(f"Unknown domain '{domain_name}' for union tokenizer.")
        self.active_domain = domain_name
        tokenizer = self.domain_tokenizers[domain_name]
        if domain_pddl_path and hasattr(tokenizer, "set_domain"):
            tokenizer.set_domain(domain_pddl_path)

    def _active_tokenizer(self) -> tuple[str, TokenizationStrategy]:
        self._check_fitted()
        if self.active_domain is None:
            raise RuntimeError("Call set_active_domain before transforming states.")
        return self.active_domain, self.domain_tokenizers[self.active_domain]

    def _place_into_block(self, domain_name: str, local_vec: np.ndarray) -> np.ndarray:
        global_vec = np.zeros(self.embedding_dim, dtype=np.float32)
        offset = self.domain_offsets[domain_name]
        width = self.domain_dimensions[domain_name]
        global_vec[offset : offset + width] = local_vec.astype(np.float32)
        return global_vec

    def transform_state(
        self,
        state_atoms: list[str],
        goal_atoms: list[str],
        objects: list[str],
        *,
        problem_pddl_path: str | None = None,
        _wl_prob=None,
    ) -> np.ndarray:
        domain_name, tokenizer = self._active_tokenizer()
        try:
            local_vec = tokenizer.transform_state(
                state_atoms,
                goal_atoms,
                objects,
                problem_pddl_path=problem_pddl_path,
                _wl_prob=_wl_prob,
            )
        except TypeError:
            try:
                local_vec = tokenizer.transform_state(
                    state_atoms,
                    goal_atoms,
                    objects,
                    problem_pddl_path=problem_pddl_path,
                )
            except TypeError:
                local_vec = tokenizer.transform_state(state_atoms, goal_atoms, objects)
        return self._place_into_block(domain_name, local_vec)

    def transform_goal(
        self,
        goal_atoms: list[str],
        objects: list[str],
        *,
        problem_pddl_path: str | None = None,
        _wl_prob=None,
    ) -> np.ndarray:
        domain_name, tokenizer = self._active_tokenizer()
        try:
            local_vec = tokenizer.transform_goal(
                goal_atoms,
                objects,
                problem_pddl_path=problem_pddl_path,
                _wl_prob=_wl_prob,
            )
        except TypeError:
            try:
                local_vec = tokenizer.transform_goal(
                    goal_atoms,
                    objects,
                    problem_pddl_path=problem_pddl_path,
                )
            except TypeError:
                local_vec = tokenizer.transform_goal(goal_atoms, objects)
        return self._place_into_block(domain_name, local_vec)

    def get_embedding_dim(self) -> int:
        self._check_fitted()
        return int(self.embedding_dim)

    def load_vocabulary(self, filepath: str) -> None:
        manifest_path = Path(filepath)
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("artifact_type") != "multi_domain_union":
            raise ValueError(f"{filepath} is not a multi-domain union tokenizer manifest.")

        self.base_tokenizer_name = payload["base_tokenizer_name"]
        self.tokenizer_kwargs = dict(payload.get("tokenizer_kwargs", {}))
        self.embedding_dim = int(payload["embedding_dim"])
        self.domain_order = list(payload["domain_order"])
        self.domain_dimensions = {
            key: int(value) for key, value in payload["domain_dimensions"].items()
        }
        self.domain_offsets = {
            key: int(value) for key, value in payload["domain_offsets"].items()
        }
        self.domain_tokenizers = {}

        for domain_name in self.domain_order:
            rel_path = str(payload["artifacts"][domain_name]).replace("\\", "/")
            tok_path = manifest_path.parent / rel_path
            tokenizer = create_tokenizer(self.base_tokenizer_name, **self.tokenizer_kwargs)
            tokenizer.load_vocabulary(str(tok_path))
            self.domain_tokenizers[domain_name] = tokenizer

        self.active_domain = None
        self._is_fitted = True


def load_tokenizer_from_manifest(manifest_path: str) -> TokenizationStrategy:
    """Load a tokenizer manifest written by the upstream transition-model run."""
    manifest = Path(manifest_path)
    with open(manifest, "r", encoding="utf-8") as f:
        payload = json.load(f)

    kind = payload.get("artifact_type")
    if kind == "multi_domain_union":
        tokenizer = MultiDomainUnionTokenizer(
            base_tokenizer_name=payload["base_tokenizer_name"],
            tokenizer_kwargs=payload.get("tokenizer_kwargs", {}),
        )
        tokenizer.load_vocabulary(str(manifest))
        return tokenizer

    if kind == "standard_tokenizer":
        tokenizer = create_tokenizer(
            payload["tokenizer_name"],
            **payload.get("tokenizer_kwargs", {}),
        )
        vocab_relpath = str(payload["vocab_relpath"]).replace("\\", "/")
        tokenizer.load_vocabulary(str(manifest.parent / vocab_relpath))
        return tokenizer

    raise ValueError(f"Unsupported tokenizer manifest type in {manifest_path}.")
