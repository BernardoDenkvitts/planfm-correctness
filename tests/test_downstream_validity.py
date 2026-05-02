from pathlib import Path
import random

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RUN_ROOT = PROJECT_ROOT / "models" / "source_transition_models"


def test_recover_gold_plan_validates_on_blocks():
    pytest.importorskip("pyperplan")

    from code.downstream.plan_utils import (
        label_plan_internal,
        load_problem_context,
        recover_gold_plan,
    )

    context = load_problem_context(
        data_dir=DATA_DIR,
        domain="blocks",
        split="train",
        problem="probBLOCKS-4-0",
    )
    plan = recover_gold_plan(
        data_dir=DATA_DIR,
        domain="blocks",
        split="train",
        problem="probBLOCKS-4-0",
    )

    assert plan == [
        "(pick-up b)",
        "(stack b a)",
        "(pick-up c)",
        "(stack c b)",
        "(pick-up d)",
        "(stack d c)",
    ]
    is_valid, is_executable = label_plan_internal(context, plan)
    assert is_valid
    assert is_executable


def test_corruption_generation_produces_invalid_examples():
    pytest.importorskip("pyperplan")

    from code.downstream.plan_utils import (
        generate_labeled_candidates_for_problem,
        label_plan_internal,
        load_problem_context,
        recover_gold_plan,
    )

    context = load_problem_context(
        data_dir=DATA_DIR,
        domain="blocks",
        split="train",
        problem="probBLOCKS-4-0",
    )
    gold_plan = recover_gold_plan(
        data_dir=DATA_DIR,
        domain="blocks",
        split="train",
        problem="probBLOCKS-4-0",
    )
    rows = generate_labeled_candidates_for_problem(
        context=context,
        gold_plan=gold_plan,
        negative_ratio=2,
        rng=random.Random(7),
        validator=lambda plan: label_plan_internal(context, plan),
    )

    labels = [row["label_valid"] for row in rows]
    assert 1 in labels
    assert 0 in labels
    assert all("solved" not in row for row in rows)


def test_xgb_frozen_feature_vector_is_finite():
    pytest.importorskip("xgboost")
    pytest.importorskip("wlplan")
    if not RUN_ROOT.exists():
        pytest.skip("paper study outputs unavailable")

    from code.downstream.families import get_source_family
    from code.downstream.features import FEATURE_NAMES, FrozenTransitionFeatureExtractor
    from code.downstream.plan_utils import recover_gold_plan

    plan = recover_gold_plan(
        data_dir=DATA_DIR,
        domain="blocks",
        split="train",
        problem="probBLOCKS-4-0",
    )
    candidate = {
        "candidate_id": "blocks::train::probBLOCKS-4-0::000::gold",
        "domain": "blocks",
        "split": "train",
        "problem": "probBLOCKS-4-0",
        "corruption_type": "gold",
        "plan": plan,
        "plan_len": len(plan),
        "gold_plan_len": len(plan),
        "label_valid": 1,
        "label_executable": 1,
    }
    extractor = FrozenTransitionFeatureExtractor(
        run_root=RUN_ROOT,
        source_data_dir=DATA_DIR,
        family=get_source_family("dd_xgb_wl_delta"),
        seed=13,
    )
    features = extractor.extract(candidate)

    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()
    assert not any("label" in name or "solved" in name or "applicable" in name for name in FEATURE_NAMES)
