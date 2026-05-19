"""Configuration constants for the downstream plan-validity project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOMAINS = ["blocks", "gripper", "logistics", "visitall-from-everywhere"]
SPLITS_EVAL = ["validation", "test-interpolation", "test-extrapolation"]
SPLITS_ALL = ["train", *SPLITS_EVAL]
SEEDS = [13, 23, 37]

DATA_DIR = PROJECT_ROOT / "data"
CORRECTNESS_DATASET_DIR = DATA_DIR / "correctness_dataset"
SOURCE_MODEL_DIR = PROJECT_ROOT / "models" / "source_transition_models"
#PRETRAINED_HEAD_DIR = PROJECT_ROOT / "models" / "validity_heads"
#RETRAINED_HEAD_DIR = PROJECT_ROOT / "outputs" / "retrained_validity_heads"

RETRAINED_HEAD_DIR = PROJECT_ROOT / "results" / "experiments"/ "retrained_correctness_heads"
# Final heads (checked into the repo) - use only when experiments are done
CORRECTNESS_HEAD_DIR = PROJECT_ROOT / "models" / "correctness_heads"

ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
