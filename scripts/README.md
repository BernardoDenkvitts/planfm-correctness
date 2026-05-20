# Scripts

This folder contains command-line entry points for the main workflows.

## Files

- `evaluate_correctness_heads.py`: evaluates the checked-in trained MLP correctness heads.
- `train_correctness_heads.py`: retrains MLP correctness heads from the checked-in feature matrices (calls `train_all_correctness_heads` with `--skip_build`).
- `add_correctness_scores_to_candidates.py`: recomputes and writes `correctness_score` fields into candidate JSONL files.
- `rebuild_features_from_candidates.py`: recomputes feature matrices from checked-in candidates and frozen source models.
- `run_full_from_candidates.py`: recomputes features and retrains all MLP correctness heads.

Run these commands from the repository root or call the scripts directly from this folder.
