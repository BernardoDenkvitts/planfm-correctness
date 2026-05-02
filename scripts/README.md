# Scripts

This folder contains command-line entry points for the main workflows.

## Files

- `evaluate_pretrained_heads.py`: evaluates the checked-in trained MLP heads.
- `train_validity_heads.py`: trains new MLP heads from the checked-in feature matrices.
- `rebuild_features_from_candidates.py`: recomputes feature matrices from checked-in candidates and frozen source models.
- `run_full_from_candidates.py`: recomputes features and retrains all MLP heads.

Run these commands from the repository root or call the scripts directly from this folder.
