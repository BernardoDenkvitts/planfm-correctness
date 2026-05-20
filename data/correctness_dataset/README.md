# Correctness Dataset

This folder contains the downstream supervised dataset.

## Files And Folders

- `candidates/`: JSONL files with candidate plans, corruption type, split, domain, problem id, and VAL-derived labels.
- `features/`: compressed NumPy matrices for each source family, source seed, and split.
- `build_manifest.json`: metadata for the checked-in candidate and feature build.

Each feature file contains:

- `X`: feature matrix with 55 columns
- `y`: legacy binary validity label (kept for compatibility; **not** the regression target)
- `correctness_scores`: correctness scores of the plan (0 to 1) — the regression target
- `candidate_ids`: stable row identifiers
- `domains`, `splits`, `problems`, `corruption_types`: row metadata
- `feature_names`: column names for the 55 feature dimensions
