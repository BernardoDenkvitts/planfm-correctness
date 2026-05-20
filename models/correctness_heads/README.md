# Correctness Heads

This folder contains the trained downstream MLP heads.

Each leaf directory has:

- `correctness_mlp.pt`: model weights, feature standardization statistics, feature names, excluded-feature settings, and training arguments.
- `history.csv`: training and validation loss per epoch.
- `metrics.csv`: split-level, domain-level, and corruption-level metrics.
- `metrics.json`: split-level metric dictionary.
- `predictions.csv`: predicted correctness score and true correctness score for each candidate plan.

I train one head per source family and seed.
