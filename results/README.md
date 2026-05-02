# Results

This directory contains aggregate evaluation outputs for the downstream validity experiments.

## Layout

- `analysis/`: summary CSV files for tuned-threshold MLP evaluation and baseline comparisons.

Per-head prediction files are stored next to their trained heads in `models/validity_heads` because those predictions are tied to a specific source family and seed.
