# Baseline Hyperparameters

This folder contains the best MLP hyperparameters for each source family,
chosen from the grid search in `notebooks/exploration/01_model_comparison.ipynb`.

## best_params.json

Each family has its own best configuration (hidden layer sizes, dropout, learning
rate, weight decay). They were selected by majority frequency across the three
training seeds.

In case of ties — where no single configuration won across all three seeds — the
tie was broken by running `train_single_correctness_head.py` with each candidate
configuration across all three seeds and picking the one with the best **mean**
validation metrics.

## YAML configs

The `*.yaml` files in this folder are the corresponding per-family experiment
configs that can be passed directly to `train_single_correctness_head.py` to
reproduce the runs.