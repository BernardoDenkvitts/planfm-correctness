# Downstream Code

This folder contains the core pipeline for the **plan correctness regression** task.
The goal is to predict how correct a plan is on a continuous scale from 0 (completely wrong)
to 1 (fully correct), using features extracted from frozen transition models.

---

## Files

### Infrastructure (shared, do not modify)

| File | Purpose |
|---|---|
| `config.py` | Central paths, domains, splits, seeds. Also defines `RETRAINED_HEAD_DIR` and `CORRECTNESS_HEAD_DIR`. |
| `families.py` | Registry of the four frozen source-model families used as feature extractors. |
| `plan_utils.py` | PDDL parsing, symbolic rollout, plan corruption, `compute_correctness_score`. |
| `features.py` | Loads frozen transition models and extracts the 55-dimensional feature vector per plan. |
| `build_correctness_dataset.py` | Builds the feature matrices (`.npz` files) stored in `data/correctness_dataset/`. |

### Regression pipeline

| File | Purpose |
|---|---|
| `baselines.py` | `MeanPredictor` and `PlanLengthRegressor` — dumb baselines to compare against. |
| `train_single_correctness_head.py` | **Main script for Experiments.** Trains one MLP experiment across all seeds. Reads a YAML config. |
| `train_all_correctness_heads.py` | **Main script for full Pipeline.** Orchestrator for the full run across all 4 families × 3 seeds, reads best_params.json to set best hyperparameters for each family. |
| `evaluate_correctness_story.py` | Aggregates predictions, runs baselines, writes summary tables. |

### Config files

| Location | Purpose |
|---|---|
| `config/baseline.yaml` | Template config — copy and modify for each new experiment. |
| `config/*.yaml` | One file per experiment you want to run. |

---

## How experiments work

### Running a single experiment (recommended for testing ideas)

1. Copy `config/baseline.yaml` and give it a descriptive name:
   ```
   config/exp_001_my_experiment.yaml
   ```

2. Edit only the fields you want to change:
   ```yaml
   experiment: exp_001_my_experiment
   description: "Your experiment"
   family: "dd_lstm_shortest_path_delta"
   ```

3. Run it:
   ```bash
   python -m code.downstream.train_single_correctness_head code/downstream/experiment_config/exp_001_my_experiment.yaml
   ```

4. Results are saved to:
   ```
   results/experiments/retrained_correctness_heads/
     dd_lstm_shortest_path_delta/
       exp_001_my_experiment/
         config.yaml          ← copy of the YAML used
         mean_metrics.json    ← mean ± std across the 3 seeds
         seed_13/             ← correctness_mlp.pt, metrics.json, history.csv, *.png
         seed_23/
         seed_37/
   ```

5. Compare `mean_metrics.json` with the baseline.

---

### Running the full pipeline (all 4 families × 3 seeds)

This uses the best hyperparameters found for each family, stored in
`experiment_config/baselines/best_params.json`:

```bash
# Experiment run (results go to results/experiments/retrained_correctness_heads/)
python -m code.downstream.train_all_correctness_heads --experiment <experiment_name>

# Final run (results go to models/correctness_heads/)
python -m code.downstream.train_all_correctness_heads --final --experiment <experiment_name>
```

The orchestrator reads `best_params.json` from `experiment_config/baselines/` and calls
`train_single_correctness_head.py` for each family × seed automatically.

---

## Output directories

| Directory | When to use |
|---|---|
| `results/experiments/retrained_correctness_heads/` | All experiment runs |
| `models/correctness_heads/` | Final heads — when experiments are complete (use `--final` flag) |

---

## Correctness score definition

The regression target is computed by `compute_correctness_score` in `plan_utils.py`:

```
correctness_score = α × execution_ratio + (1 − α) × goal_satisfaction
```

where `execution_ratio` is the fraction of the plan that remains applicable before the
first failure, and `goal_satisfaction` is the fraction of goal conditions met at the end. The score is in [0, 1].

---

## Feature vectors

Each candidate plan is represented by a **55-dimensional** feature vector extracted by rolling out
the plan through a frozen transition model. The features capture residuals, cosine
distances, norm statistics, and goal distances at each step. Unknown or inapplicable actions become stalled no-op transitions in the feature sequence, which keeps the feature extraction interface fixed-size without giving the regressor an explicit symbolic failure flag. They are produced by
`features.py` and stored in `data/correctness_dataset/features/`.
