# PlanFM Downstream Plan Correctness

This repository contains the downstream plan-correctness adaptation for frozen PlanFM transition models. The implementation keeps the upstream transition models fixed, converts each candidate plan into transition-consistency features, and trains a regressor MLP head that predicts the correctness score of the plan.

## Directory Layout

```text
code/                           implementation for candidate handling, feature extraction, and MLP training

data/pddl/                      PDDL domain and problem files used for symbolic rollout

data/correctness_dataset/       candidate plans and feature matrices consumed by the correctness heads

models/source_transition_models/ frozen source transition models and tokenizer vocabularies used to build features

models/correctness_heads/       trained downstream MLP heads, metrics, histories, and predictions

notebooks/onboarding_notebooks/ beginner demos for data inspection, training, inference, and results (Starts here)

notebooks/exploration/          Initial notebooks for the downstream task

results/analysis/               aggregate evaluation tables

scripts/                        command-line entry points for the main workflows

```

## Downstream Task

The downstream task is a regression over complete candidate plans. A plan is positive if its grounded action sequence is executable and reaches the PDDL goal. A plan is negative if the sequence fails execution or finishes in a state that does not satisfy the goal.

Each problem contributes one positive candidate from the recovered gold plan. Negative candidates are generated with plan corruptions: truncation, deletion, adjacent swap, action replacement, action insertion, and short segment repetition. The checked-in candidate set was labeled with VAL.

## VAL

VAL is the planner verifier for the plan-validity labels used by this downstream task. During candidate generation, each recovered gold plan and each corrupted candidate plan is written as a temporary plan file and checked against the corresponding PDDL domain and problem with VAL. The resulting `label_valid` field is 1 only when VAL reports that the plan is valid. The separate `label_executable` field records whether the plan executes successfully before the final goal check.

The checked-in candidate JSONL files already contain these VAL-derived labels. Training and evaluation from the checked-in dataset read those labels and do not invoke VAL again. VAL is required when candidate plans are regenerated from raw state trajectories. The internal labeler is included only for local smoke tests when VAL is unavailable.

## Frozen Transition Features

The source transition models are not fine-tuned for correctness prediction. **They are used as fixed feature generators**. For each step in a candidate plan, the code rolls out the symbolic state with pyperplan, embeds the current state, next state, and goal, and compares the observed transition with the frozen model prediction.

The downstream feature vector has 55 dimensions:

- 9 transition series: residual L2, residual L1 mean, cosine distance, predicted-state norm, predicted-delta norm, candidate-delta norm, current-goal L2, predicted-goal L2, and goal-progress L2.
- 5 summaries per series: mean, standard deviation, minimum, maximum, and final value.
- 10 scalar features: plan length, log plan length, plan-budget ratio, initial and final goal distances, goal-distance change, and LSTM hidden-state norm summaries.

## Source Families

The downstream adaptation uses the weighted-best frozen source families from the PlanFM tokenizer study:

- `ad_lstm_wl_delta`: all-domain LSTM, WL tokenizer, delta target
- `ad_xgb_wl_delta`: all-domain XGBoost, WL tokenizer, delta target
- `dd_lstm_shortest_path_delta`: domain-dependent LSTM, shortest-path tokenizer, delta target
- `dd_xgb_wl_delta`: domain-dependent XGBoost, WL tokenizer, delta target

Only WL and shortest-path tokenization code is included because those are the only tokenizers needed by these four source families. SimHash, GraphBPE, and random tokenization are not part of this downstream correctness adaptation.

## MLP Correctness Head

The correctness head is implemented in `code/downstream/train_single_correctness_head.py`. The MLP architecture varies per source family:

- input: 55 standardized transition-consistency features
- hidden layers: see `code/downstream/experiment_config/baselines/best_params.json`
- normalization: `LayerNorm` after each hidden linear layer
- activation: GELU
- dropout: see `code/downstream/experiment_config/baselines/best_params.json`
- loss: `MSELoss`
- optimizer: AdamW
- scheduler: `ReduceLROnPlateau`
- patience: 20
- validation selection: best validation loss with early stopping

One MLP head is trained for each source family and source seed. The checked-in trained heads are stored in `models/correctness_heads`.

## Dataset Counts

The included labeled candidate dataset contains:

| Split | Total | Valid | Invalid |
| --- | ---: | ---: | ---: |
| train | 1156 | 231 | 925 |
| validation | 165 | 33 | 132 |
| test-interpolation | 260 | 52 | 208 |
| test-extrapolation | 1365 | 273 | 1092 |

## Setup

All commands below assume the current directory is the repository root:

```bash
cd planfm-correctness
python -m venv .venv
```

Activate the environment on Windows:

```bash
.\.venv\Scripts\activate
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The feature-rebuild path uses PyTorch, XGBoost, pyperplan, and `wlplan`. The evaluation and notebook inspection paths use the checked-in feature matrices and trained heads, so they are the fastest way to verify the repository.


## Full Development Workflow

This section describes the complete pipeline in order used to generate the reported results.

```
Step 0 — Hyperparameter search (only when tuning is needed)
  notebooks/exploration/01_model_comparison.ipynb
    → results/model_comparison/<family>/seed_<seed>/best_params.json  (per-seed)
    → commit the chosen params to:
       code/downstream/experiment_config/baselines/best_params.json

Step 1 — Compare the three tested regressor models, pick the best one
  notebooks/exploration/02_analysis_model_comparison.ipynb

Step 2 — (Optional) Rebuild feature matrices from candidate plans
  python -m code.downstream.build_correctness_dataset --skip_candidates --overwrite_features
    → data/correctness_dataset/features/<family>/seed_<seed>/<split>.npz

Step 3 — Run individual experiments based on modified YAML files 
  python -m code.downstream.train_single_correctness_head path/to_config.yaml
    → results/experiments/retrained_correctness_heads/<family>/<experiment_name>
  
  - Inside each experiment_name folder, there is one folder for each seed, along with a file containing the mean results across the three seeds for that specific experiment.

Step 4 — Run experiments (reads best_params.json automatically)
  python -m code.downstream.train_all_correctness_heads --experiment <name>
    → results/experiments/retrained_correctness_heads/<family>/<name>/seed_<seed>/

Step 5 — Promote the winning heads to models/correctness_heads/ (tracked by git)
  python -m code.downstream.train_all_correctness_heads --final --experiment <name>
    → models/correctness_heads/<family>/<name>/seed_<seed>/

Step 6 — Generate aggregate evaluation tables
  python -m code.downstream.evaluate_correctness_story
    → results/analysis/*.csv
```


## Run The Main Code

### 1. Evaluate The Checked-In Trained Heads

This command reads the trained heads prediction files from `models/correctness_heads`, computes MLP metrics, computes baseline metrics, and writes aggregate CSV files to `results/analysis`.

```bash
python -m code.downstream.evaluate_correctness_story
```

Expected outputs:

```text
results/analysis/baseline_metrics.csv
results/analysis/baseline_summary.csv
results/analysis/correctness_metrics.csv
results/analysis/correctness_summary.csv
results/analysis/story_metrics.csv
results/analysis/story_summary.csv
```

Options can be passed through the script to `code.downstream.evaluate_correctness_story`, for example:

```bash
python -m code.downstream.evaluate_correctness_story --families ad_lstm_wl_delta dd_lstm_shortest_path_delta
python -m code.downstream.evaluate_correctness_story --seeds 13
```

### 2. Retrain Correctness Heads From Checked-In Feature Matrices

This command skips feature construction and trains new MLP regression heads from `data/correctness_dataset/features` using a YAML-driven orchestration. It reads the best hyperparameters from `code/downstream/experiment_config/baselines/best_params.json` and runs `train_single_correctness_head.py` for each family × seed. The new checkpoints, metrics, predictions, and aggregated metrics are written under `results/experiments/retrained_correctness_heads/`.

```bash
python -m code.downstream.train_all_correctness_heads
```

Expected outputs:

```text
results/experiments/retrained_correctness_heads/<family>/<experiment>/seed_<seed>/correctness_mlp.pt
results/experiments/retrained_correctness_heads/<family>/<experiment>/seed_<seed>/metrics.json
results/experiments/retrained_correctness_heads/<family>/<experiment>/seed_<seed>/history.csv
results/experiments/retrained_correctness_heads/<family>/<experiment>/mean_metrics.json
```

To run a small CPU check before training every head:

```bash
python -m code.downstream.train_all_correctness_heads --families ad_lstm_wl_delta --seeds 13 --epochs 3 --device cpu
```

To save final heads to `models/correctness_heads/` (tracked by git) once experiments are complete:

```bash
python -m code.downstream.train_all_correctness_heads --final
```

### 3. Rebuild Features From Checked-In Candidate Plans

This command reads the candidate JSONL files in `data/correctness_dataset/candidates`, loads the frozen source transition models from `models/source_transition_models`, recomputes transition-consistency features, and overwrites feature matrices in `data/correctness_dataset/features`.

```bash
python -m code.downstream.build_correctness_dataset --skip_candidates --overwrite_features
```

Expected outputs:

```text
data/correctness_dataset/features/<family>/seed_<seed>/<split>.npz
data/correctness_dataset/build_manifest.json
data/correctness_dataset/logs/commands.log
```

To rebuild a single family and seed:

```bash
python -m code.downstream.build_correctness_dataset --skip_candidates --overwrite_features --source_families ad_lstm_wl_delta --seeds 13 --device cpu
```

### 4. Rebuild Features And Retrain Heads

This command performs the main end-to-end downstream path using the checked-in candidate plans. It recomputes frozen-transition features and then trains MLP correctness heads.

```bash
python -m code.downstream.build_correctness_dataset --skip_candidates --overwrite_features
python -m code.downstream.train_all_correctness_heads --experiment <experiment_name>
```

Expected outputs combine the feature matrices from step 3 and the retrained heads from step 2.

To run a smaller single-family path before launching every family and seed:

```bash
python -m code.downstream.build_correctness_dataset --skip_candidates --overwrite_features --source_families ad_lstm_wl_delta --seeds 13 --device cpu
python -m code.downstream.train_all_correctness_heads --families ad_lstm_wl_delta --seeds 13 --epochs 3 --device cpu --experiment <experiment_name>
```

### 5. Regenerate Candidate Plans From Raw State Trajectories

The checked-in workflow uses prebuilt candidate JSONL files. Candidate regeneration requires the full raw state-trajectory directory and a VAL executable for external plan labeling.

When the full state trajectories are available, run:

```bash
python -m code.downstream.build_correctness_dataset \
  --source_data_dir path/to/full_state_data \
  --run_root models/source_transition_models \
  --output_dir data/correctness_dataset \
  --source_families weighted_best \
  --seeds 13 23 37 \
  --labeler val \
  --val_path path/to/Validate \
  --rebuild_candidates \
  --overwrite_features \
  --device cpu
```

Use backslashes or forward slashes according to the shell. For Linux or macOS, replace line-continuation `^` with `\`.

For local smoke tests without VAL, the internal labeler can be used:

```bash
python -m code.downstream.build_correctness_dataset --labeler internal --max_problems 2 --rebuild_candidates --overwrite_features --device cpu
```

The internal labeler is intended only for local code-path checks. Reported downstream results should use VAL labels.

## Demo Notebooks

The notebooks provide a beginner-oriented tour of the repository:

`./notebooks/onboarding_notebooks/`

The notebooks cover candidate-plan inspection, feature-matrix inspection, CPU-only MLP training on a small subset, pretrained inference, and aggregate result summaries. They use the same checked-in data and model files as the command-line workflows.


## Included Data And Models

The repository includes the downstream candidate files, feature matrices, pretrained correctness heads, and source transition-model files needed to reproduce the downstream feature extraction path. The full raw state-trajectory corpus is not included because it is much larger than the downstream data.
