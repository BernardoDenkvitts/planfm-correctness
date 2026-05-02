# PlanFM Downstream Plan Validity

This repository contains the downstream plan-validity adaptation for frozen PlanFM transition models. The implementation keeps the upstream transition models fixed, converts each candidate plan into transition-consistency features, and trains a binary MLP head that predicts whether the candidate plan is valid.

## Directory Layout

```text
code/                           implementation for candidate handling, feature extraction, and MLP training
data/pddl/                      PDDL domain and problem files used for symbolic rollout
data/states/                    small Blocks trajectory used by smoke tests
data/validity_dataset/          candidate plans and feature matrices consumed by the validity heads
models/source_transition_models/ frozen source transition models and tokenizer vocabularies used to build features
models/validity_heads/          trained downstream MLP heads, metrics, histories, and predictions
notebooks/                      beginner demos for data inspection, training, inference, and results
results/analysis/               aggregate evaluation tables
scripts/                        command-line entry points for the main workflows
tests/                          focused tests for the downstream code path
```

## Downstream Task

The downstream task is binary classification over complete candidate plans. A plan is positive if its grounded action sequence is executable and reaches the PDDL goal. A plan is negative if the sequence fails execution or finishes in a state that does not satisfy the goal.

Each problem contributes one positive candidate from the recovered gold plan. Negative candidates are generated with plan corruptions: truncation, deletion, adjacent swap, action replacement, action insertion, and short segment repetition. The checked-in candidate set was labeled with VAL.

## Ground-Truth Verification With VAL

VAL is the ground-truth verifier for the plan-validity labels used by this downstream task. During candidate generation, each recovered gold plan and each corrupted candidate plan is written as a temporary plan file and checked against the corresponding PDDL domain and problem with VAL. The resulting `label_valid` field is 1 only when VAL reports that the plan is valid. The separate `label_executable` field records whether the plan executes successfully before the final goal check.

The checked-in candidate JSONL files already contain these VAL-derived labels. Training and evaluation from the checked-in dataset read those labels and do not invoke VAL again. VAL is required when candidate plans are regenerated from raw state trajectories. The internal labeler is included only for local smoke tests when VAL is unavailable.

## Frozen Transition Features

The source transition models are not fine-tuned for validity classification. They are used as fixed feature generators. For each step in a candidate plan, the code rolls out the symbolic state with pyperplan, embeds the current state, next state, and goal, and compares the observed transition with the frozen model prediction.

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

Only WL and shortest-path tokenization code is included because those are the only tokenizers needed by these four source families. SimHash, GraphBPE, and random tokenization are not part of this downstream validity adaptation.

## MLP Validity Head

The validity head is implemented in `code/downstream/train_validity.py`. It uses:

- input: 55 standardized transition-consistency features
- hidden layers: 64 and 32 units
- normalization: `LayerNorm` after each hidden linear layer
- activation: ReLU
- dropout: 0.1
- loss: class-weighted `BCEWithLogitsLoss`
- optimizer: AdamW
- validation selection: best validation loss with early stopping

One MLP head is trained for each source family and source seed. The checked-in trained heads are stored in `models/validity_heads`.

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
cd planfm-validity
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

## Run The Main Code

### 1. Evaluate The Checked-In Trained Heads

This command reads the trained heads and prediction files in `models/validity_heads`, computes tuned-threshold MLP metrics, computes baseline metrics, and writes aggregate CSV files to `results/analysis`.

```bash
python scripts/evaluate_pretrained_heads.py
```

Expected outputs:

```text
results/analysis/tuned_threshold_metrics.csv
results/analysis/tuned_threshold_summary.csv
results/analysis/baseline_metrics.csv
results/analysis/baseline_summary.csv
results/analysis/story_metrics.csv
results/analysis/story_summary.csv
```

Useful options can be passed through the script to `code.downstream.evaluate_validity_story`:

```bash
python scripts/evaluate_pretrained_heads.py --threshold_objective f1
python scripts/evaluate_pretrained_heads.py --families ad_lstm_wl_delta dd_lstm_shortest_path_delta
python scripts/evaluate_pretrained_heads.py --seeds 13
```

### 2. Retrain Validity Heads From Checked-In Feature Matrices

This command skips feature construction and trains new MLP heads from `data/validity_dataset/features`. The new checkpoints, metrics, histories, predictions, and logs are written under `outputs/`.

```bash
python scripts/train_validity_heads.py
```

Expected outputs:

```text
outputs/retrained_validity_heads/<family>/source_seed_<seed>/head_seed_<seed>/validity_mlp.pt
outputs/retrained_validity_heads/<family>/source_seed_<seed>/head_seed_<seed>/metrics.json
outputs/retrained_validity_heads/<family>/source_seed_<seed>/head_seed_<seed>/predictions.csv
data/validity_dataset/logs/commands.log
```

To run a small CPU check before training every head:

```bash
python scripts/train_validity_heads.py --source_families ad_lstm_wl_delta --seeds 13 --mlp_epochs 3 --mlp_device cpu
```

To train all checked-in families with the default experiment settings:

```bash
python scripts/train_validity_heads.py --mlp_device cpu
```

### 3. Rebuild Features From Checked-In Candidate Plans

This command reads the candidate JSONL files in `data/validity_dataset/candidates`, loads the frozen source transition models from `models/source_transition_models`, recomputes transition-consistency features, and overwrites feature matrices in `data/validity_dataset/features`.

```bash
python scripts/rebuild_features_from_candidates.py
```

Expected outputs:

```text
data/validity_dataset/features/<family>/seed_<seed>/<split>.npz
data/validity_dataset/build_manifest.json
data/validity_dataset/logs/commands.log
```

To rebuild a single family and seed:

```bash
python scripts/rebuild_features_from_candidates.py --source_families ad_lstm_wl_delta --seeds 13 --device cpu
```

### 4. Rebuild Features And Retrain Heads

This command performs the main end-to-end downstream path using the checked-in candidate plans. It recomputes frozen-transition features and then trains MLP validity heads.

```bash
python scripts/run_full_from_candidates.py
```

Expected outputs combine the feature matrices from step 3 and the retrained heads from step 2.

To run a smaller single-family path before launching every family and seed:

```bash
python scripts/run_full_from_candidates.py --source_families ad_lstm_wl_delta --seeds 13 --mlp_epochs 3 --device cpu --mlp_device cpu
```

### 5. Regenerate Candidate Plans From Raw State Trajectories

The checked-in workflow uses prebuilt candidate JSONL files. Candidate regeneration requires the full raw state-trajectory directory and a VAL executable for external plan labeling. The `data/states` directory in this repository only contains the small Blocks trajectory used by tests.

When the full state trajectories are available, run:

```bash
python -m code.downstream.build_validity_dataset ^
  --source_data_dir path\to\full_state_data ^
  --run_root models\source_transition_models ^
  --output_dir data\validity_dataset ^
  --source_families weighted_best ^
  --seeds 13 23 37 ^
  --labeler val ^
  --val_path path\to\Validate.exe ^
  --rebuild_candidates ^
  --overwrite_features ^
  --device cpu
```

Use backslashes or forward slashes according to the shell. For Linux or macOS, replace line-continuation `^` with `\`.

For local smoke tests without VAL, the internal labeler can be used:

```bash
python -m code.downstream.build_validity_dataset --labeler internal --max_problems 2 --rebuild_candidates --overwrite_features --device cpu
```

The internal labeler is intended only for local code-path checks. Reported downstream results should use VAL labels.

## Demo Notebooks

The notebooks provide a beginner-oriented tour of the repository:

```bash
jupyter notebook notebooks
```

The notebooks cover candidate-plan inspection, feature-matrix inspection, CPU-only MLP training on a small subset, pretrained inference, and aggregate result summaries. They use the same checked-in data and model files as the command-line workflows.

## Tests

Run the focused tests from the repository root:

```bash
pytest -q -p no:cacheprovider tests
```

Some tests are skipped when optional runtime dependencies such as PyTorch, XGBoost, pyperplan, or `wlplan` are unavailable in the active environment.

## Included Data And Models

The repository includes the downstream candidate files, feature matrices, pretrained validity heads, and source transition-model files needed to reproduce the downstream feature extraction path. The full raw state-trajectory corpus is not included because it is much larger than the downstream data. The checked-in `data/states` folder contains only the small Blocks trajectory used by the smoke tests.
