# PlanFM Downstream Plan Validity

This repository snapshot shows how I adapted frozen PlanFM transition models to a downstream plan-validity classifier. I keep the upstream transition models frozen, convert each candidate plan into transition-consistency features, and train a small binary MLP head that predicts whether the candidate plan is valid.

I organized this folder for a GitHub reader, not as a Python package. The important folders are:

```text
code/                         implementation for candidate handling, feature extraction, and MLP training
data/pddl/                    PDDL domain and problem files needed for symbolic rollout
data/states/                  one small Blocks trajectory for smoke tests
data/validity_dataset/        checked-in candidate plans and feature matrices used by the MLP heads
models/source_transition_models/ frozen source transition models and tokenizer vocabularies used to build features
models/validity_heads/        trained downstream MLP heads, metrics, histories, and predictions
notebooks/                    beginner demos for data inspection, training, inference, and results
results/analysis/             aggregate evaluation tables
scripts/                      command-line entry points for the main workflows
tests/                        focused tests for the downstream code path
```

## What I Am Modeling

The downstream task is binary classification over complete candidate plans. A plan is positive if its grounded action sequence is executable and reaches the PDDL goal. A plan is negative if it fails execution or fails to reach the goal.

I build one positive candidate from the recovered gold plan for each problem. I then generate negative candidates with plan corruptions: truncation, deletion, adjacent swap, action replacement, action insertion, and short segment repetition. The full checked-in candidate set was labeled with VAL.

## Why Frozen Transition Models Are Used

I do not fine-tune the source transition models for validity classification. Instead, I use them as fixed feature generators. For each step in a candidate plan, I roll out the symbolic state using pyperplan, embed the current state, next state, and goal, then compare the observed transition with the frozen model prediction.

The downstream feature vector has 55 dimensions:

- 9 transition series: residual L2, residual L1 mean, cosine distance, predicted-state norm, predicted-delta norm, candidate-delta norm, current-goal L2, predicted-goal L2, and goal-progress L2.
- 5 summaries per series: mean, standard deviation, minimum, maximum, and final value.
- 10 scalar features: plan length, log plan length, plan-budget ratio, initial and final goal distances, goal-distance change, and LSTM hidden-state norm summaries.

## Source Families

I use the weighted-best frozen source families from the PlanFM tokenizer study:

- `ad_lstm_wl_delta`: all-domain LSTM, WL tokenizer, delta target
- `ad_xgb_wl_delta`: all-domain XGBoost, WL tokenizer, delta target
- `dd_lstm_shortest_path_delta`: domain-dependent LSTM, shortest-path tokenizer, delta target
- `dd_xgb_wl_delta`: domain-dependent XGBoost, WL tokenizer, delta target

Only WL and shortest-path tokenization code is included because those are the only tokenizers needed by these four source families. SimHash, GraphBPE, and random tokenization are not part of this downstream validity adaptation.

## MLP Head

The validity head is implemented in `code/downstream/train_validity.py`. It uses:

- input: 55 standardized transition-consistency features
- hidden layers: 64 and 32 units
- normalization: `LayerNorm` after each hidden linear layer
- activation: ReLU
- dropout: 0.1
- loss: class-weighted `BCEWithLogitsLoss`
- optimizer: AdamW
- validation selection: best validation loss with early stopping

I train one MLP head for each source family and source seed. The checked-in trained heads are in `models/validity_heads`.

## Dataset Counts

The included labeled candidate dataset contains:

| Split | Total | Valid | Invalid |
| --- | ---: | ---: | ---: |
| train | 1156 | 231 | 925 |
| validation | 165 | 33 | 132 |
| test-interpolation | 260 | 52 | 208 |
| test-extrapolation | 1365 | 273 | 1092 |

## Setup

From this folder:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Main Workflows

Start with the demo notebooks:

```bash
jupyter notebook notebooks
```

The notebooks walk through candidate plans, feature matrices, a CPU-only training demo, pretrained prediction inspection, and aggregate result summaries.

Evaluate the included trained heads:

```bash
python scripts/evaluate_pretrained_heads.py
```

Retrain all validity heads from the included feature matrices:

```bash
python scripts/train_validity_heads.py
```

Recompute features from the included candidate plans and frozen source models, then retrain the heads:

```bash
python scripts/run_full_from_candidates.py
```

The candidate JSONL files are already included, so the default full workflow does not regenerate candidates from raw state trajectories. To regenerate candidates from scratch, provide the full state-trajectory directory and call `code.downstream.build_validity_dataset` without `--skip_candidates`.

## Notes On Included Data

I include the complete downstream candidate files, feature matrices, pretrained validity heads, and source transition-model files needed to reproduce the downstream feature extraction path. I do not include the full raw state-trajectory corpus because it is much larger than the downstream data. The checked-in `data/states` folder contains only the small Blocks trajectory used by the smoke tests.
