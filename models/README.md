# Models

This directory contains the trained models needed by the downstream plan-correctness workflow.

## Layout

- `source_transition_models/`: frozen PlanFM transition models and tokenizer vocabularies used to compute the 55-dimensional feature vectors.
- `correctness_heads/`: trained MLP correctness heads, one for each source family and seed.

The source transition models are frozen. The correctness heads are the downstream regressors trained on top of the frozen-model feature matrices.
