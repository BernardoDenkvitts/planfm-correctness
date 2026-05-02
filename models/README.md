# Models

This directory contains the trained models needed by the downstream validity workflow.

## Layout

- `source_transition_models/`: frozen PlanFM transition models and tokenizer vocabularies used to compute the 55-dimensional feature vectors.
- `validity_heads/`: trained MLP validity heads, one for each source family and seed.

The source transition models are frozen. The validity heads are the downstream classifiers trained on top of the frozen-model feature matrices.
