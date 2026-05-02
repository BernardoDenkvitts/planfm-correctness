# Source Transition Models

This folder contains the frozen source models used to compute downstream validity features.

## Included Source Families

- all-domain LSTM with WL delta features
- all-domain XGBoost with WL delta features
- domain-dependent LSTM with shortest-path delta features
- domain-dependent XGBoost with WL delta features

The directory layout mirrors the original transition-model run so `code/downstream/features.py` can load checkpoints and tokenizer vocabularies without path translation.
