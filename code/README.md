# Code

This directory contains the implementation for the downstream plan-correctness adaptation.

## Layout

- `downstream/`: candidate-plan handling, feature extraction, MLP training, and post-training evaluation.
- `modeling/`: LSTM transition-model definitions needed to load frozen LSTM source models.
- `tokenization/`: WL and shortest-path tokenizers needed by the frozen source families.

I intentionally do not include paper-building scripts, figure-generation scripts, or unrelated tokenizer-study runners here. This directory is scoped to the plan-correctness pipeline.
