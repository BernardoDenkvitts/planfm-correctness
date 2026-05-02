# Downstream Code

This folder contains the core downstream validity implementation.

## Files

- `config.py`: central paths, domains, splits, and seeds for this GitHub snapshot.
- `families.py`: registry of the four frozen source-model families used for validity features.
- `plan_utils.py`: PDDL parsing, pyperplan grounding, gold-plan recovery, symbolic rollout, plan corruption, JSONL IO, and VAL labeling.
- `features.py`: frozen transition-model loading and 55-dimensional feature extraction.
- `build_validity_dataset.py`: candidate construction and feature-matrix generation.
- `train_validity.py`: MLP validity-head training and per-split prediction output.
- `evaluate_validity_story.py`: threshold tuning, baseline evaluation, and summary-table generation.
- `run_validity_experiments.py`: workflow runner that rebuilds features from included candidates and retrains all MLP heads.

## Implementation Summary

I treat plan validity as a supervised binary classification problem. The label comes from symbolic plan validation. The input to the classifier is not a raw action sequence. Instead, I use frozen source transition models to score how plausible each symbolic transition looks in the learned state representation.

The feature extractor rolls out each candidate plan with grounded PDDL operators. Unknown or inapplicable actions become stalled no-op transitions in the feature sequence, which keeps the feature extraction interface fixed-size without giving the classifier an explicit symbolic failure flag.
