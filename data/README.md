# Data

This directory contains the data needed by the downstream plan-validity code.

## Layout

- `pddl/`: planning domains and problem files used by pyperplan for symbolic rollout.
- `states/`: one small Blocks trajectory used by tests that recover a gold plan from state transitions.
- `validity_dataset/`: the checked-in downstream candidate plans and frozen-model feature matrices.

The full raw trajectory corpus is not included here. The downstream candidate files and feature matrices are included, so the main training and evaluation workflows run without the full raw trajectory corpus.
