# Feature Matrices

This folder contains one compressed NumPy feature matrix per source family, source seed, and split.

The path pattern is:

```text
<source_family>/seed_<seed>/<split>.npz
```

Each matrix contains the 55-dimensional frozen transition-model feature vectors consumed by `code/downstream/train_correctness.py`.
