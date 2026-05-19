# Demo Notebooks

This folder contains beginner-friendly notebooks for the downstream plan-correctness repository.

## Notebooks

- `01_candidate_dataset_tour.ipynb`: loads the candidate JSONL files, explains labels and corruption types, and inspects example plans.
- `02_feature_matrix_tour.ipynb`: loads the frozen transition-model feature matrices and examines feature names, label balance, and feature differences.
- `03_downstream_training_demo.ipynb`: trains a small CPU-only scikit-learn MLP on the included features. The official repository training path is still `code/downstream/train_correctness.py`.
- `04_pretrained_inference_demo.ipynb`: inspects pretrained validity-head prediction files and queries candidate-level predictions.
- `05_results_summary.ipynb`: reads aggregate result CSV files and compares source families and baselines.

The notebooks use only the checked-in files in `data`, `models`, and `results`. They are meant to make the pipeline easy to inspect before running the heavier Torch-based training script.
