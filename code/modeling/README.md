# Modeling Code

This folder contains neural transition-model definitions that are needed when I load frozen LSTM source models.

## Files

- `models.py`: defines `StateCentricLSTM` and `StateCentricLSTM_Delta`.

The downstream MLP regressor is not defined here. It is defined in `code/downstream/train_single_correctness_head.py`. This folder is only for loading the frozen LSTM transition checkpoints.