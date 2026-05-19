Each family has its own best hiperparameters

They were chosen based on frequency from `model_comparison.ipynb` output.

In case of ties (no single best hiperparameter within the three seeds), the `train_correctness` was used to break the tie, I used the specific seed configuration on the three seeds and checked which configuration had the best MEAN validation metrics between the seeds.