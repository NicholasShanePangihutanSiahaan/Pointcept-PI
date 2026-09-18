_base_ = ["./cls-ptv3-v1m1-0-laptop-10pct.py"]

# Fix both the dataset partition and model initialization for fair comparisons.
seed = 42

data = dict(
    train=dict(split="train10_train"),
    val=dict(split="train10_val"),
    # Do not use this official test split for hyperparameter selection.
    test=dict(split="test"),
)
