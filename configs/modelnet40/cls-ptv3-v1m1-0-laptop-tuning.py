_base_ = ["./cls-ptv3-v1m1-0-laptop-10pct-split.py"]

# Development runs evaluate only validation data. Official testing is separate.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]
