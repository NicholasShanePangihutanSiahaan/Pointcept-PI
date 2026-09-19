_base_ = ["./cls-ptv3-v1m1-0-laptop-tuning.py"]

# Single serialized ordering; deterministic order selection at input.
model = dict(backbone=dict(order=("hilbert",), shuffle_orders=False))
