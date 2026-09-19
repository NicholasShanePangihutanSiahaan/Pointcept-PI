_base_ = ["./cls-ptv3-v1m1-0-laptop-tuning.py"]

# Alternate between two non-transposed serialized orderings.
model = dict(backbone=dict(order=("z", "hilbert"), shuffle_orders=True))
