_base_ = ["./cls-ptv3-v1m1-0-laptop-tuning.py"]

# Retain all baseline orderings but do not randomly permute their input order.
model = dict(
    backbone=dict(
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        shuffle_orders=False,
    )
)
