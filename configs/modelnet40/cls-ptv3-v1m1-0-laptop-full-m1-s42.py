_base_ = ["./cls-ptv3-v1m1-0-laptop-full-c0-s42.py"]

# Proposed stage-aware attention ordering: Z, Z, Z, Hilbert, Hilbert.
model = dict(
    backbone=dict(
        order=("z", "hilbert"),
        shuffle_orders=False,
        pool_shuffle_orders=False,
        enc_order_indices=(0, 0, 0, 1, 1),
    )
)
