_base_ = ["./cls-ptv3-v1m1-0-laptop-order-z-lr5e4.py"]

# Stage-aware ordering: Z, Hilbert, Z, Hilbert, Z.
model = dict(
    backbone=dict(
        order=("z", "hilbert"),
        shuffle_orders=False,
        pool_shuffle_orders=False,
        enc_order_indices=(0, 1, 0, 1, 0),
    )
)
