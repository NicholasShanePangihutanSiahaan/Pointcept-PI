_base_ = ["./cls-ptv3-v1m1-0-laptop-order-z-lr5e4.py"]

# Full official training split. Do not inspect the official test set during
# training; evaluate model_last once after the fixed 100-epoch schedule.
seed = 42
evaluate = False

model = dict(
    backbone=dict(
        order=("z",),
        shuffle_orders=False,
        pool_shuffle_orders=False,
        enc_order_indices=None,
    )
)

data = dict(
    train=dict(split="train", num_points=1024, loop=1),
    val=dict(split="test", num_points=1024),
    test=dict(split="test", num_points=1024),
)

# Fixed final evaluation protocol; no automatic test after training.
test = dict(num_repeat=10)
