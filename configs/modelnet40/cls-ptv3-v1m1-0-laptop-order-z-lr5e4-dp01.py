_base_ = ["./cls-ptv3-v1m1-0-laptop-order-z-lr5e4.py"]

# Reduce stochastic-depth regularization from the control value of 0.3.
model = dict(backbone=dict(drop_path=0.1))
