_base_ = ["./cls-ptv3-v1m1-0-laptop-order-z.py"]

# Main LR 3e-4; transformer-block LR remains one tenth of the main LR.
optimizer = dict(lr=0.0003)
param_dicts = [dict(keyword="block", lr=0.00003)]
scheduler = dict(max_lr=[0.0003, 0.00003])
