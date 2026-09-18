_base_ = ["./cls-ptv3-v1m1-0-laptop-tuning.py"]
model = dict(backbone=dict(enc_patch_size=(128, 128, 128, 128, 128)))
