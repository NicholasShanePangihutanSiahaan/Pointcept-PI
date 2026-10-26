# Full ModelNet40 protocol for Windows/WSL2

## Experimental question

Does M1 stage-aware serialized attention ordering (Z-Z-Z-H-H) outperform the
C0 control (Z-Z-Z-Z-Z) when trained with the entire official ModelNet40
training split?

## Dataset and leakage control

- Official train: 9,843 objects, all used for training.
- Official test: 2,468 objects, never evaluated during training.
- Training uses a fixed 100-epoch schedule and saves model_last.
- After both models finish, each model_last is evaluated once with the same
  10-repeat voting protocol.
- Report the final voting evaluation, not a maximum test score selected after
  observing labels.

This is a final confirmation protocol. Do not use official-test results to tune
new variants afterward.

## Controlled settings

Both models use seed 42, 1,024 points/object, patch size 1,024, batch size 4,
AdamW, main/block LR 5e-4/5e-5, drop path 0.3, 100 epochs, identical transforms,
and 38,842,792 parameters. Only attention ordering differs.

## Windows requirement

Use Ubuntu through WSL2, not native Windows. Pointcept relies on Linux-oriented
CUDA packages including spconv, Flash Attention, and compiled point-cloud
operations. The Windows host must have an NVIDIA GPU with a WSL-compatible
driver. Verify inside Ubuntu:

    nvidia-smi
    python -c "import torch; print(torch.cuda.is_available())"

Transfer the complete modified repository (or push/clone a Git commit containing
the stage-aware source changes), the conda environment specification, and the
modelnet40_normal_resampled dataset. The upstream repository alone does not
contain this project's M1 modification.

## Preflight

    conda activate pointcept-torch2.5.0-cu12.4
    cd ~/Pointcept
    python -m tools.validate_ptv3_stage_ordering
    wc -l data/modelnet40_normal_resampled/modelnet40_train.txt
    wc -l data/modelnet40_normal_resampled/modelnet40_test.txt
    python tools/summarize_modelnet_full_c0_m1.py

Expected counts are 9,843 train and 2,468 test.

## Train both models

    sh scripts/run_modelnet_full_c0_m1_s42.sh

The runner trains C0 first and M1 second. It refuses to overwrite an incomplete
experiment. Full-data training is expected to take many hours; exact duration
depends on the professor's GPU. Do not run the two models simultaneously on one
GPU.

## Final official test

Only after both training jobs finish:

    sh scripts/test_modelnet_full_c0_m1_s42.sh
    python tools/summarize_modelnet_full_c0_m1.py

Report official-test allAcc and mAcc for both models, parameter count, training
duration, GPU model, and peak VRAM if measured. M1 is supported only if it
outperforms C0 under this identical protocol.
