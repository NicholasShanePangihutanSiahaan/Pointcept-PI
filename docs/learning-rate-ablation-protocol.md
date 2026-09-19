# Protocol: PTv3 Z-order learning-rate ablation

## Question

Does reducing the learning rate improve the selected Z-order model under the
same 100-epoch OneCycle schedule?

## Controlled variables

- Model: PT-v3m1 classifier, Z-order only, initial order shuffle disabled.
- Encoder patch size: 1024; seed: 42.
- Dataset: fixed 798 train / 202 validation split.
- Input points: 1024; batch size: 4; workers: 2; epochs: 100.
- Architecture, augmentations, loss, weight decay and regularization unchanged.
- Official test is not run automatically.

## Learning-rate groups

| Run | Main LR / OneCycle max | Transformer-block LR / max | Status |
|---|---:|---:|---|
| ordering-z-lr3e4-s42 | 0.0003 | 0.00003 | pending |
| ordering-z-lr5e4-s42 | 0.0005 | 0.00005 | pending |
| ordering-z-s42 | 0.0010 | 0.00010 | complete control |

The block group remains at one tenth of the main group. Both optimizer group
LRs and the two OneCycle `max_lr` entries change together. Other OneCycle
parameters remain unchanged (`pct_start=0.05`, `div_factor=10`, and
`final_div_factor=1000`). Therefore, lower-LR runs also begin and end at
proportionally lower rates.

## Run and summarize

```bash
conda activate pointcept-torch2.5.0-cu12.4
sh scripts/run_modelnet_lr_ablation.sh
python tools/summarize_modelnet_lr_ablation.py
```

Select by validation `allAcc`; record `mAcc`, macro F1, selected epoch and
duration at the same run. Do not use the official test for selection. This is
a one-seed screening experiment. If a lower LR wins, it must later be
confirmed across seeds before replacing the current Z-order configuration.
