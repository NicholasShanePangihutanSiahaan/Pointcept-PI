# Protocol: multi-seed confirmation of baseline vs Z-order

## Purpose

Confirm whether the apparent Z-order improvement persists across training
randomness. Dataset membership remains fixed from split seed 42. Only the
training RNG seed changes; this affects initialization, augmentation,
DataLoader randomness, dropout and order shuffling where applicable.

## Runs

| Model | Seed 42 | Seed 123 | Seed 501 |
|---|---|---|---|
| Baseline all orders | complete | baseline-10pct-split-s123 | baseline-10pct-split-s501 |
| Z-order | complete | ordering-z-s123 | ordering-z-s501 |

Both candidates use encoder patch size 1024. The baseline uses all four
orderings with initial order shuffle; Z-order uses only `z` with initial
shuffle disabled. All new runs inherit the development hook list, so official
test is not run automatically.

## Run and summarize

```bash
conda activate pointcept-torch2.5.0-cu12.4
sh scripts/run_modelnet_multiseed_confirmation.sh
python tools/summarize_modelnet_multiseed.py
```

Report each seed and `mean +/- sample SD`; do not report only the best seed.
Select a final candidate primarily by mean validation allAcc, with mAcc,
macro F1 and runtime as supporting measures. Official test remains untouched
until this comparison and subsequent hyperparameter decisions are complete.
