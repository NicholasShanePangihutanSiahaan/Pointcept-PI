# Protocol: PTv3 serialized-ordering ablation

## Controlled variables

- Dataset: ModelNet40 `train10_train` (798 objects).
- Validation: `train10_val` (202 objects).
- Official test is not run during tuning.
- Dataset and training seed: 42.
- Input points: 1024; encoder patch size: 1024.
- Batch size: 4; workers: 2; epochs: 100.
- Optimizer, scheduler, augmentation, architecture and loss remain unchanged.

## Experiments

| ID | Order | `shuffle_orders` | Run name |
|---|---|---:|---|
| O0 | z, z-trans, hilbert, hilbert-trans | true | baseline-10pct-split |
| O1 | z | false | ordering-z-s42 |
| O2 | hilbert | false | ordering-hilbert-s42 |
| O3 | z, hilbert | true | ordering-z-hilbert-s42 |
| O4 | z, z-trans, hilbert, hilbert-trans | false | ordering-all-noshuffle-s42 |

O0 is the completed baseline and is not rerun. Each new experiment starts
from random initialization, not from a baseline checkpoint.

In this PTv3 implementation, backbone `shuffle_orders` controls the random
permutation produced by the initial `Point.serialization()` call. The
`SerializedPooling` modules are constructed with their own default
`shuffle_orders=True`; therefore O4 should be described precisely as
"all initial orderings without input-order shuffle", not as disabling every
shuffle operation throughout the network.

## Run

Activate the environment and execute all pending runs sequentially:

```bash
conda activate pointcept-torch2.5.0-cu12.4
sh scripts/run_modelnet_ordering_ablation.sh
```

The script skips completed runs and refuses to overwrite an incomplete run.
To run only one experiment, use `scripts/train.sh`, for example:

```bash
sh scripts/train.sh -g 1 -d modelnet40 \
  -c cls-ptv3-v1m1-0-laptop-order-z \
  -n ordering-z-s42
```

## Selection rule

Select the candidate with the highest validation `allAcc`. Record `mAcc`,
macro F1, selected epoch and training duration at that same epoch/run.
Do not select an ordering using official-test results. With 202 validation
objects, one correct prediction changes `allAcc` by about 0.495 percentage
points, so close results need confirmation with additional training seeds.
