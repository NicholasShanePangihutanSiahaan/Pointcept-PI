#!/usr/bin/env python3
"""Validate stage-aware ordering and backward-compatible PTv3 defaults."""

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (
    PointTransformerV3,
)


COMMON = dict(
    in_channels=6,
    order=("z", "hilbert"),
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 2, 2),
    enc_channels=(8, 16, 24, 32, 40),
    enc_num_head=(1, 2, 3, 4, 5),
    enc_patch_size=(8, 8, 8, 8, 8),
    enc_mode=True,
    enable_flash=False,
    upcast_attention=False,
    upcast_softmax=False,
)


def attention_indices(model):
    return [
        [stage._modules[f"block{i}"].attn.order_index for i in range(2)]
        for stage in model.enc._modules.values()
    ]


def pooling_shuffle_flags(model):
    return [
        stage._modules["down"].shuffle_orders
        for stage in model.enc._modules.values()
        if "down" in stage._modules
    ]


def main():
    stage_aware = PointTransformerV3(
        **COMMON,
        shuffle_orders=False,
        pool_shuffle_orders=False,
        enc_order_indices=(0, 0, 0, 1, 1),
    )
    assert attention_indices(stage_aware) == [
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
    ]
    assert pooling_shuffle_flags(stage_aware) == [False, False, False, False]

    legacy = PointTransformerV3(**COMMON)
    assert attention_indices(legacy) == [[0, 1]] * 5
    assert pooling_shuffle_flags(legacy) == [True, True, True, True]

    invalid_values = ((0, 0), (0, 0, 0, 1, 2))
    for value in invalid_values:
        try:
            PointTransformerV3(**COMMON, enc_order_indices=value)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Invalid enc_order_indices accepted: {value}")

    print("stage-aware indices:", attention_indices(stage_aware))
    print("stage-aware pooling shuffle:", pooling_shuffle_flags(stage_aware))
    print("legacy indices:", attention_indices(legacy))
    print("legacy pooling shuffle:", pooling_shuffle_flags(legacy))
    print("validation: PASS")


if __name__ == "__main__":
    main()
