#!/usr/bin/env python3
"""Split a ModelNet list into reproducible, class-stratified train/validation lists."""

import argparse
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create disjoint stratified train/validation ModelNet split files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/modelnet40_normal_resampled"),
    )
    parser.add_argument(
        "--source-split",
        default="train10",
        help="Source split without the modelnet40_ prefix (default: train10).",
    )
    parser.add_argument("--train-split", default="train10_train")
    parser.add_argument("--val-split", default="train10_val")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation fraction selected within every class (default: 0.20).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sample_class(sample_name):
    label, separator, sample_id = sample_name.rpartition("_")
    if not separator or not sample_id.isdigit():
        raise ValueError(f"Unexpected ModelNet sample name: {sample_name!r}")
    return label


def split_path(root, split):
    return root / f"modelnet40_{split}.txt"


def main():
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    source = split_path(args.data_root, args.source_split)
    train_target = split_path(args.data_root, args.train_split)
    val_target = split_path(args.data_root, args.val_split)
    if not source.is_file():
        raise FileNotFoundError(f"Source split not found: {source}")
    for target in (train_target, val_target):
        if target.exists() and not args.force:
            raise FileExistsError(f"Output exists: {target}; use --force to replace it")

    groups = defaultdict(list)
    source_samples = []
    for sample in source.read_text(encoding="utf-8").splitlines():
        sample = sample.strip()
        if sample:
            source_samples.append(sample)
            groups[sample_class(sample)].append(sample)

    rng = random.Random(args.seed)
    train_samples = []
    val_samples = []
    print("Class                    Train / Val / Total")
    print("-" * 48)
    for label, samples in sorted(groups.items()):
        rng.shuffle(samples)
        # At least one item remains on each side for every represented class.
        val_count = round(len(samples) * args.val_ratio)
        val_count = min(max(1, val_count), len(samples) - 1)
        val_samples.extend(samples[:val_count])
        train_samples.extend(samples[val_count:])
        print(
            f"{label:22s} {len(samples) - val_count:5d} /"
            f" {val_count:3d} / {len(samples):5d}"
        )

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    source_set = set(source_samples)
    train_set = set(train_samples)
    val_set = set(val_samples)
    assert train_set.isdisjoint(val_set)
    assert train_set | val_set == source_set
    assert len(train_samples) == len(train_set), "duplicate samples in train split"
    assert len(val_samples) == len(val_set), "duplicate samples in validation split"

    train_target.write_text("\n".join(train_samples) + "\n", encoding="utf-8")
    val_target.write_text("\n".join(val_samples) + "\n", encoding="utf-8")

    print("-" * 48)
    print(f"Source     : {len(source_samples):4d} ({source})")
    print(f"Train      : {len(train_samples):4d} ({train_target})")
    print(f"Validation : {len(val_samples):4d} ({val_target})")
    print(f"Seed       : {args.seed}")
    print("Integrity  : no overlap; union equals source")


if __name__ == "__main__":
    main()
