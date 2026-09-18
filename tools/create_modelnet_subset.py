#!/usr/bin/env python3
"""Create a reproducible class-stratified ModelNet40 training subset."""

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a stratified subset from a ModelNet40 split file."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/modelnet40_normal_resampled"),
        help="ModelNet40 root containing modelnet40_train.txt.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.10,
        help="Fraction selected from every class (default: 0.10).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-split", default="train", help="Source split name (default: train)."
    )
    parser.add_argument(
        "--output-split",
        default=None,
        help="Output split name; default is train<percentage>, e.g. train10.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing output split."
    )
    return parser.parse_args()


def class_name(sample_name):
    # Model names end in a numeric ID, for example night_stand_0001.
    name, separator, sample_id = sample_name.rpartition("_")
    if not separator or not sample_id.isdigit():
        raise ValueError(f"Unexpected ModelNet40 sample name: {sample_name!r}")
    return name


def main():
    args = parse_args()
    if not 0 < args.ratio <= 1:
        raise ValueError("--ratio must be greater than 0 and at most 1")

    percentage = round(args.ratio * 100)
    output_split = args.output_split or f"train{percentage}"
    source = args.data_root / f"modelnet40_{args.source_split}.txt"
    target = args.data_root / f"modelnet40_{output_split}.txt"

    if not source.is_file():
        raise FileNotFoundError(f"Source split not found: {source}")
    if target.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {target}; use --force to replace it")

    groups = defaultdict(list)
    for sample in source.read_text(encoding="utf-8").splitlines():
        sample = sample.strip()
        if sample:
            groups[class_name(sample)].append(sample)

    rng = random.Random(args.seed)
    selected = []
    print("Class                 Selected / Total")
    print("-" * 40)
    for label, samples in sorted(groups.items()):
        rng.shuffle(samples)
        count = max(1, math.ceil(len(samples) * args.ratio))
        selected.extend(samples[:count])
        print(f"{label:22s} {count:4d} / {len(samples):4d}")

    # Shuffle class blocks while retaining determinism from the same seed.
    rng.shuffle(selected)
    target.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print("-" * 40)
    print(f"Saved {len(selected)} of {sum(map(len, groups.values()))} samples")
    print(f"Split name : {output_split}")
    print(f"Output     : {target}")


if __name__ == "__main__":
    main()
