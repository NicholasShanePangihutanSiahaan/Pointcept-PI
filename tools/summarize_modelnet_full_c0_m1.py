#!/usr/bin/env python3
"""Summarize fixed-epoch full-ModelNet40 C0 and M1 experiments."""

import datetime as dt
import re
from pathlib import Path


RUNS = (
    ("C0", "Z-Z-Z-Z-Z", "full-c0-s42"),
    ("M1", "Z-Z-Z-H-H", "full-m1-s42"),
)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
TEST_PATTERN = re.compile(r"Val result: mAcc/allAcc ([0-9.]+)/([0-9.]+)")


def timestamp(line):
    return dt.datetime.strptime(line[1:24], TIME_FORMAT)


def training_result(path):
    if not path.is_file():
        return "pending", None
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [line for line in lines if "Start Training" in line]
    completed = [line for line in lines if "Train: [100/100]" in line]
    if not starts or not completed:
        return "running", None
    checkpoint_lines = [line for line in lines if "Saving checkpoint to:" in line]
    end_line = checkpoint_lines[-1] if checkpoint_lines else completed[-1]
    return "complete", timestamp(end_line) - timestamp(starts[0])


def official_test_result(path):
    if not path.is_file():
        return None
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TEST_PATTERN.search(line)
        if match:
            records.append(tuple(map(float, match.groups())))
    # Report the final cumulative voting evaluation, not a maximum selected
    # after observing official-test labels.
    return records[-1] if records else None


def main():
    root = Path("exp/modelnet40")
    print(
        f"{'model':5s} {'stage order':11s} {'training':10s} "
        f"{'duration':>10s} {'test mAcc':>10s} {'test allAcc':>11s}"
    )
    print("-" * 67)
    for model, stage_order, directory in RUNS:
        run_dir = root / directory
        status, duration = training_result(run_dir / "train.log")
        test_result = official_test_result(run_dir / "test.log")
        duration_text = str(duration).split(".")[0] if duration else "-"
        if test_result is None:
            macc_text = allacc_text = "pending"
        else:
            macc, allacc = test_result
            macc_text = f"{100 * macc:.2f}"
            allacc_text = f"{100 * allacc:.2f}"
        print(
            f"{model:5s} {stage_order:11s} {status:10s} "
            f"{duration_text:>10s} {macc_text:>10s} {allacc_text:>11s}"
        )


if __name__ == "__main__":
    main()
