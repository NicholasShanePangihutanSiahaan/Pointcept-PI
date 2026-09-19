#!/usr/bin/env python3
"""Summarize validation metrics from the controlled ModelNet40 ablations."""

import datetime as dt
import re
from pathlib import Path


EXPERIMENTS = (
    ("baseline", "baseline-10pct-split"),
    ("patch-128", "tuning-patch128-s42"),
    ("patch-256", "tuning-patch256-s42"),
    ("patch-512", "tuning-patch512-s42"),
    ("order-z", "ordering-z-s42"),
    ("order-hilbert", "ordering-hilbert-s42"),
    ("order-z-hilbert", "ordering-z-hilbert-s42"),
    ("order-all-noshuffle", "ordering-all-noshuffle-s42"),
)

TIME_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
VAL_PATTERN = re.compile(
    r"Val result: mIoU/macroF1/mAcc/mPrecision/allAcc "
    r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)\."
)


def timestamp(line):
    return dt.datetime.strptime(line[1:24], TIME_FORMAT)


def summarize(log_path):
    lines = log_path.read_text(encoding="utf-8").splitlines()
    start_lines = [line for line in lines if "Start Training" in line]
    records = []
    for line in lines:
        match = VAL_PATTERN.search(line)
        if match:
            records.append((timestamp(line), tuple(map(float, match.groups()))))
    if not start_lines or not records:
        return None

    # Python's max retains the first record when allAcc values are tied.
    best_index, (best_time, metrics) = max(
        enumerate(records, start=1), key=lambda item: item[1][1][-1]
    )
    duration = records[-1][0] - timestamp(start_lines[0])
    miou, macro_f1, macc, precision, allacc = metrics
    return dict(
        epochs=len(records),
        best_epoch=best_index,
        miou=miou,
        macro_f1=macro_f1,
        macc=macc,
        precision=precision,
        allacc=allacc,
        duration=duration,
    )


def main():
    root = Path("exp/modelnet40")
    print(
        f"{'experiment':24s} {'status':10s} {'epoch':>5s} "
        f"{'allAcc':>7s} {'mAcc':>7s} {'F1':>7s} {'duration':>10s}"
    )
    print("-" * 83)
    for label, directory in EXPERIMENTS:
        log_path = root / directory / "train.log"
        if not log_path.is_file():
            print(f"{label:24s} {'pending':10s}")
            continue
        result = summarize(log_path)
        if result is None:
            print(f"{label:24s} {'incomplete':10s}")
            continue
        status = "complete" if result["epochs"] == 100 else "running"
        duration = str(result["duration"]).split(".")[0]
        print(
            f"{label:24s} {status:10s} {result['best_epoch']:5d} "
            f"{100 * result['allacc']:7.2f} {100 * result['macc']:7.2f} "
            f"{100 * result['macro_f1']:7.2f} {duration:>10s}"
        )


if __name__ == "__main__":
    main()
