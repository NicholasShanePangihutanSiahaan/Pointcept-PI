#!/usr/bin/env python3
"""Summarize the controlled PTv3 drop-path ablation on seed 42."""

import datetime as dt
import re
from pathlib import Path


RUNS = (
    (0.1, "ordering-z-lr5e4-dp01-s42"),
    (0.2, "ordering-z-lr5e4-dp02-s42"),
    (0.3, "ordering-z-lr5e4-s42"),
)
VAL_PATTERN = re.compile(
    r"Val result: mIoU/macroF1/mAcc/mPrecision/allAcc "
    r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)\."
)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S,%f"


def timestamp(line):
    return dt.datetime.strptime(line[1:24], TIME_FORMAT)


def summarize(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [line for line in lines if "Start Training" in line]
    records = []
    for line in lines:
        match = VAL_PATTERN.search(line)
        if match:
            records.append((timestamp(line), tuple(map(float, match.groups()))))
    if not starts or not records:
        return None
    best_epoch, (_, metrics) = max(
        enumerate(records, start=1), key=lambda item: item[1][1][-1]
    )
    return {
        "epochs": len(records),
        "epoch": best_epoch,
        "macro_f1": metrics[1],
        "macc": metrics[2],
        "allacc": metrics[4],
        "duration": records[-1][0] - timestamp(starts[0]),
    }


def main():
    root = Path("exp/modelnet40")
    print(
        f"{'drop_path':>9s} {'status':10s} {'epoch':>5s} {'allAcc':>7s} "
        f"{'mAcc':>7s} {'F1':>7s} {'duration':>10s}"
    )
    print("-" * 68)
    for drop_path, directory in RUNS:
        path = root / directory / "train.log"
        if not path.is_file():
            print(f"{drop_path:9.1f} {'pending':10s}")
            continue
        result = summarize(path)
        if result is None:
            print(f"{drop_path:9.1f} {'incomplete':10s}")
            continue
        status = "complete" if result["epochs"] == 100 else "running"
        duration = str(result["duration"]).split(".")[0]
        print(
            f"{drop_path:9.1f} {status:10s} {result['epoch']:5d} "
            f"{100 * result['allacc']:7.2f} {100 * result['macc']:7.2f} "
            f"{100 * result['macro_f1']:7.2f} {duration:>10s}"
        )


if __name__ == "__main__":
    main()
