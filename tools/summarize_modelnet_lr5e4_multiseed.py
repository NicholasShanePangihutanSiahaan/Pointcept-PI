#!/usr/bin/env python3
"""Compare Z-order LR 1e-3 and LR 5e-4 across three fixed seeds."""

import datetime as dt
import re
import statistics
from pathlib import Path


RUNS = {
    "LR 1e-3": {
        42: "ordering-z-s42",
        123: "ordering-z-s123",
        501: "ordering-z-s501",
    },
    "LR 5e-4": {
        42: "ordering-z-lr5e4-s42",
        123: "ordering-z-lr5e4-s123",
        501: "ordering-z-lr5e4-s501",
    },
}
VAL_PATTERN = re.compile(
    r"Val result: mIoU/macroF1/mAcc/mPrecision/allAcc "
    r"([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)\."
)
TIME_FORMAT = "%Y-%m-%d %H:%M:%S,%f"


def timestamp(line):
    return dt.datetime.strptime(line[1:24], TIME_FORMAT)


def read_result(log_path):
    lines = log_path.read_text(encoding="utf-8").splitlines()
    starts = [line for line in lines if "Start Training" in line]
    records = []
    for line in lines:
        match = VAL_PATTERN.search(line)
        if match:
            records.append((timestamp(line), tuple(map(float, match.groups()))))
    if not starts or len(records) != 100:
        return None
    epoch, (_, metrics) = max(
        enumerate(records, start=1), key=lambda item: item[1][1][-1]
    )
    return {
        "epoch": epoch,
        "macro_f1": metrics[1],
        "macc": metrics[2],
        "allacc": metrics[4],
        "duration": records[-1][0] - timestamp(starts[0]),
    }


def percentage_stats(values):
    return statistics.mean(values) * 100, statistics.stdev(values) * 100


def main():
    root = Path("exp/modelnet40")
    grouped = {}
    print(
        f"{'setting':9s} {'seed':>5s} {'status':10s} {'epoch':>5s} "
        f"{'allAcc':>7s} {'mAcc':>7s} {'F1':>7s} {'duration':>10s}"
    )
    print("-" * 80)
    for setting, seeds in RUNS.items():
        grouped[setting] = []
        for seed, directory in seeds.items():
            path = root / directory / "train.log"
            if not path.is_file():
                print(f"{setting:9s} {seed:5d} {'pending':10s}")
                continue
            result = read_result(path)
            if result is None:
                print(f"{setting:9s} {seed:5d} {'incomplete':10s}")
                continue
            grouped[setting].append(result)
            duration = str(result["duration"]).split(".")[0]
            print(
                f"{setting:9s} {seed:5d} {'complete':10s} {result['epoch']:5d} "
                f"{100 * result['allacc']:7.2f} {100 * result['macc']:7.2f} "
                f"{100 * result['macro_f1']:7.2f} {duration:>10s}"
            )

    print("\nAggregate (mean +/- sample SD across exactly three seeds)")
    print(f"{'setting':9s} {'allAcc':>18s} {'mAcc':>18s} {'macro F1':>18s}")
    print("-" * 67)
    for setting, results in grouped.items():
        if len(results) != 3:
            print(f"{setting:9s} {'pending':>18s}")
            continue
        fields = []
        for metric in ("allacc", "macc", "macro_f1"):
            mean, sd = percentage_stats([result[metric] for result in results])
            fields.append(f"{mean:.2f} +/- {sd:.2f}")
        print(f"{setting:9s} {fields[0]:>18s} {fields[1]:>18s} {fields[2]:>18s}")


if __name__ == "__main__":
    main()
