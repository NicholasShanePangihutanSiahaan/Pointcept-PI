#!/usr/bin/env python3
"""Report per-seed results and aggregate statistics for baseline vs Z-order."""

import datetime as dt
import re
import statistics
from pathlib import Path


RUNS = {
    "baseline": {
        42: "baseline-10pct-split",
        123: "baseline-10pct-split-s123",
        501: "baseline-10pct-split-s501",
    },
    "order-z": {
        42: "ordering-z-s42",
        123: "ordering-z-s123",
        501: "ordering-z-s501",
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
    miou, macro_f1, macc, precision, allacc = metrics
    return {
        "epoch": epoch,
        "miou": miou,
        "macro_f1": macro_f1,
        "macc": macc,
        "precision": precision,
        "allacc": allacc,
        "duration": records[-1][0] - timestamp(starts[0]),
    }


def percentage_stats(values):
    mean = statistics.mean(values) * 100
    # Sample SD (n-1), appropriate for reporting variability across seeds.
    sd = statistics.stdev(values) * 100
    return mean, sd


def main():
    root = Path("exp/modelnet40")
    grouped_results = {}
    print(
        f"{'model':10s} {'seed':>5s} {'status':10s} {'epoch':>5s} "
        f"{'allAcc':>7s} {'mAcc':>7s} {'F1':>7s} {'duration':>10s}"
    )
    print("-" * 81)
    for model, seeds in RUNS.items():
        grouped_results[model] = []
        for seed, directory in seeds.items():
            log_path = root / directory / "train.log"
            if not log_path.is_file():
                print(f"{model:10s} {seed:5d} {'pending':10s}")
                continue
            result = read_result(log_path)
            if result is None:
                print(f"{model:10s} {seed:5d} {'incomplete':10s}")
                continue
            grouped_results[model].append(result)
            duration = str(result["duration"]).split(".")[0]
            print(
                f"{model:10s} {seed:5d} {'complete':10s} {result['epoch']:5d} "
                f"{100 * result['allacc']:7.2f} {100 * result['macc']:7.2f} "
                f"{100 * result['macro_f1']:7.2f} {duration:>10s}"
            )

    print("\nAggregate (mean +/- sample SD across exactly three seeds)")
    print(f"{'model':10s} {'allAcc':>18s} {'mAcc':>18s} {'macro F1':>18s}")
    print("-" * 68)
    for model, results in grouped_results.items():
        if len(results) != 3:
            print(f"{model:10s} {'pending':>18s}")
            continue
        fields = []
        for metric in ("allacc", "macc", "macro_f1"):
            mean, sd = percentage_stats([result[metric] for result in results])
            fields.append(f"{mean:.2f} +/- {sd:.2f}")
        print(f"{model:10s} {fields[0]:>18s} {fields[1]:>18s} {fields[2]:>18s}")


if __name__ == "__main__":
    main()
