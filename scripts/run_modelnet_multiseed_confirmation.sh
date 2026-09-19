#!/bin/sh
set -eu

cd "$(dirname "$(dirname "$0")")"

run_experiment() {
    config="$1"
    name="$2"
    log="exp/modelnet40/$name/train.log"

    if [ -f "$log" ] && grep -q "Best allAcc:" "$log"; then
        echo "Skipping completed experiment: $name"
        return
    fi
    if [ -e "exp/modelnet40/$name" ]; then
        echo "Refusing to overwrite incomplete experiment: $name" >&2
        echo "Resume it manually or choose a new experiment name." >&2
        exit 1
    fi

    sh scripts/train.sh \
        -g 1 \
        -d modelnet40 \
        -c "$config" \
        -n "$name"
}

run_experiment cls-ptv3-v1m1-0-laptop-baseline-s123 baseline-10pct-split-s123
run_experiment cls-ptv3-v1m1-0-laptop-baseline-s501 baseline-10pct-split-s501
run_experiment cls-ptv3-v1m1-0-laptop-order-z-s123 ordering-z-s123
run_experiment cls-ptv3-v1m1-0-laptop-order-z-s501 ordering-z-s501
