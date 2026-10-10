#!/bin/sh
set -eu

cd "$(dirname "$(dirname "$0")")"

test_experiment() {
    config="$1"
    name="$2"
    checkpoint="exp/modelnet40/$name/model/model_last.pth"

    if [ ! -f "$checkpoint" ]; then
        echo "Missing completed checkpoint: $checkpoint" >&2
        exit 1
    fi

    sh scripts/test.sh \
        -g 1 \
        -d modelnet40 \
        -c "$config" \
        -n "$name" \
        -w model_last
}

test_experiment cls-ptv3-v1m1-0-laptop-full-c0-s42 full-c0-s42
test_experiment cls-ptv3-v1m1-0-laptop-full-m1-s42 full-m1-s42
