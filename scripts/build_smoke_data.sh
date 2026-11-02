#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-data/processed/smoke}
BENCH=configs/data/benchmarks.yaml

onegpu-posttrain build-finemath \
  --config configs/data/finemath.yaml \
  --benchmark-config "$BENCH" \
  --output "$OUT/finemath" \
  --train-token-budget 500000 \
  --holdout-token-budget 50000

onegpu-posttrain build-openr1 \
  --config configs/data/openr1.yaml \
  --benchmark-config "$BENCH" \
  --output "$OUT/openr1" \
  --sft-target 200 \
  --alignment-target 300 \
  --preference-target 200 \
  --preference-eval-target 50 \
  --grpo-target 200 \
  --holdout-target 100
