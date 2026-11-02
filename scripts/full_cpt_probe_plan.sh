#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed/smoke}"
NATIVE_ROOT="${2:-artifacts/native/ms-swift/smoke}"
RUN_ROOT="${3:-runs/smoke}"

mkdir -p "$NATIVE_ROOT" "$RUN_ROOT"

onegpu-posttrain adapt-ms-swift --stage cpt \
  --input "$DATA_ROOT/finemath/canonical/cpt_train.jsonl" \
  --output "$NATIVE_ROOT/cpt.jsonl"

onegpu-posttrain plan-full-cpt-probe \
  --config recipes/smoke/11-full-cpt-probe.yaml \
  --dataset "$NATIVE_ROOT/cpt.jsonl" \
  --output "$RUN_ROOT/cpt-full-probe-s42"
