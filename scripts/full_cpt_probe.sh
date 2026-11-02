#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed/smoke}"
NATIVE_ROOT="${2:-artifacts/native/ms-swift/smoke}"
RUN_ROOT="${3:-runs/smoke}"
PEFT_RUN="$RUN_ROOT/cpt-peft-smoke-s42"
FULL_RUN="$RUN_ROOT/cpt-full-probe-s42"

mkdir -p "$NATIVE_ROOT" "$RUN_ROOT"

onegpu-posttrain adapt-ms-swift --stage cpt \
  --input "$DATA_ROOT/finemath/canonical/cpt_train.jsonl" \
  --output "$NATIVE_ROOT/cpt.jsonl"

# OOM is intentionally recorded as a valid probe outcome and returns success.
onegpu-posttrain run-full-cpt-probe \
  --config recipes/smoke/11-full-cpt-probe.yaml \
  --dataset "$NATIVE_ROOT/cpt.jsonl" \
  --output "$FULL_RUN"

if [[ -f "$PEFT_RUN/run.json" ]]; then
  onegpu-posttrain compare-cpt-training \
    --peft "$PEFT_RUN/run.json" \
    --full "$FULL_RUN/run.json" \
    --output "$RUN_ROOT/cpt-feasibility"
  echo "CPT feasibility report: $RUN_ROOT/cpt-feasibility/cpt_feasibility.md"
else
  echo "PEFT run.json not found; probe result written to $FULL_RUN/run.json"
fi
