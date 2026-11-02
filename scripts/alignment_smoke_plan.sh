#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed/smoke}"
NATIVE_ROOT="${2:-artifacts/native/ms-swift/smoke}"
RUN_ROOT="${3:-runs/smoke}"
M2="${4:-$RUN_ROOT/sft-smoke-s42/merged/M2-SFT}"

mkdir -p "$NATIVE_ROOT" "$RUN_ROOT"

onegpu-posttrain adapt-ms-swift --stage dpo \
  --input "$DATA_ROOT/openr1/canonical/preference.jsonl" \
  --output "$NATIVE_ROOT/preference.jsonl"

onegpu-posttrain plan-dpo \
  --config recipes/smoke/30-dpo.yaml \
  --model "$M2" \
  --dataset "$NATIVE_ROOT/preference.jsonl" \
  --output "$RUN_ROOT/dpo-smoke-s42"

onegpu-posttrain plan-simpo \
  --config recipes/smoke/31-simpo-original.yaml \
  --model "$M2" \
  --dataset "$NATIVE_ROOT/preference.jsonl" \
  --output "$RUN_ROOT/simpo-original-smoke-s42"

for MODEL in "$M2" "$RUN_ROOT/dpo-smoke-s42/merged/M3-DPO" "$RUN_ROOT/simpo-original-smoke-s42/merged/M3-SimPO"; do
  onegpu-posttrain eval-preference \
    --model "$MODEL" \
    --input "$DATA_ROOT/openr1/canonical/preference_eval.jsonl" \
    --output "$RUN_ROOT/plans/$(basename "$MODEL")-preference" \
    --limit 10 \
    --dry-run
done
