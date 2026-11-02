#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed/smoke}"
NATIVE_ROOT="${2:-artifacts/native/ms-swift/smoke}"
RUN_ROOT="${3:-runs/smoke}"
M1="${4:-$RUN_ROOT/cpt-peft-smoke-s42/merged/M1-CPT}"
SFT_RUN="$RUN_ROOT/sft-smoke-s42"

mkdir -p "$NATIVE_ROOT" "$RUN_ROOT"

onegpu-posttrain adapt-ms-swift --stage sft \
  --input "$DATA_ROOT/openr1/canonical/sft.jsonl" \
  --output "$NATIVE_ROOT/sft.jsonl"

onegpu-posttrain plan-sft \
  --config recipes/smoke/20-sft.yaml \
  --model "$M1" \
  --dataset "$NATIVE_ROOT/sft.jsonl" \
  --output "$SFT_RUN"

onegpu-posttrain run-model-eval \
  --config recipes/smoke/00-base-eval.yaml \
  --run-id sft-smoke-s42 \
  --stage sft \
  --model "$SFT_RUN/merged/M2-SFT" \
  --revision "" \
  --internal-holdout "$DATA_ROOT/openr1/canonical/internal_holdout.jsonl" \
  --output "$SFT_RUN/evaluation" \
  --dry-run
