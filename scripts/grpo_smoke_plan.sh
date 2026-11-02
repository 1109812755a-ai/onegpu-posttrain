#!/usr/bin/env bash
set -euo pipefail

NATIVE_ROOT="${1:-artifacts/native/ms-swift/smoke}"
RUN_ROOT="${2:-runs/smoke}"
M2="${3:-$RUN_ROOT/sft-smoke-s42/merged/M2-SFT}"
mkdir -p "$NATIVE_ROOT" "$RUN_ROOT"

onegpu-posttrain plan-grpo \
  --config recipes/smoke/40-grpo-standard.yaml \
  --model "$M2" \
  --dataset "$NATIVE_ROOT/grpo.jsonl" \
  --output "$RUN_ROOT/grpo-standard-smoke-s42"

onegpu-posttrain plan-grpo \
  --config recipes/smoke/41-grpo-lowmem.yaml \
  --model "$M2" \
  --dataset "$NATIVE_ROOT/grpo.jsonl" \
  --output "$RUN_ROOT/grpo-lowmem-smoke-s42"
