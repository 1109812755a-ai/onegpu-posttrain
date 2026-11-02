#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed/smoke}"
NATIVE_ROOT="${2:-artifacts/native/ms-swift/smoke}"
RUN_ROOT="${3:-runs/smoke}"

mkdir -p "$NATIVE_ROOT" "$RUN_ROOT"

# Ensure the CPT training file is in ms-swift pre-training native format.
onegpu-posttrain adapt-ms-swift --stage cpt \
  --input "$DATA_ROOT/finemath/canonical/cpt_train.jsonl" \
  --output "$NATIVE_ROOT/cpt.jsonl"

onegpu-posttrain plan-peft-cpt \
  --config recipes/smoke/10-peft-cpt.yaml \
  --dataset "$NATIVE_ROOT/cpt.jsonl" \
  --output "$RUN_ROOT/cpt-peft-smoke-s42"

onegpu-posttrain run-model-eval \
  --config recipes/smoke/00-base-eval.yaml \
  --run-id base-smoke-s42 \
  --stage base \
  --model Qwen/Qwen3-0.6B-Base \
  --internal-holdout "$DATA_ROOT/openr1/canonical/internal_holdout.jsonl" \
  --domain-holdout "$DATA_ROOT/finemath/canonical/cpt_holdout.jsonl" \
  --output "$RUN_ROOT/base-smoke-s42/evaluation" \
  --dry-run

# Planned post-CPT merged model location. Empty revision explicitly prevents the
# Base HF revision from being inherited before the local directory exists.
onegpu-posttrain run-model-eval \
  --config recipes/smoke/00-base-eval.yaml \
  --run-id cpt-peft-smoke-s42 \
  --stage cpt \
  --model "$RUN_ROOT/cpt-peft-smoke-s42/merged/M1-CPT" \
  --revision "" \
  --internal-holdout "$DATA_ROOT/openr1/canonical/internal_holdout.jsonl" \
  --domain-holdout "$DATA_ROOT/finemath/canonical/cpt_holdout.jsonl" \
  --output "$RUN_ROOT/cpt-peft-smoke-s42/evaluation" \
  --dry-run
