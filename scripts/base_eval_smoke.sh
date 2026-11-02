#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed/smoke}"
OUT="${2:-runs/smoke/base-smoke-s42/evaluation}"

onegpu-posttrain run-model-eval \
  --config recipes/smoke/00-base-eval.yaml \
  --run-id base-smoke-s42 \
  --stage base \
  --model Qwen/Qwen3-0.6B-Base \
  --internal-holdout "$DATA_ROOT/openr1/canonical/internal_holdout.jsonl" \
  --domain-holdout "$DATA_ROOT/finemath/canonical/cpt_holdout.jsonl" \
  --output "$OUT"
