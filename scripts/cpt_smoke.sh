#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed/smoke}"
NATIVE_ROOT="${2:-artifacts/native/ms-swift/smoke}"
RUN_ROOT="${3:-runs/smoke}"
BASE_RUN="$RUN_ROOT/base-smoke-s42"
CPT_RUN="$RUN_ROOT/cpt-peft-smoke-s42"
M1="$CPT_RUN/merged/M1-CPT"

mkdir -p "$NATIVE_ROOT" "$RUN_ROOT"

onegpu-posttrain adapt-ms-swift --stage cpt \
  --input "$DATA_ROOT/finemath/canonical/cpt_train.jsonl" \
  --output "$NATIVE_ROOT/cpt.jsonl"

# Evaluate Base on the same official/internal/domain protocol used after CPT.
onegpu-posttrain run-model-eval \
  --config recipes/smoke/00-base-eval.yaml \
  --run-id base-smoke-s42 \
  --stage base \
  --model Qwen/Qwen3-0.6B-Base \
  --internal-holdout "$DATA_ROOT/openr1/canonical/internal_holdout.jsonl" \
  --domain-holdout "$DATA_ROOT/finemath/canonical/cpt_holdout.jsonl" \
  --output "$BASE_RUN/evaluation"

# Train LoRA PEFT-CPT and merge the latest checkpoint into M1.
onegpu-posttrain run-peft-cpt \
  --config recipes/smoke/10-peft-cpt.yaml \
  --dataset "$NATIVE_ROOT/cpt.jsonl" \
  --output "$CPT_RUN" \
  --merge

# Stage-end evaluation of the merged standalone M1 model.
onegpu-posttrain run-model-eval \
  --config recipes/smoke/00-base-eval.yaml \
  --run-id cpt-peft-smoke-s42 \
  --stage cpt \
  --model "$M1" \
  --revision "" \
  --internal-holdout "$DATA_ROOT/openr1/canonical/internal_holdout.jsonl" \
  --domain-holdout "$DATA_ROOT/finemath/canonical/cpt_holdout.jsonl" \
  --output "$CPT_RUN/evaluation"

onegpu-posttrain compare-evals \
  --base "$BASE_RUN/evaluation/eval.json" \
  --candidate "$CPT_RUN/evaluation/eval.json" \
  --output "$CPT_RUN/comparison"

echo "CPT smoke complete. See: $CPT_RUN/comparison/stage_comparison.md"
