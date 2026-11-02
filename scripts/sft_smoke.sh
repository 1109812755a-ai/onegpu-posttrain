#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed/smoke}"
NATIVE_ROOT="${2:-artifacts/native/ms-swift/smoke}"
RUN_ROOT="${3:-runs/smoke}"
M1="${4:-$RUN_ROOT/cpt-peft-smoke-s42/merged/M1-CPT}"
CPT_RUN="$RUN_ROOT/cpt-peft-smoke-s42"
SFT_RUN="$RUN_ROOT/sft-smoke-s42"
M2="$SFT_RUN/merged/M2-SFT"

if [[ ! -d "$M1" ]]; then
  echo "Merged M1 model not found: $M1" >&2
  echo "Run scripts/cpt_smoke.sh first, or pass M1 as the fourth argument." >&2
  exit 2
fi

mkdir -p "$NATIVE_ROOT" "$RUN_ROOT"

onegpu-posttrain adapt-ms-swift --stage sft \
  --input "$DATA_ROOT/openr1/canonical/sft.jsonl" \
  --output "$NATIVE_ROOT/sft.jsonl"

onegpu-posttrain run-sft \
  --config recipes/smoke/20-sft.yaml \
  --model "$M1" \
  --dataset "$NATIVE_ROOT/sft.jsonl" \
  --output "$SFT_RUN" \
  --merge

onegpu-posttrain run-model-eval \
  --config recipes/smoke/00-base-eval.yaml \
  --run-id sft-smoke-s42 \
  --stage sft \
  --model "$M2" \
  --revision "" \
  --internal-holdout "$DATA_ROOT/openr1/canonical/internal_holdout.jsonl" \
  --output "$SFT_RUN/evaluation"

if [[ -f "$CPT_RUN/evaluation/eval.json" ]]; then
  onegpu-posttrain compare-evals \
    --base "$CPT_RUN/evaluation/eval.json" \
    --candidate "$SFT_RUN/evaluation/eval.json" \
    --output "$SFT_RUN/comparison"
  echo "SFT comparison: $SFT_RUN/comparison/stage_comparison.md"
fi
