#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed/smoke}"
NATIVE_ROOT="${2:-artifacts/native/ms-swift/smoke}"
RUN_ROOT="${3:-runs/smoke}"
M2="${4:-$RUN_ROOT/sft-smoke-s42/merged/M2-SFT}"
SFT_RUN="$RUN_ROOT/sft-smoke-s42"
DPO_RUN="$RUN_ROOT/dpo-smoke-s42"
SIMPO_RUN="$RUN_ROOT/simpo-original-smoke-s42"
M3_DPO="$DPO_RUN/merged/M3-DPO"
M3_SIMPO="$SIMPO_RUN/merged/M3-SimPO"
PREF_EVAL="$DATA_ROOT/openr1/canonical/preference_eval.jsonl"

if [[ ! -d "$M2" ]]; then
  echo "Merged M2 model not found: $M2" >&2
  echo "Run scripts/sft_smoke.sh first, or pass M2 as the fourth argument." >&2
  exit 2
fi
if [[ ! -f "$PREF_EVAL" ]]; then
  echo "Held-out preference dataset not found: $PREF_EVAL" >&2
  echo "Re-run scripts/build_smoke_data.sh with the dev5 data contract." >&2
  exit 2
fi

mkdir -p "$NATIVE_ROOT" "$RUN_ROOT"

onegpu-posttrain adapt-ms-swift --stage dpo \
  --input "$DATA_ROOT/openr1/canonical/preference.jsonl" \
  --output "$NATIVE_ROOT/preference.jsonl"

# Common public preference baseline on the frozen M2 parent.
onegpu-posttrain eval-preference \
  --model "$M2" \
  --input "$PREF_EVAL" \
  --output "$SFT_RUN/preference_evaluation"

onegpu-posttrain run-dpo \
  --config recipes/smoke/30-dpo.yaml \
  --model "$M2" \
  --dataset "$NATIVE_ROOT/preference.jsonl" \
  --output "$DPO_RUN" \
  --merge

onegpu-posttrain run-model-eval \
  --config recipes/smoke/00-base-eval.yaml \
  --run-id dpo-smoke-s42 \
  --stage dpo \
  --model "$M3_DPO" \
  --revision "" \
  --internal-holdout "$DATA_ROOT/openr1/canonical/internal_holdout.jsonl" \
  --output "$DPO_RUN/evaluation"

onegpu-posttrain eval-preference \
  --model "$M3_DPO" \
  --input "$PREF_EVAL" \
  --output "$DPO_RUN/preference_evaluation"

onegpu-posttrain run-simpo \
  --config recipes/smoke/31-simpo-original.yaml \
  --model "$M2" \
  --dataset "$NATIVE_ROOT/preference.jsonl" \
  --output "$SIMPO_RUN" \
  --merge

onegpu-posttrain run-model-eval \
  --config recipes/smoke/00-base-eval.yaml \
  --run-id simpo-original-smoke-s42 \
  --stage simpo \
  --model "$M3_SIMPO" \
  --revision "" \
  --internal-holdout "$DATA_ROOT/openr1/canonical/internal_holdout.jsonl" \
  --output "$SIMPO_RUN/evaluation"

onegpu-posttrain eval-preference \
  --model "$M3_SIMPO" \
  --input "$PREF_EVAL" \
  --output "$SIMPO_RUN/preference_evaluation"

if [[ -f "$SFT_RUN/evaluation/eval.json" ]]; then
  onegpu-posttrain compare-alignment \
    --sft-eval "$SFT_RUN/evaluation/eval.json" \
    --dpo-eval "$DPO_RUN/evaluation/eval.json" \
    --simpo-eval "$SIMPO_RUN/evaluation/eval.json" \
    --sft-preference "$SFT_RUN/preference_evaluation/preference_metrics.json" \
    --dpo-preference "$DPO_RUN/preference_evaluation/preference_metrics.json" \
    --simpo-preference "$SIMPO_RUN/preference_evaluation/preference_metrics.json" \
    --output "$RUN_ROOT/alignment-comparison"
fi

echo "Alignment smoke comparison: $RUN_ROOT/alignment-comparison/alignment_comparison.md"
