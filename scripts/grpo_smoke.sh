#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed/smoke}"
NATIVE_ROOT="${2:-artifacts/native/ms-swift/smoke}"
RUN_ROOT="${3:-runs/smoke}"
M2="${4:-$RUN_ROOT/sft-smoke-s42/merged/M2-SFT}"
SFT_RUN="$RUN_ROOT/sft-smoke-s42"
STD_RUN="$RUN_ROOT/grpo-standard-smoke-s42"
LOW_RUN="$RUN_ROOT/grpo-lowmem-smoke-s42"
GRPO_NATIVE="$NATIVE_ROOT/grpo.jsonl"

if [[ ! -d "$M2" ]]; then
  echo "Merged M2 model not found: $M2" >&2
  echo "Run scripts/sft_smoke.sh first, or pass M2 as the fourth argument." >&2
  exit 2
fi

mkdir -p "$NATIVE_ROOT" "$RUN_ROOT"

onegpu-posttrain adapt-ms-swift --stage grpo \
  --input "$DATA_ROOT/openr1/canonical/grpo.jsonl" \
  --output "$GRPO_NATIVE"

run_profile() {
  local config="$1"
  local run_dir="$2"
  local merged_name="$3"
  onegpu-posttrain run-grpo \
    --config "$config" \
    --model "$M2" \
    --dataset "$GRPO_NATIVE" \
    --output "$run_dir" \
    --merge

  local status
  status="$(python - "$run_dir/run.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding='utf-8'))
print((p.get('train') or {}).get('status', 'UNKNOWN'))
PY
)"
  echo "$(basename "$run_dir") status: $status"

  if [[ "$status" == "PASS" ]]; then
    onegpu-posttrain run-model-eval \
      --config recipes/smoke/00-base-eval.yaml \
      --run-id "$(basename "$run_dir")" \
      --stage grpo \
      --model "$run_dir/merged/$merged_name" \
      --revision "" \
      --internal-holdout "$DATA_ROOT/openr1/canonical/internal_holdout.jsonl" \
      --output "$run_dir/evaluation"
  fi
}

# Standard is attempted first. OOM is a valid feasibility result and does not
# prevent the beta=0 low-memory fallback from being tested.
run_profile recipes/smoke/40-grpo-standard.yaml "$STD_RUN" M3-GRPO-standard
run_profile recipes/smoke/41-grpo-lowmem.yaml "$LOW_RUN" M3-GRPO-lowmem

STD_EVAL=""
LOW_EVAL=""
[[ -f "$STD_RUN/evaluation/eval.json" ]] && STD_EVAL="$STD_RUN/evaluation/eval.json"
[[ -f "$LOW_RUN/evaluation/eval.json" ]] && LOW_EVAL="$LOW_RUN/evaluation/eval.json"

ARGS=(
  --sft-eval "$SFT_RUN/evaluation/eval.json"
  --standard-run "$STD_RUN/run.json"
  --lowmem-run "$LOW_RUN/run.json"
  --output "$RUN_ROOT/grpo-comparison"
)
[[ -n "$STD_EVAL" ]] && ARGS+=(--standard-eval "$STD_EVAL")
[[ -n "$LOW_EVAL" ]] && ARGS+=(--lowmem-eval "$LOW_EVAL")
onegpu-posttrain compare-grpo "${ARGS[@]}"

onegpu-posttrain summarize-smoke \
  --run-root "$RUN_ROOT" \
  --output "$RUN_ROOT/smoke-summary"

echo "GRPO comparison: $RUN_ROOT/grpo-comparison/grpo_comparison.md"
echo "Smoke summary:   $RUN_ROOT/smoke-summary/smoke_summary.md"
