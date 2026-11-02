#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/adapt_smoke_data.sh [processed_root] [native_dir]
# processed_root is the root produced by scripts/build_smoke_data.sh.
ROOT="${1:-data/processed/smoke}"
NATIVE_DIR="${2:-artifacts/native/ms-swift/smoke}"
mkdir -p "$NATIVE_DIR"

onegpu-posttrain adapt-ms-swift --stage cpt \
  --input "$ROOT/finemath/canonical/cpt_train.jsonl" \
  --output "$NATIVE_DIR/cpt.jsonl"
onegpu-posttrain adapt-ms-swift --stage sft \
  --input "$ROOT/openr1/canonical/sft.jsonl" \
  --output "$NATIVE_DIR/sft.jsonl"
onegpu-posttrain adapt-ms-swift --stage dpo \
  --input "$ROOT/openr1/canonical/preference.jsonl" \
  --output "$NATIVE_DIR/preference.jsonl"
onegpu-posttrain adapt-ms-swift --stage grpo \
  --input "$ROOT/openr1/canonical/grpo.jsonl" \
  --output "$NATIVE_DIR/grpo.jsonl"
