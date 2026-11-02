#!/usr/bin/env bash
set -euo pipefail

# This is a command-contract walkthrough. It does not download models or train.
DATA_ROOT="${1:-data/processed/smoke}"
NATIVE_ROOT="${2:-artifacts/native/ms-swift/smoke}"
RUN_ROOT="${3:-runs/smoke}"

scripts/cpt_smoke_plan.sh "$DATA_ROOT" "$NATIVE_ROOT" "$RUN_ROOT"
scripts/full_cpt_probe_plan.sh "$DATA_ROOT" "$NATIVE_ROOT" "$RUN_ROOT"
scripts/sft_smoke_plan.sh "$DATA_ROOT" "$NATIVE_ROOT" "$RUN_ROOT"
scripts/alignment_smoke_plan.sh "$DATA_ROOT" "$NATIVE_ROOT" "$RUN_ROOT"
scripts/grpo_smoke_plan.sh "$NATIVE_ROOT" "$RUN_ROOT"
