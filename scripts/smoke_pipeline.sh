#!/usr/bin/env bash
set -euo pipefail

# Full RTX-4060 smoke pipeline. Run from the repository root after installing
# `pip install -e '.[all]'` in the pinned Python 3.10-3.12 environment.
DATA_ROOT="${1:-data/processed/smoke}"
NATIVE_ROOT="${2:-artifacts/native/ms-swift/smoke}"
RUN_ROOT="${3:-runs/smoke}"

scripts/build_smoke_data.sh "$DATA_ROOT"
scripts/adapt_smoke_data.sh "$DATA_ROOT" "$NATIVE_ROOT"
scripts/cpt_smoke.sh "$DATA_ROOT" "$NATIVE_ROOT" "$RUN_ROOT"
scripts/full_cpt_probe.sh "$DATA_ROOT" "$NATIVE_ROOT" "$RUN_ROOT"
scripts/sft_smoke.sh "$DATA_ROOT" "$NATIVE_ROOT" "$RUN_ROOT"
scripts/alignment_smoke.sh "$DATA_ROOT" "$NATIVE_ROOT" "$RUN_ROOT"
scripts/grpo_smoke.sh "$DATA_ROOT" "$NATIVE_ROOT" "$RUN_ROOT"
