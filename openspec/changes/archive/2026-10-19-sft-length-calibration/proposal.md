# Proposal

## Why

The current 1,024-token SFT limit is a conservative smoke setting, not yet an evidence-based limit for longer local math traces. A bounded comparison can show how rendered-sequence retention and single-GPU resource cost change at 1,024, 2,048, and 4,096 tokens without weakening verification rules or silently changing the official Mini recipe.

## What Changes

- Add a reproducible, offline SFT length-calibration workflow for locally cached DeepMath data, initially comparing 1,024/2,048/4,096 rendered tokens.
- Report source/sample identity, tokenizer and template identity, eligible counts and denominators, length distribution, verification coverage, and explicit unavailable states.
- Support bounded short training probes with the same parent model, fixed sample order, seed, and training controls; vary only max sequence length and record completion/OOM, duration, and peak VRAM.
- Keep calibration artifacts separate from official Mini datasets and preserve the existing 1,024-token smoke recipe unless evidence is reviewed and a separate change explicitly updates it.

## Capabilities

### New Capabilities
- `sft-length-calibration`: Reproducible offline comparison of SFT rendered-length eligibility and bounded hardware calibration.

### Modified Capabilities

## Impact

- Affected areas: local parquet source adaptation, SFT rendered-length measurement, calibration report/artifact schemas, and bounded training-probe orchestration.
- No required network access, new core dependency, official Mini readiness change, or default recipe change is intended.
- Existing `recipes/smoke/20-sft.yaml` remains the 1,024-token smoke contract; calibration output is exploratory until reviewed.
