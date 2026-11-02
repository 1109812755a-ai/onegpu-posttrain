# Proposal

## Why

The 1,536-token SFT probe passed, but the full run OOMed at step 20/599. Its run now records complete, truthful lineage. To keep Mini moving, derive a shorter cohort from the existing local data rather than rescanning the source or treating a target count as an execution gate.

## What Changes

- Register the existing immutable 1,536-token calibration cohort as a canonical dataset artifact, with a real registration/build Run and verified canonical/native digests.
- Preserve the full 1,536-token SFT OOM as evidence, then derive a 1,024-token subset from the existing cohort without rescanning DeepMath; record its parent dataset, digest, counts, and length-rejection reasons.
- Resolve training inputs to the exact registered dataset identity and preserve parent/data/config/model ancestry through SFT, merge, and evaluation.
- Complete one exploratory SFT pass on the available 1,024-token subset, merge M2, and evaluate M1/M2 with the existing protocol if training succeeds.
- Report actual retained counts, review limitations, resource use, and failures. Do not wait for the 5,000 target or make an API generation request in this change.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

None. Existing `dataset-lineage`, `run-model-lineage`, `evaluation-lineage`, and `sft-length-calibration` requirements cover the required behavior; this change brings the calibration-to-training path into conformance without changing those contracts. Specs remain intentionally skipped.

## Impact

Adds a focused calibration-dataset registration/derivation path, training-lineage resolution and tests, plus versioned Mini SFT recipe/run evidence. No new dependency or public schema version. Historical JSONL and calibration manifest contents remain unchanged; new registration/derivation evidence is additive.
