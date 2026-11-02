# Proposal

## Why

Mini readiness currently reports quality as `NOT EVALUATED` for every dataset because reports contain structural checks and summaries but do not provide a stage-aware account of evidence coverage. This change makes the existing evidence useful for review and next-step decisions without turning proxy signals into semantic correctness or weakening the official Mini acceptance gates.

## What Changes

- Extend the versioned quality assessment and Mini readiness output to distinguish evidence coverage/completeness from an approved quality judgment.
- Define and report supported, stage-specific data quality evidence for CPT, SFT, Preference, GRPO, and evaluation datasets, including denominators, method/version, coverage, limitations, and explicit unavailable states.
- Preserve the separation between dataset-quality evidence, build diagnostics, and model/recipe suitability; do not create a universal score.
- Keep existing budget, lineage, split-isolation, and reviewed quality-policy gates intact. Evidence completeness alone cannot mark data ready or claim model improvement.
- Add tests and a read-only reassessment of retained Mini artifacts; do not rewrite source datasets or historical build reports.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `mini-dataset-readiness`: define stage-specific evidence coverage, readiness interpretation, and report behavior when metrics are measured, unavailable, or not applicable.

## Impact

- Affected implementation: `src/onegpu_posttrain/data/evidence.py`, `src/onegpu_posttrain/data/mini_readiness.py`, their CLI/report serialization and focused tests.
- Affected artifacts: new quality/readiness reports only; historical source datasets and reports remain immutable.
- No new dependency, external quality framework, API generation, GPU training, or frontend redesign is included. Existing consumers of versioned JSON reports must retain a clear schema/version contract.
