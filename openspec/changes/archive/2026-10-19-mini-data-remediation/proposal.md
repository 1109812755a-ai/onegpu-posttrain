# Proposal

## Why

The retained Mini build meets CPT and GRPO budgets but is short on SFT (375/5,000) and Preference (4/2,000), while all stage-level quality approvals remain `NOT_EVALUATED`. A prior relaxed SFT policy increased yield but also included many upstream incorrect labels, so filling quotas by silently weakening frozen rules would damage both reproducibility and the project's evidence standard.

## What Changes

- Add a versioned, stage-aware quality review profile that separates measured evidence, independent verification, human review, unsupported checks, and final review decisions. No universal composite score or inferred PASS is introduced.
- Make the distinction between exploratory datasets and official Mini candidates explicit throughout data artifacts and readiness output. Exploratory or intentionally noisy data can support learning and controlled policy comparisons, but cannot satisfy official Mini budgets or gates by relabeling.
- Provide a reproducible remediation workflow that consumes existing local source pools and explicitly supplied generated candidates, records yield/rejection and independent verifier coverage, and emits new digest-bound artifacts without rewriting build-02 or weakening its profile.
- Reuse the existing policy comparison and targeted-generation interfaces to plan and trace supplementary data; remote generation remains opt-in and is never triggered by readiness, preview, or review.
- Preserve existing Mini targets, split/contamination constraints, and model-effect boundaries. Readiness may become ready only when the declared official requirements are met and review decisions are supported by their versioned policy.

## Capabilities

### New Capabilities

- `stage-quality-review`: versioned review profiles and per-dataset review outcomes for stage-specific evidence, including verifier/checker identity, coverage, limitations, and explicit unsupported/pending states.

### Modified Capabilities

- `mini-dataset-readiness`: distinguish exploratory artifacts from official Mini candidates and consume applicable versioned quality-review outcomes without changing existing targets or automatically promoting data.

## Impact

- Affected areas: `src/onegpu_posttrain/data/quality_protocol.py`, `evidence.py`, `quality_assessment.py`, `mini_readiness.py`, policy/synthetic artifact integration, CLI/report serialization, and focused tests.
- New artifacts are additive and content-addressed; build-02, historical quality reports, and readiness reports remain immutable.
- No required database, external quality framework, provider call, network access, GPU execution, or training scheduler. Any real provider request or training run remains separately and explicitly invoked by the user.
- Operator guidance and project status documents will explain that exploratory success is not official Mini readiness and that `NOT_EVALUATED` is not a quality failure or pass.
