# Design

## Context

See [proposal.md](proposal.md) for motivation and [specs](specs/) for observable behavior. `stage-data-quality-v1` emits digest-bound evidence inventories, but its semantic quality gate is intentionally `NOT_EVALUATED`. The existing optional math verifier is used for candidate answer checks; it does not establish reasoning quality, and the lightweight answer comparison can return false when verification is unavailable. Data policies already distinguish `exploratory` and `baseline`, while ordinary Mini artifacts without a usage field predate that distinction.

The implementation must retain the official `mini-v1` targets and safeguards, support local/offline review, and avoid interpreting source correctness metadata as fresh independent verification.

## Goals / Non-Goals

**Goals:**

- Add a versioned stage-quality review profile and immutable review artifacts, separate from measured quality reports.
- Reuse existing source labels and supported verifier facilities while recording verifier coverage and unsupported outcomes precisely.
- Allow Mini readiness to consume an explicit review bundle and reject identity, digest, stage, policy, or path mismatches.
- Keep exploratory, noisy, baseline-candidate, and legacy-unclassified data distinguishable; only reviewed official candidates can meet official readiness.
- Produce remediation data as new lineage-linked artifacts, using existing local policy and targeted-generation workflows.

**Non-Goals:**

- No universal quality score, automatic threshold discovery, automatic approval, or automatic promotion of exploratory data.
- No alteration to `mini-v1` budgets, frozen source eligibility, split/contamination rules, or past reports.
- No provider calls, dataset downloads, training, evaluation runs, GPU work, database, general-purpose quality framework, or scheduler during review/readiness.
- No claim that final-answer equivalence proves reasoning correctness or that a data-quality change caused model improvement.

## Decisions

### 1. Keep measurement and review as separate artifacts

Keep `QualityReport` as the immutable observation layer (`measured`, `estimated`, `not_measured`, `not_applicable`). Introduce `QualityReviewProfile` and digest-bound `QualityReview` / index artifacts as a separate decision layer. A review records profile ID/version, dataset ID and digest, stage, assessment reference, per-check disposition, reviewer or verifier identity, timestamp, rationale, and any superseded review. This avoids rewriting manifests and historical reports and allows a new review policy to be applied without pretending the underlying measurements changed.

Alternative: add `approved: true` to `QualityReport`. Rejected because it conflates observations with a judgment that can change under a new policy.

### 2. Version policy decisions, not universal thresholds

The profile enumerates stage-applicable checks and declares whether each is required, diagnostic, or not applicable, along with any evidence-backed threshold and unsupported-evidence consequence. Default profiles contain no guessed thresholds. A profile requiring a verifier cannot pass that check if the verifier is missing or lacks reference support. Manual review can document a judgment and scope but cannot relabel an unavailable automated check as measured.

For mathematical SFT and Preference candidates, reuse the existing answer parser and optional `math_verify` capability only when a trusted reference is present. Persist `correct`, `incorrect`, `unparseable`, `unsupported`, and `unverified` separately. Pair review evaluates both sides and ordering. Never infer reasoning completeness from answer equivalence. CPT semantic review remains unavailable unless a separately declared review method is supplied; no generic LLM-as-judge is introduced here.

Alternative: install `math_verify` as a mandatory dependency. Rejected to preserve the lightweight offline install; the profile records the resulting coverage gap.

### 3. Make usage classification explicit at readiness boundaries

Use the existing policy usage vocabulary (`exploratory`, `baseline`) where present. New review/readiness inputs must classify the intended role. Treat missing legacy usage metadata as `legacy_unknown`, not as official by default. Exploratory or intentionally noisy records remain usable for clearly labeled experiments, but cannot satisfy official Mini targets even if counts are high. Promotion requires a new reviewed artifact/manifest or explicit classification record; it never edits the old manifest in place.

### 4. Validate review bundles as explicit overlays

Add a local review command that reads current quality evidence and an explicit profile, writing to a new directory with unique review IDs and an index. `validate-mini-data` accepts a review bundle only when explicitly supplied and validates its report digests, containment, dataset key/stage, dataset ID, content digest, and profile identity. Without a review bundle, readiness retains existing behavior and reports quality approval as missing/not evaluated. Invalid bundles are errors; no fallback to a different review is permitted.

Review outputs are additive under a new review directory. Existing reassessment bundles remain valid; a review references their report identity rather than copying or changing them. Remediation builds reuse the existing policy builder and synthetic candidate export, with source pool, split, usage, and generation ancestry carried through their manifests.

### 5. Keep model-effect evidence downstream and separate

This change does not execute training or use a prior held-out set as fresh final evaluation. Once an official candidate dataset has passed budget and review gates, the existing preregistered policy comparison workflow can link its quality/build artifact to new runs, models, and evaluations. Any previously used holdout must be reclassified as diagnostic; a new final holdout is required for a fresh final claim.

## Risks / Trade-offs

- **[Optional verifier is absent or has limited mathematical coverage]** → report unsupported/unverified denominators; do not silently install dependencies or mark the dataset approved.
- **[Manual review can be subjective]** → require policy version, reviewer, timestamp, rationale, scope, and immutable supersession links; keep manual judgment distinct from measured metrics.
- **[Legacy manifests lack usage classification]** → preserve compatibility for reading, but require explicit classification before official readiness.
- **[Relaxed policies create higher yield but noisier labels]** → keep them exploratory and report label denominators; preserve official gates and split checks.
- **[Reassessment/review could be attached to the wrong artifact]** → require exact dataset and content digests and validate report identity/path containment.
- **[Scope expands into costly generation/training]** → CLI review remains offline/read-only; generation and training require separate explicit user invocation and budgets.

## Migration Plan

1. Add review-profile and review-artifact schemas plus fixtures for measured, unsupported, and manual-review cases.
2. Add offline stage review and optional verifier execution only when explicitly configured and locally available.
3. Add review-bundle validation to Mini readiness, preserving prior behavior when no bundle is supplied and rejecting invalid identity bindings.
4. Exercise policy-produced exploratory and baseline candidates, and generated candidates through their existing lineage path; never modify build-02.
5. Reassess readiness against the official profile and report residual budget and quality blockers. Do not start OVAT until official readiness requirements are met.

Rollback: stop supplying review bundles to readiness. New review artifacts remain immutable and auditable; old datasets and reports need no restoration.
