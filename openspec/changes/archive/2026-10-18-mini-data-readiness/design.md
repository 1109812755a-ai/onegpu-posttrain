# Design

## Context

See `proposal.md` for motivation and `specs/mini-dataset-readiness/spec.md` for externally visible requirements. Existing `build-openr1` and `build-finemath` commands already emit versioned manifests, processing reports, static quality reports, and data-build/quality-assessment lineage. OpenR1 reports target counts, selected counts, rejection counters, and a stop reason; FineMath reports token budgets and document splitting. `validate-lineage --require-complete` checks local identities, references, and digests. Current quality reports intentionally leave uncalibrated semantic checks as `not_measured` and the aggregate gate as `NOT_EVALUATED`.

The gap is a Mini-specific, repeatable acceptance view over those existing records. The only durable budget evidence currently lives in project planning text and per-builder configs; existing CLI commands do not produce one consolidated readiness result across source datasets/splits. Mini records must remain separate from downstream model-effect claims.

## Goals / Non-Goals

**Goals:**

- Provide one versioned Mini profile for expected budgets, counting units, required report evidence, and explicitly reviewed blocking checks.
- Produce a read-only readiness report that identifies each dataset/split, its profile target, observed eligible amount, quality-evidence state, processing-evidence state, and lineage-integrity state.
- Reuse current dataset manifests, processing/quality reports, and lineage validation rather than introducing a second lineage model.
- Preserve the existing rule that unknown semantic quality is not a pass and that no quality metric alone proves downstream model improvement.
- Make target shortfalls actionable through saved stop reasons and rejection/yield evidence; do not change filters to reach quotas.

**Non-Goals:**

- Building or downloading Mini datasets as a side effect of readiness validation.
- Training models, running evaluation, or performing the four OVAT experiments.
- Defining new semantic quality thresholds from the observed Mini distributions in advance of review.
- Adding a database, scheduler, web UI, or mandatory third-party data-quality framework.
- Claiming causal model effects from processing, quality, or readiness evidence.

## Decisions

### 1. Keep builders authoritative; add a separate read-only readiness evaluator

Keep `build-openr1` and `build-finemath` as the only data-construction paths. Add a Mini profile and a read-only evaluator (proposed CLI boundary: `validate-mini-data --root <artifact-root> --profile <profile>`). The evaluator consumes saved manifests/reports and reuses lineage validation; it does not reach the network, rerun transforms, or mutate datasets. This is preferred over embedding release policy in both builders, which would duplicate logic and couple generic Smoke builds to Mini acceptance.

### 2. Version the profile and express target units explicitly

The profile records the existing milestone budgets: 5,000,000 CPT training tokens, 5,000 eligible SFT examples, 2,000 verified preference training pairs, and 2,000 verifiable GRPO prompts. Counts are read from stage-appropriate evidence, never inferred from raw source rows. The 2,000-pair preference budget applies to the training split; preference evaluation remains a distinct held-out split governed by its own declared evaluation target and is not added to the training budget. CPT holdout is likewise reported separately from the 5M training-token target. If a profile or artifact does not specify a required count source or unit, the evaluator reports it as unavailable and cannot mark that dimension ready.

Changing target values, eligibility protocol references, or blocking-check policy requires a new profile/policy version. The profile must not silently change historical dataset identities or rewrite source reports.

FineMath separates the **split identity** from the **record identity**. The split identity remains the existing deterministic URL-derived document ID, so two extraction variants for one URL cannot leak across train/holdout. A chunk record ID additionally incorporates a deterministic fingerprint of the emitted chunk text; distinct extracted text under the same URL therefore receives distinct IDs without removing either row. Byte-identical repeated source rows still produce duplicate IDs and remain blocked by the existing structural quality check. This changes FineMath build protocol/identity semantics and must be versioned (for example, `finemath-build-v2`); manifests, content digests, and reports must be regenerated into a fresh retained output root, never edited in place.

### 3. Return independent readiness dimensions, not a composite quality score

The report separates `budget`, `quality_evidence`, `processing_evidence`, and `lineage` outcomes. A small overall readiness state may summarize blockers, but it must not replace these dimensions or present a quality score. Quality evidence states distinguish missing, invalid, complete-but-not-evaluated, warning, and reviewed under an explicit policy. Report presence and structural schema checks alone do not establish semantic-quality approval.

Candidate quality checks remain non-blocking until their policy is explicitly versioned and reviewed using coverage and observed Mini distributions. Existing hard structural failures (invalid schema, duplicate IDs, or train/eval overlap) remain blocking according to current builder behavior. A not-measured item must stay visible; it may only be non-blocking when the selected policy explicitly says so.

### 4. Validate provenance with the existing artifact contract

For each selected Mini artifact root, run the existing manifest evidence checks and `validate-lineage --require-complete` semantics, retaining structured issue codes and paths in the readiness report. Required local missing references and digest mismatches block acceptance. External evidence already marked unavailable is reported separately; it is never relabeled as verified. If current validator behavior cannot distinguish a policy-approved external unknown from a local integrity error, resolve that at the readiness layer without weakening global validation semantics.

### 5. Use stable, content-linked reports and preserve history

Write the readiness summary as a new versioned report (proposed `reports/mini_readiness.json` plus a human-readable Markdown view), containing profile ID/version/hash, artifact-root scope, dataset IDs and content digests, per-dimension statuses, blockers/warnings, and validation time. Do not edit builder reports or historical Smoke artifacts. Separate output locations or append-only run IDs preserve prior Mini assessments.

### Alternatives considered

- **Put Mini thresholds directly into generic builders:** rejected because it couples all data-building use cases to one release profile and makes Smoke/Mini semantics easy to confuse.
- **Treat any existing quality report as a pass:** rejected because current reports intentionally distinguish measured structural checks from uncalibrated semantic quality.
- **Add Great Expectations or another framework:** rejected for this change; it would not supply project-specific preference/GRPO semantics and would add install burden without solving the readiness join.
- **Use a database or frontend as the readiness source of truth:** rejected; local JSON/Markdown artifacts remain inspectable and usable without a service.

## Risks / Trade-offs

- [Preference yield remains far below 2K] → Preserve the shortfall report and inspect source coverage/eligible-pair yield; do not relax verifier, length, contamination, or split rules in this change.
- [Profile targets and builder config drift] → Record both profile and resolved build config hashes and fail readiness on incompatible target/protocol declarations.
- [Quality report completeness is mistaken for semantic quality] → Keep evidence completeness and policy outcome separate; show `NOT_EVALUATED` and limitations explicitly.
- [Complete-lineage checks reject legitimate external evidence] → Keep validator's unavailable status distinct from local integrity errors; document exceptions as unavailable rather than bypassing checks.
- [Large artifact roots make repeated validation slow] → Scope the profile to explicit artifact roots and reuse bounded/local scans; optimize only after measuring Mini-sized roots.

## Migration Plan

1. Add the versioned Mini profile and readiness report schema without changing builder output semantics.
2. Run the evaluator against fixtures, incomplete/legacy artifacts, and current dev7 evidence; verify it reports blockers without rewriting inputs.
3. Version the collision-safe FineMath record identity while preserving URL-keyed split assignment; verify distinct text variants receive distinct IDs and byte-identical duplicates remain blocking.
4. Build official Mini datasets into fresh output directories and retain all build artifacts.
5. Run readiness and complete-lineage validation, review candidate blocking checks and observed distributions, then freeze the profile/policy used for official Mini experiments.
6. Rollback is to stop consuming the readiness report and continue using the existing builders; no existing data or lineage format is migrated in place. Previously emitted Mini artifacts remain immutable and are not equivalent to artifacts emitted by the versioned identity protocol.

## Open Questions

- None required before implementation. The exact preference-evaluation held-out target is intentionally not assigned by this change; it remains an independently declared evaluation target and does not alter the 2K training-pair budget.
