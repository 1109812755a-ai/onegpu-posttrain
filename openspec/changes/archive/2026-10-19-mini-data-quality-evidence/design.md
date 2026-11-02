# Design

## Context

See [proposal.md](proposal.md) for motivation. `QualityReport` currently records canonical-schema validity, exact ID duplication, split overlap, source-provided correctness metadata, and length/source summaries. It emits a `NOT_EVALUATED` gate when structural invariants pass because semantic policy is not calibrated. Mini readiness then treats that gate state as a generic quality blocker. The retained build-02 report confirms this gap across stages; it does not mean every metric is inherently unmeasurable.

The existing metric contract already carries value, unit, status, class, method, numerator/denominator, and limitations. Readiness and quality artifacts are content-addressed and written additively; manifests, datasets, and historical reports are not to be rewritten during reassessment.

## Goals / Non-Goals

**Goals:**

- Establish a versioned catalog of evidence expected by stage, and make every catalog item explicit as measured, estimated, not measured, or not applicable.
- Distinguish complete accounting of applicable evidence from a reviewed quality approval.
- Reassess retained Mini datasets offline and bind results to the exact dataset digest, protocol, and checker/verifier identity.
- Make readiness explain missing evidence, unsupported checks, quality-policy status, and existing budget/processing/lineage blockers separately.
- Preserve report compatibility for historical `1.0` artifacts and keep existing split/lineage safety gates unchanged.

**Non-Goals:**

- Establish universal quality thresholds, a composite quality score, or automatic promotion of exploratory data.
- Treat source labels as independent verification, heuristics as semantic truth, or GRPO rollout outcomes as intrinsic dataset quality.
- Add external data-quality frameworks, remote services, model generation, GPU training, or frontend redesign.
- Change the frozen Mini targets or eligibility filters.

## Decisions

### 1. Use a versioned stage evidence catalog with metric-level provenance

Add a protocol such as `stage-data-quality-v1` that selects applicable metric definitions by stage. The initial catalog should cover only evidence supported by the current data and build contracts:

- **CPT:** canonical structure, exact duplicate identities, token-length distribution, source distribution, train/holdout separation, and available contamination-screen evidence.
- **SFT:** canonical structure, prompt/target parse or format checks where fields permit, rendered token lengths, source/task distribution, and upstream correctness-label coverage. Independent correctness is measured only when an already-installed supported verifier is explicitly available; source metadata is never relabeled as a fresh verification.
- **Preference:** canonical complete pairs, same-problem/identity integrity where represented, chosen/rejected label evidence and verifier coverage, chosen/rejected length diagnostics, and source/task distribution. Pair yield and rejection counts remain build diagnostics, not quality scores.
- **GRPO:** static schema/reference/verifier-support and prompt-length evidence where represented. Reward variance, pass rate, completion lengths, and truncation remain a separate calibration artifact tied to model, reward/verifier, seed, and recipe.
- **Evaluation:** schema and scoring coverage, source/protocol and task distribution, parser failures, and available train/evaluation overlap or contamination evidence.

For every expected metric, emit a record even when unsupported. Record numerator/denominator for rates, checker and protocol versions, method, assessment scope, and limitations. Empty denominators remain unavailable rather than zero. Stage catalog entries are not thresholds and do not claim representativeness beyond observed dimensions.

Alternative considered: add a general-purpose framework and map its outputs into the report. Rejected because current needs are stage-specific and can be measured with existing project schemas; a mandatory framework would increase install and runtime cost without improving evidence semantics.

### 2. Separate evidence accounting from quality judgment

Extend the quality report with an evidence-coverage summary that indicates whether every metric required by the named stage catalog has an explicit status. `complete` means accounted for, not that all metrics were measured or passed. Keep the existing gate as a distinct reviewed judgment, with its policy identity and checks; absent a reviewed semantic policy, quality approval remains `NOT_EVALUATED`.

In Mini readiness, expose evidence coverage separately from quality approval. Missing or invalid report identity remains an integrity failure. Incomplete catalog coverage is reported as incomplete evidence. Complete coverage with unsupported or not-measured metrics remains visible, and cannot itself satisfy a quality-approval gate. Existing budget, processing, lineage, and isolation dimensions keep their independent outcomes.

Alternative considered: mark quality as `PASS` once all metrics are enumerated. Rejected because recording an unavailable metric is not evidence that the dataset meets a quality threshold.

### 3. Reassess existing artifacts into a separate, digest-bound assessment bundle

Add an explicit local command to read an artifact root and Mini profile, load canonical dataset records under existing path/size safeguards, recompute supported metrics, and write a new uniquely identified assessment bundle under a caller-selected or default report directory. The bundle contains per-dataset quality reports plus an index joining dataset key, dataset ID, canonical content digest, report identity, and relative report path. It never edits source records, manifests, prior quality reports, or prior readiness output.

Allow `validate-mini-data` to consume this bundle explicitly. It must verify that each supplied assessment matches the manifest's dataset ID and canonical-record digest and that referenced files remain inside the assessment bundle; mismatches are invalid evidence, not silently ignored overrides. Without an explicit bundle, readiness continues using the manifest-linked report.

Alternative considered: update each dataset manifest to point at the newest quality report. Rejected because it would mutate historical build evidence and make prior readiness runs ambiguous. A content-addressed overlay preserves provenance while allowing old data to be reassessed.

### 4. Add report fields without invalidating historical artifacts

Emit the new quality/readiness schema version for new assessments, while readers continue to accept existing `1.0` reports and interpret their evidence coverage as legacy/unknown rather than complete. New fields must be additive and validated; report IDs remain unique per assessment, while dataset ID and content digest remain stable. Do not reinterpret historical `NOT_EVALUATED` reports retroactively.

Alternative considered: rewrite build-02 reports and manifests in place. Rejected because it breaks reproducibility and lineage digests.

### 5. Keep current requirements as reportable signals, not newly calibrated blockers

Structural validity, duplicate identities, split isolation, content/report identity, and required lineage retain their existing enforcement semantics. Newly surfaced distribution, verifier-coverage, label, and unsupported-metric signals are diagnostic until a separately reviewed versioned policy sets consequences. No threshold is inferred from the recent 20-example SFT comparison.

## Risks / Trade-offs

- **Large retained JSONL may increase reassessment time and memory use** → keep reassessment explicit, report processed counts and stop/failure state, and avoid changing default read-only readiness cost.
- **Source metadata may be mistaken for independent truth** → include method labels and limitations in the metric itself and test UI/Markdown wording at report level.
- **New fields may be consumed by older scripts** → preserve old readers, version new output, and document the schema transition.
- **An assessment bundle could be paired with the wrong dataset** → require dataset ID plus canonical digest matches and validate bundle paths before readiness uses it.
- **A complete evidence inventory may look like approval** → present separate coverage and approval statuses and add explicit negative-claim scenarios/tests.
- **Some requested dimensions are absent from current records** → report `not_measured` with a reason; defer collecting new source annotations or adopting an external checker.

## Migration Plan

1. Add the versioned metric catalog, assessment fields, and backward-compatible readers.
2. Add explicit reassessment and assessment-bundle validation; retain ordinary readiness behavior when no bundle is supplied.
3. Run the new assessment against retained Mini build-02 into a new output directory, then run readiness against that bundle. Verify the build-02 files and hashes are unchanged.
4. Review evidence and remaining blockers. Do not alter Mini rules or claim a model effect as part of this change.

Rollback consists of stopping use of the optional assessment bundle and returning to the existing manifest-linked reports; the additive outputs remain auditable and historical files require no restoration.
