# Tasks

## 1. Mini Profile and Readiness Contract

- [x] 1.1 Add a versioned Mini profile declaring 5M CPT training tokens, 5K eligible SFT examples, 2K verified preference training pairs, and 2K verifiable GRPO prompts, with explicit count sources/units and separate preference-eval/CPT-holdout targets; verify profile parsing and unit/count mapping with focused tests.
- [x] 1.2 Add typed readiness-report models for per-dataset budget, quality-evidence, processing-evidence, lineage, and overall blocker states; verify valid/invalid status combinations and serialization with schema tests.
- [x] 1.3 Document profile versioning, the distinction between training and evaluation splits, and the non-equivalence of budget attainment, data quality, lineage, and model effect; verify links and examples against the profile and schema.

## 2. Readiness Evaluation and Evidence Integrity

- [x] 2.1 Implement a read-only evaluator that joins profile targets to dataset manifests and their processing/quality reports using dataset IDs and content digests, not directory names; verify it derives counts only from the declared stage-appropriate evidence through unit tests for each stage.
- [x] 2.2 Preserve missing, invalid, `not_measured`, `NOT_EVALUATED`, warning, and blocking states independently; verify fixtures for absent reports, mismatched dataset IDs/digests, structural failure, uncalibrated semantic checks, and target shortfalls never become false passes.
- [x] 2.3 Integrate local manifest/report and complete-lineage validation, retaining issue codes and paths while distinguishing unavailable external evidence from local integrity errors; verify missing local references, digest mismatch, duplicate IDs, legacy/incomplete records, and explicitly unavailable external references with tests.
- [x] 2.4 Emit a versioned JSON readiness report and human-readable Markdown summary containing profile identity/hash, dataset IDs/digests, stage units, dimension statuses, blockers/warnings, and validation time without modifying input artifacts; verify repeated evaluation leaves source hashes unchanged and output links are resolvable.

## 3. CLI and User Documentation

- [x] 3.1 Add `validate-mini-data` CLI support for explicit artifact root and profile, with a nonzero exit when any required readiness dimension blocks acceptance; verify help text, valid readiness, blocked readiness, and malformed-profile exit behavior in CLI tests.
- [x] 3.2 Document the end-to-end data build → readiness validation commands, output locations, exit-status meaning, and how to review unavailable evidence without weakening gates; verify every documented command against CLI help or fixture execution.

## 4. Mini Data Evidence and Release Gate

- [x] 4.1 Version FineMath chunk record IDs with deterministic content fingerprints while retaining URL-keyed train/holdout assignment; verify different text variants get distinct IDs and byte-identical duplicates remain blocking, then build Mini data into fresh, retained output roots using the frozen eligibility rules and verify target counts, units, source revisions, stop reasons, rejection/yield evidence, and stage-quality coverage for all required splits. The full source scan completed in `data/processed/mini/2026-10-17-build-02`; shortfalls are retained as evidence and do not relax eligibility.
- [x] 4.2 Run `validate-mini-data` and `validate-lineage --require-complete` against the resulting roots; verify local integrity failures are resolved and all explicitly unavailable external evidence is listed without being marked verified. The readiness gate is `NOT READY` for budget/quality blockers; complete-lineage integrity is valid with no local errors or warnings.
- [x] 4.3 Review Mini distributions and checker coverage, version and record any approved blocking quality policy without inventing post-hoc thresholds; verify the readiness report shows each independent gate and does not claim model-effect evidence. Existing `structural-evidence-v1` remains the only blocking quality policy; semantic checks remain `NOT_EVALUATED`, and no post-hoc thresholds were added.
- [x] 4.4 Update the Mini milestone and implementation status from saved artifacts, recording achieved/shortfall budgets and unresolved limitations; verify documentation values and statuses match the readiness report exactly.
