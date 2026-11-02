# Tasks

## 1. Stage-specific quality evidence protocol

- [x] 1.1 Define a versioned metric catalog for CPT, SFT, Preference, GRPO, and evaluation data, with applicability, metric class, units, method/version, coverage semantics, and explicit unavailable reasons; verify catalog fixtures enumerate every expected metric without adding thresholds.
- [x] 1.2 Extend quality assessment to compute only signals supported by canonical records and existing build evidence, including stage-appropriate distributions, pair/label evidence, verifier coverage, and static-vs-calibration separation; verify each stage using hand-checkable fixtures and denominator assertions.
- [x] 1.3 Add evidence-coverage accounting distinct from gate/approval state and emit the new quality report schema; verify complete-but-unmeasured evidence is not represented as quality PASS and empty populations remain unavailable rather than zero.
- [x] 1.4 Preserve historical report compatibility by reading schema 1.0 as legacy/unknown coverage and writing new versioned reports without mutating inputs; verify old fixtures still validate and new report round-trips preserve dataset/content identities.

## 2. Mini readiness and reassessment bundle

- [x] 2.1 Extend Mini readiness reports to expose evidence coverage separately from quality approval while retaining independent budget, processing, lineage, split, and quality-gate outcomes; verify a complete evidence inventory with unreviewed quality policy cannot produce READY.
- [x] 2.2 Implement explicit offline reassessment output containing digest-bound per-dataset reports and an index; verify reports are additive, use unique assessment identities, and source artifact hashes are unchanged.
- [x] 2.3 Allow `validate-mini-data` to consume an explicitly supplied assessment bundle, validating dataset ID, canonical digest, report identity, and path containment; verify mismatched, stale, missing, or path-escaping references are rejected rather than silently ignored.
- [x] 2.4 Update CLI help and Chinese data-quality/Mini-readiness guidance for the reassessment and optional bundle workflow; verify documented commands run on project-owned fixtures without network access or external services.

## 3. Retained Mini evidence and integration verification

- [x] 3.1 Reassess retained Mini build-02 into a new report directory and run readiness against the bundle; verify per-stage evidence coverage, remaining budget/quality/lineage blockers, and model-effect `NOT_EVALUATED` are reported accurately.
- [x] 3.2 Snapshot and compare hashes for retained datasets, manifests, build reports, and historical readiness reports around reassessment; verify none are modified and new reports bind to the exact existing dataset digests.
- [x] 3.3 Run focused data evidence/readiness/CLI tests, Ruff, compile checks, and `openspec validate mini-data-quality-evidence --strict`; record exact results and ensure no external framework, provider request, or GPU run is required.

## Verification record (2026-10-19)

- `.venv/Scripts/python.exe -m pytest tests/test_data_evidence.py tests/test_mini_readiness.py tests/test_cli_mini_data.py -q` — **41 passed**.
- `.venv/Scripts/python.exe -m ruff check src/onegpu_posttrain/data/evidence.py src/onegpu_posttrain/data/quality_protocol.py src/onegpu_posttrain/data/quality_assessment.py src/onegpu_posttrain/data/mini.py src/onegpu_posttrain/data/mini_readiness.py src/onegpu_posttrain/cli.py tests/test_data_evidence.py tests/test_mini_readiness.py tests/test_cli_mini_data.py` — **All checks passed**.
- `.venv/Scripts/python.exe -m compileall -q src/onegpu_posttrain` — **passed**.
- `openspec validate mini-data-quality-evidence --strict` — **valid**.
- Retained build-02: offline assessment wrote 7 stage reports; readiness correctly returned `not_ready` (exit 2), with 9 explicit budget/quality blockers, lineage and processing passing, and model effect `not_evaluated`. Hash snapshot of 49 pre-existing files remained unchanged.
- No external quality framework, provider request, network access, or GPU run was used.
