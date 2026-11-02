# Tasks

## 1. Verified Catalog Recency

- [x] 1.1 Enrich catalog entries from the verified source training Run with `completed_at`, keep the separate merge Run distinct, and sort available models newest-first with missing/invalid timestamps last; extend `tests/test_model_playground_catalog.py` to verify order, tie/fallback behavior, and unchanged eligibility.
- [x] 1.2 Extend the catalog API contract with display metadata while retaining immutable IDs and existing loopback restrictions; add API tests proving explicit model IDs and provenance remain present and no model/session/GPU action occurs on catalog reads.

## 2. Workspace Display-Name Metadata

- [x] 2.1 Implement bounded workspace-local alias read/write/clear storage keyed only by registered `model_id`, with schema/input validation, serialized atomic writes, and malformed-file recovery; add unit tests for persistence, clearing, size/control-character rejection, stale keys, and failed-write safety.
- [x] 2.2 Add loopback-only alias mutation endpoints protected by the local launcher token and strict model-ID validation; add API tests for valid update/clear, missing token, unknown model, invalid names, and path-like input rejection.
- [x] 2.3 Enrich Run model-artifact responses from the same alias metadata without rewriting Run or lineage files; extend Run workspace/API tests to prove names are joined by exact model ID and raw identity/lineage data remains unchanged.

## 3. Discovery and Naming UI

- [x] 3.1 Update the Model Playground picker/details/results to show alias or readable fallback plus stage, training completion time, training Run, merge Run where applicable, raw model ID, and digest; verify `cd web && npm run build` succeeds and add UI tests for these display states.
- [x] 3.2 Add accessible edit/clear name controls usable from the selected model and successful Run artifact view, with plain-text rendering and consistent refresh across views; add UI tests for save, rename, clear, keyboard operation, and visible validation errors.
- [x] 3.3 Make catalog sorting select the newest available model only when there is no explicit Run-handoff model ID, without auto-loading or creating inference work; add UI tests for default selection, deep-link precedence, unavailable deep-link behavior, and no implicit session request.

## 4. Integration Verification

- [x] 4.1 Run `pytest tests/test_model_playground_catalog.py tests/test_ui_run_workspace.py tests/test_model_playground_jobs.py` and `cd web && npm run build`; resolve regressions and confirm existing Run/model identity and lineage tests remain green.

