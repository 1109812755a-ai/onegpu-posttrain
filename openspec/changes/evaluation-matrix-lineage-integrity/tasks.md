# Tasks

## 1. Matrix source identity validation

- [x] 1.1 Audit the current matrix builder, evaluation run/lineage schemas, and their path conventions; document the accepted identity fields and ensure the validator distinguishes absent legacy evidence from contradictory evidence.
- [x] 1.2 Update matrix generation to verify run ID, eval ID when present, configured/recorded model identity, terminal status, and declared metric-output reference where available; preserve metric values while emitting structured provenance status/issues. Add fixture tests for matching, missing, failed, partial, mismatched-ID, mismatched-model, and mismatched-output cases.
- [x] 1.3 Verify only provenance-valid completed models can yield `comparable`; update focused matrix tests and Chinese documentation for the integrity states and legacy behavior.

## 2. Mini artifact reconciliation

- [x] 2.1 Trace the M2, DPO, SimPO, GRPO-standard, and GRPO-lowmem matrix entries to exact metric files, evaluation run records, lineage sidecars, and model IDs; produce a source-to-ID audit table and leave any unprovable entry unresolved.
- [x] 2.2 Correct the Mini matrix source configuration only for verified artifacts, regenerate into a new versioned output (preserving the prior matrix), and update the Mini summary with verified/unresolved lineage status; verify source metric and historical sidecar hashes remain unchanged.

## 3. Consumer and end-to-end verification

- [x] 3.1 Extend the read-only UI/API and frontend tests to show provenance status/issues without hiding available metrics; verify mismatched and unresolved lineage cannot appear complete or comparable.
- [x] 3.2 Run focused matrix, lineage, API, and frontend tests; validate the change with `openspec validate evaluation-matrix-lineage-integrity --strict`; inspect the real Mini matrix in the running UI and record the final evidence coverage without claiming official Mini readiness.

## Workflow follow-up

- Archive this change after implementation and evidence review; keep the separate `mini-sft-2048-pilot` OOM follow-up distinct from this lineage-integrity change.
