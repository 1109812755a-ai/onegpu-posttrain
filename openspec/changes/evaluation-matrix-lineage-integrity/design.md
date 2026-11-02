# Design

## Context

See `proposal.md` for motivation. The current matrix builder loads `eval.json`, `evaluation_run.json` and `evaluation_lineage.json` beside a configured path, then primarily derives completion from metrics and the run status. It does not cross-check the run IDs, eval IDs, configured model identity, or whether the run record's output points at the metric artifact. The general lineage validator checks evaluation sidecars against known runs, but not against a matrix entry.

## Goals / Non-Goals

**Goals:**
- Make matrix completion and comparability depend on verifiable, mutually consistent source evidence.
- Preserve observed metric values while exposing specific provenance errors.
- Reconcile the current Mini matrix only from source artifacts that can be positively identified.

**Non-Goals:**
- Editing or deleting historical evaluation metrics, run records, or lineage sidecars.
- Re-running training or evaluation, changing model/data artifacts, or promoting exploratory Mini evidence to official readiness.
- Inferring source paths or identities from display labels such as `M3-GRPO-standard`.

## Decisions

1. **Treat configured `eval_path` as the source anchor and validate adjacent evidence.** Require metric payload, evaluation run record, and lineage sidecar to agree on `run_id` / `produced_by_run_id`, `eval_id` when present, model ID, and terminal status. Validate declared run outputs against the configured metric artifact when output identity is available. This keeps the current file-based workflow; a database or global artifact registry is unnecessary. An alternative would be to trust the sidecar alone, but that would not detect the observed split-source case.

2. **Represent evidence integrity separately from metric availability.** Keep already-read values and denominators, add bounded machine-readable provenance issues, and mark an entry partial/unverified when required evidence is absent or inconsistent. Full matrix comparability requires all included models to have completed, internally consistent provenance and the existing shared-protocol checks. Do not turn an identity problem into a zero score or silently drop the model. This is safer than rejecting the entire matrix, which would hide useful but qualified observations.

3. **Validate against fixtures and the real Mini artifacts before regenerating.** Add fixtures for matching evidence, run/eval/model mismatch, missing sidecars, failed runs, and mismatched output references. For Mini, inspect the actual eval JSON, evaluation run records, sidecars, and run outputs; regenerate only when the source set can be established. If no successful GRPO evaluation source can be proven, leave it explicitly unresolved rather than manufacturing a successful link. Keep prior matrix/report bytes as historical evidence and write the reconciled result as a new generated artifact or documented replacement with provenance.

4. **Keep UI behavior additive.** Reuse matrix status and issue fields through the existing read-only matrix API; metrics remain visible, while integrity warnings are explicit and cannot be mistaken for a lineage-resolved result. Avoid a UI redesign or new route.

## Risks / Trade-offs

- [Legacy evaluations lack stable IDs or output declarations] → Continue displaying their metrics with an explicit legacy/incomplete integrity state; do not block file reading.
- [Some valid historical run records may encode output paths differently] → Test documented path-resolution rules and only assert exact identity when the evidence contains a resolvable output reference.
- [A real Mini GRPO source may remain unavailable] → Finish code-level integrity behavior and mark those specific entries unresolved; do not fabricate lineage or claim the full matrix is verified.

## Migration Plan

No schema migration is required. Add backward-compatible matrix fields, rebuild the Mini matrix only from validated sources, verify existing UI and lineage tests, and retain the old report as historical evidence if its source references cannot be proven. Rollback is to stop consuming the new integrity fields; original eval artifacts remain untouched.
