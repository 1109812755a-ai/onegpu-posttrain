# Design

## Context

See `proposal.md` for motivation and `specs/model-playground-discovery/spec.md` for observable behavior. The current model catalog already verifies local registrations and successful producer lineage, but emits only model identity/source IDs and sorts by `(stage, model_id)`. Run summaries expose `completed_at`. The catalog endpoint is loopback-only; inference requests already resolve by immutable `model_id`. Run detail currently renders model artifacts directly from Run data.

## Goals / Non-Goals

**Goals:**

- Enrich catalog entries from verified training Run summaries and provide one authoritative display-name lookup for model and Run views.
- Persist labels as workspace-local presentation metadata, separately from immutable model/Run artifacts.
- Keep explicit Run-to-playground model selection authoritative while making the default useful for direct entry.

**Non-Goals:**

- Renaming model files, IDs, Run labels, lineage, or model digests.
- Loading a model, acquiring GPU, or creating a session as a side effect of discovery/default selection.
- Synchronizing user labels across machines or accounts.

## Decisions

1. **Use the producing training Run's `completed_at` as recency evidence.** Build an indexed lookup from `RunWorkspace` candidates and attach the timestamp for the verified `source_run_id`; for merged models, do not mistake the separate merge Run for the training completion time. Parse timestamps for ordering, place missing/invalid values after valid values, and use stable ID tie-breakers. Do not substitute filesystem modification time, which is not reliable training evidence. Consider a model recent/eligible only after the catalog's existing availability and lineage checks; digest verification at actual resolution remains in force.

2. **Keep aliases in a separate workspace metadata sidecar.** Store a bounded mapping keyed by validated immutable `model_id` under `.onegpu-posttrain/` (for example `model-display-names.json`). The sidecar is not a model artifact and is never included in Run lineage or digest computation. Use strict JSON/schema and size limits, atomic replacement, and an in-process lock for writes. Do not accept filesystem paths from the client. Only allow edits for model IDs present in the workspace model registry; do not rewrite model or Run records. Unknown/stale entries are ignored on reads rather than mapped to another model.

3. **Expose labels through the existing local catalog boundary.** Extend model catalog responses with training completion time and user display name, and add narrowly scoped local alias update/clear endpoints. Mutations must validate model ID, trim and bound the plain-text label, reject control characters, require the existing local launcher capability token, and remain loopback-only. Run detail can enrich its model artifact presentation by joining the same workspace catalog metadata, without modifying the stored Run response/lineage record. Inference endpoints continue accepting only `model_id`.

4. **Separate the primary label from identity evidence in the UI.** Show the alias (or a deterministic stage + training time + short Run/model ID fallback) as the scan-friendly title. Keep raw `model_id`, training `source_run_id`, optional merge `produced_by_run_id`, and digest in secondary/details content. Provide an edit/clear affordance at the successful Run artifact and/or selected model view; render all user text as escaped plain text and keep controls keyboard-operable and responsive.

5. **Resolve initial selection after catalog load without implicit inference.** If a valid `initialModelId` came from an explicit Run handoff, retain it. Otherwise select the first available item after recency sorting. Do not call session/comparison endpoints until the user explicitly starts an experience. If the requested deep-linked model is no longer available, surface that fact and do not silently replace it with a different model; allow an intentional user selection afterward.

## Risks / Trade-offs

- [Legacy Run records may not contain parseable completion timestamps] → Keep those models discoverable after dated entries and label time as unknown; never infer recency from file timestamps.
- [A workspace sidecar can be malformed or concurrently edited] → Bound reads, validate every entry, preserve the previous file on failed writes, serialize updates, and surface a recoverable local error rather than modifying model artifacts.
- [The catalog identity scan is asynchronous and may be expensive] → Reuse its existing snapshot and Run summary scan; enrich existing entries in the same pass and avoid per-model Run rescans.
- [A label may be confused with a verified model identity] → Keep immutable IDs and provenance visible, use explicit “用户显示名” wording in edit UI, and never send aliases to inference resolution.

## Migration Plan

No migration is required. Existing workspaces without a display-name sidecar behave as if all models are unnamed. Create the sidecar only on the first successful label update. Rollback consists of removing the new sidecar and reverting the UI/API code; existing model and Run artifacts remain untouched.

