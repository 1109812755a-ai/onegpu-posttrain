# Design

## Context

See `proposal.md` for motivation and `specs/sft-generation-workflow/spec.md` for user-visible behavior. The project already has an explicit targeted-generation CLI, canonical SFT JSONL validation/import, immutable dataset manifests, a Mini SFT Launcher that accepts registered dataset IDs, and an evaluation flow that records model/run lineage. The missing connection is a safe, observable UI execution path between these existing components.

The current targeted page only stores a local draft. Targeted generation requires a validated plan and supporting source evidence; generation outputs are currently private artifacts and are not necessarily in the processed-data registry scanned by the SFT Launcher. Generation candidate count is not the same as the number of usable registered records.

## Goals / Non-Goals

**Goals:**
- Bridge the existing explicit generation service to the local UI without introducing arbitrary command, path, or environment-variable execution.
- Preserve generation, dataset, training, model, and evaluation identities and their existing lineage evidence.
- Make partial generation and its true usable-record count visible, without turning a configured count into a quality gate.
- Keep the generated dataset immutable and opt-in before it becomes selectable for training.

**Non-Goals:**
- Do not create a generic workflow scheduler or a new metadata database.
- Do not enable CPT, Preference, or GRPO generation in this change.
- Do not add a new provider SDK, change model downloads, or invoke external providers during tests.
- Do not change the established Mini SFT recipe or evaluation protocol.

## Decisions

1. **Use a controlled local generation job, not arbitrary subprocess input.** Add a dedicated generation-workflow service behind the existing loopback-only UI server. It accepts a bounded, typed request referencing a validated generation profile/template and registered seed inputs; the server resolves paths under the workspace and invokes the existing generation code. The browser must not submit shell commands, filesystem paths, secrets, or environment overrides. Keep the existing local token and same-origin checks, and require explicit confirmation for execution. A persisted job record and bounded log/status endpoint allow polling and page reload recovery. Reusing the launcher process supervisor was considered, but generation has different inputs, artifacts, and lifecycle; it should share security conventions, not couple generation state to training state.

2. **Separate usable-record target from generation/request budgets.** The UI records an optional desired usable SFT record count and an independent hard maximum for provider requests/candidates. Existing generator candidate counts remain explicit in the execution evidence. The job may stop at the hard budget and report a shortfall; it must never claim the usable target was met based solely on candidate count. Current generation remains bounded and uses existing validation/policy filtering; no unbounded retry loop is introduced.

3. **Require an explicit dataset registration action.** Generated output first remains in the existing private synthetic-artifact area. After generation, the user reviews its actual count and validation/quality evidence and explicitly registers a non-empty valid SFT result. Registration creates a new immutable dataset artifact under the processed-data registry with a stable dataset ID and canonical content hash; its manifest identifies `model_generated` provenance and links back to the generation execution/artifact. It does not overwrite or expose the private source artifact. Reusing the private artifact in the launcher was rejected because the launcher and generic data catalog intentionally discover registered processed datasets, not arbitrary private outputs.

4. **Reuse the registered-dataset launcher and evaluation chain.** Once registered, the dataset appears through existing `/api/launcher/capabilities` and preflight by `dataset_id`; training and evaluation continue through their existing APIs. Extend the launcher dataset scan only if necessary so valid registered generated datasets are discoverable; do not introduce a second training path. Evaluation lineage remains tied to the actual training run, produced model artifact, dataset identity, and evaluation protocol.

5. **Reuse canonical JSONL import and lineage primitives.** Registration must validate canonical SFT records and generate the same class of manifests, content digests, quality evidence, and dataset-build lineage as other registered datasets. Preserve parent/source IDs and execution/config fingerprints as metadata. Avoid inventing a second dataset schema or rewriting existing lineage files.

6. **Keep execution and secrets local.** Provider profiles contain references to environment variable names; credentials are read only by the backend at execution time and never returned by APIs or written into logs, plans, job state, or dataset manifests. Add a separate `--enable-synthetic-generation` opt-in instead of reusing the policy editor or training-launcher flag; reject remote binding whenever it is enabled. This makes the potentially billable external-call capability visible and independently disableable.

## Risks / Trade-offs

- [Generation yields fewer usable rows than requested] → Show candidate, accepted, rejected, request, and shortfall counts separately; permit explicit registration of a non-empty valid partial dataset with its actual count.
- [A UI job is interrupted or the page reloads] → Persist job state and terminal evidence under the workspace; expose status/log polling and never silently restart a job on reload.
- [Generated data is mistakenly treated as validated high-quality data] → Preserve quality status and model-generated provenance; require review/registration action and show warnings in launcher preflight.
- [Provider secrets leak through diagnostics] → Redact provider errors and logs, store only profile ID and environment-variable name, and add regression tests asserting secret values are absent.
- [Existing local artifacts include malformed or unsafe paths] → Resolve only allowlisted workspace-relative inputs, reject symlinks and path escapes, and validate manifests/content before registration.

## Migration Plan

No migration of existing runs or datasets is required. The feature is opt-in behind the existing local UI controls. Existing targeted-generation CLI, manifests, launcher API, and evaluation artifacts remain readable. Rollback consists of disabling the UI generation capability; already registered immutable datasets and their lineage remain valid and are not deleted.
