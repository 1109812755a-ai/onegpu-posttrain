# Tasks

## 1. Generation workflow backend

- [x] 1.1 Add a workspace-contained SFT generation job service that loads only registered plan inputs, applies bounded user-approved request/candidate limits, invokes the existing generation and candidate-policy pipeline, persists an independent execution ID/status/log summary, and verifies behavior with fake-provider tests (zero calls before explicit start; no network in tests).
- [x] 1.2 Add a separately gated `--enable-synthetic-generation` opt-in and loopback/token-protected UI APIs for SFT generation preflight, explicit start, status/log polling, and cooperative cancellation; verify remote-bind rejection, invalid IDs, path traversal/symlink inputs, cross-origin requests, secret redaction, reload recovery, and no automatic execution with API tests.

## 2. Immutable dataset registration and lineage

- [x] 2.1 Add an explicit registration operation for non-empty, schema-valid generated SFT output that creates an immutable registered dataset manifest with actual count, content hash, model-generated source metadata, quality state, generation execution parent, and no overwrite; verify manifest/hash/lineage and reject empty or invalid outputs in tests.
- [x] 2.2 Extend the generated dataset discovery/preview path only as needed so registered SFT output is visible to the data catalog and Mini SFT Launcher by dataset ID; verify preview and launcher preflight use the same ID/hash and preserve warnings for partial/exploratory data.

## 3. User-facing SFT vertical slice

- [x] 3.1 Update the targeted-data UI to select an eligible registered SFT plan, display preflight identity and separate usable-record target/candidate/request budgets, require explicit cost-aware start confirmation, and show persisted progress/counts/shortfall; verify editing/browsing makes no provider call and refresh resumes status.
- [x] 3.2 Add review and explicit dataset-registration controls for completed or partial valid outputs, then link the registered dataset to the existing Mini SFT Launcher; verify empty/invalid outputs cannot be registered and successful registration becomes selectable without editing recipe files.
- [x] 3.3 Document the complete local SFT route, prerequisites, provider credential handling, partial-data semantics, security boundary, and generated dataset lineage in the Chinese user docs; verify every documented command/API/UI step matches the implementation.

## 4. End-to-end regression

- [x] 4.1 Add an offline fake-provider integration test covering preflight → explicit generation → review/registration → launcher dataset preflight and asserting lineage connects source seed → generation execution → dataset → training input; verify the test makes zero network calls and does not require GPU or model downloads.
- [x] 4.2 Run focused UI/backend tests, the full test suite, compile checks, and OpenSpec strict validation; verify reported results and leave unresolved real-provider/GPU training as an explicit user-run acceptance step.
