# Tasks

## 1. Verified model catalog

- [x] 1.1 Implement a server-side catalog of locally loadable base/merged artifacts resolved from model registry and Run lineage; verify ID, digest, path containment, and file availability with unit tests for missing, mutated, and unsupported artifacts.
- [x] 1.2 Add a safe model-catalog API returning stage, model ID, digest, producing Run, and availability reason without exposing arbitrary filesystem paths; verify response filtering and local API access controls.

## 2. Shared GPU task lease

- [x] 2.1 Implement an atomic workspace GPU lease with owner/task metadata, busy response, stale-owner detection, and explicit recovery behavior; verify mutual exclusion and crash/recovery cases with tests.
- [x] 2.2 Integrate training and formal evaluation starts with the shared lease while preserving their existing Run states and outputs; verify training, evaluation, and competing starts cannot overlap in integration tests.

## 3. Bounded local inference session

- [x] 3.1 Implement an ephemeral local model session using only catalog-resolved models, bounded conversation history/output/decoding settings, and safe local-only loading; verify fake-loader tests cover prompt rendering, bounds, and model identity.
- [x] 3.2 Implement cancellation, unload, GPU lease release, and sanitized failure cleanup for normal completion, cancellation, OOM, and exceptions; verify each path releases the model and lease in tests.

## 4. Inference job API and serial comparison

- [x] 4.1 Add start/status/cancel APIs for single-model chat with server-side model resolution and non-persistent prompt handling; verify job lifecycle, busy-GPU rejection, and that prompts/completions are absent from logs and general Run artifacts.
- [x] 4.2 Add same-prompt comparison for at most three models, loading and unloading each sequentially and retaining completed outputs when a later model fails; verify ordering, model isolation, maximum selection, partial failure, and lease lifetime.

## 5. Workspace experience and Run handoff

- [x] 5.1 Add the “模型体验” subview under training/evaluation with single-model multi-turn chat, up-to-three same-prompt comparison, generation settings, activity states, and qualitative/unverified labels; verify keyboard accessibility and layout at 390px and desktop widths.
- [x] 5.2 Add a “试用此模型” action for eligible successful Run outputs that passes model ID and return Run ID, not a path; verify it preselects the correct verified model and returns to the source Run.
- [x] 5.3 Document model eligibility, GPU exclusivity, session privacy, comparison limits, and the distinction from formal evaluation; verify documented behavior matches UI/API contracts.

## 6. Integrated verification

- [x] 6.1 Run focused backend/API/UI suites plus the full project test suite and frontend production build; verify existing training and formal evaluation workflows remain green and no chat session creates a Run, evaluation result, or dataset.
