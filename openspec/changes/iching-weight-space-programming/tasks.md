# Tasks

## 1. Experiment identity and data isolation

- [x] 1.1 Define a versioned experiment/iteration manifest for the pinned Qwen3.5 target model and revision, Docker/runtime identity, processor/chat template, generation settings, practice and final dataset IDs/digests, protocol, roles, and baseline; verify validation rejects missing identities and practice/final ID or digest collisions.
- [x] 1.2 Register and resolve `Qwen/Qwen3.5-0.8B` revision `05f056712f8c582c8708395156518d963b46fb29` plus the 300/81 datasets through the existing registries; verify setup reports their expected identities and rejects unavailable or changed artifacts. Keep the legacy Qwen3-0.6B-Base result separate.
- [x] 1.3 Add holdout isolation checks for iteration prompts and tool results; verify final-holdout question bodies, answers, and per-question outputs never enter provider requests.
- [x] 1.4 Pin the WSL Docker image digest and the tested PyTorch/Transformers runtime; verify the host project's Transformers `<5` environment remains unchanged, the target model loads through `AutoProcessor`/`AutoModelForMultimodalLM`, and the container sees the selected GPU.
- [x] 1.5 Add preflight verification for the read-only target-model snapshot and dataset mounts; verify the snapshot revision and 81-question input digest match the pinned values and no alternate model is silently substituted.

## 2. Bailian profile and provider tool-call support

- [x] 2.1 Add a dedicated fixed `qwen3.8-max` provider profile using the existing local credential mechanism without changing the Qwen3.6 profile; verify profile loading does not return or persist credential values.
- [x] 2.2 Extend the OpenAI-compatible provider contract to encode tool declarations, tool-call responses, and tool results; verify request/response serialization against fixtures for the selected profile.
- [x] 2.3 Add bounded call count, token, and timeout controls plus sanitized provider failures and usage records; verify missing credentials, timeout, rate limiting, and malformed responses do not disclose secrets or claim an edit was applied.
- [x] 2.4 Add an opt-in live smoke command for a minimal Qwen3.8-Max request; verify it reports model identity, success/failure category, latency, and token usage without printing credentials or prompt contents.

## 3. Bounded weight tools and candidate lifecycle

- [x] 3.1 Implement fixed server-side tensor inspection and edit tool schemas with bounded outputs; verify unknown tools/tensors and out-of-budget reads are rejected.
- [x] 3.2 Implement the bounded model/tool iteration loop and validate parent digest, tensor, shape, dtype, finite values, edit range, and element budget; verify invalid calls leave parent and candidate files unchanged.
- [x] 3.3 Apply valid edits transactionally into isolated candidate checkpoints and write operation/diff/digest lineage; verify the pinned base is byte-identical and each candidate resolves to its recorded parent.
- [x] 3.4 Implement candidate retention, discard, and explicit restore state transitions; verify discarded candidates cannot be finalized without a recorded restore.
- [x] 3.5 Document supported tensor tools and edit limits; verify the model can complete a bounded mocked tool round without arbitrary code execution.

## 4. Practice and final evaluation lineage

- [x] 4.1 Register edited Qwen3.5 candidates as derived models with parent and edit-artifact lineage; verify model discovery resolves each candidate's correct path and digest without mutating the read-only base snapshot.
- [x] 4.2 Add practice-set scoring that reuses the existing objective-choice parser and metrics while recording the practice role and actual denominators; verify per-type/overall scores, parse failures, and coverage against fixed examples.
- [x] 4.3 Connect final evaluation to the pinned 81-question suite and `choice-accuracy-v1` protocol using the pinned Qwen3.5 processor template and generation settings; verify the baseline reproduces 33/81 and candidate reports use matching dataset digest, runtime, prompt configuration, protocol, sample coverage, and comparable metrics.
- [x] 4.4 Extend evaluation lineage with iteration, role, candidate digest, parent candidate, model/runtime/processor identity, and generation configuration; verify practice and final results remain distinguishable when loaded by existing report readers.

## 5. Dedicated workbench UI

- [ ] 5.1 Add a first-class route and navigation entry for the weight-space workbench; verify direct navigation, current-page state, keyboard focus, and accessible naming.
- [ ] 5.2 Implement provider readiness, data-egress notice, model/call budget and cost disclosure, and explicit per-iteration confirmation; verify unknown pricing is stated as unknown and no provider call occurs before confirmation.
- [ ] 5.3 Implement experiment setup, live iteration status, candidate review, practice metrics, and final comparison screens; verify UI state reflects backend records and sanitized provider failures.
- [ ] 5.4 Add responsive layout and state-specific empty/blocked/running/partial/failed/discarded views; verify the page remains usable at 390 CSS pixels and does not report incomplete runs as scores.

## 6. Integration and documentation

- [ ] 6.1 Wire backend endpoints to the loopback-only experiment UI and existing registered model/dataset/evaluation services; verify remote mode cannot invoke weight mutation or evaluation actions.
- [ ] 6.2 Document target-model Docker setup and pinning, local credential setup, Qwen3.8-Max editor profile, data sent to Bailian, cost confirmation, paper-adaptation limits, and rollback; verify the documented flow matches the UI and generated artifacts.
- [ ] 6.3 Run the complete workbench flow from the pinned Qwen3.5 baseline through a confirmed provider request, candidate practice round, and final holdout report; verify artifacts and scores link to the pinned model/runtime, datasets, candidate, provider identity, and protocol.
