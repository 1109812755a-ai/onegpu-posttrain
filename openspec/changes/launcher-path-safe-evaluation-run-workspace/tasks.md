# Tasks

## 1. Path-safe Launcher evaluation

- [x] 1.1 Move new Launcher EvalScope work/output to a shallow evaluation-run-ID directory, store the resolved relative output path in evaluation state, and keep explicit training-run/model lineage; add tests proving result/lineage endpoints still resolve both new shallow paths and legacy nested paths.
- [x] 1.2 Add a Windows-aware preflight path-budget check using the expected longest review-cache filename and a safety margin; test that the current long checkout root is accepted with the shallow layout, impossible roots fail before process creation, and no global long-path setting is required.
- [x] 1.3 Add regression tests for a successful EvalScope subprocess, failed/partial status, and no-score behavior; verify the retry-suffixed review path no longer exceeds the supported path budget in a Windows-compatible fixture.

## 2. Read-only Runs API

- [x] 2.1 Implement bounded normalization/discovery for supported CLI run records, evaluation runs and Launcher state; add fixtures for complete, legacy, duplicate-ID, malformed, failed and interrupted executions and verify missing metadata is never inferred.
- [x] 2.2 Add filtered list and stable-ID detail APIs for stage/status/time/label, config, metrics, logs, outputs and lineage; test containment, traversal, symlink, file-size, record-count, pagination and unresolved-parent behavior.
- [x] 2.3 Document supported run-record sources and read-only/path-safety semantics in the Chinese local UI guide; verify its documented routes and displayed fields match API responses.

## 3. Runs user interface

- [x] 3.1 Activate Runs navigation and add responsive list/detail routes with URL-persisted filters/selection, loading/empty/error/legacy/partial states, keyboard focus and accessible labels; add frontend tests for navigation, refresh/back, filters and 390px layout.
- [x] 3.2 Render metrics with denominators, evaluation failure/log state, output links and dataset/model/evaluation lineage without inventing scores; link the Launcher completion view to its Runs detail and verify missing/unresolved provenance fixtures.
- [x] 3.3 Update the Chinese frontend roadmap/guide with the Runs list/detail workflow and verify no Database, scheduling or arbitrary process controls were added.

## 4. Integrated verification

- [x] 4.1 Run full backend and frontend regression suites, production build, targeted Ruff and strict OpenSpec validation; verify existing Data, comparison, policy, targeted-data and Launcher routes remain functional. Full backend: 431 passed, 4 skipped; frontend production build and 11 UI tests pass. Changed Python files pass Ruff. Full-repository Ruff still reports 12 issues in unrelated existing files. Strict OpenSpec validation passes.
- [x] 4.2 From the current long checkout path, start the existing completed Launcher run's fixed 5+5 evaluation through the UI API, confirm it completes without a path error, and verify actual metric denominators plus evaluation lineage link back to the training run. Real evaluation completed from `runs/evaluations/run-launcher-mini-math-eval-84f79f72dae5`: GSM8K 5/5, MATH-500 5/5 (Level 1–5, one each); both scores were 0/5. Path preflight estimated 235 characters under the 240 safety limit. Runs detail links to parent `run-mini-sft-1024-0c5512ed2ca6`.
- [x] 4.3 Use the Runs API and UI regression suite to inspect that real training/evaluation pair, verify filters, URL refresh/back behavior and lineage validation on produced artifacts; report warnings honestly and confirm no failed/partial run is shown as successful. The new evaluation and training pair resolve as completed with explicit parent linkage. `validate-lineage --root runs` reports 0 errors and 131 warnings across existing artifacts; for this new run, warnings are limited to external/unavailable benchmark dataset manifests. Interrupted attempts remain visibly interrupted and are not presented as successful.

## Workflow follow-up

- Preserve existing nested Launcher evaluation artifacts; only new evaluations use the shallow output layout.
- Do not move or rename the user's workspace during implementation. A shorter checkout path may be used as a manual workaround, but passing verification at the current path is required.
