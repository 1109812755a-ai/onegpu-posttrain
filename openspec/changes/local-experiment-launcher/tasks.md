# Tasks

## 1. Fixed SFT recipe and preflight

- [x] 1.1 Add a single typed Launcher catalog entry for `recipes/mini/23-sft-1024-fallback.yaml`; verify it resolves only the pinned Mini M1, 307-example dataset and 1,024-token config, rejects all user path/parameter overrides, and passes digest/path/manifest tests.
- [x] 1.2 Expose shared side-effect-free SFT preflight using existing plan/resolution logic; verify missing or mismatched model/data, output collision, stale inputs and absent GPU resources produce clear blockers/warnings with no training artifacts or downloads.
- [x] 1.3 Add optional validated `--run-id` to `run-sft` only; verify launcher-supplied ID is identical in `run.json` and `lineage.json`, while direct CLI still creates a unique ID and other stage commands remain unchanged.

## 2. Local SFT process supervision API

- [x] 2.1 Add default-disabled `--enable-experiment-launcher`; reject remote/non-loopback launch mode and test default read-only behavior, local token/origin enforcement and startup refusal.
- [x] 2.2 Implement the one-active-SFT supervisor with fixed argv and `shell=False`, one-use preflight confirmation, atomic status and bounded logs, process-group shutdown/interrupted state, and no overwrite/retry; verify success, failure, OOM, duplicate/racing requests and shutdown with fake child processes only.
- [x] 2.3 Add capability, preflight, confirmed-start, active/completed run and log-tail APIs; test browser-safe bounded responses, no client-controlled path/command/environment, no rejected-request side effects, and links to actual CLI outputs/lineage.

## 3. Frontend SFT launch experience and guidance

- [x] 3.1 Add the opt-in Launcher page for the fixed recipe with preflight blockers/warnings, exact config confirmation, one-active-run state and clear notices for narrow VRAM headroom and service shutdown; verify disabled mode has no launch action and keyboard/mobile interaction tests pass.
- [x] 3.2 Add run monitoring for state, elapsed time, bounded logs, unknown progress, actual model/lineage links and “evaluation pending”; verify refresh, service interruption, stale/error/empty states and no fabricated zero or success.
- [x] 3.3 Update English and Chinese startup/UI roadmap guidance with the exact SFT recipe, opt-in command, exploratory caveats, browser-close vs service-stop behavior, and CLI evaluation follow-up; verify every documented flag and command exists.

## 4. Vertical-slice verification

- [x] 4.1 Exercise a fake-CLI end-to-end browser/API path from capability discovery through preflight, user confirmation, status/log refresh and final lineage display; verify the test launches no real GPU training or network call.
- [x] 4.2 Run focused backend/frontend suites, production build, lineage regression checks and `openspec validate local-experiment-launcher --strict`; verify existing read-only UI, policy editor, evaluation matrix and all non-SFT CLI commands remain unaffected.
