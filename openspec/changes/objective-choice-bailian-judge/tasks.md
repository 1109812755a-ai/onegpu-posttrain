# Tasks

## 1. Bailian Judge Protocol

- [ ] 1.1 Add the versioned judge prompt and strict structured outcome parser; verify unit coverage for correct, incorrect, undetermined, malformed response, and rationale validation.
- [ ] 1.2 Reuse the approved DashScope provider profile and local credential loader for per-question judging; verify timeout, rate limit, provider request ID, secret redaction, and missing-credential behavior with a fake transport.
- [ ] 1.3 Persist per-question outcomes and protocol metadata, including decided accuracy, decision coverage, non-decision counts, provider/model identity, prompt hash, effective settings, and comparability key; verify generated artifacts reproduce all aggregates from the rows.
- [x] 1.4 Document judge protocol semantics and Bailian credential setup in the Chinese evaluation guide; verify documented profile names and environment variable match the checked-in provider configuration.

## 2. Evaluation Run and Preflight

- [ ] 2.1 Extend strict evaluation request/preflight contracts for explicit scoring strategy and the configured Bailian profile; verify exact-match remains the default and invalid/unconfigured judge profiles block before Run creation.
- [ ] 2.2 Bind explicit launch confirmation to model, dataset digest, sample count, scoring strategy, provider/model, prompt version, and expected request count; verify changing any bound input invalidates confirmation and no provider request occurs during preflight.
- [ ] 2.3 Run supervised per-question judging with bounded concurrency/retry behavior and partial recovery; verify timeout, uncertain request, provider error, and resumed/partial status remain distinct and do not inflate incorrect counts.
- [ ] 2.4 Persist evaluation lineage and metrics under the new protocol while preserving legacy `choice-accuracy-v1` readers; verify both protocols load, and comparability rejects cross-protocol delta calculations.
- [ ] 2.5 Add a row-scoped manual retry operation that reuses stored model output, validates the source run and `provider_error` outcome, records every attempt separately, and leaves original artifacts and metrics unchanged; verify successful, repeated-failure, invalid-row, and uncertain-request cases with a fake provider.
- [x] 2.6 Derive report-level current metrics from original outcomes plus each row's latest completed manual retry decision, falling back to the original outcome when retries only fail; verify accuracy, coverage, and outcome counts without changing `eval.json` or lineage.

## 3. User Interface and Result Review

- [ ] 3.1 Add the exact-match/Bailian strategy selector and preflight disclosure of model, suite, sample count, expected requests, and data sent to Alibaba Cloud; verify explicit confirmation gates launch and responsive keyboard-accessible UI tests pass.
- [ ] 3.2 Extend the per-question report with judge outcome, rationale, request identity, and undecided/provider-error states; verify filtering and aggregate coverage agree with persisted rows without changing exact-match presentation.
- [ ] 3.3 Add API/UI integration coverage for a complete run, partial provider failure, missing credentials, stale confirmation, and cross-protocol comparison; update the Chinese user guide and verify links and described result fields.
- [x] 3.4 Expose live judge progress with completed/total counts, outcome counters, current processing phase, and bounded safe progress events; verify a slow fake-provider run updates without page reload and logs omit questions, answers, model output, and credentials.
- [ ] 3.5 Add per-row retry controls only for API-error judgments, show original and supplemental retry history together, and require confirmation that discloses resent content and possible duplicate billing; verify keyboard operation, no request before confirmation, and no retry control for `undetermined` or decided rows.
- [x] 3.6 Display original and retry-adjusted current metrics as separate views, including accuracy, coverage, and unresolved outcome counts; verify successful, undetermined, and repeated-provider-error retries are reflected without altering original metrics.
- [x] 3.7 Group each metric view into overall, single-choice, multiple-choice, and true/false sections, keeping accuracy, coverage, and unresolved counts together; verify both views use matching order and responsive layout.
