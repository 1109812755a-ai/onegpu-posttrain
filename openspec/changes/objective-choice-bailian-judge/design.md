# Design

## Context

See `proposal.md` for motivation and `specs/unified-model-evaluation/spec.md` for the observable contract. The current objective evaluator is deterministic and local. The repository already has an OpenAI-compatible DashScope provider profile, an HTTP adapter with JSON response support and sanitized provider errors, and a local credential loader for `.env.local`.

## Goals / Non-Goals

**Goals:**

- Add a user-selected Bailian scoring path without changing existing `choice-accuracy-v1` outputs.
- Reuse the established provider profile and credential boundary.
- Preserve per-item evidence and partial outcomes, including provider request identity when available.
- Make protocol identity sufficient to distinguish judge runs from deterministic runs.

**Non-Goals:**

- Replacing exact matching, silently upgrading historical results, or treating the judge as ground truth.
- Sending complete dataset files, local paths, model credentials, or unrelated record metadata to Bailian.
- Adding arbitrary provider endpoints, free-form API parameters, or browser-managed secrets.
- Claiming model-judge scores are directly comparable to exact-match accuracy.

## Decisions

### Reuse the configured DashScope provider profile

Use the existing OpenAI-compatible provider abstraction and the configured local Bailian profile, which currently declares the DashScope compatible-mode endpoint and `qwen3.6-plus`. Resolve credentials server-side through the existing environment/`.env.local` mechanism; the browser submits only the approved profile identity. Do not add another provider dependency or accept an endpoint from the request.

Alternative considered: implement a judge-specific HTTP client. This would duplicate authentication, timeout, response validation, and error sanitization already present in the provider layer.

### Keep exact match as the default and isolate the new protocol

The evaluation request explicitly carries a scoring strategy. Missing strategy continues to resolve to `choice-accuracy-v1`; the Bailian path uses a new protocol version such as `choice-llm-judge-v1`. Comparability identity includes dataset digest and count, protocol, provider profile/model, judge prompt version/hash, and effective decoding/request settings. UI comparisons treat these results as separate metrics.

Alternative considered: add a fallback to LLM judging when exact parsing fails. Rejected because the score would vary by sample depending on parser outcome and conceal which rule produced each decision.

### Make one bounded judge request per question

Send only the question, options, reference answer, question type, and evaluated model's raw output. Require a structured result with `correct`, `incorrect`, or `undetermined` plus a short rationale. Store a validated, sanitized response and provider request ID where available. A per-question request makes timeouts, malformed responses, retries, and partial completion attributable to a single row; no batch request may silently lose or reorder records.

Alternative considered: batch many questions into one request to reduce latency/cost. Rejected for the first version because a single malformed or truncated batch response makes per-question attribution and recovery ambiguous.

### Separate decision accuracy from coverage

For LLM judging, accuracy is `correct / (correct + incorrect)` and decision coverage is `(correct + incorrect) / total`. `undetermined`, malformed responses, and provider failures are listed separately and are not silently counted as incorrect. Overall Run status distinguishes complete from partial/error runs; a metric with zero decided records is `not_measured`.

Alternative considered: count every non-decision as wrong. Rejected because it conflates model quality with network, provider, and judge uncertainty.

### Require an explicit external-data confirmation

Preflight shows the configured Bailian model, registered suite and sample count, expected request count, and the exact classes of content sent to Alibaba Cloud. The confirmation is bound to model/dataset identities, scoring strategy, provider profile/model, and prompt version; changing any input invalidates it. The request excludes local IDs and paths unless needed for local correlation, and credentials never enter request artifacts or browser state.

Alternative considered: rely on a one-time global consent. Rejected because suites and outputs vary and may contain user-authored or sensitive content.

### Make manual retries separate, row-scoped supplemental attempts

Offer retry only for rows with a Bailian `provider_error`; `undetermined` is a valid judge outcome and is not retryable through this action. Reuse the stored question evidence and evaluated-model raw output so a retry does not rerun local generation. Require a per-attempt confirmation that names the content being resent, and warn that an uncertain original request may already have been processed and billed.

Persist each manual retry as a separate supplemental attempt linked to the source evaluation run and question ID. Keep the source evaluation's predictions, lineage, and aggregate metrics immutable. The report displays the original outcome and retry history together; supplemental outcomes do not silently rewrite the original score. A later explicit retry creates another attempt rather than replacing prior evidence.

Alternative considered: edit the original prediction row and recompute its metrics in place. Rejected because this would erase the original provider error and make the evaluation artifact and its lineage change after completion.

### Derive a current metric view from the latest completed retry judgment

Keep the original `eval.json` metrics and lineage as the immutable original evaluation. In the choice report, derive a second current metric view by selecting, per question, the latest manual retry outcome among `correct`, `incorrect`, or `undetermined`; if no manual retry has produced one, use the original outcome. A later `provider_error` attempt remains in retry history and does not replace a prior completed retry decision. Recompute per-type and overall accuracy, decision coverage, and outcome counts from this derived row set using the existing judge formulas.

Alternative considered: show only retry history and require users to mentally reconcile it with original aggregate metrics. Rejected because a successful retry would leave the main reported error count and accuracy stale and make the corrected review hard to interpret.

### Group metrics by question type

Present overall metrics first, then group single-choice, multiple-choice, and true/false metrics separately. Keep accuracy and decision coverage adjacent within each group, and include the corresponding unresolved outcome counts. When both original and retry-adjusted views are available, use the same group order and layout in each view so users can compare matching metric categories directly.

Alternative considered: keep one flat card grid sorted by metric key. Rejected because accuracy, coverage, and question-type labels become difficult to associate when all cards have equal visual weight.

## Risks / Trade-offs

- [The configured model name may be a mutable provider alias] → Record the declared model, provider profile/revision if available, protocol and prompt hash, plus provider request IDs; label revision as unavailable when not supplied.
- [Judge decisions can be inconsistent or biased by the reference answer] → Preserve rationale and raw evaluated output, report coverage, version the prompt, and keep this score separate from exact matching.
- [Per-question requests increase latency and provider cost] → Show expected request count at preflight, bound concurrency and retries, persist progress per record, and allow partial runs to remain inspectable.
- [Evaluation content leaves the local machine] → Require run-specific confirmation and transmit only the minimum fields needed for judging; never expose credentials.
- [A timeout may occur after the provider processed a billable request] → Mark the record as uncertain/provider error with request state where knowable; do not retry uncertain requests invisibly.
- [A manual retry may duplicate a billable request or produce a different judgment] → Require a separate user confirmation for every attempt, warn about uncertain prior requests, and retain all attempts without mutating original metrics.

## Migration Plan

1. Add the new protocol and judge result shape while leaving the existing deterministic evaluator and stored results readable unchanged.
2. Add provider-profile preflight, external-data disclosure, confirmation binding, and supervised execution.
3. Add separately labeled aggregate metrics and extend the per-question report to show judge outcome, rationale, request identity, and non-decision status.
4. Keep historical `choice-accuracy-v1` runs on their original protocol. Rollback by hiding the new strategy and refusing new judge runs; preserve already written results as read-only evidence.
5. Add row-scoped manual retry attempts as supplemental evidence linked to the original evaluation; keep original evaluation artifacts and metrics read-only.
6. Derive and display current metrics from the original rows plus the latest completed retry judgment for each row, while retaining the original metric snapshot unchanged.
7. Group original and current metrics by overall result and question type, with accuracy, coverage, and unresolved counts associated in each group.

## Open Questions

None. The initial implementation uses the configured Bailian profile; adding UI model selection among multiple Bailian profiles can be a later scoped change.
