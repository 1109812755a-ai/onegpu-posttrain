# Spec Delta

## ADDED Requirements

### Requirement: Select an objective-choice scoring strategy
客观题评测 MUST allow the user to select deterministic exact matching or Alibaba Bailian LLM judging, with exact matching as the default; each strategy MUST produce a separately identified protocol result.

#### Scenario: Keep existing exact-match behavior by default
- **WHEN** the user starts objective-choice evaluation without changing the scoring strategy
- **THEN** the evaluator uses the existing exact-match protocol and produces the same metric semantics as before

#### Scenario: Explicitly select Bailian judging
- **WHEN** the user selects LLM judging during evaluation setup
- **THEN** preflight identifies the configured Bailian provider profile, judge model, protocol version, dataset, sample count, and expected request count before launch

### Requirement: Confirm evaluation data sent to Bailian
Before making external judge requests, the system MUST disclose which evaluation content will be sent to Alibaba Cloud and require explicit user confirmation; provider credentials MUST NOT be exposed in the browser or persisted in evaluation artifacts.

#### Scenario: Confirm external evaluation request
- **WHEN** the user reviews preflight for Bailian judging
- **THEN** the interface states that questions, options, reference answers, and evaluated model outputs will be sent to Alibaba Cloud, and no request is made until the user explicitly starts the evaluation

#### Scenario: Bailian credential is unavailable
- **WHEN** the configured provider credential is missing or the provider profile is invalid
- **THEN** preflight blocks launch with a safe, actionable error and does not create a completed evaluation result

### Requirement: Report LLM-judge decisions and coverage separately
Each LLM-judged record MUST preserve the judge outcome and rationale; undecided and provider-error outcomes MUST remain distinct from incorrect answers, and aggregate results MUST report decision coverage separately from accuracy.

#### Scenario: Judge decides a record
- **WHEN** Bailian returns a valid correct or incorrect decision
- **THEN** the per-question report records that decision, rationale, and judge identity, and aggregate accuracy uses only decided records as its denominator

#### Scenario: Judge cannot decide or request fails
- **WHEN** Bailian returns an undecided result, malformed response, timeout, or provider error
- **THEN** the report records the corresponding non-decision state, excludes it from the decided accuracy denominator, includes it in coverage accounting, and does not count it as incorrect

### Requirement: Keep LLM-judge results traceable and incomparable to exact-match scores
Every LLM-judge result MUST bind its metrics to the registered dataset identity and digest, scoring protocol, provider/model identity, judge prompt version, and effective request settings; it MUST NOT be compared with exact-match metrics as if they used the same protocol.

#### Scenario: Reproduce and inspect a judge run
- **WHEN** a completed Bailian-judged evaluation is opened
- **THEN** its lineage and per-question report identify the dataset, protocol, configured provider/model, prompt version, judge decisions, and non-decision counts without exposing credentials

#### Scenario: Compare different scoring strategies
- **WHEN** a user views results produced by exact matching and LLM judging
- **THEN** the UI displays them as different protocols and does not calculate a default score delta or rank them as directly comparable

### Requirement: Show live progress for per-question LLM judging
While a Bailian objective-choice evaluation is running, the system MUST expose completed and total question counts, outcome counts, current processing phase, and safe progress events; progress logs MUST NOT contain question content, reference answers, model outputs, or credentials.

#### Scenario: A question finishes judging
- **WHEN** a question's result is durably recorded
- **THEN** the visible completed count and the matching outcome counter advance, while undetermined and provider errors remain separate from incorrect answers

#### Scenario: A question is still being processed
- **WHEN** local generation or a Bailian request is in progress
- **THEN** the run status identifies the current phase and the interface updates automatically without requiring a page reload

#### Scenario: Progress events are displayed
- **WHEN** the interface displays live judge progress
- **THEN** it shows bounded status/count information and sanitized error categories without exposing evaluation content or secrets

### Requirement: Manually retry failed LLM judgments
The system MUST allow a user to manually retry an individual `provider_error` judgment using the stored evaluated-model output, while preserving the original evaluation result and each retry attempt as separate traceable evidence.

#### Scenario: Retry an API-error judgment
- **WHEN** the user opens a row whose latest original evaluation outcome is `provider_error` and confirms retry
- **THEN** the system sends a new Bailian judgment request for that row without rerunning local model generation, and reports the retry outcome alongside the unchanged original error

#### Scenario: Do not offer retry for a judge decision
- **WHEN** a row outcome is `correct`, `incorrect`, or `undetermined`
- **THEN** the report does not offer the API-error retry action for that row

#### Scenario: Retry request fails again
- **WHEN** a manually retried judgment returns a provider error
- **THEN** the system preserves both attempts, displays the latest provider-error category, and allows a later explicit retry

### Requirement: Confirm external submission for a manual judgment retry
Before a manual retry sends evaluation content to Bailian, the system MUST identify the content being resent and disclose that an uncertain prior request may have been processed and that retrying may incur another charge.

#### Scenario: Confirm retry and possible duplicate charge
- **WHEN** the user starts a retry for a row whose prior request outcome is uncertain
- **THEN** the interface identifies the question, options, reference answer, and stored model output that will be resent, warns that another charge may occur, and sends no request until the user confirms

#### Scenario: Retry a request known to have failed before processing
- **WHEN** the user starts a retry for a provider error marked as not uncertain
- **THEN** the interface still identifies the evaluation content that will be resent and requires explicit confirmation before sending it

### Requirement: Recompute current evaluation metrics from successful manual retries
The system MUST present original evaluation metrics unchanged and derive a separate current metric view using each question's latest completed judge decision from a manual retry when available.

#### Scenario: A retry completes with a judge decision
- **WHEN** a manual retry returns `correct`, `incorrect`, or `undetermined`
- **THEN** the current metric view uses that outcome for the question while the original metric view remains unchanged

#### Scenario: A retry fails after a completed manual judgment
- **WHEN** a question already has a completed manual retry decision and a later retry returns `provider_error`
- **THEN** the current metric view retains the most recent completed judgment and the retry history still shows the later provider error

#### Scenario: Compute current accuracy and coverage
- **WHEN** the current metric view is displayed
- **THEN** accuracy uses correct divided by correct plus incorrect, decision coverage uses correct plus incorrect divided by the full sample count, and unresolved provider errors and undetermined outcomes remain separately counted

#### Scenario: No retry has completed successfully
- **WHEN** a question has only provider-error retry attempts or no manual retry attempts
- **THEN** the current metric view uses the original outcome for that question

### Requirement: Group objective-choice evaluation metrics by question type
The report MUST group objective-choice metrics into overall, single-choice, multiple-choice, and true/false sections, with accuracy and decision coverage presented together within each applicable section.

#### Scenario: Review grouped objective-choice metrics
- **WHEN** a user opens an objective-choice evaluation report
- **THEN** the report presents overall metrics first, followed by separate single-choice, multiple-choice, and true/false groups, with accuracy and coverage visually associated by group

#### Scenario: Review original and retry-adjusted metrics
- **WHEN** the report contains both original and retry-adjusted metric views
- **THEN** both views use the same question-type grouping and ordering so users can compare corresponding metrics
