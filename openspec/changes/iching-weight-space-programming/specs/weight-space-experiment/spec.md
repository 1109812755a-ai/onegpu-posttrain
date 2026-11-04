# Spec Delta

## Purpose

Provides a local, reviewable workflow for paper-inspired weight-space experiments that iteratively inspect model behavior, apply bounded weight edits to isolated candidates, and compare final results on a protected holdout.

## ADDED Requirements

### Requirement: Experiment identity and fixed dataset roles
An experiment MUST pin the starting model identity and revision, inference runtime identity, the practice dataset identity, the final holdout identity, and the protocol before the first edit; practice feedback MUST NOT be represented as final-holdout evidence. The target model and the external edit-planning model MUST be recorded as distinct roles.

#### Scenario: Create a Qwen3.5 I Ching experiment
- **WHEN** a user selects the registered Qwen3.5-0.8B checkpoint at revision `05f056712f8c582c8708395156518d963b46fb29`, the registered 300-question practice set, and the protected 81-question final set
- **THEN** the workbench records the target checkpoint and processor identity, the separate Bailian `qwen3.8-max` editor identity, runtime image digest and package versions, dataset IDs and content digests, marks the 300 questions as iterative practice and the 81 questions as final holdout, and rejects identical dataset IDs or digests

#### Scenario: Target-model Docker runtime is unavailable
- **WHEN** the pinned runtime image, requested model revision, verified model snapshot, or GPU is unavailable
- **THEN** preflight blocks the experiment and reports the missing identity or capability without substituting another model, image, or revision

#### Scenario: Holdout data requested for an iteration
- **WHEN** an iteration request includes final-holdout questions, answers, or derived per-question feedback
- **THEN** the workbench rejects the request and records that the holdout remains reserved for final evaluation

### Requirement: Bailian Qwen3.8-Max performs the weight-space iteration
The workbench MUST call the configured Bailian `qwen3.8-max` model for each user-started iteration and MUST mediate its tensor inspection and edit tool calls through a bounded local tool interface.

#### Scenario: Start a practice iteration
- **WHEN** the user confirms a practice iteration with a ready provider profile
- **THEN** the workbench sends the pinned candidate identity and practice feedback to `qwen3.8-max`, services only declared weight-inspection or edit tools, and records the model/tool turns under that iteration

#### Scenario: Provider is unavailable
- **WHEN** credentials are missing, the request times out, or Bailian returns an error
- **THEN** the workbench records a sanitized failure, preserves the parent and candidate checkpoints, and offers retry without implying an edit was applied

### Requirement: Provider credentials and request costs are disclosed
Provider credentials MUST remain on the local backend, and the workbench MUST disclose the selected model, request budget, and estimated cost status before sending billable iteration requests.

#### Scenario: Confirm a billable iteration
- **WHEN** a user reviews the iteration request before submission
- **THEN** the workbench displays the model ID and maximum planned tool/model calls, shows a cost estimate when configured or marks pricing unknown, and waits for explicit user confirmation before sending

#### Scenario: Provider credential is unavailable
- **WHEN** the configured credential is not available to the local backend
- **THEN** the workbench reports the missing configuration without exposing credential contents or sending a request

### Requirement: Candidate weight edits are isolated and reversible
Every accepted edit MUST be applied to a new candidate derived from the pinned starting or explicitly selected parent checkpoint, and MUST preserve an auditable record of the operation and resulting weight changes.

#### Scenario: Apply a valid edit
- **WHEN** a validated edit tool call is accepted for an iteration
- **THEN** the system writes a distinct candidate artifact, records its parent, edit payload, tensor names, changed element counts, and before/after digests, and leaves the registered starting checkpoint unchanged

#### Scenario: Reject an invalid tool call
- **WHEN** the model requests an unknown tensor, exceeds declared bounds, supplies an invalid value, or references a stale parent digest
- **THEN** the call is rejected, no checkpoint is modified, and a validation result is returned to the model and shown in the workbench

#### Scenario: Discard a candidate
- **WHEN** a user rejects an iteration candidate
- **THEN** the candidate is marked discarded and cannot be selected for final evaluation unless explicitly restored through a recorded action

### Requirement: Practice feedback and final evaluation are separate
The workbench MUST allow iterative scoring on the practice set and final scoring on the fixed holdout, while clearly separating both result types and their denominators.

#### Scenario: Run a practice iteration
- **WHEN** a candidate is scored during an iteration
- **THEN** the report identifies the practice dataset, records per-type and overall accuracy with actual denominators and parse failures, and labels the result as iterative feedback

#### Scenario: Complete final holdout evaluation
- **WHEN** a user explicitly evaluates a selected candidate on the pinned 81-question holdout
- **THEN** the report identifies the final-holdout dataset and protocol, records coverage and failures, and compares the result with the pinned baseline without exposing practice feedback as holdout evidence

### Requirement: Results describe adaptation limits
The workbench MUST identify the workflow as a paper-inspired adaptation whenever the selected model, task, or dataset differs from the cited paper's experiment.

#### Scenario: View experiment summary
- **WHEN** the selected model is Qwen3.5-0.8B and the task uses I Ching objective questions
- **THEN** the summary identifies the task, practice/holdout datasets, and scoring differences from the paper and does not label the experiment an exact reproduction

### Requirement: Reproduce the pinned Qwen3.5 inference baseline
The target model MUST be evaluated in the pinned Docker runtime with the recorded Qwen3.5 processor chat template, deterministic generation settings, and existing `choice-accuracy-v1` scoring; candidate results MUST use the same inference configuration as the baseline.

#### Scenario: Verify the registered holdout baseline
- **WHEN** the pinned Qwen3.5-0.8B revision is evaluated against the registered 81-question holdout
- **THEN** the report records the input digest, runtime and model identities, `max_new_tokens=16`, greedy decoding, sample count, per-type metrics, and parse failures; the reference run is 33/81 with zero parse failures

#### Scenario: Compare an edited candidate
- **WHEN** an edited candidate is compared with the Qwen3.5 baseline
- **THEN** the workbench allows a direct comparison only when dataset digest, runtime image, model/processor revision, chat template, generation configuration, scoring protocol, and sample coverage match
