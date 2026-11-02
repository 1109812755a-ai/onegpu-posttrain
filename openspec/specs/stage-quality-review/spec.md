# stage-quality-review Specification

## Purpose

Defines reviewable, versioned quality decisions for stage-specific training data while keeping measurements, independent verification, human judgment, and unsupported checks distinguishable and traceable.

## Requirements

### Requirement: Quality review profiles are versioned by stage
The system MUST evaluate datasets against an explicit versioned review profile that names applicable evidence, methods, units, decision rules, and treatment of unsupported checks; a profile MUST NOT create a universal composite quality score.

#### Scenario: Review uses a declared profile
- **WHEN** a dataset quality review is created
- **THEN** the report records the profile ID/version and each applicable check's method, evidence state, and decision consequence

#### Scenario: A check has no calibrated rule
- **WHEN** the profile has no reviewed rule for a metric
- **THEN** the metric remains diagnostic or pending and is not silently promoted to PASS

### Requirement: Independent verification is distinguishable from source labels
The system MUST report source-provided labels separately from independently executed verification and MUST identify verifier version, supported coverage, and the assessed denominator when independent verification is available.

#### Scenario: Upstream correctness metadata exists
- **WHEN** a report contains upstream correctness labels but no verifier execution record
- **THEN** it reports label evidence only and leaves independent correctness unmeasured

#### Scenario: Verifier supports only part of a dataset
- **WHEN** an independent verifier cannot parse or support every assessed reference
- **THEN** the report exposes verified, unsupported, and failed counts with denominators and does not treat unsupported examples as verified

### Requirement: Review outcomes are immutable and digest-bound
Every review result MUST bind to dataset ID, canonical content digest, review profile version, and checker/verifier identities; a later review MUST create a distinct additive artifact and MUST NOT rewrite the dataset, manifest, or prior review.

#### Scenario: Dataset content changes after review
- **WHEN** readiness evaluates a review whose dataset ID or canonical digest differs from the current manifest
- **THEN** the review is rejected as mismatched evidence and cannot satisfy official quality review

#### Scenario: Reviewer changes a decision
- **WHEN** an authorized local review decision is revised
- **THEN** the system records a new decision artifact with timestamp, method, rationale, and reference to the superseded review while retaining the original

### Requirement: Unavailable checks retain explicit non-approval states
The system MUST distinguish `not_measured`, `unsupported`, `not_applicable`, and measured outcomes, and MUST apply only consequences declared by the review profile.

#### Scenario: Optional verifier is not installed
- **WHEN** review runs without a supported independent verifier
- **THEN** no dependency is downloaded automatically, affected checks remain not measured, and the report records why

#### Scenario: A metric is not applicable to a stage
- **WHEN** a review profile marks a check not applicable for the dataset stage
- **THEN** the report records that disposition explicitly rather than emitting a numeric zero or passing the check implicitly
