# Spec Delta

## ADDED Requirements

### Requirement: Official Mini readiness requires a matching quality review
The system MUST require an approved, versioned stage-quality review bound to the dataset ID, canonical content digest, and applicable review profile before an official Mini dataset can satisfy its quality dimension.

#### Scenario: Approved review matches the dataset
- **WHEN** a dataset has a valid approved review for its current identity and stage
- **THEN** readiness records the review profile and evidence reference and may satisfy quality review subject to all other Mini requirements

#### Scenario: Review is pending, rejected, or mismatched
- **WHEN** the review is pending, rejected, stale, or bound to another dataset identity or digest
- **THEN** readiness reports the specific review state and the dataset cannot satisfy official Mini quality acceptance

### Requirement: Exploratory datasets cannot satisfy official Mini acceptance
The system MUST preserve dataset usage classification and MUST prevent exploratory or intentionally noisy artifacts from satisfying official Mini budget, quality, or completion requirements.

#### Scenario: Exploratory policy output reaches its target
- **WHEN** an exploratory dataset reaches the configured Mini-sized count
- **THEN** readiness reports the quantity as exploratory evidence but does not treat it as an official Mini input

#### Scenario: A policy output lacks a usage classification
- **WHEN** readiness encounters a legacy or unclassified dataset artifact
- **THEN** it reports the classification as legacy/unknown and requires explicit reviewed classification before official acceptance

### Requirement: Remediation preserves frozen Mini contracts
The system MUST keep official Mini targets, split isolation, contamination controls, and eligibility protocols versioned; remediation MUST create new artifacts rather than silently changing or relabeling historical inputs.

#### Scenario: A user changes an eligibility rule to improve yield
- **WHEN** a remediation build uses a changed rule or verifier
- **THEN** it records a distinct policy/protocol identity and new dataset digest, and readiness does not treat it as equivalent to the frozen Mini build

#### Scenario: Remediation uses generated candidates
- **WHEN** generated candidates are included in a remediation dataset
- **THEN** readiness can trace their generation and validation ancestry, while candidates lacking required independent verification remain exploratory or explicitly unverified
