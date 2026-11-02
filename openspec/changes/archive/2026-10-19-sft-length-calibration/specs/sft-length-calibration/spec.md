# Spec Delta

## Purpose

Provides reproducible evidence for choosing SFT sequence-length budgets on a single GPU by separating rendered-length retention, data-verification coverage, and bounded resource outcomes.

## ADDED Requirements

### Requirement: Offline source handling
The calibration workflow MUST use an explicitly identified local source snapshot and MUST NOT download source data implicitly.

#### Scenario: Local source is available
- **WHEN** the user starts calibration with a locally cached source
- **THEN** the report records its source revision or snapshot identity and calibration proceeds without network access

#### Scenario: Local source is unavailable
- **WHEN** the requested source snapshot is not available locally
- **THEN** calibration stops with an actionable missing-source status and does not start a download

### Requirement: Matched rendered-length comparison
The workflow MUST compare configured SFT length candidates using the same pinned tokenizer, final chat-template rendering, source sample, and eligibility rules, with the candidate length as the only changed data-selection variable.

#### Scenario: Candidate lengths are compared
- **WHEN** the user compares the default calibration candidates 1,024, 2,048, and 4,096 tokens
- **THEN** each result records its candidate budget, rendered token-count method, source/sample identity, and eligible count with an explicit denominator

#### Scenario: Candidate data is produced
- **WHEN** calibration emits candidate training data for a length budget
- **THEN** the candidate has a distinct manifest bound to its length rule and content digest, and longer examples are rejected rather than silently truncated

### Requirement: Quality evidence remains distinct from length eligibility
The calibration report MUST distinguish length eligibility from semantic correctness and MUST identify verifier coverage and any quality metrics that were not measured.

#### Scenario: Length-only pilot has no verifier evidence
- **WHEN** a candidate is measured only for rendered length
- **THEN** the report labels semantic correctness as not measured and does not describe length-retained records as verified training examples

### Requirement: Bounded training resource probe
An optional training probe MUST use the same parent model, fixed source sample pool/order, seed, and training controls across candidates; only the sequence-length budget may vary, and candidate-specific eligible dataset identities MUST be recorded alongside execution and resource measurements.

#### Scenario: Probe completes or fails
- **WHEN** a candidate probe completes, runs out of memory, or fails for another reason
- **THEN** the report records that status and available duration and peak VRAM, while unavailable measurements remain explicitly unavailable

#### Scenario: Probe results are interpreted
- **WHEN** the report presents short-probe results
- **THEN** it describes them as hardware-feasibility evidence and does not infer comparative model-quality improvement from a short run

### Requirement: Calibration is non-authoritative for official Mini data
Calibration artifacts MUST remain separate from official Mini datasets and MUST NOT change the existing smoke length default or mark Mini readiness as passed.

#### Scenario: Calibration completes
- **WHEN** the calibration report is generated
- **THEN** it identifies its exploratory status and leaves the official Mini data, readiness state, and 1,024-token smoke recipe unchanged
