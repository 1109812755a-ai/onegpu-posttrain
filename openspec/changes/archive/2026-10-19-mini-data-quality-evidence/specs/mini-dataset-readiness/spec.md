# Spec Delta

## MODIFIED Requirements

### Requirement: Quality evidence is linked and reviewable
Every official Mini dataset split MUST have a versioned quality report linked to its dataset identity and content digest, with applicable metric units and denominators, checker or protocol version, coverage, limitations, and explicit unavailable or not-measured states. The report MUST distinguish evidence coverage from any quality approval or gate judgment.

#### Scenario: Quality evidence is complete for review
- **WHEN** a Mini dataset is considered for official use
- **THEN** its manifest resolves to a quality report for the same dataset content, and applicable structural, stage-specific verification, yield/rejection, length/distribution, and overlap/contamination evidence is available or explicitly marked unavailable

#### Scenario: Evidence is present but semantic quality is not evaluated
- **WHEN** a report contains structural checks but no calibrated semantic gate
- **THEN** the system preserves the unevaluated quality status and does not infer overall quality approval from report presence or target attainment

#### Scenario: Evidence coverage differs from approval
- **WHEN** all applicable evidence fields have an explicit measured, estimated, not-measured, or not-applicable status but no reviewed quality policy approves the dataset
- **THEN** the report may identify evidence coverage as complete while keeping quality approval and Mini readiness not evaluated or not ready

#### Scenario: Verification runtime fails or times out
- **WHEN** answer verification fails because a dependency, parser, worker, or deadline prevents a decision
- **THEN** the quality evidence records those outcomes separately from incorrect answers, includes their denominator and coverage, and does not use them as a correctness failure or approval

## ADDED Requirements

### Requirement: Quality evidence uses stage-appropriate metrics
Quality assessment MUST report supported metrics according to the dataset stage and label each metric's meaning, method, coverage, and limitations; it MUST NOT substitute a universal quality score.

#### Scenario: SFT labels come from upstream metadata
- **WHEN** an SFT report summarizes source correctness labels without rerunning an independent verifier
- **THEN** it labels the measure as upstream label evidence with its denominator and selection limitations, not as independently verified semantic correctness

#### Scenario: Preference pairs use correctness ordering
- **WHEN** a Preference dataset declares correctness-based chosen/rejected ordering
- **THEN** the report distinguishes complete pair counts and build yield from pair-label evidence, and reports each with the correct unit, denominator, method, and verifier coverage

#### Scenario: GRPO evidence includes model-dependent calibration
- **WHEN** a GRPO quality report has static prompt evidence and a calibration run
- **THEN** static dataset measures remain separate from model/recipe-dependent reward, variance, completion-length, and truncation measures, which identify their calibration model and recipe

### Requirement: Mini readiness preserves evidence and acceptance boundaries
Mini readiness MUST report evidence coverage separately from quality approval and MUST NOT mark a dataset ready solely because its evidence report is complete or its budget and lineage pass.

#### Scenario: All evidence is reported but quality policy is unreviewed
- **WHEN** required budget, processing, lineage, and evidence-coverage checks pass but no applicable reviewed quality policy exists
- **THEN** readiness reports the remaining quality-approval state and does not label the dataset or Mini build ready

#### Scenario: An applicable metric is unsupported or not measured
- **WHEN** a metric cannot be evaluated for the dataset or checker coverage
- **THEN** readiness preserves the explicit unavailable status and reason, distinguishes it from a measured zero, and applies only the consequence defined by the versioned readiness policy

### Requirement: Quality reassessment is additive and reproducible
Reassessment of retained data MUST bind its report to the exact dataset identity and content digest, assessment protocol, checker/verifier versions, and coverage; it MUST create a new report rather than mutate the dataset, manifest, or historical reports.

#### Scenario: Checker protocol changes
- **WHEN** a retained dataset is reassessed with a changed metric or checker version
- **THEN** the new report has a distinct assessment identity and records the changed protocol while prior reports remain readable and unchanged

#### Scenario: Reassessment has no independent verifier
- **WHEN** the local environment lacks a supported independent verifier for a metric
- **THEN** reassessment completes with that metric marked not measured and records the missing capability without downloading dependencies or inventing a result
