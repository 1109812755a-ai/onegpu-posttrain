# Spec Delta

## MODIFIED Requirements

### Requirement: Quality evidence is linked and reviewable
Every official Mini dataset split MUST have a versioned quality report linked to its dataset identity and content digest, with applicable metric units and denominators, checker or protocol version, coverage, limitations, and explicit unavailable or not-measured states.

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
