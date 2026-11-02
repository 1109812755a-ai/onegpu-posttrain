# Spec Delta

## Purpose

Defines bounded, reproducible mathematical final-answer verification so that equivalent answers can be distinguished from incorrect answers without confusing parser, dependency, timeout, or runtime failures with negative correctness evidence.

## ADDED Requirements

### Requirement: Verification outcomes remain distinguishable
The system MUST report final-answer verification as explicit outcomes and MUST treat only a positive verifier result as verified correctness or incorrectness.

#### Scenario: Exact normalized answers agree
- **WHEN** an extracted answer and trusted reference are equal after the declared normalization
- **THEN** the result is `correct` with the normalization method recorded

#### Scenario: A semantic verifier decides a nonliteral answer
- **WHEN** a nonliteral extracted answer and supported trusted reference are successfully compared
- **THEN** the result is `correct` or `incorrect` according to the verifier result, with the verifier identity and version recorded

#### Scenario: Verification cannot reach a decision
- **WHEN** a reference is unsupported or missing, parsing fails, the verifier is unavailable, execution times out, or a runtime error occurs
- **THEN** the result is an explicit non-decision outcome and MUST NOT be counted as `incorrect`

### Requirement: Verification execution is bounded and cross-platform
The system MUST execute batch answer verification with a finite resource bound and MUST NOT spawn an unbounded per-answer process tree on Windows.

#### Scenario: Windows batch verification completes
- **WHEN** a user verifies a bounded local batch on Windows
- **THEN** verifier work completes or reaches its configured deadline without creating a child process per parse or comparison, and each affected record receives a result

#### Scenario: Batch deadline or worker failure occurs
- **WHEN** a verifier batch exceeds its deadline or its worker fails
- **THEN** affected records are marked `timeout` or `verifier_error`, remaining work follows the declared recovery policy, and no affected record is marked incorrect by default

### Requirement: Verification evidence is reproducible and limited in claim
Every verification report MUST bind outcomes to exact input content and record verifier configuration, version, coverage, and limitations.

#### Scenario: Verification report is produced
- **WHEN** a local answer batch is verified
- **THEN** the report records input identity and digest, method and version, outcome counts, and a denominator from which coverage can be reproduced

#### Scenario: Answer verification is used as quality evidence
- **WHEN** final answers are verified against trusted references
- **THEN** the report states that this does not certify reasoning correctness or overall dataset quality
