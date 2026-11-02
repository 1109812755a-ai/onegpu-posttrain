# Spec Delta

## MODIFIED Requirements

### Requirement: Quality evidence remains distinct from length eligibility
The calibration report MUST distinguish length eligibility from semantic correctness and MUST identify verifier coverage and any quality metrics that were not measured.

#### Scenario: Length-only pilot has no verifier evidence
- **WHEN** a candidate is measured only for rendered length
- **THEN** the report labels semantic correctness as not measured and does not describe length-retained records as verified training examples

#### Scenario: Semantic verification is explicitly enabled
- **WHEN** the user opts into semantic final-answer verification for a calibration candidate
- **THEN** the report records the verifier method and version, denominator, coverage, and counts for each outcome, while keeping verifier failures and timeouts distinct from incorrect answers

#### Scenario: Verification mode affects candidate selection
- **WHEN** semantic verification is enabled to select candidate SFT examples
- **THEN** candidate manifests bind to the verification protocol and verifier identity, while the default calibration mode remains exact normalized answer matching
