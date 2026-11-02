# Spec Delta

## ADDED Requirements

### Requirement: SFT source cohorts remain independently auditable
Mini readiness MUST report each configured SFT source cohort's eligible budget, quality-review state, processing evidence, and lineage separately unless a versioned profile explicitly defines a combined cohort.

#### Scenario: Multiple SFT sources are configured
- **WHEN** a Mini profile includes more than one SFT source cohort
- **THEN** readiness reports each source's count and acceptance dimensions independently and does not silently sum them toward the 5,000-example target

#### Scenario: A versioned profile explicitly combines SFT sources
- **WHEN** a Mini profile declares a combined SFT cohort and names its source members and selection protocol
- **THEN** readiness reports both the combined total and each member's separate count, review state, and lineage reference
