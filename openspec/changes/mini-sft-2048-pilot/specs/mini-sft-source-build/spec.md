# Spec Delta

## Purpose

Build reproducible Mini SFT candidate datasets from a pinned local mathematical corpus, with explicit length, quality, split, and provenance evidence before a candidate can be used as an official Mini input.

## ADDED Requirements

### Requirement: Mini SFT source identity is pinned and offline
The build MUST bind every candidate to a dataset revision, local snapshot identity, source-file digests, source license metadata, and tokenizer/template identity, and MUST NOT download missing resources implicitly.

#### Scenario: Required local snapshot is available
- **WHEN** the operator builds a candidate from the configured DeepMath snapshot
- **THEN** the output records the immutable revision and source digests and completes without network access

#### Scenario: Snapshot or license evidence is unavailable
- **WHEN** the configured snapshot, required files, or reviewable source/license evidence is missing or mismatched
- **THEN** the build stops with an actionable reason and does not silently fetch, substitute, or accept another source

### Requirement: Mini SFT split and selection are deterministic
The build MUST assign source problems to splits before selecting training examples, preserve problem-level disjointness, and make the selected 5,000-example training budget reproducible from the declared seed and source identity.

#### Scenario: Eligible training data exceeds the Mini budget
- **WHEN** more than 5,000 eligible examples remain after split, contamination, and quality-selection rules
- **THEN** the build selects at most 5,000 deterministically and records the selection method, denominator, and excluded counts

#### Scenario: Eligible training data is below the Mini budget
- **WHEN** fewer than 5,000 eligible examples remain
- **THEN** the build emits the observed count and shortfall without weakening eligibility or mixing evaluation/holdout problems into training

### Requirement: SFT length eligibility uses the complete rendered sequence
The build MUST count the pinned tokenizer's final rendered prompt-plus-answer sequence against the configured 2,048-token budget and MUST reject over-budget examples instead of silently truncating them.

#### Scenario: Candidate is within the configured budget
- **WHEN** a source trace renders to no more than 2,048 tokens
- **THEN** its measured rendered length and tokenizer/template identity are recorded in the candidate evidence

#### Scenario: Candidate exceeds the configured budget
- **WHEN** a source trace renders to more than 2,048 tokens
- **THEN** it is rejected with a length-gate reason and is not shortened to fit

### Requirement: SFT candidate quality evidence is separate from selection and approval
The build MUST report structural validity, answer-check method and coverage, reasoning-review state, duplicate and contamination outcomes, and limitations separately; reaching the count target or matching an answer string MUST NOT itself approve the candidate.

#### Scenario: Candidate has incomplete quality evidence
- **WHEN** semantic verification, human reasoning review, or contamination evidence is unavailable or incomplete
- **THEN** the candidate remains explicitly exploratory or pending review and cannot be represented as an approved Mini dataset

#### Scenario: Candidate passes a matching quality review
- **WHEN** all required checks have reviewable evidence bound to the candidate content digest and the applicable versioned profile is approved
- **THEN** the output records that review identity and may proceed to Mini readiness evaluation

### Requirement: Candidate artifacts preserve complete processing and model lineage
Every built candidate MUST emit immutable dataset manifests, ordered processing evidence, stable problem/record/source-trace identities, content digests, and lineage linking its source and build configuration.

#### Scenario: Candidate build completes
- **WHEN** a candidate dataset is emitted
- **THEN** each output split resolves to its matching manifest, processing report, quality evidence, and lineage references with reconciled counts

#### Scenario: A repeated build uses the same inputs and protocol
- **WHEN** the source revision, split seed, tokenizer/template, selection rules, and length budget are unchanged
- **THEN** the selected record identities and content digest are reproducible, while prior artifacts remain unchanged
