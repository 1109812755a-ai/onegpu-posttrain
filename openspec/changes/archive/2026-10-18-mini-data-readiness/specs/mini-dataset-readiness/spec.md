# Spec Delta

## Purpose

Defines when stage-specific datasets qualify as official Mini inputs by requiring explicit budget accounting, reviewable quality evidence, processing provenance, and independently validated lineage. It prevents a target count or a passing integrity check from being misrepresented as proof of data quality or model improvement.

## ADDED Requirements

### Requirement: Mini budgets use stage-appropriate units
The system MUST evaluate Mini data targets using the declared stage-specific unit and MUST report the observed eligible amount and target for each dataset split.

#### Scenario: Stage targets are reported
- **WHEN** a Mini build is evaluated for readiness
- **THEN** CPT is measured in training tokens, SFT in eligible examples, preference optimization in verified preference pairs, and GRPO in verifiable prompts, with the milestone targets of 5M, 5K, 2K, and 2K respectively

#### Scenario: Preference records contain pairs
- **WHEN** the preference train or evaluation dataset is assessed
- **THEN** its budget count is the number of complete chosen/rejected pairs, not raw rows, generations, or problems

### Requirement: Budget shortfalls preserve eligibility rules
The system MUST report an unmet Mini target and its observed stop reason without weakening the dataset's declared eligibility, verification, contamination, split, or length rules.

#### Scenario: Source or scan limit prevents target completion
- **WHEN** a Mini build reaches its configured scan cap or source exhaustion before meeting a target
- **THEN** readiness reports the target as unmet, includes available shortfall and rejection evidence, and does not mark that dataset ready

#### Scenario: User changes a qualification rule
- **WHEN** a rule affecting dataset eligibility is changed
- **THEN** the output identifies the changed, versioned protocol and is not silently treated as equivalent to a dataset built under the prior rule

#### Scenario: Same source URL has distinct extracted text
- **WHEN** FineMath contains source rows with the same URL but different emitted chunk text
- **THEN** record IDs include a deterministic chunk-content fingerprint so the distinct records do not collide, while train/holdout assignment remains keyed to the URL-derived document identity; byte-identical duplicate records remain subject to the existing duplicate-ID blocking check

### Requirement: Quality evidence is linked and reviewable
Every official Mini dataset split MUST have a versioned quality report linked to its dataset identity and content digest, with applicable metric units and denominators, checker or protocol version, coverage, limitations, and explicit unavailable or not-measured states.

#### Scenario: Quality evidence is complete for review
- **WHEN** a Mini dataset is considered for official use
- **THEN** its manifest resolves to a quality report for the same dataset content, and applicable structural, stage-specific verification, yield/rejection, length/distribution, and overlap/contamination evidence is available or explicitly marked unavailable

#### Scenario: Evidence is present but semantic quality is not evaluated
- **WHEN** a report contains structural checks but no calibrated semantic gate
- **THEN** the system preserves the unevaluated quality status and does not infer overall quality approval from report presence or target attainment

### Requirement: Processing evidence accounts for the dataset build
Every official Mini dataset MUST be traceable to an ordered, versioned processing report that records source and protocol references, step input/output counts with units, available first-rejection reasons, stop reason, and limitations.

#### Scenario: User traces a final split to source processing
- **WHEN** an official Mini dataset is reviewed
- **THEN** its lineage resolves to the build report and ordered processing steps that produced that split, without requiring reconstruction from directory names

#### Scenario: Processing evidence is incomplete
- **WHEN** a required step, count unit, report reference, or digest is missing or invalid
- **THEN** the evidence is marked incomplete or invalid and cannot satisfy the Mini processing-evidence gate

### Requirement: Mini readiness keeps acceptance dimensions separate
The system MUST report budget attainment, quality-evidence review state, and lineage integrity as separate dimensions; an overall Mini dataset status MUST NOT imply a quality pass solely from target attainment or a lineage pass.

#### Scenario: All required dimensions qualify
- **WHEN** a Mini split meets its declared budget, required quality evidence is present and reviewed under a versioned policy, and required local lineage validation succeeds
- **THEN** it may be marked ready for official Mini experiments with each dimension and its evidence reference visible

#### Scenario: One readiness dimension fails
- **WHEN** budget, quality evidence, or lineage integrity is unmet, failed, or not evaluated under its applicable policy
- **THEN** the report identifies that dimension and prevents the dataset from being represented as fully ready

#### Scenario: Model effect has not been tested
- **WHEN** no registered controlled comparison links a processing choice to downstream evaluations
- **THEN** Mini data readiness reports model effect as not evaluated and makes no causal improvement claim

### Requirement: Complete provenance is validated before official acceptance
Official Mini dataset artifacts MUST pass local lineage and manifest-evidence integrity validation; externally unavailable evidence MUST remain explicitly identified and MUST NOT be silently treated as verified.

#### Scenario: Local references and digests are valid
- **WHEN** the Mini artifact root is checked with complete-provenance validation
- **THEN** missing required local manifests, reports, runs, references, duplicate identities, and digest mismatches are reported as errors, and official acceptance is blocked until resolved

#### Scenario: External evidence cannot be resolved locally
- **WHEN** a source revision, benchmark artifact, or other external evidence is unavailable locally
- **THEN** the report names that evidence as unavailable and distinguishes it from local integrity failures; it does not claim that the unavailable evidence was content-verified
