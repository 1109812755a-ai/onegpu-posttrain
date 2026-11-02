# Mini Dataset Readiness Specification

## Purpose

Defines when stage-specific datasets qualify as official Mini inputs by requiring explicit budget accounting, reviewable quality evidence, processing provenance, and independently validated lineage. It prevents a target count or a passing integrity check from being misrepresented as proof of data quality or model improvement.

## Requirements

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

#### Scenario: All evidence is reported but quality policy is unreviewed
- **WHEN** required budget, processing, lineage, and evidence-coverage checks pass but no applicable reviewed quality policy exists
- **THEN** readiness reports the remaining quality-approval state and does not label the dataset or Mini build ready

#### Scenario: An applicable metric is unsupported or not measured
- **WHEN** a metric cannot be evaluated for the dataset or checker coverage
- **THEN** readiness preserves the explicit unavailable status and reason, distinguishes it from a measured zero, and applies only the consequence defined by the versioned readiness policy

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

### Requirement: Quality reassessment is additive and reproducible
Reassessment of retained data MUST bind its report to the exact dataset identity and content digest, assessment protocol, checker/verifier versions, and coverage; it MUST create a new report rather than mutate the dataset, manifest, or historical reports.

#### Scenario: Checker protocol changes
- **WHEN** a retained dataset is reassessed with a changed metric or checker version
- **THEN** the new report has a distinct assessment identity and records the changed protocol while prior reports remain readable and unchanged

#### Scenario: Reassessment has no independent verifier
- **WHEN** the local environment lacks a supported independent verifier for a metric
- **THEN** reassessment completes with that metric marked not measured and records the missing capability without downloading dependencies or inventing a result

### Requirement: Complete provenance is validated before official acceptance
Official Mini dataset artifacts MUST pass local lineage and manifest-evidence integrity validation; externally unavailable evidence MUST remain explicitly identified and MUST NOT be silently treated as verified.

#### Scenario: Local references and digests are valid
- **WHEN** the Mini artifact root is checked with complete-provenance validation
- **THEN** missing required local manifests, reports, runs, references, duplicate identities, and digest mismatches are reported as errors, and official acceptance is blocked until resolved

#### Scenario: External evidence cannot be resolved locally
- **WHEN** a source revision, benchmark artifact, or other external evidence is unavailable locally
- **THEN** the report names that evidence as unavailable and distinguishes it from local integrity failures; it does not claim that the unavailable evidence was content-verified

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

### Requirement: Mini readiness preserves evidence and acceptance boundaries
Mini readiness MUST report evidence coverage separately from quality approval and MUST NOT mark a dataset ready solely because its evidence report is complete or its budget and lineage pass.

#### Scenario: All evidence is reported but quality policy is unreviewed
- **WHEN** required budget, processing, lineage, and evidence-coverage checks pass but no applicable reviewed quality policy exists
- **THEN** readiness reports the remaining quality-approval state and does not label the dataset or Mini build ready

#### Scenario: An applicable metric is unsupported or not measured
- **WHEN** a metric cannot be evaluated for the dataset or checker coverage
- **THEN** readiness preserves the explicit unavailable status and reason, distinguishes it from a measured zero, and applies only the consequence defined by the versioned readiness policy
