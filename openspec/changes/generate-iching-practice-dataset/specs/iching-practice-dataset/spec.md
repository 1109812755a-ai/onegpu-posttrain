# Spec Delta

## Purpose

This capability provides a traceable, import-ready I Ching practice corpus for weight-space experiments while preserving the registered 81-question suite as an unseen final evaluation set.

## ADDED Requirements

### Requirement: Deliver a fixed-size objective-choice practice corpus
The project SHALL provide exactly 300 Chinese objective-choice practice records covering the I Ching topics represented in the registered 81-question evaluation suite.

#### Scenario: Verify corpus coverage and question-type mix
- **WHEN** the practice corpus is delivered
- **THEN** it contains 100 single-choice, 89 multiple-choice, and 111 true/false questions, with coverage of trigram images and attributes, hexagram order and composition, judgments and line texts, Ten Wings passages, and yin-yang/line-position concepts

### Requirement: Practice records are importable and attributable
Every delivered record SHALL conform to the existing objective-choice import source format, and accompanying provenance SHALL identify the primary-text source and answer evidence without asserting unverified rights or correctness.

#### Scenario: Preview the full corpus for import
- **WHEN** all 300 records are submitted to the existing objective-choice import preview
- **THEN** all rows are accepted with no schema errors, the preview reports 300 valid records, and source/answer-evidence metadata remains available in the delivered provenance material

### Requirement: Keep practice data disjoint from final evaluation
The practice corpus SHALL be checked against the registered 81-question suite and SHALL NOT modify, replace, or be represented as that held-out suite.

#### Scenario: Detect overlap before delivery
- **WHEN** the practice corpus is compared with the held-out suite
- **THEN** exact and reviewable near-duplicate checks are reported, identified overlaps are removed or rewritten from independent primary-text evidence, and the 81-question suite remains byte-for-byte unchanged

### Requirement: Register the corpus as a distinct dataset
The practice corpus SHALL be registered through the existing local objective-choice import flow under a distinct dataset identity and SHALL retain its original-source digest, canonical digest, count, source declaration, and registration lineage.

#### Scenario: Register the validated practice corpus
- **WHEN** the validated JSONL is registered using the objective-choice import flow
- **THEN** registration creates or reuses only the content-matching practice dataset identity, records 300 accepted rows and available source evidence, and leaves the held-out dataset identity and artifacts unchanged
