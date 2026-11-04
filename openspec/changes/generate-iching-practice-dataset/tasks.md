# Tasks

## 1. Pin source material and build the question plan

- [x] 1.1 Select a primary-text Zhouyi/Ten Wings edition, pin its revision, record attribution and verified/unknown license status, and verify the source references resolve.
- [x] 1.2 Derive the topic map from the held-out suite without copying its question content; record target counts totaling 300 across the observed topic areas and the 100/89/111 question-type mix.

## 2. Generate and review the corpus

- [x] 2.1 Create 300 original questions in UTF-8 `objective-choice-jsonl-v1` format and a sidecar with stable IDs, answer explanations, exact primary-text citations, and generation/review provenance; verify counts and required fields.
- [x] 2.2 Independently review every answer, option set, multi-select key, and cited passage; replace ambiguous or unsupported items and record review coverage.
- [x] 2.3 Check duplicate IDs, exact normalized overlap, and bounded near-duplicate overlap against the 81-question suite; record method, findings, and limitations, and verify the 81-question dataset files and digest remain unchanged.

## 3. Preview and register the practice dataset

- [x] 3.1 Run the existing objective-choice import preview on the final file and verify exactly 300 valid rows, zero rejected rows, and correct single/multiple/true-false counts.
- [x] 3.2 Register the corpus through the existing import flow with practice-specific source/name metadata; verify the new dataset ID, eval-only role, raw/canonical digests, 300-record count, and lineage.
- [x] 3.3 Verify the registered practice corpus is distinct from the held-out suite and document the exact preview/registration steps and the rule that final benchmark reporting continues to use the original 81 questions.
