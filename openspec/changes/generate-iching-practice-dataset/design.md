# Design

## Context

The registered held-out dataset `ds-eval-upload-54ca42ca021d295d8390` contains 81 records: 27 single-choice, 24 multiple-choice, and 30 true/false. Questions cover the Eight Trigrams, all parts of the 64-hexagram Zhouyi, line positions, and the Ten Wings. The existing import source format is `objective-choice-jsonl-v1` with `questionType`, `question`, `correctAnswer`, and `options`; importer registration assigns the eval-only role. Canonical EvalRecord rejects undeclared fields, so answer explanations and detailed source citations cannot be embedded in imported rows.

## Goals / Non-Goals

**Goals:**

- Deliver 300 independent, Chinese practice questions matching the observed type mix and subject scope.
- Make the JSONL pass the existing preview and registration contracts without changing importer behavior.
- Preserve answer evidence, primary-text citations, source revision, generation/review method, and overlap evidence in a companion provenance file.
- Register as a distinct eval-only practice dataset and preserve the existing 81-question suite unchanged.

**Non-Goals:**

- Changing application import APIs, canonical schema, or evaluation protocol.
- Using the 81 held-out questions as generation prompts or answer material.
- Claiming the generated set is an independently validated benchmark or a training dataset.

## Decisions

### Use the existing objective-choice import contract

Export each line with only `questionType`, `question`, `correctAnswer`, and `options`, using source-format values `single_choice`, `multiple_choice`, and `true_false`. For multiple choice, encode `correctAnswer` as an array of labels; for true/false, use the accepted boolean form. Keep explanations and detailed citations in a line-aligned sidecar keyed by stable `id` or record locator because extra canonical fields are rejected.

Alternative: extend EvalRecord with practice-purpose and evidence fields. Rejected for this corpus-only change because existing import preview already accepts the required quiz fields and API changes would expand the scope.

### Build an original corpus from pinned primary-text evidence

Internet access is permitted during generation to retrieve and verify primary-text sources. Prefer the Chinese Wikisource Zhouyi text and relevant Ten Wings pages; pin each page's revision (the inspected Zhouyi landing-page revision is `oldid=7907208`) and record its URL, title, revision, retrieval date, and attribution in provenance. The inspected page identifies the work as public domain and states that Wikisource page text is provided under CC BY-SA 4.0; preserve applicable attribution/share-alike notices for material derived from that transcription, and verify the terms of every additional source used. Write new question stems and distractors from cited passages; do not copy question wording from the 81-question suite or modern copyrighted translations/commentaries. Record source/license status accurately and do not claim broader redistribution rights than the evidence supports.

Alternative: bulk-download a hosted question benchmark. Rejected because benchmark question licenses and overlap risk are harder to establish, and using pre-existing questions would weaken independence from the held-out suite.

### Match the held-out suite's distribution and subject map

Generate exactly 100 single-choice, 89 multiple-choice, and 111 true/false rows. Allocate topics across trigram images/attributes and family/order relations, hexagram structure/order, judgments and line texts, Ten Wings, and yin-yang/line-position concepts. Use the full 81-question content only to derive this high-level topic map and type counts; do not expose its questions, choices, or answer keys to generation.

Alternative: use only the eight trigram symbols. Rejected because inspection shows the held-out suite also tests the 64 hexagrams and several Ten Wings sections.

### Validate content before import

Run deterministic JSONL/schema checks, answer-label checks, duplicate-ID checks, and exact normalized text overlap against the held-out suite. Use semantic/near-duplicate review as a documented, bounded manual check; report its coverage and limitations rather than claiming complete contamination detection. Independently review every answer against the cited primary text, including multiple-answer completeness and distractor validity. Then run the real import preview and require 300 accepted/zero rejected before registration.

Alternative: treat successful structural import as quality approval. Rejected because importer validation does not establish answer correctness, source rights, or topic representativeness.

### Register with a clearly separate identity

Use a dataset name and source declaration that explicitly identify this corpus as weight-space practice, preserve the raw and canonical digests, and record eval-only role. Registration is a local data operation; it does not make this dataset the final evaluation suite. The experiment workflow must continue to select the original held-out dataset for final reported scores.

Alternative: merge the practice rows into the 81-question suite. Rejected because it would contaminate the protected final score and change its comparability identity.

## Risks / Trade-offs

- [Generated questions or keyed answers may be wrong] → Cite exact primary-text evidence and independently review all rows before registration; mark unresolved items and replace them rather than silently accepting them.
- [Primary-text edition or digital transcription rights may be unclear] → Preserve source/revision/attribution and explicit license status; do not claim open redistribution until verified.
- [Near-duplicate detection cannot prove semantic independence] → Keep the evaluation suite out of generation context and report exact-match plus manual near-duplicate review coverage and limitations.
- [Eval-only registration makes practice data selectable as an evaluation set] → Use unmistakable practice naming and document that only the 81-question suite is the held-out final metric source.
- [A 300-question set can still be small or imbalanced] → Match known question-type proportions, publish topic counts, and describe it as an experiment practice corpus rather than a validated benchmark.

## Migration Plan

1. Create the JSONL and provenance sidecar without touching registered data.
2. Validate content, answer evidence, type counts, and overlap; revise rejected rows.
3. Run objective-choice import preview and confirm 300 valid rows.
4. Register as a separate local dataset; verify resulting dataset ID, role, digests, count, and lineage.
5. Verify the original 81-question dataset manifest and canonical digest remain unchanged.

Rollback: remove the new local registration/artifact using supported repository lifecycle tooling if available; never alter or replace the held-out suite. If deletion is unavailable, leave the practice dataset registered but clearly labeled and do not select it for final evaluation.

## Open Questions

None. The source edition/license details must be verified and recorded during implementation; unresolved redistribution rights do not block local import but do block any claim that the corpus is openly redistributable.
