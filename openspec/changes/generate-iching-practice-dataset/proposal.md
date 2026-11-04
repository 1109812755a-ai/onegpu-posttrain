# Proposal

## Why

The 81-question registered objective suite covers the Eight Trigrams, the 64 hexagrams, line texts, the Ten Wings, and yin-yang/line-position concepts. Weight-space programming needs separate practice prompts for iterative diagnosis and forward-pass feedback, while this suite must remain held out for final evaluation.

## What Changes

- Add a 300-question, Chinese, standalone I Ching practice corpus in UTF-8 JSONL, shaped for the existing objective-choice import workflow.
- Match the source suite's question-type mix: 100 single-choice, 89 multiple-choice, and 111 true/false questions.
- Cover the source suite's observed scope: Eight Trigrams and their images/attributes, hexagram ordering and composition, judgments and line texts, Tuan/Xiang/Wenyan/Xici/Zagua passages, and yin-yang/line positions.
- Include answer evidence and primary-text citations in accompanying provenance material; retain only fields accepted by the importer in each JSONL record.
- Check exact and practical near-duplicate overlap against the held-out 81-question suite, validate all 300 rows with the existing import preview, and register the practice corpus as a separately identified dataset without modifying the held-out suite.

## Capabilities

### New Capabilities

- `iching-practice-dataset`: A traceable, import-ready practice corpus kept separate from the protected 81-question evaluation suite.

### Modified Capabilities

None.

## Impact

Adds the practice JSONL and source/provenance documentation under `data/`; uses the existing `objective-choice-jsonl-v1` preview and registration flow. No importer API or evaluator changes are planned. The corpus is for weight-space practice and is not evidence of model improvement or an independently validated benchmark.
