# Smoke Data Build

This document describes the first implemented execution slice after Design v0.1.1 Freeze.

The 2026-10-16 additive report implementation is documented in [Data Evidence](DATA_EVIDENCE.md): both builders now emit typed processing traces and initial static quality reports without changing stage-selection rules. Historical scope below describes the original smoke-build slice.

## Scope

Implemented:

- Hugging Face streaming or local JSONL source adapters;
- pinned Qwen3 tokenizer integration;
- FineMath document-level train/holdout split before chunking;
- nested/prefix-compatible CPT token budgets;
- secondary FineMath benchmark contamination verification;
- OpenR1 normalization and exact dedup;
- benchmark decontamination before split;
- Math-Verify-based positive/negative eligibility using upstream correctness flags;
- complete + parseable reasoning gate;
- Qwen3 chat-template rendered Length Eligibility Gate;
- deterministic problem-level SFT / Alignment / Holdout split;
- length-controlled Preference pair construction;
- verifiable GRPO prompt + ground-truth construction;
- JSON + Markdown Data Reports;
- dataset manifests with source/tokenizer revisions and SHA256 hashes.

Not implemented in this slice:

- ms-swift native-format export;
- runtime Math-Verify reward implementation;
- Base evaluation;
- training commands/checkpoint lineage execution;
- real RTX 4060 smoke measurements.

## Gate order — OpenR1

```text
raw rows
  ↓
normalize + exact dedup
  ↓
GSM8K/MATH-500 contamination scan
  ↓
Verifiable Math Core
  ↓
problem-level deterministic split
  ├── SFT Pool
  ├── Alignment Core
  └── Internal Holdout
       ↓
stage-specific tokenizer-rendered Length Gate
       ↓
SFT / Preference / GRPO / Eval canonical artifacts
```

Preference pairs require both responses to be complete and parseable. The chosen response must be Math-Verify correct; the rejected response must be Math-Verify wrong. Among valid negatives, the response with token length closest to the chosen response is selected, subject to the default ratio `[0.5, 2.0]`.

## Gate order — FineMath

```text
raw document
  ↓
stable document_id
  ↓
document-level train/holdout assignment
  ↓
secondary benchmark-contamination scan
  ↓
tokenizer chunking
  ↓
fixed prefix token budget
```

The last chunk is not truncated merely to hit an exact token number. This intentionally preserves nested dataset-size experiments: a smaller CPT corpus is a prefix/subset of the larger corpus under the same pinned source revision and traversal order.

## Offline E2E validation

The repository tests use synthetic rows shaped like the real OpenR1 and FineMath schemas. The end-to-end CLI has also been exercised with local JSONL input and produced:

```text
canonical/*.jsonl
manifests/*.yaml
reports/*_data_report.json
reports/*_data_report.md
reports/*contamination*.jsonl
```

The test tokenizer is intentionally marked `whitespace-test-only`; official builds must use the pinned Qwen3 tokenizer.
