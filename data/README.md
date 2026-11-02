# Data pipeline

`data/examples/` contains tiny canonical examples only. Full source datasets are never committed.

Smoke build output is written under `data/processed/smoke/` by default:

```text
data/processed/smoke/
├── finemath/
│   ├── canonical/cpt_train.jsonl
│   ├── canonical/cpt_holdout.jsonl
│   ├── manifests/
│   └── reports/
└── openr1/
    ├── canonical/sft.jsonl
    ├── canonical/preference.jsonl
    ├── canonical/grpo.jsonl
    ├── canonical/internal_holdout.jsonl
    ├── manifests/
    └── reports/
```

## OpenR1 gate order

```text
normalize/dedup
→ benchmark decontamination
→ verifiable math core
→ deterministic problem-level split
→ tokenizer-rendered length gate
→ stage transform
```

Preference pairs use complete/parser-valid Math-Verify positives and negatives from the same problem, with a default rejected/chosen rendered-token ratio in `[0.5, 2.0]`.

GRPO excludes proof/notfound-style ground truth and keeps the answer only as verifier metadata; it is not an SFT target.

## FineMath gate order

```text
document id
→ document-level train/holdout split
→ secondary benchmark contamination scan
→ token chunking
→ fixed token budget
```

The train token budget is prefix-nested under a fixed dataset revision and traversal order, so 5M/10M/20M experiments can be constructed as nested budgets rather than unrelated samples.

## Offline testing

All builders accept local JSONL sources. `--test-tokenizer` uses a deterministic whitespace tokenizer for fixtures only. Never publish official length-gated results with it.

## Processing and static quality reports

Both build commands now emit `reports/processing_report.json/.md` and `reports/quality/<stage>.json/.md` alongside the existing data reports. Manifests link those reports by relative path, report ID and SHA256. The new processing reports expose ordered steps, count units, first-rejection reasons, non-quality quota exclusions, available before/after evidence and explicit stop reasons. Dataset IDs include content/configuration provenance; the legacy manifest filenames remain locators for existing recipes.

These are initial static checks, not independently reverified semantic correctness or model-effect results. Missing/empty benchmark indexes are marked skipped, and empty-population rates remain unavailable. See [implementation and limitations](../docs/implementation/data/DATA_EVIDENCE.md).

## Held-out preference evaluation

OpenR1 smoke build also emits:

```text
openr1/canonical/preference_eval.jsonl
```

These pairs come only from the top-level holdout pool. They use the same verified-complete, parseable, length-controlled pair construction rules as training preference pairs, but are marked `evaluation_only=true` and never passed to DPO/SimPO training.

Required invariant:

```text
training preference problem_id ∩ preference_eval problem_id = ∅
```

It is valid for `preference_eval` and `internal_holdout` to share problem IDs because both are evaluation-only views of the same frozen holdout pool.
