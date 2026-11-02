# Base Evaluation

Base evaluation has two independent tracks that are written into the same run directory.

## 1. Official benchmarks

EvalScope runs:

- `gsm8k`
- `math_500`

The model revision, seed and deterministic generation configuration are taken from repository configs. A smoke run uses only five examples per benchmark; the Mini pipeline removes that small smoke limit.

Create a plan without model inference:

```bash
onegpu-posttrain plan-base-eval \
  --config recipes/smoke/00-base-eval.yaml \
  --output runs/base-smoke-s42/evaluation \
  --internal-holdout artifacts/openr1/canonical/internal_holdout.jsonl
```

Run:

```bash
onegpu-posttrain run-base-eval \
  --config recipes/smoke/00-base-eval.yaml \
  --output runs/base-smoke-s42/evaluation \
  --internal-holdout artifacts/openr1/canonical/internal_holdout.jsonl \
  --limit 5 \
  --internal-limit 20
```

## 2. Internal math holdout

The project evaluator stores one JSONL row per prediction with:

- raw output
- extracted final answer
- parse success
- correctness

It first uses deterministic answer normalization and can optionally use `math_verify` when available.

## Smoke vs official results

Smoke scores are only pipeline validation. They must **not** be copied to the project README as model-quality claims. Official results require the frozen Mini evaluation protocol and complete artifact lineage.
