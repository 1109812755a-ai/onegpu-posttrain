# Contributing to OneGPU-PostTrain

Thanks for helping make post-training experiments more reproducible and understandable.

## Contribution principles

1. **Reproducibility before headline numbers.**
2. **One experimental variable at a time when making causal claims.**
3. **No hidden data leakage or silent truncation.**
4. **Do not compare trainer-specific metric scales as if they were universal.**
5. **Do not claim GPU results that were not actually measured.**

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Run:

```bash
onegpu-posttrain doctor
onegpu-posttrain validate-example-data data/examples
```

## Pull requests

A code PR should include:

- a clear problem statement;
- tests for the changed contract;
- updated documentation when behavior changes;
- no generated checkpoints, large datasets, caches or secrets;
- no placeholder benchmark claims.

If a PR changes a frozen v0.1 semantic decision, update the design revision first rather than silently changing implementation behavior.

## Planning future changes with OpenSpec

Use a scoped OpenSpec change before implementing new behavior or changing data, lineage, recipe, evaluation or user-workflow contracts. Give the change a concrete outcome and acceptance criteria, then review its proposal, design, capability specs and tasks before coding. Keep implementation, evidence and documentation aligned with that scope; validate and archive the change when its gates pass.

The existing frozen design and project plans in `docs/` remain authoritative for their stated releases. Reference the affected documents in a new OpenSpec change, and create a versioned design revision before altering frozen experimental semantics. Start with the next concrete development change; do not bulk-import historical work simply to fill `openspec/specs/`. Factual documentation corrections that do not change behavior or contracts can be made directly.

## New training recipes

A new recipe should document:

```text
parent checkpoint
training data / manifest
independent variable
control variables
training budget
seed
resource expectations
evaluation protocol
```

Avoid changing multiple unrelated variables in an experiment that is presented as an ablation.

## Result contributions

A result PR should include, where applicable:

- run ID;
- git commit;
- model + revision;
- dataset + revision / manifest;
- sample IDs or deterministic build recipe;
- resolved training config;
- seed;
- hardware / driver / CUDA / dependency metadata;
- raw evaluation artifact;
- resource metrics;
- limitations / failed runs / OOMs.

OOM and regressions are valid results when the experiment is correctly specified. Do not discard them only because they are inconvenient.

## Data changes

Any new training/evaluation data path should state:

- source and license;
- split unit;
- deduplication strategy;
- contamination checks;
- length gates;
- whether any field is visible to the model vs reward/evaluator only.

## Style

Prefer small, inspectable changes. The project values clear contracts over abstraction for its own sake.
