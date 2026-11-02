# Verified Results

This directory is for results backed by real run artifacts.

## Publication rule

A result is eligible for this directory only when it can be traced to:

- model revision;
- dataset revision / manifest;
- parent checkpoint lineage;
- resolved configuration;
- seed;
- evaluation protocol;
- hardware and dependency metadata;
- raw metric artifacts.

## Current status

No RTX 4060 result is published yet.

The current repository contains executable smoke contracts and CPU-side validation, but the reference GPU execution still needs to be performed. This is intentional: placeholder or fabricated benchmark values are not allowed.

## Planned v0.1 result groups

```text
smoke/
  feasibility-and-instrumentation.md

mini/
  stage-comparison.md
  cpt-dataset-size.md
  sft-learning-rate.md
  dpo-vs-simpo.md
  grpo-reward.md
```
