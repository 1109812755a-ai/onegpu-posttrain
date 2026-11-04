# Roadmap

OneGPU-PostTrain starts as a reproducible post-training experiment lab for understanding what changes across CPT, SFT, preference optimization and RL on a single GPU, then grows into a deployable platform for real research workflows. The lab is the validated entry point; a practical open-source platform is the v1.0 goal. See the [Product Vision](docs/design/PRODUCT_VISION.md), [System Architecture](docs/design/SYSTEM_ARCHITECTURE.md) and versioned [Open-Source Release Plan](docs/project/OPEN_SOURCE_PLAN.md). Scope expands only after the previous release has reproducible artifacts.

The connected design for raw-data processing traces, quality evidence and downstream model-effect experiments is [Data Processing, Quality Change, and Model Effect Design](docs/design/DATA_PROCESSING_EFFECT_DESIGN.md).

## v0.1 — Single-GPU Baseline

Release gate:

- real RTX 4060 8GB smoke artifacts;
- Base → PEFT-CPT → SFT → DPO / SimPO / GRPO pipeline validated;
- Full-CPT feasibility result captured;
- GRPO-standard / lowmem feasibility captured;
- Mini dataset recipes promoted from smoke;
- at least four verified OVAT experiments;
- public result provenance with configs, manifests and environment metadata;
- file-based data → run → model → evaluation lineage for official Mini/OVAT results.
- versioned, stage-specific quality reports for official data artifacts, with denominators, checker coverage and limitations (no universal quality score; thresholds reviewed using Mini evidence).
- ordered data-build processing traces with input/output units, rejection reasons and comparable quality snapshots where measured; link datasets to consuming runs and evaluations.

Primary experiments:

- CPT dataset size;
- SFT learning rate;
- DPO vs SimPO-original;
- GRPO reward design;
- optional DPO beta and GRPO generation count.

## Product release v0.1.1 — Local Experiment UI

Build a local interface over existing run/data artifacts. Artifact browsing stays read-only; the data policy editor can be explicitly enabled for bounded local previews and new policy drafts. The UI does not start training or own run lifecycle.

- Overview of stage lineage, run outcomes and available hardware/resource metrics;
- Runs list and run detail with resolved config, lineage, training curves, evaluation, logs and artifact links;
- Stage comparison, data processing/quality/lineage/distribution/leakage views and GRPO/preference analysis; show model impact only when linked controlled comparisons exist.
- artifact parsing that tolerates missing/older fields and clearly labels unavailable metrics;
- no database, scheduler, authentication service or training control in this release.

The UI release gate is correct rendering of representative real artifacts, safe local file access, documented startup/shutdown, and validated opt-in policy editing. Model effects remain unassessed until compatible registered training/evaluation evidence exists. See the [frontend roadmap](docs/design/frontend/FRONTEND_ROADMAP.md) and [policy experiment guide](docs/guides/POLICY_EXPERIMENT_UI_zh.md).

## Workspace Experiment Assistant — first increment implemented

The first product-workflow assistant increment is implemented on the current development branch, distinct from v0.6 Agent / Tool RL, which studies post-training models for tool use. The read-only chat assistant searches workspace runs, evaluations, datasets and model lineage to help locate recorded experiment issues and understand evaluation changes. Answers cite recorded evidence, separate facts from interpretation, and surface missing or incompatible evidence. Bailian context sharing requires consent for the current session; the assistant does not execute commands or start training.

Future work may accept an experiment goal, produce a reviewable plan and invoke platform-owned generation, training or evaluation workflows after user approval. Skills, MCP tools, web research and a broader agent-guided improvement loop remain future extensions, not delivered capabilities.

The long-term direction is an evidence-driven recursive improvement loop: evaluate → diagnose weak scenarios → prepare and review targeted data → train a versioned model → re-evaluate on protected holdouts. Existing lineage, diagnostics and explicit data-generation workflows are building blocks; the full agent-guided loop is not yet implemented.

## v0.2 — Model Scale and Local Experiment Launcher

- add selected 1B–3B models;
- LoRA-rank ablation;
- more compute/quality trade-off reporting;
- compare 8GB and larger single-GPU profiles;
- improve experiment-report automation;
- add an opt-in local launcher for approved recipes, limited parameter overrides, process status and live logs. Keep CLI as the source of execution semantics; the UI must call the supported CLI/API boundary rather than reimplement training.

## v0.3 — Data-Centric Post-Training and Experiment Studio

- pipeline-effect reports and controlled data-quality ablations using versioned quality dimensions and matched controls (not a universal composite score);
- data-size / difficulty / source-distribution ablations;
- data mixture experiments;
- synthetic-data generation and filtering;
- stronger contamination analysis;
- define OVAT experiments, compare multiple runs/seeds, show curves/tables/deltas and export reports.

## v0.4 — Alignment/RL Breadth and Registry Foundations

Candidate additions only when each can share the same evaluation/experiment contract:

- KTO / ORPO / CPO-family experiments;
- additional GRPO-family variants;
- reward design ablations;
- optional process-level verification experiments;
- begin registry UI for datasets, models and experiments, backed by explicit provenance and stable schemas.

## v0.5 — Data Flywheel / Model Flywheel

- failure mining;
- new-data proposal generation;
- data CI gates;
- iterative train → evaluate → gap analysis → data update loops;
- stronger lineage across data and model versions.

## v0.6 — Agent / Tool RL

- tool-call environments;
- deterministic tool/task rewards;
- trajectory datasets;
- small-model agent post-training experiments.

## v1.0 — Reproducible Post-Training Research Platform

Platform-level release goal:

- deployable end-to-end Data / Training / Evaluation / Experiment workflows with stable CLI/API/schema contracts;
- dataset, model, run and evaluation registries with import/export and lineage;
- job lifecycle, bounded task scheduling, resource visibility, failure/retry behavior and operational diagnostics;
- installation, deployment, upgrade, backup/recovery, security and contributor operations suitable for real use;
- reproducible public result catalog and richer resource/effectiveness comparisons;
- optional larger or multi-GPU profiles without removing the single-GPU learning path.

Deployment mode and scheduler complexity will follow validated users and workloads; v1.0 must be usable as a real deployed project without requiring a proprietary service.

## Non-goal

The roadmap does not imply that OneGPU-PostTrain should become another general-purpose distributed training framework. Training engines remain external dependencies unless a missing capability directly blocks the project's experiment methodology.
