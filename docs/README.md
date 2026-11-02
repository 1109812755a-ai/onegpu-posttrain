# OneGPU-PostTrain Documentation

This directory is the canonical documentation entry point for the project.

> Reading order: **Quick Start → Design → Data/Adapter/Evaluation → Training Stages → Experiments/Results → Project Status**.


## 0. Start Here

- [`design/PRODUCT_VISION.md`](design/PRODUCT_VISION.md) — the product north star and feature decision filter
- [`design/LINEAGE_DESIGN.md`](design/LINEAGE_DESIGN.md) — file-based data/model/run/evaluation lineage contract (v1 design)
- [`design/DATA_QUALITY_DESIGN.md`](design/DATA_QUALITY_DESIGN.md) — processing trace, before/after quality evidence, stage metrics and controlled data→model effect studies
- [`design/DATA_PROCESSING_EFFECT_DESIGN.md`](design/DATA_PROCESSING_EFFECT_DESIGN.md) — detailed artifact contracts, stage metrics, controlled comparisons, UI workflow and rollout acceptance
- 中文：[产品愿景](design/PRODUCT_VISION_zh.md) · [系统整体设计](design/SYSTEM_ARCHITECTURE_zh.md)
- 中文项目路线：[整体开发计划](project/DEVELOPMENT_PLAN_zh.md) · [开源发布计划](project/OPEN_SOURCE_PLAN_zh.md)
- [`guides/QUICKSTART.md`](guides/QUICKSTART.md) — CPU validation, smoke planning, real CUDA execution and artifact interpretation
- [`../README.md`](../README.md) — English project overview
- [`../README_zh.md`](../README_zh.md) — 中文项目首页

## 1. Design and Contracts

Living architecture/contract documents:

- [`design/PRODUCT_VISION.md`](design/PRODUCT_VISION.md) — project north star
- [`design/SYSTEM_ARCHITECTURE.md`](design/SYSTEM_ARCHITECTURE.md) — how product modules and release-specific interfaces fit together
- [`design/LINEAGE_DESIGN.md`](design/LINEAGE_DESIGN.md) — file-based data/model/run/evaluation provenance contract
- [`design/DATA_QUALITY_DESIGN.md`](design/DATA_QUALITY_DESIGN.md) — processing trace, stage-specific quality dimensions, before/after reporting and provisional gate policy
- [`design/DATA_PROCESSING_EFFECT_DESIGN.md`](design/DATA_PROCESSING_EFFECT_DESIGN.md) — detailed data processing → quality → model-effect design

中文版本：[产品愿景](design/PRODUCT_VISION_zh.md) · [系统整体设计](design/SYSTEM_ARCHITECTURE_zh.md) · [数据质量设计](design/DATA_QUALITY_DESIGN_zh.md) · [数据处理与模型效果详细设计](design/DATA_PROCESSING_EFFECT_DESIGN_zh.md)

The v0.1 implementation is governed by frozen **design revision v0.1.1**. This is a design-package version, distinct from the planned product release v0.1.1 local Experiment UI:

- [`design/v0.1.1/README.md`](design/v0.1.1/README.md) — design bundle index
- [`design/v0.1.1/00-overall-design.md`](design/v0.1.1/00-overall-design.md) — overall architecture and scope
- [`design/v0.1.1/01-data-design.md`](design/v0.1.1/01-data-design.md) — data contracts, split, decontamination, length gates
- [`design/v0.1.1/02-training-recipe-design.md`](design/v0.1.1/02-training-recipe-design.md) — CPT/SFT/DPO/SimPO/GRPO recipes and 8 GB constraints
- [`design/v0.1.1/03-evaluation-design.md`](design/v0.1.1/03-evaluation-design.md) — unified evaluation protocol
- [`design/v0.1.1/04-experiment-design.md`](design/v0.1.1/04-experiment-design.md) — OVAT and controlled experiment protocol
- [`design/v0.1.1/05-design-review.md`](design/v0.1.1/05-design-review.md) — design review findings and closure
- [`design/v0.1.1/DESIGN_FREEZE.md`](design/v0.1.1/DESIGN_FREEZE.md) — frozen decisions
- [`design/v0.1.1/CHANGELOG-v0.1.1.md`](design/v0.1.1/CHANGELOG-v0.1.1.md) — design revision log

Frozen design documents are historical decision records. Implementation docs may evolve, but changes that alter frozen semantics should first be reflected in a new design revision.

## 1.1 Frontend Design (Data browser and policy editor implemented; full UI planned)

- [`design/frontend/FRONTEND_PRODUCT_DESIGN.md`](design/frontend/FRONTEND_PRODUCT_DESIGN.md) — product intent, read-only artifact boundary, opt-in policy editor, safety and success criteria
- [`design/frontend/INFORMATION_ARCHITECTURE.md`](design/frontend/INFORMATION_ARCHITECTURE.md) — navigation, core entities and artifact sources
- [`design/frontend/PAGE_SPEC.md`](design/frontend/PAGE_SPEC.md) — requirements for the six initial pages
- [`design/frontend/FRONTEND_ROADMAP.md`](design/frontend/FRONTEND_ROADMAP.md) — v0.1.1 UI through v0.4+ registry milestones

中文版本：[前端产品设计](design/frontend/FRONTEND_PRODUCT_DESIGN_zh.md) · [信息架构](design/frontend/INFORMATION_ARCHITECTURE_zh.md) · [页面规格](design/frontend/PAGE_SPEC_zh.md) · [前端路线图](design/frontend/FRONTEND_ROADMAP_zh.md) · [策略实验操作指南](guides/POLICY_EXPERIMENT_UI_zh.md) · [定向补数规划指南](guides/TARGETED_SYNTHETIC_DATA_zh.md) · [CPT 生成—审核—登记闭环](guides/CPT_GENERATION_WORKFLOW_zh.md) · [SFT 生成—审核—训练闭环](guides/SFT_GENERATION_WORKFLOW_zh.md)

## 2. Data, Adapter and Evaluation

### Data

- [`implementation/data/SMOKE_DATA_BUILD.md`](implementation/data/SMOKE_DATA_BUILD.md) — FineMath/OpenR1 smoke build, gates, split and reports
- [`implementation/data/DATA_EVIDENCE.md`](implementation/data/DATA_EVIDENCE.md) — implemented processing trace, static quality reports, manifest references and current limitations（中文）
- [`implementation/data/QUALITY_REVIEW_zh.md`](implementation/data/QUALITY_REVIEW_zh.md) — offline stage-quality review profiles, usage classes, verifier coverage and CLI workflow
- [`implementation/data/MINI_BUILD02_REVIEW_zh.md`](implementation/data/MINI_BUILD02_REVIEW_zh.md) — retained Mini build-02 review outcomes and integrity/test record
- [`guides/POLICY_EXPERIMENT_UI_zh.md`](guides/POLICY_EXPERIMENT_UI_zh.md) — local SFT/Preference policy preview, experiment plan and report workflow
- [`guides/SFT_GENERATION_WORKFLOW_zh.md`](guides/SFT_GENERATION_WORKFLOW_zh.md) — register SFT generation inputs, run locally with explicit cost confirmation, review/register dataset and continue to Mini SFT
- [`guides/MODEL_PLAYGROUND_zh.md`](guides/MODEL_PLAYGROUND_zh.md) — eligible local model chat, serial same-prompt comparison, GPU exclusivity, privacy, and qualitative-result limits
- [`design/frontend/ADR-002-policy-editor.md`](design/frontend/ADR-002-policy-editor.md) — opt-in editor boundary, safety and UX decisions
- [`design/DATA_QUALITY_DESIGN.md`](design/DATA_QUALITY_DESIGN.md) — proposed per-stage data quality contract
- [`../data/README.md`](../data/README.md) — data directory, generated artifacts and local usage

### Framework Adapter

- [`implementation/adapters/MS_SWIFT_ADAPTER.md`](implementation/adapters/MS_SWIFT_ADAPTER.md) — Canonical Data → ms-swift native format

### Evaluation

- [`implementation/evaluation/BASE_EVALUATION.md`](implementation/evaluation/BASE_EVALUATION.md) — EvalScope + internal holdout + domain PPL evaluation contract

## 3. Training Stage Implementation

### CPT / SFT

- [`implementation/training/PEFT_CPT_SMOKE.md`](implementation/training/PEFT_CPT_SMOKE.md) — official 8 GB PEFT-CPT smoke path
- [`implementation/training/FULL_CPT_PROBE.md`](implementation/training/FULL_CPT_PROBE.md) — diagnostic Full-CPT feasibility probe
- [`implementation/training/SFT_SMOKE.md`](implementation/training/SFT_SMOKE.md) — M1 → SFT → merged M2

### Preference Alignment

- [`implementation/alignment/ALIGNMENT_SMOKE.md`](implementation/alignment/ALIGNMENT_SMOKE.md) — M2 → DPO / SimPO-original
- [`implementation/alignment/PREFERENCE_EVALUATION.md`](implementation/alignment/PREFERENCE_EVALUATION.md) — held-out common policy-pair evaluator

### Reinforcement Learning

- [`implementation/rl/GRPO_SMOKE.md`](implementation/rl/GRPO_SMOKE.md) — GRPO-standard / GRPO-lowmem
- [`implementation/rl/GRPO_REWARDS.md`](implementation/rl/GRPO_REWARDS.md) — accuracy + boxed-format reward contract

## 4. Project Status and History

- [`project/PROJECT_HISTORY.md`](project/PROJECT_HISTORY.md) — original product intent and decision evolution reconstructed from the supplied project history
- [`project/OPEN_SOURCE_PLAN.md`](project/OPEN_SOURCE_PLAN.md) — open-source user promises, compliance and release gates by version
- [`project/IMPLEMENTATION_STATUS.md`](project/IMPLEMENTATION_STATUS.md) — current implementation status and real-GPU execution checklist
- [`project/CHANGELOG-implementation.md`](project/CHANGELOG-implementation.md) — implementation history from dev1 through dev6
- [`project/DEVELOPMENT_PLAN.md`](project/DEVELOPMENT_PLAN.md) — project-wide phases, version labels, OpenSpec change workflow and release gates

中文版本：[分版本开源计划](project/OPEN_SOURCE_PLAN_zh.md)。

## 5. Pipeline Map

```text
Design revision v0.1.1 (frozen for product release v0.1)
        │
        ▼
Data Build
FineMath + OpenR1
        │
        ▼
Canonical Data
        │
        ▼
ms-swift Adapter
        │
        ▼
Base Evaluation
        │
        ▼
PEFT-CPT ── Full-CPT Feasibility Probe
        │
        ▼
      M1-CPT
        │
        ▼
       SFT
        │
        ▼
      M2-SFT
        │
        ├── DPO ───────────────→ M3-DPO
        ├── SimPO-original ────→ M3-SimPO
        ├── GRPO-standard ─────→ M3-GRPO-standard
        └── GRPO-lowmem ───────→ M3-GRPO-lowmem
        │
        ▼
Unified Evaluation + Controlled Comparison
```

## 6. Experiments and Results

- [`experiments/README.md`](experiments/README.md) — report contract for Mini/standard OVAT experiments
- [`results/README.md`](results/README.md) — publication gate for verified real-hardware results

No placeholder benchmark values are allowed in either section.

## 7. Project and Contribution Docs

- [`../ROADMAP.md`](../ROADMAP.md) — release progression and gates
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — contribution and reproducibility requirements
- [`../SECURITY.md`](../SECURITY.md) — vulnerability and operational safety guidance
