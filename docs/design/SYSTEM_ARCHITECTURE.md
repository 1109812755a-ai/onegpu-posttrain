# OneGPU-PostTrain System Architecture

**Document type:** living architecture overview  
**Status:** v0.1 architecture baseline; future-version components are proposals until their release gates are met  
**Authority:** Product intent comes from [Product Vision](PRODUCT_VISION.md); v0.1 experiment contracts remain in the frozen [Design v0.1.1](v0.1.1/00-overall-design.md).

中文版本：[系统整体设计](SYSTEM_ARCHITECTURE_zh.md)。

## 1. Purpose and product boundary

OneGPU-PostTrain is an open-source, reproducible post-training lab for understanding what changes across CPT, SFT, preference optimization and RL on a single consumer GPU. It is not a new trainer. It composes data preparation, mature training/evaluation tools, controlled experiments and inspectable artifacts so an engineer can reproduce and interpret stage-level changes.

The five product workstreams are:

```text
Data Lineage · Data Quality · Training · Evaluation · Controlled Experiments
```

Data quality is not a late-stage add-on: each official data build records its ordered processing steps, rejection reasons and available before/after quality evidence. Dataset lineage connects the resulting data versions to training runs, models and evaluations. A model-effect claim requires a controlled comparison; the processing trace alone only describes data changes. See the [Data Quality Design](DATA_QUALITY_DESIGN.md) and [Lineage Design](LINEAGE_DESIGN.md); the frozen v0.1.1 contracts remain unchanged until an explicit revision.

This document connects the frozen v0.1 contracts, current local artifact workflow, lineage design and staged UI/launcher plans. It does not supersede their detailed specifications.

The detailed source-processing → dataset-quality → model-effect contract is in [Data Processing, Quality Change, and Model Effect Design](DATA_PROCESSING_EFFECT_DESIGN.md); this overview and the release roadmap point to it as the cross-component design.

## 2. Context and system boundary

```text
Project user / contributor
       │
       ├── CLI + recipes ───────────────┐
       │                                ▼
       │                       OneGPU-PostTrain core
       │                    ┌───────────┼───────────┐
       │                    ▼           ▼           ▼
       │                  Data       Training    Evaluation
       │                    │       (ms-swift)  (EvalScope +
       │                    │                    project evaluators)
       │                    └───────────┼───────────┘
       │                                ▼
       │                     Local filesystem artifacts
       │                    manifests / runs / reports
       │                                │
       └── future local read-only UI ───┘
```

### In scope

- Data source declarations, preparation, validation, split/contamination controls and framework adapters;
- stage recipes and a thin execution boundary around external training engines;
- common evaluation and experiment comparison;
- file-based run, data, model and evaluation provenance;
- local CLI and, in later versions, optional local UI/launcher layers.

### Out of scope for the initial release

- implementing Transformer/trainer algorithms already provided by mature frameworks;
- remote multi-tenant service, distributed scheduler, database-backed registry or mandatory hosted experiment tracking;
- automatic hyperparameter search before controlled experiments and evidence are reliable;
- presenting illustrative/placeholder values as results.

## 3. Core domain and experiment graph

Core objects are `ProcessingReport`, `DatasetArtifact`, `QualityReport`, `Recipe`, `CodeRevision`, `Run`, `ModelArtifact`, `EvaluationArtifact`, `DataEffectExperiment` and `DataEffectReport`. Dataset/run/model/evaluation identity rules are specified in [Data and Model Lineage Design](LINEAGE_DESIGN.md); processing and data-effect contracts are in [Data Processing Effect Design](DATA_PROCESSING_EFFECT_DESIGN.md).

```text
External source(s)
   ↓ versioned processing steps (order is builder-specific)
Processing report + canonical problem pool + dataset quality reports
   ↓ problem-level split before stage transforms
Train pools ──────────────── Independent holdout / benchmark data
   ├── CPT corpus                         │
   ├── SFT examples                       │
   ├── Preference pairs                   │
   └── GRPO prompts                       │
          │                               │
Base model → CPT run → M1 → SFT run → M2  │
                                   ┌───────┴────────┐
                                   │ same M2 parent │
                              DPO / SimPO / GRPO runs
                                   │
                          adapter → merge → model artifacts
                                   │
                                   └──── evaluations
                                            ↓
                               controlled data-effect report
```

The DAG is the source of truth for relationships. DPO, SimPO and GRPO are sibling experiments from a common M2 parent unless an explicitly separate sequential-pipeline experiment is declared. Dataset lineage and model lineage join at `run_id`; evaluation links back to the exact model, data/protocol and evaluator version.

## 4. Component responsibilities

| Component | Owns | Does not own |
|---|---|---|
| Data | Source/revision/license references, ordered processing trace, stable problem/record IDs, filtering, split, leakage checks, dataset manifests, stage-specific quality and distribution reports, canonical→framework adapters | Trainer semantics or hidden changes to source data |
| Recipes / experiment definitions | Parent, datasets, stage, declared independent variable, controls, seed, budget, resource expectations and evaluation plan; data-selection studies link policy, matched controls and model/evaluation outputs | Mutable run status or actual results |
| CLI / orchestration | Validate a plan, resolve a recipe, invoke supported commands, capture exit/status/log/resource metadata, connect inputs and outputs | Reimplementation of ms-swift algorithms; in v0.1, arbitrary scheduling |
| Training adapter | Translate canonical recipe/data to pinned ms-swift invocation and capture resolved arguments | Project-level claim that a run improved capability |
| Evaluation | Run common task/domain/preference/RL evaluators; record protocol, denominators, metrics and raw references | Editing training artifacts or comparing incompatible protocols silently |
| Experiment analysis | Validate controls, calculate compatible deltas, summarize limitations/resources and produce reports | Inventing missing values or causal claims beyond the design |
| Artifact/provenance layer | Stable IDs, manifests, processing/quality/effect reports, hashes/status, relative references and schema versions in local files | Database service in the current release |
| Future UI | Read and visualize normalized artifacts, lineage and compatible comparisons | Owning experiment truth or changing source artifacts in v0.1.1 |

## 5. Data and model contracts

The project keeps canonical stage formats understandable and visible: raw-text CPT, `messages` SFT, `prompt/chosen/rejected` preference data, and prompt plus verifier reference for GRPO. For the math baseline, FineMath feeds CPT; OpenR1-derived data feeds SFT and alignment; evaluation uses disjoint holdout and fixed benchmarks.

Critical invariants:

- Deduplicate and split at the problem level before deriving stage-specific formats.
- Preserve `problem_id` separation between training pools and internal evaluation.
- DPO and SimPO consume the same preference artifact when compared as an OVAT.
- GRPO target answers are available to verifier/reward code but not exposed as supervised policy targets.
- Any changed data, recipe, reward or evaluation semantics is versioned and visible in provenance.

Frozen schemas, filtering and lengths live in [Data Design](v0.1.1/01-data-design.md); the additive quality-report proposal lives in [Data Quality Design](DATA_QUALITY_DESIGN.md); trainer parameters live in [Training Recipe Design](v0.1.1/02-training-recipe-design.md).

## 6. Execution and artifact lifecycle

### v0.1 implementation shape

```text
recipe + data manifests
        ↓
CLI validation / plan
        ↓
external process (ms-swift / evaluator)
        ↓
run.json + environment + logs + trainer outputs
        ↓
lineage.json + eval/comparison/report artifacts
```

- CLI and local filesystem are the supported user interface and persistence boundary.
- Each attempt—including OOM/failure—gets a distinct run identity and retained status/evidence.
- `run.json` remains the operational execution record; `lineage.json` records typed relationships without duplicating mutable runtime state.
- Dataset manifests and model/evaluation references connect the graph. Paths locate artifacts; stable IDs and digests establish identity.
- Missing or legacy fields stay explicitly unknown/unavailable. Do not backfill by guessing or rewrite historical artifacts silently.
- Run artifacts do not include raw third-party datasets, secrets or model weights by default.

## 7. Reproducibility and interpretation

An interpretable result should link, where applicable, to:

- model and tokenizer source/revision plus parent/checkpoint identity;
- dataset source/revision/license, manifest, split and sample/problem IDs;
- recipe and resolved config digest, code revision/dirty state, seed and software environment;
- evaluation dataset/protocol/evaluator version, metric denominator and raw report;
- hardware, duration, throughput and peak VRAM, with measurement method;
- failed/OOM runs, missing fields and validity limitations.

OVAT comparisons require compatible parent/data/evaluation and a declared independent variable. Metrics from different objectives are not assumed to share a scale. Smoke establishes feasibility/instrumentation, not model-quality superiority.

## 8. Versioned interaction architecture

| Version | User interaction | Architecture boundary |
|---|---|---|
| v0.1 | CLI + local artifacts | No web service, database or scheduler; recipe/CLI remain execution truth. |
| v0.1.1 | Optional local read-only UI | FastAPI adapter reads allowlisted roots; UI renders artifacts and lineage; no write/process endpoints. |
| v0.2 | Optional constrained local launcher | UI requests a validated recipe through the supported CLI boundary; only an explicit recipe/parameter allowlist; no arbitrary shell. |
| v0.3 | Experiment Studio | Versioned OVAT definitions and multi-run/report views; local serial queue only if batch-run demand demonstrates it is needed. |
| v0.4+ | Registry views | Add only after IDs/provenance/lifecycle contracts are stable; persistence choice requires a separate decision. |

For UI-specific interaction and safety rules, see the [Frontend Product Design](frontend/FRONTEND_PRODUCT_DESIGN.md) and [Frontend Roadmap](frontend/FRONTEND_ROADMAP.md). The UI is subordinate to the experiment system, not its source of truth.

## 9. Open-source and security constraints

- Keep the core path usable from the public source tree without a proprietary hosted service.
- Pin/document dependencies, recipes, setup and hardware expectations; make optional integrations visibly optional.
- Track third-party code, model and data licenses separately. Repository license does not relicense upstream assets.
- Avoid committing large datasets/checkpoints, credentials, personal/restricted records or local machine secrets.
- Future filesystem UI access must use configured roots, path containment, artifact allowlists, bounded reads and sanitized previews.
- Future process launch must validate recipe IDs/parameters, avoid arbitrary command strings, retain logs/status and require explicit local operation.

Versioned release requirements are in the [Open-Source Release Plan](../project/OPEN_SOURCE_PLAN.md).

## 10. Source-of-truth map

- Product intent and feature filter: [Product Vision](PRODUCT_VISION.md)
- Original rationale and decision evolution: [Project History](../project/PROJECT_HISTORY.md)
- Frozen v0.1 experiment contracts: [Design v0.1.1](v0.1.1/00-overall-design.md)
- Data and model/run/evaluation relationships: [Lineage Design](LINEAGE_DESIGN.md)
- Stage-specific data assessment: [Data Quality Design](DATA_QUALITY_DESIGN.md)
- Data processing and model-effect flow: [Data Processing Effect Design](DATA_PROCESSING_EFFECT_DESIGN.md)
- UI behavior: [Frontend design](frontend/FRONTEND_PRODUCT_DESIGN.md)
- Current phase and evidence: [Development Plan](../project/DEVELOPMENT_PLAN.md) and [Implementation Status](../project/IMPLEMENTATION_STATUS.md)
- Version sequencing: [Roadmap](../../ROADMAP.md) and [Open-Source Release Plan](../project/OPEN_SOURCE_PLAN.md)
