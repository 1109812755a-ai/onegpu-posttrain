# Frontend Information Architecture

## Product goal

OneGPU-PostTrain is a reproducible post-training lab for a single consumer GPU. The UI is organized around the work users need to do: prepare and understand data, train a selected stage, inspect the resulting run and model, then evaluate and plan the next experiment. The first release remains local-first, bounded to one active GPU task, and based on approved profiles.

## Primary navigation

The workspace has four primary areas. Secondary views contain the concrete operations for each area.

| Area | User task | Secondary views | Main objects and handoff |
|---|---|---|---|
| Data | Acquire, inspect, and register training data | Datasets; Processing & Quality; Rule Comparison | Sources, samples, quality reports, lineage, registered dataset IDs; hand a registered dataset to training |
| Training & Evaluation | Select a stage and dataset, then explicitly launch training | Training & Evaluation | Stage setup and preflight for CPT, SFT, DPO, SimPO, and GRPO; open the resulting Run after launch |
| Experiment Runs | Observe execution and trace real artifacts | Run History | Training/generation/evaluation status, logs, metrics, models, and data lineage; explicitly evaluate a verified model |
| Analysis & Improvement | Understand outcomes and choose the next experiment | Stage Comparison; Weak Scenarios | Comparable evaluation matrices, executed policy-effect reports, weak scenarios; targeted generation carries source evaluation/model/run context into Data |

Overview and alignment-analysis placeholders are not primary navigation destinations until they contain usable workflows. Unsupported stage profiles show their actual blockers in context.

## End-to-end object flow

```text
Data source / local import / model generation
                    ↓
          Process, assess, register
                    ↓ dataset_id
       Stage training and explicit preflight
                    ↓ run_id
              Run and model
                    ↓ separate evaluation run
           Analysis and next improvement
                    └──── targeted goal returns to Data
```

Training stage and data source are independent dimensions. Data owns generation, review, quality checks, registration, and dataset lineage. Training consumes an explicit registered dataset ID. Runs is the source of truth for execution state and model artifacts. Analysis reads actual evaluation evidence. Parent-model links are shown through model lineage; upstream stages are never launched implicitly.

## Page responsibilities

- **Datasets:** Browse registered data, filter by stage/source, preview samples, and open one dataset detail with samples, source/license, quality, processing evidence, dataset lineage, and related Runs.
- **Processing & Quality:** Inspect processing reports, quality, and distributions. Rule editing and preview also belong here. Format validity, semantic quality, license state, and training readiness remain separate facts.
- **Rule Comparison:** Configure rules and preview filtering differences on the same source. Executed model-effect reports belong in Analysis & Improvement and link through the same policy/experiment identity.
- **Training & Evaluation:** Show stage, approved profile, compatible dataset ID, parent model, and blockers. Each training run requires explicit preflight and confirmation. Only implemented and verifiable profiles are launchable.
- **Run History:** A list and shared Run detail show status, logs, config summary, training data/parent, outputs, evaluation, and lineage. Evaluation is a separate execution and does not replace the training terminal state.
- **Stage Comparison:** Calculate deltas only when protocol, data version, denominator, and coverage match. Show other metrics separately with a clear reason they are not comparable.
- **Weak Scenarios:** Show the source evaluation, model, Run, and target. “Generate data” transfers those identities into the Data flow; generation, review, and registration use the shared Data workflow.

Ordinary and targeted generation share the applicable execution, quality, review, and registration operations while retaining the distinct inputs each mode requires. Ordinary generation does not ask users to enter an internal plan ID. Active task identity and state are independent of the current form draft.

## URLs and browser history

Existing `page` values remain valid and resolve to the corresponding secondary view:

| Existing URL | New area |
|---|---|
| `page=workbench` or an unfiltered root URL | Data / Datasets |
| No `page` with legacy `scope/build/stage/tab` report context, or `page=data` | Data / Processing & Quality, restoring legacy filters |
| `page=policies` | Data / Rule Comparison |
| `page=launcher` | Training & Evaluation |
| `page=runs&run=...` | Experiment Runs / requested Run |
| `page=comparison&matrix=...` | Analysis & Improvement / requested comparison |
| `page=targeted` | Analysis & Improvement / Weak Scenarios |

Each area writes only its own URL state: Data owns data filters, Training receives the applicable stage and dataset ID, Runs owns run identity/filters, and Analysis owns a matrix or explicit improvement context. Moving between areas does not carry unrelated build, tab, or search parameters. Browser back/forward restores the target view and object identity.

## Object identity and traceability

- **Dataset:** `dataset_id`, stage, source/revision, content digest, quality/license evidence, and parent-dataset lineage.
- **Training Run:** `run_id`, stage/profile, actual dataset ID/hash, parent model ID/hash, config digest, status, and output references.
- **Model:** model ID, producing Run, parent-model link, adapter/merged type, and verifiable digest.
- **Evaluation Run:** separate execution identity, target model, benchmark/data, protocol, denominator, metric states, and source training Run.
- **Data-improvement experiment:** scenario, source evaluation/model/Run, policy or generation-plan version, and resulting dataset ID. Evaluation holdout samples are not used as training-generation input.

The UI defaults to readable names, stages, counts, and states. Exact IDs, hashes, raw configs, and paths are available on demand and are not required for cross-area navigation.

## State and interaction principles

- Each area handles loading, empty, error, partial, and unsupported states independently. A failed catalog endpoint must not hide Runs that are still available.
- Metrics include their meaning, unit, denominator, protocol, and state. Not measured, unavailable, failed, and zero are distinct.
- Editing a draft invalidates its preflight. A started execution/run survives draft reset, view changes, and reload, and is never restarted automatically.
- Primary actions are visible and ordered by the current workflow state. Browsing or navigation never starts generation, training, or evaluation implicitly.
- Support desktop, 390 CSS pixel narrow screens, and keyboard use. Narrow layouts use horizontally scrollable secondary tabs; long logs scroll within their own area.

## Current navigation implementation scope

The four primary areas and legacy URL mapping land first, while existing Data/rule, Training/Run, and Analysis/comparison views continue to provide their current behavior. A unified dataset detail, shared generation/registration UI, cross-area task recovery, and general stage-training entry are implemented in later OpenSpec tasks; navigation consolidation alone does not imply those capabilities are finished.
