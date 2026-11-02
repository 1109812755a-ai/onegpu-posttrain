# OneGPU-PostTrain Development Plan

This document is the project-level plan and phase index. Frozen design revision v0.1.1 targets product release v0.1 and remains the authority for its experiment contracts; milestone documents track execution state and evidence.

## Product goal

The north star is the [Product Vision](../design/PRODUCT_VISION.md), with the original rationale preserved in [Project History](PROJECT_HISTORY.md): start with a reproducible single-GPU post-training lab for understanding what changes across CPT, SFT, preference optimization and RL, then grow it into a deployable platform for real research workflows by v1.0. The lab is the validated starting point, not the final product form.

The project owns data contracts, data lineage, stage-specific data quality, unified evaluation, controlled comparisons and bounded interpretation while delegating training to mature frameworks. Data quality is a first-class workstream, not deferred to the later data flywheel: every official dataset/run should carry interpretable quality evidence. A result is useful only when its parent checkpoint, data IDs/revision, quality protocol/evidence, controls, budget, seed, evaluation protocol and resource measurements are recoverable; a preset benchmark score is not the goal.

## Near-term scope guardrails

- Through v0.1, optimize for an open-source, reproducible single-consumer-GPU lab that can be installed and run locally; keep the CLI and portable file artifacts as the core workflow.
- Do not make a database, distributed scheduler, multi-user service, multi-GPU support or deployed web stack a prerequisite for the MVP or v0.1 release.
- Keep early UI read-only, optional and local. Add a constrained local launcher only in v0.2; add a serial queue only when OVAT workflows justify it; evaluate heavier scheduling, persistence and deployment at later gates using real users and workloads.
- The v1.0 platform objective guides compatible interfaces and provenance now, but does not pull its infrastructure cost into early milestones. Every phase must preserve the practical single-GPU path.

## Version labels and change workflow

| Label | Meaning | Current example |
|---|---|---|
| Package version | Version declared by the installable Python package | `0.1.0.dev6` |
| Design revision | Version of a frozen design package; it does not name a software release | `docs/design/v0.1.1/` is design revision v0.1.1 for product release v0.1 |
| Product release | User-facing delivery milestone | v0.1 is the single-GPU baseline; product release v0.1.1 is the planned complete read-only UI |

Future development that changes behavior, data or lineage contracts, evaluation semantics, recipes or user workflows uses a scoped OpenSpec change before implementation. The change references affected frozen design decisions and release gates; changing frozen semantics requires an explicit versioned design revision. Existing design/history documents remain in place, and historical work is not backfilled into OpenSpec merely to populate it. Small factual documentation corrections do not need an OpenSpec change. See [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md) for the contributor workflow.

## Phases

| Phase | Scope | Exit gate | State |
|---|---|---|---|
| 0. Design freeze | Freeze schemas, stage boundaries, evaluation and OVAT contracts. | Design revision v0.1.1 reviewed and frozen for product release v0.1. | Complete |
| 1. Smoke implementation | Implement Base eval, PEFT-CPT, Full-CPT probe, SFT, DPO, original SimPO, GRPO standard/lowmem, reporting and orchestration. | Commands, artifacts and tests exist for each stage. | Complete |
| 2. RTX 4060 execution | Run the smoke workflow on the 8GB reference GPU and capture real checkpoints, evaluations and resource logs. | All training smoke stages have explicit PASS/OOM/FAILED artifacts; Base/M1/M2 evals and preference/GRPO metrics are recorded. | Complete; see run report |
| 3. Smoke hardening + Lineage/Data Quality v1 design (dev7) | Verify preference target scanning, calibrate GRPO completion length, evaluate all M3 variants, clear Ruff findings, and define file-based lineage and stage-specific quality-report contracts. | Dev7 execution and design gates complete with documented source-yield, clipping and lineage limitations. | Complete with limitations |
| 4. Mini pipeline | Assess stage data budgets/quality and promote recipes to a complete training/evaluation matrix with emitted lineage. Add policy-based data exploration to study filtering changes without changing frozen Mini gates. | 5M CPT tokens, 5K eligible SFT examples, 2K verified preference pairs, 2K verifiable GRPO prompts; dataset manifests, quality/build reports, run/model/eval lineage and stage-end evals present. Mini distributions and verifier coverage reviewed. | Reduced-budget exploratory chain completed: 5M CPT → 307-example/1,024-token SFT → four alignment branches and same-protocol smoke-scale evaluation, with verified lineage. Not official Mini: only 4 preference pairs, reviews pending, 2,048-token full SFT OOM, and GRPO was 3 steps with 100% completion clipping. Readiness remains NOT READY. |
| 5. Required OVAT | Test CPT data size, SFT LR, DPO vs. original SimPO and GRPO reward design under frozen controls. | Four reviewed experiment reports; SFT LR representative point has seeds 42/43/44. | Planned |
| 6. v0.1 release | Package the single-GPU baseline and reproducible evidence. | Smoke hardening, Mini, required OVAT, provenance and release docs complete. | Planned |
| 7. Model scale and experiment breadth (v0.2) | Add selected 1B–3B models, rank ablations and broader compute/quality comparisons. | Each new profile shares the data/run/evaluation contract and has resource evidence. | Planned |
| 8. Data quality research and flywheel (v0.3/v0.5) | Build on the v0.1 quality and processing-trace foundation with controlled cleaning/quality ablations, mixture/source/difficulty experiments, synthetic-data filtering, failure mining, data CI and lineage-aware train/evaluate/update loops. | Quality dimensions and selection rules are versioned; matched-control data changes are measurable and traceable to model outcomes; generated data has complete provenance. | Planned |
| 9. Alignment breadth and agent/tool RL (v0.4/v0.6) | Add compatible alignment/RL methods and deterministic tool environments; mature registry and public result catalog. | New methods use comparable protocols; all artifacts retain provenance and the single-GPU path remains supported. | Planned |
| 10. Platform hardening and v1.0 | Turn the proven lab into a deployable project: stable APIs/schemas, job lifecycle and scheduling, registry/data/model/run/evaluation workflows, installation/deployment, observability, security and contributor operations. Select local/single-user or broader deployment modes from validated use cases. | A clean install can run, monitor, inspect, compare, export and reproduce a supported end-to-end workflow; failures/retries/resources and lineage are auditable; deployment and upgrade docs work; local single-GPU learning path remains supported; no mandatory proprietary service. | Planned |

## Current evidence and limits

The smoke execution is recorded in [`../../runs/smoke/RUN_REPORT_zh.md`](../../runs/smoke/RUN_REPORT_zh.md). Dev7 hardening and M2/M3 unified evaluation are complete with limited smoke evidence, not model-quality gains. The current Mini pilot has a deterministic DeepMath 5K SFT cohort (max 2,048 tokens), a completed 5M-token CPT and M1, and an M1 unified evaluation. A 3-step 2,048 SFT probe passed, but the complete SFT OOMed at step 19/624; no M2 exists, so the Mini training chain and M1/M2 comparison are not complete. A refreshed readiness report remains **NOT READY** with 22 blockers due pending/legacy data reviews and the small OpenR1 preference cohort. See [`MINI_SFT_2048_PILOT_zh.md`](../implementation/MINI_SFT_2048_PILOT_zh.md) and [`pilot summary`](../../data/processed/mini/2026-10-19-mini-sft-2048-pilot/pilot_summary_zh.md) for run evidence. UI/API tests and browser smoke pass; the UI surfaces the DeepMath cohort and its SFT probe/OOM lineage without JS errors.

The OpenR1 builder already has target-based stopping and an explicit `max_raw_rows` option. Dev7 must verify these semantics against the real stream and explain the low preference yield using eligibility/rejection counters before changing sampling policy.

Lineage v1 is drafted as a file-based contract in [`../design/LINEAGE_DESIGN.md`](../design/LINEAGE_DESIGN.md); stage-specific data assessment is drafted in [`../design/DATA_QUALITY_DESIGN.md`](../design/DATA_QUALITY_DESIGN.md). The quality design adds linked `quality_report.json` evidence to dataset manifests, but does not retroactively change frozen v0.1.1 eligibility rules. The v0.1 approach is report-and-warn first; candidate thresholds are reviewed against Mini distributions and checker coverage, not guessed to fill quotas. Hashes, unknown provenance and unavailable metrics must be explicit; no database or generic event graph is in scope.

The detailed processing-step → quality-change → model-effect design is in [`../design/DATA_PROCESSING_EFFECT_DESIGN.md`](../design/DATA_PROCESSING_EFFECT_DESIGN.md). Policy identity, offline preview/build, preregistration, conservative result association and the constrained local editor are implemented and CPU-fixture tested in [`data-policy-experiments`](../../openspec/changes/data-policy-experiments/proposal.md). The owned-data walkthrough returns `NOT_EVALUATED`; real controlled training comparisons and sufficient Mini data remain outstanding. The bounded evaluation-driven synthetic-data slice (scenario report, optional provider, candidate validation/policy export, lineage, local UI draft/export and effect summaries) is being delivered under [`targeted-synthetic-data`](../../openspec/changes/targeted-synthetic-data/proposal.md); it is not evidence of real API use or a model gain, nor an autonomous flywheel.

## Frontend roadmap (Data, Launcher and comparison slices delivered; Runs workspace remains planned)

The frozen v0.1 training design remains CLI-first. The Data browser, explicitly enabled policy editor, targeted-data draft/report page, Stage Comparison and a bounded local SFT Launcher preview are implemented. The Launcher is an early developer slice; its benchmark evaluation currently fails on a Windows EvalScope review-cache path and must be repaired before calling the UI flow complete. A general read-only Runs list/detail workspace remains the next v0.1.1 UI slice. The targeted UI has no provider or training launch control. OVAT Studio remains v0.3 and registry views v0.4+. See [`../design/frontend/FRONTEND_ROADMAP.md`](../design/frontend/FRONTEND_ROADMAP.md), [ADR-001](../design/frontend/ADR-001-data-ui-slice.md) and [ADR-002](../design/frontend/ADR-002-policy-editor.md).

## Change control

- Do not weaken correctness, completion, contamination or length gates merely to fill a target. Any changed eligibility rule requires a versioned contract and rebuilt manifests.
- Do not change more than the declared independent variable in an OVAT run.
- Any OOM fallback, sequence-length change or reward change must be visible in resolved config and run provenance.
- Update [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md), the relevant milestone and the run report from saved artifacts; do not copy claims from chat without evidence.

## Milestones

- [v0.1 smoke execution](milestones/v0.1-smoke.md)
- [v0.1.0.dev7 smoke hardening](milestones/v0.1-dev7-hardening.md)
- [v0.1 Mini and release](milestones/v0.1-mini.md)
- [v0.2 and beyond](milestones/v0.2-roadmap.md)

## Design references

- [Overall design v0.1.1](../design/v0.1.1/00-overall-design.md)
- [Training recipe design](../design/v0.1.1/02-training-recipe-design.md)
- [Evaluation design](../design/v0.1.1/03-evaluation-design.md)
- [Experiment and ablation design](../design/v0.1.1/04-experiment-design.md)
- [Frontend product design and roadmap](../design/frontend/FRONTEND_PRODUCT_DESIGN.md)
- [Data and model lineage design](../design/LINEAGE_DESIGN.md)
- [Stage-specific data quality design](../design/DATA_QUALITY_DESIGN.md)
- [System architecture](../design/SYSTEM_ARCHITECTURE.md)
- [Open-source release plan](OPEN_SOURCE_PLAN.md)
- [Quick Start](../guides/QUICKSTART.md)
