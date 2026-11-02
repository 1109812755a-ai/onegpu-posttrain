# Open-Source Release Plan

**Status:** project-wide release plan; version scopes are planned until their release gates are met.  
**License baseline:** repository code declares Apache-2.0; third-party data, models and dependencies retain their own terms.

中文版本：[开源发布计划](OPEN_SOURCE_PLAN_zh.md)。

## 1. Purpose

Open source is a product requirement, not a final publishing step. Each release should let an external user understand what is being run, reproduce the supported path locally, inspect the evidence, and contribute without depending on private infrastructure.

This plan covers the open-source user experience and release obligations by version. It complements the technical [Roadmap](../../ROADMAP.md), [System Architecture](../design/SYSTEM_ARCHITECTURE.md) and [Product Vision](../design/PRODUCT_VISION.md).

The cross-release contract for ordered data processing, quality evidence and controlled model-effect studies is [Data Processing, Quality Change, and Model Effect Design](../design/DATA_PROCESSING_EFFECT_DESIGN.md).

## 2. Cross-version release rules

### License and third-party assets

- The repository currently declares Apache-2.0. Keep the root `LICENSE`, `pyproject.toml`, package metadata and release announcement consistent.
- Repository licensing applies to project-owned code/docs only. Every external code, dataset, model/tokenizer, benchmark and generated artifact needs a recorded source and applicable license/terms; do not imply that Apache-2.0 covers them.
- Do not commit full source datasets or model weights by default. Ship small project-authored examples only when their provenance and redistribution rights are clear; document fetch/build steps for external assets.
- Maintain a third-party asset/dependency inventory and provide attribution/NOTICE when required by the applicable terms. Resolve unknown or incompatible terms before making a public release claim.

### Reproducibility and evidence

- Version recipes, schemas and evaluation protocols; record exact source revisions where available, resolved configuration, seed, environment, hardware and lineage.
- Keep failed/OOM runs and limitations visible. Never use illustrative metrics as results or omit a denominator that changes interpretation.
- Publish reports only when the underlying artifacts are available and internally consistent; label smoke, Mini and OVAT evidence separately.
- Provide a CPU-safe contract/validation path where feasible and state clearly which steps require CUDA, large downloads or external credentials.

### Security and contributor experience

- Never publish secrets, private local paths when avoidable, restricted records, or unreviewed logs/config dumps.
- Keep `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` and `CITATION.cff` accurate. Document supported Python/dependency versions and expected hardware.
- Changes to frozen experiment semantics require an explicit design revision; data/model provenance and leakage checks are required for new recipes.
- Keep optional W&B or other hosted tracking optional; the core run must remain inspectable from local artifacts.

## 3. Version-by-version open-source roadmap

| Release | Open-source user promise | Required public artifacts / docs | Release gate |
|---|---|---|---|
| **Current development: Mini readiness / v0.1 OVAT preparation** (package remains `0.1.0.dev6`; dev7 complete) | Contributors can inspect RTX 4060 smoke, Mini data readiness and the policy-comparison tooling; no claim that Mini training or OVAT is complete. | Status, smoke/Mini reports, Lineage v1, policy/quality/experiment contracts and operator guide; reconcile README/changelog/status claims. | Mini readiness is `NOT READY`; owned-fixture example is labeled `fixture`/`NOT_EVALUATED`; real run status is consistent; no secrets/restricted assets; dev7 limitations recorded. |
| **v0.1** | Reproduce the supported single-GPU Qwen3-0.6B math post-training baseline and understand the stage/data-quality/evaluation contracts. | CLI/recipes, data examples and builders, manifests/lineage, ordered data-processing reports, stage-specific quality reports, environment/doctor guidance, unified evaluation, required OVAT reports, result provenance, bilingual project overview, install/license/contribution/security/citation docs. | RTX 4060 smoke hardening; Mini budgets; four required OVATs and seed requirements; Lineage v1 emitted/validated; official data builds expose processing steps/count units/rejection reasons and link quality reports with protocols, denominators, checker coverage and limitations; candidate thresholds reviewed against Mini evidence; license and asset inventory reviewed; all published metrics trace to artifacts. |
| **v0.1.1** | Inspect runs, data quality, lineage and comparisons locally. Artifact browsing is read-only; policy preview/save is a bounded opt-in local exception. | Optional UI install/start and policy guide, demo data if rights permit, API/schema docs, safe artifact-viewing policy and known limitations. | UI tested against complete/partial real artifacts; quality states, missing values and limitations render correctly; path containment, secret/weight exclusion, bounded reads and malformed-file handling verified; policy editor remains loopback-only and opt-in; CLI remains independently usable; no DB/remote service required. |
| **v0.2** | Try selected 1B–3B profiles and launch an explicitly allowlisted local recipe from the UI/CLI boundary. | Compatibility matrix by model/GPU, recipe resource expectations, parameter allowlist, launch/logging docs and failure troubleshooting. | Each profile has real resource evidence; process launch prevents arbitrary command execution; config is validated and preserved; local process lifecycle, interruption, OOM and logs are tested; launcher is opt-in. |
| **v0.3** | Define/review OVAT experiment sets, compare runs/seeds and export provenance-complete reports alongside controlled cleaning/data-quality/source/mixture studies. | Versioned experiment-spec format, comparison compatibility rules, report export, pipeline-effect and matched-budget quality-ablation docs/examples. | Multi-run outputs are traceable; incompatible protocols are not silently compared; cleaning studies distinguish full-pipeline effects from matched-budget policy effects, link dataset-quality/coverage changes to model evaluations, and expose controls/limitations; dataset/license provenance is complete. Any local queue remains serial and bounded unless separately justified. |
| **v0.4** | Explore additional alignment/RL objectives and read stable dataset/model/experiment registry views. | Method cards/recipes, compatibility table, registry schema/API docs, migration and export formats. | New objectives share evaluation/provenance contracts; registry records have stable IDs and exportable files; persistence/deployment is justified and reviewed separately; no lock-in to a hosted service. |
| **v0.5** | Reproduce data-quality, synthetic-data and data/model flywheel experiments. | Synthetic-data provenance/labeling, source/license policy, filtering reports, failure-mining method, data CI and lineage examples. | Generated data is distinguishable from human/source data, linked to generator/model/prompt/filter versions, audited for contamination/privacy and evaluated with fixed controls. |
| **v0.6** | Run documented agent/tool-RL experiments in deterministic local environments. | Environment/task definitions, tool interfaces, reward/verifier docs, sandbox/resource limits, trajectory schemas and safety notes. | Tool actions are constrained; environment versions and trajectories are reproducible; network/filesystem side effects are disabled or explicitly allowlisted; reward and task-evaluation evidence are auditable. |
| **v1.0** | Use a deployable, platform-grade open-source post-training research project that manages data, training jobs, models, evaluations, lineage and controlled experiments, while retaining the single-GPU local learning path. | Stable CLI/API/schema compatibility policy; configurable deployment; job lifecycle/scheduling and resource observability; dataset/model/run/evaluation registries and import/export; full installation, operations, user/contributor guides, release notes, citations and result catalog. | Clean install and deployment from public source; supported end-to-end research workflows can run, inspect, compare and reproduce; job state, failure/recovery and resources are auditable; schema/API compatibility is defined; license, security, upgrade/backup/recovery are reviewed; public results are traceable; no mandatory proprietary service. |

## 4. Current repository readiness snapshot

The repository already contains an Apache-2.0 `LICENSE`, contributor guide, code of conduct, security policy and citation metadata, plus a Python package and real run artifacts. These are a useful foundation, not proof that every dependency/data/model asset has been cleared for redistribution.

Before v0.1 publication, reconcile all release-facing files. A mismatch was found during this plan's preparation—the root `CHANGELOG.md` said real RTX 4060 execution was pending while the implementation status and smoke report recorded it complete—and has now been corrected. Recheck cross-document consistency at every release.

## 5. Release checklist template

For each public version, verify:

1. Version number, tag, package metadata, changelog and release notes agree.
2. README and docs distinguish supported features from planned work.
3. Code/license/dependency/asset provenance is reviewed; no unlicensed data/model weights or secrets are included.
4. Setup, examples, validation/tests and known hardware limits are reproducible from a clean environment.
5. Every reported metric links to run, dataset, model, evaluation protocol and environment evidence.
6. Security-sensitive behavior (downloads, file access, process launch, tool execution) has explicit bounds and tests for that version.
7. Limitations, failed/OOM runs, breaking changes and migration steps are visible.
8. Citation and contribution paths are current; optional hosted services are not mistaken for core requirements.

## 6. Scope control

Early releases target an open, reproducible single-GPU lab: MVP/v0.1 does not require a web UI, scheduler, database, registry, multi-GPU support or automatic HPO; any UI remains optional and local-first. v0.2 adds only a constrained local launcher, and a serial queue follows when OVAT usage requires it. Heavier scheduling, persistence and deployment infrastructure belong to later releases and must be justified by real users and workloads.

The platform-grade v1.0 is the end goal. Early work should establish compatible schemas, IDs, lineage and CLI/API boundaries without importing platform operations cost ahead of schedule. Every release keeps the local single-GPU path usable; future UI/registry layers must remain exportable through local files, recipes and CLI so users do not lose their experiments when they leave the layer.

The targeted-synthetic-data workbench is an early optional research/UI slice, not a release claim that remote generation or training ran. It exports local non-registered configuration drafts and summarizes allowlisted evidence; users explicitly execute provider calls and training through CLI. Fake-provider examples and project-owned fixtures are for offline contract tests only.
