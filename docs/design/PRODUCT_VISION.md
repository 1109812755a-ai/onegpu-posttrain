# OneGPU-PostTrain Product Vision

**Status:** Product north star  
**Applies to:** Current project direction and future roadmap decisions

中文版本：[产品愿景](PRODUCT_VISION_zh.md)。

## One-sentence definition

> **Starting as an open-source, reproducible post-training lab on one consumer GPU and growing into a deployable platform for real post-training research workflows—so users can understand and reliably run CPT, SFT, preference optimization and RL.**

中文版：

> **一个面向单卡消费级 GPU 的开源、可复现 LLM 后训练实验室，用统一数据、训练、评测和单变量实验，理解 CPT、SFT、偏好优化和 RL 到底改变了什么。**

## What we are building

OneGPU-PostTrain is not another trainer or fine-tuning framework. Its near-term form is an open-source, runnable research baseline and learning lab: an individual engineer can run a full post-training workflow on a consumer GPU and inspect the evidence behind every stage and experiment. Its long-term form is a deployable, platform-grade project that organizes data, training jobs, models, evaluations, lineage and controlled experiments into a complete workflow for sustained use and extension.

```text
Data → Training → Evaluation → Controlled Experiments
```

The core path is:

```text
Base → CPT → SFT ─┬→ DPO
                  ├→ SimPO
                  └→ GRPO
```

DPO, SimPO and GRPO branch in parallel from the same SFT checkpoint for controlled comparison; they are not implicitly chained into one another.

## Questions the project must help answer

- What data entered each stage, what was filtered out, and what data contract was applied?
- Which processing step changed the data's measured quality and coverage, and did a controlled downstream experiment show a corresponding model-metric change?
- What changed after CPT and SFT, and which measurements support that conclusion?
- Under matched data and controls, how do DPO and SimPO differ?
- When GRPO reward rises, does task capability rise too? What do reward, KL and completion behavior show?
- What happens when one variable—learning rate, beta, generation count, data size or reward design—changes?
- Why might training loss improve while benchmark performance does not?
- What are the real time, throughput and peak-memory costs on the target consumer GPU?

## The differentiator

The differentiator is not merely supporting CPT/SFT/DPO/GRPO; mature external frameworks already provide training implementations. The project contribution is the reproducible, comparable and interpretable experiment system around them:

- stage-specific data quality evidence alongside data lineage, distribution and leakage checks; quality reports expose their methods, denominators and limitations rather than hiding them in one score;
- ordered source-to-dataset processing traces, comparable before/after quality evidence, and controlled links from dataset versions to downstream model evaluations;
- consistent data lineage and stage boundaries;
- explicit data → run → model → evaluation lineage that remains inspectable in the open-source artifacts;
- shared task, domain, preference and RL evaluation where applicable;
- One Variable At A Time experiments with declared controls;
- checkpoint, configuration, seed, dataset and evaluation provenance;
- resource-cost evidence alongside outcome metrics;
- reports that state limitations and do not overclaim from smoke-scale samples.

The near-term experience is a **post-training experiment textbook + runnable lab + open research baseline**. The long-term goal is a **practical, open-source post-training research platform**, not a collection of scripts or a static demo.

## Product principles and invariants

1. **Single-GPU first:** a consumer GPU is a first-class constraint and teaching context, not an afterthought.
2. **Open source by design:** keep experiment definitions, interfaces, implementation and evidence inspectable; make local reproduction possible without a proprietary hosted service; provide a clear path for external contributions.
3. **Observe every stage:** data, resolved configuration, lineage, training signals, evaluations and resource costs should be inspectable.
4. **Compare under controls:** change one declared variable at a time; use common parents, data, seeds and evaluation protocols where the question requires them.
5. **Evidence before claims:** separate feasibility smoke runs from quality/effect-size conclusions; show sample sizes, uncertainty/context and missing measurements.
6. **CLI and external trainers first:** keep training execution delegated to mature frameworks and preserve CLI workflows. UI and orchestration are supporting layers.
7. **The lab is the start; the platform is the destination:** prove a trustworthy single-GPU research loop first, then build scheduling, experiment/data/model registries, deployment and extension capabilities in response to real usage. Each layer must support the complete research workflow.

## Open-source product commitments

- Keep the core workflow usable locally from the documented source and pinned dependencies; optional hosted integrations must remain optional.
- Publish enough configuration, data manifests/identifiers, environment details, evaluation protocols and run provenance for others to understand and reproduce claims.
- Make contribution and extension points explicit; avoid requiring private services or undocumented manual steps for the main path.
- Treat upstream licenses and terms as part of data/model provenance. Open-sourcing this repository does not grant permission to redistribute third-party datasets, model weights or generated artifacts; document how to obtain them and keep unlicensed assets out of the repository.
- Do not publish secrets, personal data or restricted raw records in logs, reports or example artifacts.

## What we are not optimizing for at the start

- Replacing ms-swift, TRL or other trainer implementations;
- becoming a general-purpose distributed training framework;
- building a heavyweight scheduler, database or distributed platform before the single-GPU loop and user needs are validated;
- maximizing leaderboard scores without reproducibility and provenance;
- adding UI features that obscure rather than explain the underlying experiment.

These are sequencing decisions, not permanent prohibitions. Platform capabilities will be introduced incrementally: constrained local launch and serial queues first, then stable registries, job lifecycle, deployment and resource management. Each needs a real workflow, open interfaces and clear operational boundaries.

## Evolution

```text
Reproducible single-GPU lab (v0.1)
        ↓
Local experiment launch, OVAT and serial task queue (v0.2–v0.3)
        ↓
Dataset/model/experiment registries and broader research (v0.4–v0.6)
        ↓
Deployable, operable and extensible open Post-Training platform (v1.0)
```

The platform is the explicit end goal; the lab is the validated starting point. Deployment, concurrency and resource-scheduling choices will be determined from user needs and evidence rather than assuming a heavyweight infrastructure stack. See [`../../ROADMAP.md`](../../ROADMAP.md) for release sequencing and [`FRONTEND_ROADMAP.md`](frontend/FRONTEND_ROADMAP.md) for UI-specific gates.

## Decision filter

Before adding a capability, answer:

1. Which post-training question or reproducibility failure does it address?
2. Does it improve understanding, controlled comparison, provenance or practical single-GPU execution?
3. Can the same result still be obtained and audited through the CLI/artifacts?
4. What new state, dependencies or operational complexity does it introduce?
5. What evidence will show the feature helped the intended user?

If a platform capability lacks a real workflow, user and operational evidence, keep it out of the current release; do not lose the v1.0 deployable-platform objective.

## Relationship to frozen design

This vision document records the product north star. The frozen [`v0.1.1 overall design`](v0.1.1/00-overall-design.md) remains the authority for v0.1 implementation contracts and explicitly keeps Web UI/FastAPI out of v0.1. Future design revisions may refine implementation, but should preserve this product intent unless the project deliberately revises its direction.

For the original discussion and how these decisions evolved, see the [Project History](../project/PROJECT_HISTORY.md).
