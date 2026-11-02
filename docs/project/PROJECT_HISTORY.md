# Project History and Decision Record

**Purpose:** preserve the original problem statement and the reasoning behind the project's current direction.  
**Source:** historical planning conversation supplied by the project owner. The transcript does not provide reliable dates for each decision, so this record uses decision sequence rather than invented timestamps.

This is a concise decision history, not a verbatim transcript. Early brainstorms and illustrative examples are not automatically binding requirements; the adopted direction and current source of truth are identified below.

## Origin: learn the full post-training process on one consumer GPU

The project began with a practical question: is there an open-source small-model project where an individual can run CPT, SFT, DPO, SimPO and GRPO, inspect each stage's data format, and observe changes in model behavior and metrics?

The need was not just to execute training commands. It was to understand what data and objective each stage uses, what changes in the model, what trade-offs appear, and what the run costs on a real consumer GPU.

The lasting product definition is recorded in the [Product Vision](../design/PRODUCT_VISION.md):

> An open-source, reproducible post-training lab for understanding what changes across CPT, SFT, preference optimization, and RL on a single GPU.

The intended open-source artifact is a **post-training experiment textbook + runnable lab + open research baseline**, not another Trainer implementation.

## Decisions that shaped the project

| Topic | Adopted direction | Alternatives explored / status | Current source of truth |
|---|---|---|---|
| First model and hardware | Start with Qwen3-0.6B-Base on a single RTX 4060 8GB so complete experiments and repeated ablations remain practical. | 0.8B, 3B and 4B were discussed. Larger models are later scale-up work, not the first teaching baseline. | [Overall design](../design/v0.1.1/00-overall-design.md), [Roadmap](../../ROADMAP.md) |
| Domain | Use mathematical reasoning, where answers can be checked and preference/RL signals can be constructed transparently. | General chat was deferred because it makes stage-level causal interpretation harder for the first release. | [Data design](../design/v0.1.1/01-data-design.md) |
| Training engine | Delegate trainer mechanics to ms-swift; build the experiment/data/evaluation layer around it. | Reimplementing Transformer, LoRA, DPO, SimPO or GRPO was never the project's differentiator. | [Overall design](../design/v0.1.1/00-overall-design.md), [adapter docs](../implementation/adapters/MS_SWIFT_ADAPTER.md) |
| Stage graph | Base → CPT → SFT, then DPO, SimPO and GRPO branch in parallel from the same SFT/M2 parent for controlled comparison. | Sequential DPO → SimPO → GRPO was explicitly rejected as the default comparison because it confounds algorithm effects. A sequential industrial pipeline could be a separate later experiment. | [Overall design](../design/v0.1.1/00-overall-design.md) |
| Data as a first-class module | Make data schemas, examples, preparation, reports, split hygiene and lineage part of the experiment—not incidental input files. | Committing full upstream datasets was rejected; provide examples/manifests/scripts and fetch licensed source data as needed. | [Data design](../design/v0.1.1/01-data-design.md), [Lineage design](../design/LINEAGE_DESIGN.md) |
| Data source and roles | FineMath-4+ for CPT; OpenR1-Math for SFT and derived alignment data; independent GSM8K/MATH/internal holdout for evaluation. | UltraFeedback/general preference data was deferred from the math-focused baseline to avoid introducing a distribution change. | [Data design](../design/v0.1.1/01-data-design.md) |
| Split discipline | Deduplicate and split by problem before deriving SFT, preference and GRPO formats; preserve train/evaluation disjointness. | Independently random-splitting each derived format was rejected because the same problem could leak across stages/evaluation. | [Data design](../design/v0.1.1/01-data-design.md) |
| DPO/SimPO control | Reuse the same preference pairs and common M2 parent; compare under aligned budget and evaluation. | Different datasets/checkpoints would make it unclear whether the algorithm or data caused a difference. | [Experiment design](../design/v0.1.1/04-experiment-design.md) |
| GRPO supervision | The policy receives the prompt; reference answers are reserved for verifier/reward logic, not exposed as SFT targets. | Feeding the solution as a supervised target would change the nature of the RL experiment. | [Training recipe](../design/v0.1.1/02-training-recipe-design.md), [GRPO docs](../implementation/rl/GRPO_REWARDS.md) |
| Evaluation | Reuse a common evaluation protocol across checkpoints and report task quality, domain/PPL, preference/RL behavior and resource cost where applicable. | Training loss alone or hand-picked examples cannot establish capability changes. | [Evaluation design](../design/v0.1.1/03-evaluation-design.md) |
| Experiment method | Use One Variable At A Time with declared controls, run provenance and evidence-based reports. | Broad parameter sweeps/automatic HPO were deferred until controlled experiments are reproducible. | [Experiment design](../design/v0.1.1/04-experiment-design.md) |
| Project scope | Keep the initial project CLI/artifact centered; web UI, FastAPI, database, scheduler, multi-GPU and registries are not v0.1 prerequisites. | Early conversations explored a V1 lab with tracking/UI and a later platform. Scope was deliberately staged so infrastructure does not displace the learning experiments. | [Overall design](../design/v0.1.1/00-overall-design.md), [Frontend roadmap](../design/frontend/FRONTEND_ROADMAP.md) |
| Long-term evolution | Grow from single-GPU lab to data-quality/flywheel work, broader alignment/RL, agent/tool RL, and potentially a research platform. | Platform capabilities are a later outcome earned by demonstrated needs, not the starting product definition. | [Product Vision](../design/PRODUCT_VISION.md), [Roadmap](../../ROADMAP.md) |

## Budget proposals: distinguish versions, not contradictions

The conversation contains multiple budgets because scope was refined over time. Early sizing examples such as 20–100M CPT tokens, 10K+ examples, and larger preference/RL sets were exploratory. A later consolidated v0.1 proposal settled on a manageable target around:

- CPT: 5M tokens;
- SFT: 5K examples;
- DPO and SimPO: the same 2K preference pairs;
- GRPO: 2K prompts;
- Internal holdout: 500 problems, disjoint by `problem_id`.

The transcript also considered a smaller `mini` smoke recipe (around 1K records) and a larger `standard` recipe. These are separate run profiles, not competing definitions of the product. The repository's current smoke counts are lower than the Mini targets and must not be presented as if the targets had already been achieved. Current executable budgets and release gates are maintained in the [Implementation Status](IMPLEMENTATION_STATUS.md) and milestone docs.

Illustrative metric tables and numeric improvements in the early discussion were explicitly examples of report shape, not expected or observed results. Only actual run artifacts support claims.

## Open-source commitments from the beginning

The project was intended to become a public GitHub project, not a private training demo. The historical design called for:

- stage-specific data examples and human-readable recipes;
- preparation scripts, fixed seeds, source revisions/manifests and reproducible evaluation;
- genuine run records and reports, never fabricated benchmark results;
- keeping large third-party corpora and model weights out of the repository unless redistribution is permitted;
- respecting upstream data/model licenses and making environment/setup constraints visible.

These commitments are now part of the [Product Vision](../design/PRODUCT_VISION.md). Repository openness does not grant rights to redistribute upstream datasets or model artifacts.

## How the plan evolved

1. **Explore the learning goal:** run the full CPT/SFT/preference/RL path on a small model and see stage-by-stage data and effects.
2. **Choose a controlled research shape:** math reasoning, Qwen3-0.6B, shared evaluation, parallel M3 branches and OVAT.
3. **Make data central:** source-to-stage data preparation, examples, reports, split/leakage checks and later data-centric experiments.
4. **Keep the first release bounded:** rely on mature training/evaluation tools and postpone UI/platform infrastructure.
5. **Run real hardware smoke:** actual RTX 4060 execution exposed preference-yield, GRPO truncation, M3 evaluation and lint hardening work; these became dev7.
6. **Formalize provenance before experiment scale-up:** file-based data/model/run/evaluation lineage is being designed now and must be emitted before Mini/OVAT results become official.
7. **Add convenience layers only when justified:** read-only UI first, a constrained local launcher later, then an Experiment Studio and possibly registry views.

## Present-day interpretation

The historical intent and current direction agree: optimize for learning and trustworthy experimental evidence, not feature count. A feature belongs when it improves reproducibility, controlled comparison, interpretability, open-source usability or practical single-GPU execution. If it mainly makes the project look like a generic platform, it should wait.

For current execution state, see [Development Plan](DEVELOPMENT_PLAN.md). For the lineage schema now being formalized, see [Data and Model Lineage Design](../design/LINEAGE_DESIGN.md).
