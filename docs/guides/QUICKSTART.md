# Quick Start

This guide separates repository validation from real GPU execution so users do not need to install the full training stack just to inspect the project.

## 1. Prerequisites

Reference runtime:

```text
Python: 3.10–3.12
GPU: NVIDIA CUDA-capable GPU
Reference GPU: RTX 4060 8GB
```

The code-contract tests are CPU-safe. Real training/evaluation requires the optional runtime dependencies and model/data downloads.

## 2. CPU-only developer validation

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -e '.[dev]'
pytest

onegpu-posttrain doctor
onegpu-posttrain validate-example-data data/examples
```

Run the test suite to check the current checkout. The expected test count changes as the project evolves; use the test result and current [implementation status](../project/IMPLEMENTATION_STATUS.md) as the source of truth.

## 3. Install the full runtime

On the CUDA machine:

```bash
pip install -e '.[all]'
onegpu-posttrain doctor
```

Review the doctor output before training. A missing CUDA runtime, incompatible Python version or unavailable optional package should be fixed before running the smoke pipeline.

## 4. Inspect before execution

Always inspect the frozen plan first:

```bash
scripts/smoke_pipeline_plan.sh
```

The plan path is intentionally separate from the execute path. It should reveal the exact ms-swift commands, model/data locations and stage lineage without allocating model VRAM.

## 5. Build smoke data

```bash
scripts/build_smoke_data.sh
```

The data build performs:

```text
FineMath:
source → document split → chunking → token budget → CPT train/PPL holdout

OpenR1:
source → normalize → exact dedup → benchmark decontamination
      → verifier/parse/length gates → problem-level split
      → SFT / preference / GRPO / internal holdout / preference holdout
```

Inspect generated data reports before training. Large unexpected rejection rates, contamination or length-distribution drift should be treated as a data issue rather than ignored.

## 6. Run the complete smoke pipeline

```bash
scripts/smoke_pipeline.sh
```

Execution order:

```text
Base evaluation
→ PEFT-CPT
→ Full-CPT feasibility probe
→ SFT
→ DPO
→ SimPO-original
→ GRPO-standard
→ GRPO-lowmem
→ final smoke summary
```

The Full-CPT and GRPO-standard paths may legitimately produce `OOM` on the 8 GB reference GPU. OOM is persisted as a feasibility artifact instead of being silently converted to a different recipe.

## 7. Important outputs

Expected run families:

```text
runs/smoke/
├── base-*/
├── cpt-peft-*/
├── cpt-full-probe-*/
├── sft-*/
├── dpo-*/
├── simpo-original-*/
├── grpo-standard-*/
├── grpo-lowmem-*/
├── alignment-comparison/
├── grpo-comparison/
└── smoke-summary/
```

Key artifacts include:

- resolved command/config;
- environment metadata;
- training logs;
- monitored wall time / sampled GPU memory;
- merged stage checkpoint location;
- unified `eval.json`;
- held-out preference metrics;
- GRPO trainer-state metrics;
- comparison reports.

## 8. Interpret smoke correctly

Smoke is a **feasibility and instrumentation gate**.

Do not use 20-step smoke numbers to claim that one training method is better than another. The correct questions at this stage are:

- Did the stage run?
- Did the lineage and merge work?
- Did the evaluation execute consistently?
- Did the run fit in the target VRAM?
- Were resource/metric artifacts recorded correctly?
- Did any data/evaluation contract break?

Effect-size claims belong to the Mini/OVAT phase after smoke is validated.

## 9. If something OOMs

Do not randomly change multiple variables.

Use the frozen fallback order defined in the training design. Record every fallback in the run metadata. In particular:

- Full-CPT OOM is allowed and should remain a diagnostic result.
- GRPO-standard OOM should be compared with the frozen GRPO-lowmem path.
- Do not silently truncate OpenR1 reasoning samples to solve memory pressure.

See:

- [`../design/v0.1.1/02-training-recipe-design.md`](../design/v0.1.1/02-training-recipe-design.md)
- [`../implementation/training/FULL_CPT_PROBE.md`](../implementation/training/FULL_CPT_PROBE.md)
- [`../implementation/rl/GRPO_SMOKE.md`](../implementation/rl/GRPO_SMOKE.md)

## 10. Choose the next step

Project milestones and release gates evolve. Check the current [implementation status](../project/IMPLEMENTATION_STATUS.md) and [development plan](../project/DEVELOPMENT_PLAN.md) before choosing what to run or contribute. Smoke runs verify that a workflow executes and records evidence; their small samples do not establish model quality or rank training methods.
