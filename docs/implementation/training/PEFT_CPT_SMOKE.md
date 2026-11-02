# PEFT-CPT Smoke — Base → M1

This milestone validates the first real training edge in OneGPU-PostTrain:

```text
Qwen3-0.6B-Base
  → LoRA PEFT-CPT
  → latest checkpoint
  → merge LoRA
  → M1-CPT
  → unified evaluation
  → Base vs CPT comparison
```

## Why the smoke profile uses LoRA, not QLoRA

Qwen3-0.6B is small enough that the first RTX 4060 8GB validation should prefer the simpler non-quantized LoRA path. The recipe retains optional BNB 4-bit fields for an OOM fallback. If the real target machine requires QLoRA, that change is treated as a hardware feasibility adjustment and must be recorded in the run metadata.

## Smoke budget

The FineMath smoke data artifact targets roughly 500K training tokens, but the first training run intentionally uses only 20 optimizer steps. Smoke is for validating:

- CUDA/VRAM feasibility;
- ms-swift argument compatibility;
- training throughput;
- checkpoint saving;
- deterministic checkpoint discovery;
- LoRA merge;
- stage-end evaluation;
- Base vs CPT result plumbing.

It is **not** an official claim about CPT effect size.

## Frozen training profile

```yaml
tuner_type: lora
torch_dtype: float16
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1e-4
lora_rank: 8
lora_alpha: 32
target_modules: all-linear
gradient_checkpointing: true
max_length: 1024
max_steps: 20
save_steps: 10
seed: 42
```

The learning rate is a smoke baseline, not an asserted optimum. Formal SFT/CPT parameter conclusions belong to later ablation experiments.

## Plan only

After building smoke data:

```bash
scripts/adapt_smoke_data.sh
scripts/cpt_smoke_plan.sh
```

The plan path does not load the model and is suitable for inspecting exact `swift pt` and EvalScope commands before using the GPU machine.

## Real RTX 4060 run

Install runtime dependencies:

```bash
pip install -e '.[all]'
```

Then:

```bash
scripts/build_smoke_data.sh
scripts/cpt_smoke.sh
```

Expected run artifacts:

```text
runs/smoke/
├── base-smoke-s42/
│   └── evaluation/
│       ├── evalscope/
│       ├── internal/
│       ├── domain/perplexity.json
│       └── eval.json
│
└── cpt-peft-smoke-s42/
    ├── environment.json
    ├── run.json
    ├── train/
    ├── logs/
    ├── merged/M1-CPT/
    ├── evaluation/
    │   ├── evalscope/
    │   ├── internal/
    │   ├── domain/perplexity.json
    │   └── eval.json
    └── comparison/
        ├── stage_comparison.json
        └── stage_comparison.md
```

## Evaluation protocol

Base and M1 use the same protocol:

- GSM8K through EvalScope;
- MATH-500 through EvalScope;
- frozen internal OpenR1-derived holdout;
- FineMath document-level holdout perplexity.

The PPL implementation accumulates token-weighted NLL before exponentiation. It does not average per-document perplexities.

## Interpretation

Smoke answers only:

1. Did the stage run on the target GPU?
2. What was peak resource behavior according to the training/runtime logs?
3. Was a valid adapter checkpoint produced?
4. Could it be merged into an independent M1 model?
5. Can Base and M1 be evaluated under the same protocol?
6. Are metric changes observable end-to-end?

A smoke result should never be described as evidence that CPT is generally beneficial or harmful.
