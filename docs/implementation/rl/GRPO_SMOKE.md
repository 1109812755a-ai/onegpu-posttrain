# GRPO Smoke

## Goal

Validate the final v0.1 post-training branch on one consumer GPU and distinguish reference-model memory cost from the core GRPO loop.

## Profiles

### Standard

```yaml
beta: 0.04
reference_enabled: true
```

A reference model is required and KL is reported.

### Low-memory

```yaml
beta: 0.0
reference_enabled: false
```

No reference model is loaded. KL is intentionally `N/A`.

Both profiles start from the exact same merged `M2-SFT`, use the same GRPO JSONL, seed, LoRA configuration, generation parameters and optimizer-step budget.

## Completion-level batch contract

```text
per_device_train_batch_size = 1 completion
gradient_accumulation_steps = 4
generation_batch_size       = 4 completions
num_generations (G)         = 2
```

Therefore the smoke budget is four training completion instances per optimizer step. With 20 smoke steps the planned training budget is 80 completion instances. This is budget metadata; actual generated-token counts must come from real execution artifacts rather than estimation.

## Length contract

```text
max prompt      = 512
max completion  = 512
truncation      = delete
```

The data builder already applies a Qwen tokenizer-rendered prompt length gate. `delete` exists as a fail-safe so a contract mismatch cannot silently alter the prompt by left truncation.

## Execution

Plan only:

```bash
scripts/grpo_smoke_plan.sh
```

Run on the RTX 4060 after M2 exists:

```bash
scripts/grpo_smoke.sh
```

The standard profile is attempted first. `OOM` is persisted as a valid 8GB feasibility outcome and the low-memory profile is still attempted.

## Artifacts

```text
runs/smoke/grpo-standard-smoke-s42/
├── run.json
├── environment.json
├── train/
├── logs/
├── grpo_metrics.json        # PASS only
├── merged/M3-GRPO-standard/ # PASS only
└── evaluation/              # PASS only

runs/smoke/grpo-lowmem-smoke-s42/
└── ... M3-GRPO-lowmem ...

runs/smoke/grpo-comparison/
├── grpo_comparison.json
└── grpo_comparison.md
```

Promoted metrics include reward mean/std, fraction of zero-std reward groups, reward components, completion length/clipping, KL (standard only), sampled peak GPU memory and wall time.

## Interpretation

Smoke answers feasibility/instrumentation questions. It does not establish that beta=0.04 or beta=0 is better. Official effect claims start only after the Mini dataset and controlled OVAT experiments.
