# Full-CPT Feasibility Probe

The official RTX 4060 path remains **PEFT-CPT**. This probe answers one narrower engineering question:

> Can Qwen3-0.6B perform a standard full-parameter AdamW CPT update on an 8 GB GPU at all?

It is deliberately not an effectiveness experiment.

## Frozen probe profile

- model: `Qwen/Qwen3-0.6B-Base`
- `swift pt`
- `tuner_type=full`
- dtype: FP16
- optimizer: `adamw_torch`
- batch size: 1
- gradient accumulation: 1
- gradient checkpointing: enabled
- max length: 512
- optimizer steps: 5
- learning rate: `1e-5`

The profile intentionally does **not** switch to an 8-bit optimizer or LoRA. Doing so would answer a different question.

## Result semantics

The command always writes `run.json`.

- `PASS`: all five configured optimizer steps completed.
- `OOM`: a CUDA/system memory allocation error occurred. This is a **valid feasibility result** and does not make the smoke pipeline invalid.
- `FAILED`: a non-memory error occurred. Fix the execution issue before drawing a feasibility conclusion.

GPU memory is sampled through `nvidia-smi memory.used` while the subprocess is running. This is system-level GPU memory, not process-exclusive PyTorch allocator peak, and the report labels it accordingly.

## Commands

Plan only:

```bash
scripts/full_cpt_probe_plan.sh
```

Execute on the target GPU:

```bash
scripts/full_cpt_probe.sh
```

If a PEFT-CPT `run.json` already exists, the script also produces:

```text
runs/smoke/cpt-feasibility/
├── cpt_feasibility.json
└── cpt_feasibility.md
```

The probe checkpoint, if produced, is **not** promoted to M1. Official v0.1 lineage remains `Base -> PEFT-CPT -> merged M1-CPT`.
