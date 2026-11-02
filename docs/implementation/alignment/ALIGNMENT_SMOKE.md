# DPO vs SimPO-original Smoke

The alignment smoke starts from the same standalone merged `M2-SFT` model and the exact same ms-swift native preference dataset.

```text
                         M2-SFT
                           │
                same preference train data
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
          fresh DPO LoRA          fresh SimPO LoRA
          beta = 0.1              beta = 2.0
          sigmoid loss            gamma = 1.0
                                  cpo_alpha = 0.0
               │                       │
               ▼                       ▼
          merge M3-DPO           merge M3-SimPO
               │                       │
               └───────────┬───────────┘
                           ▼
              same unified task evaluation
                           +
              same held-out preference eval
```

## SimPO-original contract

ms-swift currently defaults `cpo_alpha` to `1.0` for SimPO/CPO training. OneGPU-PostTrain explicitly sets:

```yaml
beta: 2.0
simpo_gamma: 1.0
cpo_alpha: 0.0
```

so this branch represents the original SimPO objective rather than SimPO mixed with the CPO NLL term.

## DPO contract

```yaml
beta: 0.1
loss_type: sigmoid
label_smoothing: 0.0
```

Both smoke branches keep the same:

- parent `M2-SFT`
- training pair IDs
- seed
- learning rate
- LoRA rank/alpha/targets
- max length
- batch / gradient accumulation
- max optimizer steps
- evaluation protocol

Algorithm-specific hyperparameters are the intended difference.

## Held-out preference data

`preference_eval.jsonl` is built from the OpenR1 holdout pool, not from alignment training data. Training preference IDs and held-out preference IDs must have zero overlap.

## Plan only

```bash
scripts/alignment_smoke_plan.sh
```

## Execute on the target GPU

After real `M2-SFT` exists:

```bash
scripts/alignment_smoke.sh
```

Artifacts:

```text
runs/smoke/
├── dpo-smoke-s42/
│   ├── train/
│   ├── merged/M3-DPO/
│   ├── evaluation/eval.json
│   └── preference_evaluation/preference_metrics.json
├── simpo-original-smoke-s42/
│   ├── train/
│   ├── merged/M3-SimPO/
│   ├── evaluation/eval.json
│   └── preference_evaluation/preference_metrics.json
└── alignment-comparison/
    ├── alignment_comparison.json
    └── alignment_comparison.md
```

Smoke output validates the controlled comparison pipeline. It is not an algorithm leaderboard claim.
