# SFT Smoke: M1-CPT -> M2-SFT

SFT starts from the standalone model created by merging the PEFT-CPT adapter into the Base model.

```text
M1-CPT (merged standalone model)
  ↓
fresh LoRA adapter
  ↓
20-step SFT smoke
  ↓
latest checkpoint
  ↓
merge LoRA
  ↓
M2-SFT (standalone model)
```

This preserves the frozen stage-boundary contract: an alignment stage never depends on a hidden stack of previous adapters.

## Data contract

Input is the ms-swift native SFT JSONL generated from the Canonical OpenR1 SFT split. The canonical builder already enforces:

- programmatically verified correct answer,
- complete reasoning,
- parseable final answer,
- rendered-token length eligibility,
- train/holdout separation,
- benchmark decontamination.

The smoke data gate uses 1024 rendered tokens, and SFT training uses the same `max_length=1024`. This avoids silently truncating samples that passed data preparation.

## Frozen smoke profile

- tuner: LoRA
- FP16
- batch size: 1
- gradient accumulation: 8
- learning rate: `1e-4`
- LoRA rank: 8
- LoRA alpha: 32
- target modules: `all-linear`
- gradient checkpointing: enabled
- max length: 1024
- optimizer steps: 20
- seed: 42

## Commands

Inspect without loading a model:

```bash
scripts/sft_smoke_plan.sh
```

Run after CPT smoke has produced `M1-CPT`:

```bash
scripts/sft_smoke.sh
```

The script evaluates `M2-SFT` with the same GSM8K / MATH-500 / internal-holdout protocol and, when a CPT evaluation artifact exists, writes a `CPT vs SFT` comparison report.

Smoke results validate plumbing and resource feasibility. They are not official SFT effect-size claims.
