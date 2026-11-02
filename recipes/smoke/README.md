# Smoke Recipes

Smoke recipes validate that each stage can execute on the target environment. They are intentionally small and are **not** official effect-size experiments.

Order:

1. `00-base-eval.yaml`
2. `10-peft-cpt.yaml`
3. `11-full-cpt-probe.yaml`
4. `20-sft.yaml`
5. `30-dpo.yaml`
6. `31-simpo-original.yaml`
7. `40-grpo-standard.yaml`
8. `41-grpo-lowmem.yaml`

For PEFT-CPT, the default smoke profile is non-quantized LoRA on Qwen3-0.6B-Base. The model is small enough that this should be simpler than QLoRA on an 8 GB card. If the real RTX 4060 run proves otherwise, enable the commented BNB 4-bit fields without changing the experiment contract.

## Alignment smoke

`30-dpo.yaml` and `31-simpo-original.yaml` intentionally share the same M2 parent, preference data, seed and smoke budget. `31-simpo-original.yaml` explicitly uses `cpo_alpha: 0.0`; changing it to ms-swift's default `1.0` changes the objective and must not be reported as the original SimPO comparison.


## GRPO smoke

`40-grpo-standard.yaml` and `41-grpo-lowmem.yaml` share the same merged M2 parent, GRPO prompt IDs, seed, LoRA config and completion budget. Standard uses `beta=0.04`; lowmem uses `beta=0` and therefore has no reference model or KL metric. Both use the project external reward plugin: `onegpu_accuracy + 0.1 * onegpu_boxed_format`.

The frozen completion contract is:

```text
per-device completion batch = 1
gradient accumulation       = 4
generation batch            = 4
num_generations (G)         = 2
```

With 20 smoke optimizer steps this is a planned training budget of 80 completion instances. Smoke is for feasibility and telemetry only.
