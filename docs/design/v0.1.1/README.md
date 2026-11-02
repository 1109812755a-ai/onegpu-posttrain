# OneGPU-PostTrain — Design Revision v0.1.1

> Target software release: **v0.1**  
> Design status: **Frozen for implementation**

`v0.1.1` here identifies the design revision, not a software release. The separate product release v0.1.1 is the planned complete read-only Experiment UI. Existing design paths remain stable for references to the frozen v0.1 contract.

## Documents

1. [`00-overall-design.md`](00-overall-design.md) — 总体定位、架构、数据/训练/评测/实验基线与 Release Gate。
2. [`01-data-design.md`](01-data-design.md) — 数据源、Canonical Schema、去污染、Verifiable Core、Length Gate、Preference Pair、Manifest/Data Report。
3. [`02-training-recipe-design.md`](02-training-recipe-design.md) — PEFT-CPT/SFT/DPO/SimPO-original/GRPO Recipe、Stage Boundary、4060 8GB 策略。
4. [`03-evaluation-design.md`](03-evaluation-design.md) — accuracy@1、PPL、Preference、GRPO、通用能力 Guardrail 与评测公平性。
5. [`04-experiment-design.md`](04-experiment-design.md) — OVAT、budget_type、官方 Ablation、Run Artifact 与实验有效性规则。
6. [`05-design-review.md`](05-design-review.md) — v0.1 Design Review 关闭记录。
7. [`DESIGN_FREEZE.md`](DESIGN_FREEZE.md) — 冻结决策与仅允许由 smoke 解析的事项。
8. [`CHANGELOG-v0.1.1.md`](CHANGELOG-v0.1.1.md) — v0.1 → v0.1.1 设计变更摘要。

## Next Phase

```text
Smoke Data Build
→ Base Eval
→ PEFT-CPT Smoke
→ Full-CPT Probe
→ SFT Smoke
→ DPO Smoke
→ SimPO-original Smoke
→ GRPO-standard Smoke
→ GRPO-lowmem Validation
→ v0.1 Mini Pipeline
```
