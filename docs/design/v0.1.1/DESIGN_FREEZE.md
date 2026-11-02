# OneGPU-PostTrain — Design Freeze v0.1.1

> **Target Release:** v0.1  
> **Freeze Date:** 2026-10-12  
> **Status:** FROZEN FOR IMPLEMENTATION

## Freeze Decision

v0.1 Design Review 的 P0/P1 修改已全部回写到正式设计。

除非 smoke test 证明 RTX 4060 8GB 存在硬性不可行问题，否则实现阶段不继续扩 scope。

## Frozen Core

```text
Project             OneGPU-PostTrain
Base Model          Qwen3-0.6B-Base
Reference GPU       RTX 4060 8GB
Domain              Mathematical Reasoning
Training            ms-swift
Evaluation          EvalScope + custom
Official CPT        PEFT-CPT
Full CPT            smoke probe only
SFT Data             OpenR1 Verifiable Math Core
Preference           verified length-controlled pairs
SimPO                original / cpo_alpha=0
GRPO                 standard beta=.04 / lowmem beta=0
Stage Boundary       fresh adapter → merge → fresh adapter
Method               OVAT
```

## Smoke May Resolve Only

- exact LR values；
- final max lengths；
- Full-CPT feasibility；
- GRPO-standard 8GB feasibility；
- exact ms-swift flags realizing the frozen contracts。

任何 resolution 必须写入 resolved recipe / protocol metadata。

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
