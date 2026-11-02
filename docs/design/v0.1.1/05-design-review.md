# OneGPU-PostTrain — v0.1 Design Review (Closed)

> **Review Date:** 2026-10-12  
> **Status:** CLOSED — P0/P1 integrated into Design v0.1.1

## 1. Review Outcome

原 v0.1 设计结论：**APPROVE WITH CHANGES**。

主体保留：Qwen3-0.6B、RTX 4060 8GB、数学主线、FineMath/OpenR1、ms-swift、Base→CPT→SFT→DPO/SimPO/GRPO、Canonical Data、OVAT、统一 Evaluation。

## 2. P0 Changes Integrated

1. 项目正式名改为 `OneGPU-PostTrain`；
2. CPT 明确区分 Full CPT 与 PEFT-CPT；
3. OpenR1 增加 tokenizer-rendered Length Eligibility Gate；
4. Core 收紧为 programmatically verifiable math；
5. Preference pair 增加完整性/可解析/长度控制；
6. SimPO 官方实验冻结 `cpo_alpha=0`；
7. GRPO 采用 completion-level batch contract；
8. GRPO 增加 standard / lowmem profile；
9. Stage boundary 固定 fresh adapter → merge → fresh adapter；
10. Preference 公共指标改为 policy_pair_accuracy/margin；
11. 所有 source/tokenizer/verifier revisions 强制 pin；
12. benchmark decontamination 前移到 split 之前。

## 3. P1 Changes Integrated

- General Text PPL；
- FineMath-only 定义为 domain-only educational stress test；
- CPT nested subsets；
- CPT document-level holdout；
- prompt/tokenizer template versioning；
- answer correctness / parse / format 分离；
- GRPO KL profile-dependent；
- 禁止跨 objective 比 loss；
- 主 deterministic 指标命名 accuracy@1；
- license attribution 进入 Release Gate；
- SFT baseline/mid point 固定 3 seeds；
- Experiment 增加 budget_type。

## 4. Remaining Smoke Decisions

设计已经冻结，但以下实现参数必须由 smoke 实测决定：

- exact LR baseline；
- final 8GB sequence length；
- Full-CPT 8GB feasibility；
- GRPO-standard reference model 是否适配 8GB；
- pinned ms-swift 版本下的确切参数映射。

这些属于实现可行性 resolution，不允许改变实验语义。
