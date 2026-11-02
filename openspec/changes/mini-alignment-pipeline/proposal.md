# Proposal

## Why

当前已具备可用的 Mini M2-SFT、4 组真实 Preference 数据和 2,000 条 GRPO 数据，但严格 Mini 目标数量尚未达到。项目需要先验证 CPT→SFT→偏好优化/RL 的实际训练链路和证据闭环；样本数量不足应限制结论强度与 official readiness，而不应阻止明确标记的探索性训练。

## What Changes

- 增加探索性 Mini 训练约定：允许使用现有真实数据按实际可用数量运行 DPO、原始 SimPO、GRPO-standard 和 GRPO-lowmem，不以 2K preference pairs 或 2K GRPO prompts 作为启动门槛。
- 每个分支从同一个已完成 Mini M2-SFT 出发；记录实际样本数/暴露量、数据质量状态、父模型、recipe 与解析配置、唯一 run/model IDs、环境和实际产物。
- 为四个分支提供独立、有限预算的 Mini 配方与可复现命令；不改变 Smoke 配方，不自动补数据、不降低验证要求，也不将小样本结果描述成正式 Mini 或有效性结论。
- 在共享评测资源可用时，对 M2 与成功产出的分支使用同一评测协议；失败或未测分支明确显示状态，不伪造分数。
- 保留 2048-token SFT OOM 作为独立历史证据；本变更不重试该运行、不覆盖其 OpenSpec 任务或已有产物。

## Capabilities

### New Capabilities
- `exploratory-training`: 在不满足正式 Mini 预算/质量门槛时，允许显式标记的受控探索运行，并保持其与 official readiness 的边界。

### Modified Capabilities
- `run-model-lineage`: 新增探索性偏好优化/RL 分支必须引用实际数据集、同一父模型和真实 run/model 产物，并保留成功、失败及缺失评测状态。

## Impact

影响 Mini recipes、训练数据适配与 CLI 调用、run/model lineage、统一评测以及中文实验记录。复用现有本地数据和训练栈，不引入数据库、服务或新的训练依赖；不下载数据或模型。
