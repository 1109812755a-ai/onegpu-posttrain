# Proposal

## Why

Smoke 已验证训练链路可运行，但当前 preference 数据产出远低于 Mini 预算，Mini 阶段数据也尚未整体通过质量证据与完整血缘验收。现在需要把“达到目标且证据完整”定义成可重复执行的 Mini 数据就绪门槛，避免把样本数量、局部报告或 Smoke 结果误当成正式实验数据。

## What Changes

- 定义 Mini 各阶段数据预算与数据就绪判定：CPT 5M tokens、SFT 5K 合格样本、Preference 2K 经验证偏好对、GRPO 2K 可验证 prompts；实际数量不足时明确报告短缺与停止原因，不放宽既有质量筛选规则。
- 对每个正式 Mini 数据集生成并关联版本化 manifest、阶段质量报告、数据构建报告及有序处理步骤；记录计数单位、拒绝/产出诊断、检查覆盖和不可用指标。
- FineMath 对同一来源 URL 下不同 chunk 文本生成包含内容指纹的稳定 record ID，同时继续用来源 URL 派生的 document identity 固定 train/holdout split；不去重或改写样本资格，完全相同的重复记录仍由结构质量 gate 阻断。该身份协议变更必须版本化并在新输出目录重建。
- 要求正式 Mini 数据集及其构建、质量评估产物具备可校验的文件级血缘；以完整性校验区分可追溯、明确不完整和校验失败，避免将缺失证据视为通过。
- 将数据目标达成、质量证据和血缘完整性作为彼此独立的验收维度；通过其中一项不得被解释为其他维度通过或模型效果提升。
- 不改变既有样本资格规则；如数据源在既定扫描范围内无法满足预算，保留失败证据并单独评估后续方案。

## Capabilities

### New Capabilities

- `mini-dataset-readiness`: 定义正式 Mini 数据集的阶段预算、数据就绪状态及质量证据和血缘完整性验收行为；复用已完成变更中 `data-build-evidence` 与 `dataset-lineage` 的既有契约，不重定义其通用语义。

### Modified Capabilities

无。当前 OpenSpec 主 `specs/` 目录为空；相关通用契约曾在已完成变更的 delta specs 中定义。本变更新增 Mini 专属的资格与验收行为，规格阶段应引用并复用这些契约，避免重复定义或无必要改变通用行为。

## Impact

- 可能涉及 Mini 数据配方/构建 CLI、阶段质量评估与报告、manifest/处理报告、lineage 校验命令、相应测试及中文 Mini 里程碑/状态文档。
- 以现有数据质量、处理轨迹和 lineage 设计为依据，不引入数据库、调度器或强制外部元数据服务。
- 不包含 Mini 模型训练、统一模型评测、四项 OVAT、完整只读前端或 launcher；这些在数据就绪后分别进入后续变更。
- OpenSpec 当前主 `specs/` 目录为空；相关通用能力曾出现在已完成变更的 delta specs 中。后续规格阶段需核对这些既有契约并明确承接。
