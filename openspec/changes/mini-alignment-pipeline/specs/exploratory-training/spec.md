# Spec Delta

## Purpose

允许开发者使用当前可用的少量真实训练数据验证训练阶段是否能端到端运行，同时明确隔离探索结论与正式 Mini 数据/模型验收。

## ADDED Requirements

### Requirement: Data quantity does not block labeled exploratory runs
训练入口 MUST NOT 仅因实际样本量低于 Mini 目标而拒绝显式标记的探索运行；运行记录 MUST 保留实际数据量、单位和质量/审查状态。

#### Scenario: Preference set is below the Mini target
- **WHEN** 显式探索运行使用少于 2,000 组的完整 preference pairs
- **THEN** 训练可按配置启动，并记录实际 pair 数和未达到正式目标的状态

#### Scenario: GRPO set is below the Mini target
- **WHEN** 显式探索运行使用少于 2,000 个可验证 prompts
- **THEN** 训练可按配置启动，并记录实际 prompt 数和未达到正式目标的状态

### Requirement: Exploratory runs remain separate from official readiness
探索运行及其产物 MUST 保留 exploratory 分类，且 MUST NOT 仅凭训练成功、数据数量或评测结果晋升为 official Mini 输入、baseline 或完成状态。

#### Scenario: Small-data run succeeds
- **WHEN** preference 或 GRPO 探索运行成功
- **THEN** 结果报告实际训练暴露量、数据质量审查限制和探索性状态，Mini 正式 readiness 不因此改变

#### Scenario: Review or lineage evidence is incomplete
- **WHEN** 数据审查未完成或血缘证据不完整
- **THEN** 运行仍可作为探索性工程验证，但不得声明数据质量已批准或完整可复现

### Requirement: Parallel alignment branches share an explicit parent
同一 Mini 对照组中的 DPO、原始 SimPO、GRPO-standard 与 GRPO-lowmem MUST 显式引用同一 M2-SFT 模型身份；无法使用该父模型的分支 MUST 标记失败或未运行。

#### Scenario: Alignment branches use the same M2
- **WHEN** 多个 Mini 对齐分支被训练或比较
- **THEN** 每个分支记录相同 M2 model ID 和可验证摘要，分支结果按实际状态展示

#### Scenario: Parent checkpoint is unavailable
- **WHEN** 指定 M2 checkpoint 或摘要无法验证
- **THEN** 相应训练不得伪称从该 M2 派生，并记录阻断或失败原因

### Requirement: Exploratory evaluation reports coverage without fabricated outcomes
探索性模型比较 MUST 使用相同的可用评测协议，并逐模型报告完成、失败、部分完成或未运行状态；缺失结果 MUST NOT 表示为零分或成功。

#### Scenario: A branch cannot be evaluated
- **WHEN** 分支训练失败、评测未运行或评测部分失败
- **THEN** 比较报告保留具体状态和缺失指标，不补造模型得分
