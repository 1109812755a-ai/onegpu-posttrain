# Spec Delta

## ADDED Requirements

### Requirement: Composed datasets preserve all parent identities
组合数据集的 lineage MUST 把每个已登记输入 dataset ID 记录为直接父级，并绑定父内容摘要、组合配置摘要、实际输出内容摘要与构建 run；训练 run 可通过组合 dataset ID 递归追溯全部来源。

#### Scenario: 从模型训练 run 回溯多个数据来源
- **WHEN** 用户检查一个消费组合数据集的训练 run
- **THEN** lineage 能解析组合 dataset ID、全部父 dataset ID 与内容摘要，并验证组合配置和输出摘要

#### Scenario: 一个父数据集后续生成新版本
- **WHEN** 组合计划被新父版本重新执行
- **THEN** 新组合记录引用新父 dataset ID 并生成新的组合身份，原组合和历史训练 run 保持不变
