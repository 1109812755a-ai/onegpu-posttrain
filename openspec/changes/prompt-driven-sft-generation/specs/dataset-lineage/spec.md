# Spec Delta

## ADDED Requirements

### Requirement: Prompt-only 数据集记录生成根来源
没有父数据集的模型生成数据 MUST 明确标记为 prompt-only 根来源，并记录生成 execution、模板/提示词配置摘要、provider profile 摘要及模型声明；不得伪造 parent dataset ID 或上游记录 ID。

#### Scenario: 登记 prompt-only SFT 数据
- **WHEN** 用户登记由提示词直接生成且不依赖输入数据集的 SFT 样本
- **THEN** dataset manifest 的父数据集集合为空，并通过来源证据关联 generation execution、提示词摘要、provider/model 身份及输出内容摘要

#### Scenario: 沿血缘查看 prompt-only 样本
- **WHEN** 用户从已登记 dataset 追溯其来源
- **THEN** 系统显示这是无数据集父项的 prompt-only 生成根，并能继续追到提示词版本、provider profile 和生成 execution，而不显示虚构祖先
