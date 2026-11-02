# Spec Delta

## ADDED Requirements

### Requirement: Exploratory alignment runs retain complete available lineage
Mini 探索性 DPO、SimPO 与 GRPO 运行 MUST 记录实际父模型身份、实际使用数据集及内容摘要、实际样本量、recipe/解析配置证据、运行状态和真实输出；未知证据 MUST 显式标为不可得，不得虚构。

#### Scenario: Successful adapter training and merge
- **WHEN** 探索性训练与模型合并均成功
- **THEN** 训练和合并以各自唯一 run_id 关联共同父模型、确切数据集与 adapter/merged model IDs

#### Scenario: Training fails before producing an adapter
- **WHEN** 探索性训练 OOM 或以其他错误结束且未产生 adapter
- **THEN** 失败 run 保留已知输入和错误状态，且不声明 adapter 或 merged model

#### Scenario: Input dataset is exploratory or quality review is pending
- **WHEN** 训练输入数据未达到目标数量或质量 review 尚未完成
- **THEN** lineage 保留原数据集分类与审查状态，不将它改写为已批准数据
