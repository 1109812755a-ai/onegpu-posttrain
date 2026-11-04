# Spec Delta

## MODIFIED Requirements

### Requirement: 带版本的评测身份与输入
每次新执行的评测 SHALL 有唯一评测 `run_id`，并产生独立 `eval_id` 的评测产物；来源信息 SHALL 包含被评模型 ID／摘要或明确的外部模型证据、可得的数据集／基准 ID 与修订、评测器／协议版本、解析后评测配置摘要，以及可追溯的训练／合并 `run_id`。`experiment_id` 仅用于可选分组，不替代来源引用。权重空间实验中的评测来源还 SHALL 包含迭代编号、数据集角色（practice 或 final_holdout）、候选检查点摘要及其父模型／父候选身份，并 SHALL 记录目标模型修订、processor/chat-template 修订、容器镜像摘要、PyTorch/Transformers 版本及生成参数。

#### Scenario: M3 DPO 基准评测
- **WHEN** 在 GSM8K 与 MATH-500 上评测 DPO 模型
- **THEN** 评测来源指向 DPO 模型产物、基准修订或明确的不可得状态、协议及评测配置

#### Scenario: 外部基准修订未知
- **WHEN** 评测器没有暴露确切基准修订
- **THEN** 来源将其标为不可得，而非填写猜测的修订

#### Scenario: Weight-space practice iteration
- **WHEN** 候选模型在练习集上完成一轮评测
- **THEN** 来源记录迭代编号、practice 角色、练习数据集摘要、候选权重摘要及父候选身份

#### Scenario: Weight-space final holdout
- **WHEN** 候选模型在固定最终留出集上完成评测
- **THEN** 来源记录 final_holdout 角色、固定留出集摘要、候选权重摘要及评测协议，且与练习轮次可区分

#### Scenario: Qwen3.5 Docker evaluation
- **WHEN** Qwen3.5-0.8B is evaluated in the isolated WSL Docker runtime
- **THEN** the evaluation lineage records model revision `05f056712f8c582c8708395156518d963b46fb29`, image digest, PyTorch and Transformers versions, processor/chat-template identity, generation parameters, and the exact evaluation input digest

#### Scenario: Compare baseline and candidate
- **WHEN** a candidate result is compared with the pinned 33/81 baseline
- **THEN** the comparison is marked directly comparable only when the model revision, runtime identity, inference template and parameters, dataset digest, scoring protocol, and sample coverage match
