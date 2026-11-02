# Unified Model Evaluation Specification

## Purpose

让不同训练分支的模型能够在同一套任务评测协议下产出可比较且可追溯的结果，避免因数据、解析器或评测限制不同而误读阶段差异。

## Requirements

### Requirement: Shared protocol for stage-model evaluation
同一比较中的阶段模型 MUST 使用一致的评测数据版本与样本 ID、提示/解码设置、答案解析器和评测上限，并记录模型身份、父模型及协议信息。

#### Scenario: M3 branches are evaluated against the common M2 baseline
- **WHEN** DPO、原始 SimPO、GRPO-standard 和 GRPO-lowmem 模型被纳入同一阶段比较
- **THEN** 每个模型均生成可追溯的评测结果，且 M2 基线及所有候选使用相同协议与评测覆盖

### Requirement: Explicit evaluation coverage and failures
阶段评测比较 MUST 显示每个预期模型和指标的完成、缺失或失败状态；缺少结果时 MUST NOT 将其解释为零分或成功。

#### Scenario: One evaluation is missing or failed
- **WHEN** 预期模型未产生评测结果，或评测执行失败/部分失败
- **THEN** 比较产物标记对应模型及缺失/失败状态，并保留可用结果而不伪造指标
