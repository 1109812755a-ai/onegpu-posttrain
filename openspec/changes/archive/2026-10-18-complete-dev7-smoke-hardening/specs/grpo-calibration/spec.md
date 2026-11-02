# Spec Delta

## Purpose

为 GRPO completion 长度选择提供受控、可追溯的校准证据，使用户能在硬件预算内权衡截断、任务信号和资源成本，而非仅凭训练配置猜测。

## ADDED Requirements

### Requirement: Controlled completion-length comparison
GRPO completion 长度校准 MUST 在候选之间仅改变声明的 completion 长度变量，并固定模型父项、prompt 样本、生成设置、数据和计算预算；每个候选运行 MUST 可由实验与单次运行标识区分。

#### Scenario: Candidate configurations are compared under shared controls
- **WHEN** 用户执行多个 completion 长度候选的校准
- **THEN** 结果表明候选共享父模型、prompt 集和其余生成/训练控制，且各次运行记录其 completion 长度值

### Requirement: Calibration outcome and resource evidence
每个 GRPO 校准候选 MUST 保存执行状态、实际生成长度、completion clipping/truncation 比率、可用 reward/verifier 指标、耗时和峰值显存；缺失或失败的测量 MUST 明确标记为不可用。

#### Scenario: A candidate completes or fails within the hardware budget
- **WHEN** 候选正常完成、OOM 或因其他错误失败
- **THEN** 校准报告记录候选状态及可用指标，未产生的指标标记为缺失，不将失败候选误记为通过

#### Scenario: No candidate meets the clipping target
- **WHEN** 所有可运行候选的 clipping/truncation 比率均未低于校准目标
- **THEN** 报告保留这一结果并指出当前受测配置中的最佳支持候选，不通过放宽答案验证或完成性规则制造达标结论
