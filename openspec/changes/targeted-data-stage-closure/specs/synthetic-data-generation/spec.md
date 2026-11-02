# Spec Delta

## ADDED Requirements

### Requirement: Stage-specific generation protocol
模型生成 MUST 显式声明目标阶段并产出对应 canonical 数据；每个候选保存生成执行身份、配置摘要、输出校验和独立验证状态。

#### Scenario: 生成非 SFT 阶段候选
- **WHEN** 用户为 CPT、Preference 或 GRPO 执行已确认的生成计划
- **THEN** 系统按阶段协议解析输出并报告格式、质量和语义验证的分别结果，不复用 SFT 记录作为其他阶段输出

### Requirement: Generation recovery preserves stage and partial evidence
阶段生成任务 MUST 持久化目标阶段、不可变输入、预算、进度、已完成候选及终止原因；恢复只能复用匹配计划的有效候选，不得丢失或覆盖其他阶段任务。

#### Scenario: 生成任务中断后恢复
- **WHEN** 用户恢复一个已中断的阶段任务
- **THEN** 系统保留原执行身份和已完成证据，按原阶段协议及剩余预算继续；不确定的远程请求仍须按现有不确定请求规则处理
