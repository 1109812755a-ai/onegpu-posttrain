# Spec Delta

## ADDED Requirements

### Requirement: Training and evaluation actions are prominent and discoverable
阶段训练与阶段评测 MUST 以清晰、高对比的主操作呈现当前唯一的下一步，并在长页面滚动时保持该操作容易找到。

#### Scenario: Prepare and preflight training
- **WHEN** 用户完成当前阶段要求的数据、父模型和参数选择
- **THEN** 页面以醒目的主按钮呈现配置检查/预检，并清楚区分它与正式启动训练

#### Scenario: Confirm training after a successful preflight
- **WHEN** 阶段训练预检通过且等待用户确认
- **THEN** 页面突出显示明确命名的“启动<阶段>训练”主操作，并在操作附近保留预检结论与关键配置摘要

#### Scenario: Confirm stage evaluation after a successful preflight
- **WHEN** 阶段评测预检通过且等待用户确认
- **THEN** 页面突出显示明确命名的阶段评测启动主操作，并在操作附近展示协议、模型和输出摘要

### Requirement: Unavailable launch actions explain the next step
当训练或评测操作不可用时，页面 MUST 在操作附近说明具体原因和可执行的下一步；活动任务期间 MUST 提供查看任务的入口且不得呈现可重复启动的主操作。

#### Scenario: Required training inputs are missing
- **WHEN** 父模型、训练数据或其他必要输入尚未满足
- **THEN** 页面说明缺少的具体输入及其可完成的位置，而不是只显示无解释的禁用按钮

#### Scenario: A local GPU task is already active
- **WHEN** 训练或评测因本机 GPU 任务占用而不能启动
- **THEN** 页面说明当前活动任务并提供查看入口，且不让用户误以为再次点击即可启动
