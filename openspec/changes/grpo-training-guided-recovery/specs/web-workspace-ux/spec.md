# Spec Delta

## ADDED Requirements

### Requirement: GRPO 数据阻断提供可执行恢复路径
当 GRPO 数据因可信 reference、reward contract 或登记证据不完整而不可训练时，工作台 MUST 显示具体阻断原因，并提供前往受支持 GRPO 数据派生流程的恢复入口；不得将不合规数据伪装为可训练。

#### Scenario: GRPO 数据均不可用
- **WHEN** 当前登记的 GRPO 数据集均未通过训练资格检查
- **THEN** 工作台显示各数据集的具体原因，并提供进入可信 seed 派生页面的操作

#### Scenario: 用户检查单个不合规数据集
- **WHEN** 用户查看缺少 reward/reference 契约或登记证据的 GRPO 数据集
- **THEN** 数据项保持不可选择，且阻断说明指出缺少的契约或登记证据

### Requirement: GRPO 数据派生往返保留训练上下文
数据模块与 GRPO 训练工作台往返时 MUST 保留目标阶段、所选 profile、父模型和新登记 dataset ID；返回后 MUST 重新预检，且 MUST NOT 自动启动训练。

#### Scenario: 从 GRPO 工作台进入数据准备
- **WHEN** 用户从 GRPO 阻断恢复入口进入数据模块
- **THEN** 数据准备页面保留返回到 GRPO 的上下文，包括兼容的 profile 和父模型

#### Scenario: GRPO 数据登记完成并返回
- **WHEN** 用户将可信 seed 派生结果登记为 GRPO 数据并选择返回训练
- **THEN** 工作台预选精确 dataset ID 及仍兼容的 profile/父模型，清除旧预检，并等待用户重新预检和显式启动
