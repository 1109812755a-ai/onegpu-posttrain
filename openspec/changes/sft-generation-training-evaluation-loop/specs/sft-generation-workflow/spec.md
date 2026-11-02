# Spec Delta

## Purpose

让用户从本地工作台显式生成 SFT 数据、检查实际产出并登记为可追溯数据集，再使用该数据集训练并查看对应评测，形成不依赖固定数据数量门槛的可复现实验闭环。

## ADDED Requirements

### Requirement: SFT 生成计划可配置且可预检
工作台 MUST 允许用户选择已登记的 SFT 训练种子、provider profile、提示模板、目标可用记录数和请求/输出预算，并在执行前显示解析后的配置摘要。

#### Scenario: 查看或预检计划
- **WHEN** 用户打开、编辑或预检 SFT 生成计划
- **THEN** 系统显示数据/模型/模板身份、目标和硬预算，不发出 provider 请求

### Requirement: SFT 生成仅由明确操作启动
工作台 MUST 在用户明确确认执行后，才可启动本地 SFT 生成任务；页面关闭或刷新不得隐式重启任务或清除其持久化状态。

#### Scenario: 用户确认开始生成
- **WHEN** 用户确认已选数据、provider、请求预算与可能费用后启动计划
- **THEN** 系统创建独立执行身份并运行该计划，界面显示状态、日志摘要、进度及取消/终止边界

#### Scenario: 页面只是浏览或编辑
- **WHEN** 用户浏览或修改未执行的生成计划
- **THEN** provider 调用数保持为零，草稿明确标记为未执行

### Requirement: 实际样本产出和计划目标如实呈现
SFT 生成工作流 MUST 分别显示目标数、生成候选数、验证/筛选后可用数、请求预算消耗和短缺原因；目标数不得作为数据质量通过或失败的依据。

#### Scenario: 生成完成但样本数低于目标
- **WHEN** 生成因预算、provider 错误、验证或筛选而未达到目标
- **THEN** 系统保留 partial 产物并报告各环节实际数量、验证状态、短缺及停止原因

### Requirement: 经验证的输出登记为独立 SFT 数据集
工作流 MUST 将用户选择保留的 canonical SFT 记录登记为新 dataset ID，保存内容摘要、来源许可状态、质量检查状态、输入祖先、生成配置和执行 lineage；不得改写父数据集或将未运行的检查显示为通过。

#### Scenario: 登记可用的生成输出
- **WHEN** 用户审阅并确认一份非空、符合 SFT canonical schema 的输出
- **THEN** 系统创建独立 dataset ID 和 manifest，保留 model_generated 来源、父 seed、实际记录数及生成/处理报告引用

#### Scenario: 输出为空或格式不合法
- **WHEN** 生成输出为空或不符合 canonical SFT schema
- **THEN** 系统拒绝将其登记为可训练数据集，并保留可诊断的失败证据

### Requirement: Mini SFT Launcher 可消费已登记的生成数据集
Launcher MUST 允许用户选择已登记且阶段兼容的 SFT 数据集，并在训练前显示 dataset ID、实际数量、来源、质量/许可状态及训练长度兼容性。

#### Scenario: 使用生成数据集启动训练
- **WHEN** 用户选择兼容的非空 SFT dataset ID 并确认训练
- **THEN** 训练 Run 明确引用该 dataset ID 和摘要，模型产物 lineage 可追溯到输入数据及 parent model

#### Scenario: 数据不足但用户选择探索运行
- **WHEN** 数据集未达到 Mini 建议数量但通过训练前完整性和配方兼容检查
- **THEN** UI 如实显示样本量和探索限制，数量不足本身不阻止用户明确启动小规模流程

### Requirement: 训练后的评测沿血缘关联
工作流 MUST 允许用户对已完成的 SFT 模型明确启动受支持评测，并展示评测状态、协议、结果和其关联的训练 Run、模型及 dataset ID。

#### Scenario: 对训练产物启动评测
- **WHEN** 用户选择完成的 SFT 模型并确认评测配置
- **THEN** 新评测记录引用模型身份、训练 Run、实际训练 dataset ID 和评测协议，结果可从 Runs/比较页面追溯

#### Scenario: 评测尚未运行或失败
- **WHEN** 评测没有启动或执行失败
- **THEN** UI 显示待执行或失败及原因，不展示虚构指标或模型收益结论
