# Spec Delta

## Purpose

让用户在本地 Launcher 中连续观察已批准的训练过程，并在训练产物核实后显式启动统一评测。该能力保持单卡、可追溯和可复现边界，不提供通用任务调度。

## ADDED Requirements

### Requirement: Training and evaluation logs update during execution
活动运行页 SHALL 自动更新当前阶段的真实训练或评测日志，并展示可从日志可靠解析的进度；无法解析时 SHALL 明确显示未知，且不得要求用户刷新页面才能看到新日志。

#### Scenario: Training emits new progress
- **WHEN** SFT 训练进程写入新的 step 或训练指标
- **THEN** Launcher 在页面保持打开时自动显示最新日志与可解析的 step/total 信息

#### Scenario: Progress format is unsupported
- **WHEN** 当前日志没有受支持的进度格式
- **THEN** 页面继续显示真实日志，并将进度标为未知而不是伪造百分比

#### Scenario: Browser reload during an active run
- **WHEN** 用户刷新 Launcher 页面或重新打开它
- **THEN** 页面恢复同一运行的实际状态和有界日志尾部，不创建或重启训练进程

### Requirement: Evaluation is available only after verified training success
Launcher SHALL 仅在训练与所需模型合并成功且目标模型产物已核实后，提供该模型的评测预检入口。

#### Scenario: SFT and merged model are complete
- **WHEN** 训练 Run 成功并登记了可读取的合并模型
- **THEN** 页面展示“运行评测”操作并绑定该确切模型产物

#### Scenario: Training did not produce a verified model
- **WHEN** 训练失败、OOM、中断、合并失败或合并模型缺失
- **THEN** 页面不允许启动该模型的统一评测，并展示真实阻断状态

### Requirement: Evaluation requires a fixed preflight and explicit confirmation
评测启动 SHALL 使用版本化、固定且有界的项目评测协议；页面 MUST 展示协议、基准覆盖、样本限制、资源/数据准备提示与输出位置，并要求用户确认后才执行。

#### Scenario: User reviews and confirms evaluation
- **WHEN** 目标模型和评测预检通过，用户确认当前评测摘要
- **THEN** 系统只执行该固定评测，不接受浏览器提供的任意模型路径、命令、参数或环境变量

#### Scenario: Evaluation preflight is blocked
- **WHEN** 评测依赖、协议、模型、数据来源或输出条件不满足
- **THEN** 页面列出阻断原因且不创建评测进程或成功结果

#### Scenario: Training or evaluation already occupies the launcher
- **WHEN** 本地 Launcher 已有活动 GPU 运行
- **THEN** 系统拒绝并发启动评测或训练，并保留当前运行状态

### Requirement: Evaluation has an independently observable run lifecycle
每次确认的评测 SHALL 作为独立、持久化的评测运行执行，具有自己的唯一 `run_id` 和评测产物身份，并在页面展示真实状态、时长、日志、输出及失败/中断信息。

#### Scenario: Evaluation is running
- **WHEN** 评测进程正在执行
- **THEN** 页面显示评测运行状态和自动更新的评测日志，而不覆盖原训练 Run 的状态或日志

#### Scenario: Evaluation completes, fails, or is interrupted
- **WHEN** 评测到达任一终态
- **THEN** 页面展示已核实的终态和实际生成的产物；UI 服务关闭导致未完成时记录为中断且不自动续跑

### Requirement: Evaluation results preserve metric and lineage evidence
评测结果 SHALL 展示可用指标、样本分母或明确的未知状态、协议版本、评测状态及来源链接，并关联被评模型和产生该模型的训练/合并运行；失败或部分结果 SHALL NOT 被呈现为完整成功分数。

#### Scenario: Evaluation result resolves to its SFT model
- **WHEN** 评测生成指标且模型/训练血缘可解析
- **THEN** 页面可从评测结果追溯到确切模型 ID/摘要、SFT 训练 Run 和所用协议

#### Scenario: Metric coverage is partial or unknown
- **WHEN** 某基准失败、缺少指标或样本分母不可得
- **THEN** 页面标明缺失、失败、部分或未知，不填入零分或推断分数
