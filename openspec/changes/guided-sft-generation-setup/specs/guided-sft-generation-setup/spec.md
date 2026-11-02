# Spec Delta

## Purpose

让用户先以少量操作完成普通 SFT 数据生成闭环；在有可靠评测证据时，再选择定向补数。无论模式如何，都保留格式与质量校验、人工审核、数据登记及可追溯血缘。

## ADDED Requirements

### Requirement: 普通 SFT 生成不依赖诊断报告
工作台 MUST 支持不指定诊断报告或薄弱场景的普通 SFT 生成。该模式使用兼容的已登记 SFT 数据集作为训练主题来源，并使用项目默认生成模板；缺少诊断证据不得阻断普通模式。

#### Scenario: 用户准备普通 SFT 生成
- **WHEN** 用户选择普通 SFT 生成并选择兼容的已登记 SFT 数据集
- **THEN** 系统允许其继续指定目标生成数量并准备方案，不要求场景报告、逐题预测、角色注册表或最终 holdout

#### Scenario: 普通模式没有已登记源数据
- **WHEN** workspace 中没有兼容的已登记 SFT 数据集，但存在项目默认生成模板和已配置 provider
- **THEN** 系统不得启动普通生成，并提示先导入或构建 SFT 种子数据集；不得虚构生成问题的来源锚点

### Requirement: 定向补数必须由诊断证据驱动
定向补数 MUST 作为与普通生成分开的可选模式；只有当匹配的评测身份、逐题结果、场景标签、训练来源和污染隔离证据满足要求时，系统才可启用该模式。

#### Scenario: 评测只有总体分数
- **WHEN** 用户选择定向补数，但最近一次评测没有题目级预测或可靠场景映射
- **THEN** 系统不得推测薄弱场景，须解释缺失证据；普通 SFT 生成仍可独立使用

#### Scenario: 诊断与 holdout 身份冲突
- **WHEN** 定向补数使用的诊断/holdout 与训练来源存在 ID、祖先或污染冲突
- **THEN** 系统拒绝准备该定向方案并解释冲突，不自动更换角色或放宽隔离规则

### Requirement: 目标数量不是质量门槛
系统 MUST 允许用户指定生成目标数量，并将其作为生成停止/完成目标，而非数据质量、训练 readiness 或登记的唯一门槛。系统 MUST 如实报告请求数、成功候选数、拒绝数和最终登记数；不得为凑数降低质量规则。

#### Scenario: 生成量少于目标
- **WHEN** provider 调用或质量过滤后得到的候选少于用户指定目标
- **THEN** 系统报告实际数量和拒绝/失败原因，允许用户审核已生成候选或明确补充生成，不得自动放宽质量阈值

#### Scenario: 小规模实验
- **WHEN** 用户选择小于正式计划样本量的生成目标
- **THEN** 系统仍允许其继续审核、登记和训练，并将规模作为实验上下文记录，不因样本数不足宣称正式质量验收通过

### Requirement: 前端默认复用 provider 且不暴露密钥
工作台 MUST 优先自动选择凭据已配置的兼容 provider profile；用户不得被要求在普通生成流程中填写 API key、endpoint 或模型名。密钥只能从本地环境变量/被忽略的本地配置读取，接口和页面 MUST NOT 返回密钥值。

#### Scenario: 唯一 provider 已配置
- **WHEN** 发现唯一凭据可用且兼容的 provider profile
- **THEN** 系统默认选中该 profile，并让用户确认使用；页面主要展示数据集和目标生成数量，预算及技术参数默认折叠

#### Scenario: 无凭据可用的 provider
- **WHEN** workspace 没有可用凭据的 provider profile
- **THEN** 系统展示简短、可执行的本地配置说明和支持的 provider 预设，不在浏览器索取或保存密钥；不得发起 provider 请求

#### Scenario: 页面发现与准备
- **WHEN** 页面打开、刷新、自动发现或准备方案
- **THEN** provider 调用数保持为零，密钥值不出现在响应、日志或 lineage 中

### Requirement: 保留显式调用、审核与 lineage
简化配置 MUST NOT 自动触发 provider 请求、数据集登记、训练或评测。普通与定向生成均 MUST 执行 canonical SFT 格式、身份及适用的一致性校验；未知的语义正确性不得标记为已验证。生成候选在人工审核和显式登记前不得作为已登记训练数据。生成执行、输入来源、provider profile 摘要、登记数据集及后续训练模型之间 MUST 保留可追溯 lineage。

#### Scenario: 启动 provider 生成
- **WHEN** 用户检查方案并显式确认启动
- **THEN** 系统才发出 provider 请求，并记录 execution 身份、输入数据/模板身份、profile 摘要、目标数、实际产出与校验结果

#### Scenario: 候选审核与登记
- **WHEN** 用户审核生成候选并显式登记
- **THEN** 系统生成 canonical 数据文件、质量报告和 dataset ID，关联生成 execution 与来源；未通过强制格式/身份/一致性检查的样本不得静默进入训练数据

#### Scenario: 下游训练
- **WHEN** 已登记数据集被训练 run 消费
- **THEN** run lineage 引用明确 dataset ID 和摘要，并能追溯到生成 execution 与原始来源（若存在）
