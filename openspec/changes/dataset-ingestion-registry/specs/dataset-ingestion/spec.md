# Spec Delta

## Purpose

让用户能够从多类来源准备适用于不同后训练阶段的数据集，并通过统一的版本、质量和血缘契约交给训练流程消费。来源类型与训练阶段是独立维度，具体可用组合由来源能力显式声明。

## ADDED Requirements

### Requirement: 来源类型与训练阶段独立声明
工作台 SHALL 将数据来源类型与目标训练阶段分开建模；每个来源适配器 SHALL 声明其支持的阶段，界面只允许选择已声明支持的 source-stage 组合。

#### Scenario: 选择受支持的来源与阶段
- **WHEN** 用户选择一个来源和其能力清单中支持的 CPT、SFT、Preference 或 GRPO 阶段
- **THEN** 工作台允许继续准备该阶段数据，并记录来源身份及所选阶段

#### Scenario: 选择不支持的来源与阶段组合
- **WHEN** 用户选择来源未声明支持的阶段
- **THEN** 工作台解释该组合当前不支持且不启动获取、生成或构建，不把来源类型与阶段硬编码成同一个概念

#### Scenario: 查看模型生成阶段支持范围
- **WHEN** 用户查看模型生成来源的能力
- **THEN** SFT 是首个支持阶段，Preference 在其后接入并作为 DPO/SimPO 数据；CPT 与 GRPO 显示为未支持，直到分别完成专门设计和验证，不允许沿用 SFT 生成流程伪装支持

### Requirement: 来源准备输出版本化阶段数据集
每次来源获取或生成及其阶段化处理 SHALL 产生可审计的构建证据，并在真实数据产物持久化后登记版本化 dataset ID。

#### Scenario: 开源来源产生阶段数据
- **WHEN** 已接入的开源来源适配器按受支持阶段读取固定或可得版本并构建训练数据
- **THEN** 数据集记录来源标识、revision 可用性、原始输入摘要、阶段处理版本、输出摘要和构建 run_id；缺失 revision 明确标未知

#### Scenario: 用户导入本地阶段数据
- **WHEN** 用户上传符合声明阶段 canonical schema 的 JSONL
- **THEN** 系统保留原文件摘要与来源/许可声明状态，并将有效内容登记为该阶段的版本化 DatasetArtifact

#### Scenario: 模型生成候选进入数据集工作台
- **WHEN** 用户明确执行已支持阶段的模型生成并通过该阶段输出格式/验证流程
- **THEN** 生成执行、provider/model/template/config 证据及生成候选通过共同 dataset ID 和 lineage 进入工作台；保存草稿或浏览页面不发起生成请求

### Requirement: 本地导入使用阶段化 canonical JSONL
本地导入 SHALL 要求用户声明目标阶段，并逐行使用该阶段项目 canonical schema 验证 JSONL。

#### Scenario: 导入 CPT、SFT、Preference 或 GRPO
- **WHEN** 用户为文件声明 `cpt`、`sft`、`preference` 或 `grpo` 并提交 JSONL
- **THEN** 系统按 CPT text、SFT messages、Preference prompt/chosen/rejected、GRPO prompt/answer 语义校验；Preference 数据可以被 DPO/SimPO recipe 选择

#### Scenario: 文件不是有效 JSONL 或阶段结构不符
- **WHEN** 文件有无法解析行、缺少必需字段或含 schema 不允许字段
- **THEN** 系统拒绝注册并报告有界的行号/字段原因，不静默跳过、自动猜字段或改写原记录

#### Scenario: 上传文件为空或超过限制
- **WHEN** 文件没有记录或超过系统声明的上传字节/行数上限
- **THEN** 系统拒绝导入并显示实际值及限制，不创建可训练数据集

### Requirement: 数据集 ID 和血缘连接来源、阶段构建与训练 Run
每个数据集版本 SHALL 记录唯一身份、阶段、来源类型/引用、原始及输出摘要、处理版本、数量、质量报告及其构建 Run，并允许训练 Run 通过明确 dataset ID 消费。

#### Scenario: 从源数据沿 lineage 回溯训练数据
- **WHEN** 用户查看某训练模型或 Run 的数据 lineage
- **THEN** 可以从训练 Run 追到被消费 dataset ID/hash，再追到阶段化构建 Run、数据来源和可得来源内容摘要

#### Scenario: 重复或改变内容的导入
- **WHEN** 相同阶段和规范化内容再次导入，或来源/规则改变导致内容改变
- **THEN** 相同内容可复用稳定身份且不覆盖产物；输出内容改变产生新 dataset ID 和摘要

#### Scenario: 来源事实缺失
- **WHEN** 无法取得来源名称、revision 或记录级祖先
- **THEN** manifest 标记 unknown/not_provided 或 dataset_only，不以文件名、行号、用户声明或哈希伪造来源证据

### Requirement: 格式、质量、来源与许可状态分别评估
工作台 SHALL 分开展示结构校验、数据质量检查和来源/许可信息完整度；格式有效或数量达标不得自动代表质量通过。

#### Scenario: 格式通过但质量含失败或未测项
- **WHEN** canonical schema 有效但重复、答案验证或其他质量检查失败/未测量
- **THEN** 保留并展示各自真实状态和覆盖，未执行检查标记 not_measured，不把 schema 通过升级为质量通过

#### Scenario: 数据量低于正式目标
- **WHEN** 非空、格式有效的数据集未达到某 Mini/实验建议数量
- **THEN** 工作台保留数据并显示实际数量与 readiness 告警，不因推荐目标自动阻止探索训练或宣称正式验收通过

### Requirement: 来源或许可未知时允许注册并要求本地训练确认
缺失来源或许可信息 SHALL NOT 阻止本地数据集登记；许可未知的数据用于本地训练前 SHALL 要求用户针对该数据集明确确认使用权，确认不等同于许可核验或公开分发授权。

#### Scenario: 注册来源或许可信息缺失的数据
- **WHEN** 数据格式有效但用户没有提供来源、revision、license 或 attribution
- **THEN** 系统允许本地登记并将对应信息标为 not_provided/unknown，不做推断

#### Scenario: 用户确认将未知许可数据用于训练
- **WHEN** 用户为指定 dataset ID 明确确认有权本地使用并启动训练
- **THEN** 训练 Run lineage 引用该 dataset ID/hash 和确认事件，许可状态仍是未知/用户声明未核验

#### Scenario: 没有本地使用确认
- **WHEN** 训练流程选择许可未知的数据但没有该数据集的明确确认
- **THEN** preflight 阻止启动并解释所需确认，不修改数据集的许可元数据

### Requirement: 训练入口只消费已登记且配方兼容的数据
训练入口 SHALL 根据配方阶段列出已登记、非空且格式兼容的数据集，并在解析配置和训练 Run lineage 中固定其 dataset ID 与内容摘要。

#### Scenario: 选择匹配阶段的数据集
- **WHEN** 用户为训练 recipe 选择支持阶段的数据集并完成所需本地使用确认
- **THEN** preflight 展示 dataset ID、hash、数量、质量和来源/许可状态，实际 Run 消费相同身份与摘要

#### Scenario: 选择不兼容或未登记的文件
- **WHEN** 用户选择阶段不匹配的数据集或绕过登记直接给任意文件路径
- **THEN** 训练入口拒绝启动并说明兼容阶段或注册步骤

### Requirement: 上传内容被限制在工作区且不覆盖既有产物
导入系统 SHALL 将上传视为不可信字节流，只在配置工作区允许的数据目录内安全落盘，并限制输入规模和失败响应。

#### Scenario: 上传文件名包含越界路径
- **WHEN** 文件名、请求字段或路径表达试图访问工作区之外的位置
- **THEN** 系统拒绝请求且不在越界位置读取或写入

#### Scenario: 导入中断或失败
- **WHEN** 校验、磁盘写入或提交过程失败
- **THEN** 系统不留下被 catalog/训练入口误认为已注册的半成品，且不覆盖先前版本
