# Spec Delta

## Purpose

让用户针对有评测证据的能力缺口，按 CPT、SFT、Preference 或 GRPO 阶段生成/导入、审查并训练明确版本的数据，再用同协议评测和血缘证据判断变化。

## ADDED Requirements

### Requirement: Stage-specific target plan
系统 MUST 为每个补数计划固定目标阶段、目标能力/领域、证据引用、父模型、输入数据角色、数据预算、训练配方和复测/保护指标；计划变更 MUST 产生新版本。

#### Scenario: 建立 SFT 定向计划
- **WHEN** 用户基于具备分子分母的场景评测创建 SFT 补数计划
- **THEN** 计划保存阶段、诊断证据、隔离种子、父模型、预算和复测协议，不预先宣称指标提升

#### Scenario: 计划来源证据不足
- **WHEN** 选择的能力没有题目级场景证据或领域数据证据
- **THEN** 系统指出缺失证据并允许保存为未诊断草稿，但不伪造薄弱场景或自动执行生成

### Requirement: End-to-end stage lifecycle
每阶段 MUST 支持计划、生成或导入、验证、审核、登记、训练、评测及效果复查的可观察状态；不支持或未执行的环节 MUST 明确显示阻塞/未测，不得显示为成功。

#### Scenario: 数据登记后继续训练和评测
- **WHEN** 用户审核并登记非空、schema 有效的数据集
- **THEN** 后续阶段预检和训练消费该确切 dataset ID，完成后可从训练模型打开关联评测和效果报告

#### Scenario: 阶段能力尚未接通
- **WHEN** 某阶段缺少可执行生成器、训练配方、reward 或评测协议
- **THEN** 页面展示具体缺项和下一步，不能用 mock 状态冒充已完成闭环

### Requirement: Data quantity is not a quality gate
样本目标 MUST 作为用户期望值而非质量通过门槛；有效 partial 数据 MUST 按实际数量审核和登记，效果报告 MUST 使用实际训练预算。

#### Scenario: 未达到期望数量
- **WHEN** 生成达到硬预算但合格样本少于期望值
- **THEN** 报告实际候选/可用/拒绝数量与短缺原因，允许用户显式登记非空有效 partial 数据，不自动补齐或宣称目标达成

### Requirement: Stage semantics are not interchangeable
生成与导入流程 MUST 使用阶段 canonical schema 和阶段适用的验证证据；不能以通用 SFT 问答记录替代 CPT 语料、Preference pair 或 GRPO prompt/reward 输入。

#### Scenario: CPT 定向语料
- **WHEN** 用户选择 CPT
- **THEN** 计划按领域语料覆盖目标构建连续文本数据，并显示来源、重复/污染、长度及可获得的领域质量证据；不要求伪造指令问答字段

#### Scenario: Preference 数据
- **WHEN** 用户生成供 DPO 或 SimPO 使用的数据
- **THEN** 每个 pair 绑定同一训练问题的 chosen/rejected 候选和排序证据；缺少独立可信排序依据时保持未验证，不能伪造正确/错误标签

#### Scenario: GRPO 数据
- **WHEN** 用户为 GRPO 建立训练数据
- **THEN** 每个 prompt 关联可执行的本地 reward/reference 契约及其版本；缺少可信 reward 契约时阻止正式训练预检

### Requirement: Diagnostic and final holdout remain isolated
诊断开发集与最终 holdout MUST 依据数据角色和祖先隔离，不得作为训练生成输入；污染命中或祖先不明 MUST 显示并阻止声明为无污染的正式效果结论。

#### Scenario: 尝试将 holdout 用作种子
- **WHEN** 用户选取 final_holdout 或其派生记录作为生成/训练输入
- **THEN** 预检拒绝该输入并解释角色冲突及可替代的训练种子

### Requirement: Dataset and model lineage span the closed loop
每个持久化生成、导入、质量审查、训练、模型合并与评测 MUST 使用各自真实身份，并通过 dataset ID、run_id、model_id 和 evaluation ID 保持可验证连接；不得为临时状态或未产生的产物伪造节点。

#### Scenario: 从评测结果回溯生成来源
- **WHEN** 用户从模型评测结果查看来源
- **THEN** 可追溯到训练 run、模型父级、训练 dataset、质量报告及可得的来源/生成配置证据，并能校验内容摘要

#### Scenario: 定向计划身份贯穿登记、训练与评测
- **WHEN** 定向计划派生的数据集通过审查并被训练 Launcher 消费
- **THEN** 数据 lineage 保留父训练种子 dataset ID，data-build/quality/train/evaluation 记录保留同一 `targeted_plan_id + version`，且训练输入继续引用实际登记的 dataset ID；缺少或不匹配的计划身份不得补写成成功证据

### Requirement: Training recipe and dataset composition are explicit
训练流程 MUST 支持用户选择一个或多个已登记且符合阶段 schema 的训练数据集。训练数据集合 MUST 与用于定向生成的 seed 分开声明，并记录每个 dataset ID、内容 hash、顺序及采样/合并策略；多个数据集通过已登记的 dataset mixture 固化为可复现训练输入，其 lineage 必须保留各父数据集。训练流程 MUST 支持受支持的内置配方，以及基于受控 schema 的自定义参数配置；解析后的完整配置摘要和配方版本 MUST 固定到预检、训练 Run 与模型 lineage。预检 MUST 验证每个数据集、配方的阶段兼容性、质量/许可提醒、参数边界和运行能力；不兼容或不安全配置 MUST 阻止启动并给出原因。

#### Scenario: 组合多个已登记数据集
- **WHEN** 用户在一个训练阶段选择多个有效数据集并确认各自采样/合并策略
- **THEN** 系统用显式合并/采样策略生成登记的 mixture dataset；训练消费该 mixture 的精确 ID/hash，训练 Run 与模型血缘递归可回溯到每个输入数据集

#### Scenario: 选择自定义训练配方
- **WHEN** 用户提交该阶段受支持 schema 中的自定义配方配置
- **THEN** 系统校验边界并显示解析后的配置摘要，且不会接受浏览器传入任意命令或路径

### Requirement: Effect report separates observed change from causal attribution
补数效果报告 MUST 展示目标指标、预登记保护指标、分母/覆盖、协议、父模型、dataset ID 及训练资源；单臂 before/after 只能标记为流水线观察，匹配预算 A/B 才能按实际控制条件讨论策略差异。

#### Scenario: 目标指标提升但保护指标退化
- **WHEN** 补数后目标指标变化为正且任一保护指标越过预定界限
- **THEN** 同时展示提升与退化，整体状态不得标为无条件成功

#### Scenario: 只有单臂前后结果
- **WHEN** 只有补数前后两个模型而无匹配对照
- **THEN** 报告训练与数据变化及资源成本，并将结论标为观察而非合成数据的因果效果

#### Scenario: GRPO 效果目标未声明
- **WHEN** 用户创建 GRPO 计划但没有明确目标指标、方向、阈值来源、阈值或最小样本数
- **THEN** 系统可保存训练/评测记录，但效果报告不得判定目标达标或失败，并显示目标判定未配置
