# Spec Delta

## Purpose

让用户把同一训练阶段的多个已登记数据集组成一个可复现的数据输入，同时明确组合比例、数据角色、质量证据和多父级血缘。组合结果作为普通 dataset artifact 被训练消费，不依赖其来源是开源、本地导入还是模型生成。

## ADDED Requirements

### Requirement: Stage-compatible datasets can be composed
系统 MUST 允许用户选择两个或更多已登记数据集组成同阶段派生数据集，并在构建前验证各输入的阶段 schema、记录可用性和允许的训练角色。

#### Scenario: 组合不同来源的同阶段 SFT 数据
- **WHEN** 用户选择本地导入和模型生成的 SFT 数据集
- **THEN** 系统按 SFT canonical schema 校验两者并允许继续配置组合；来源差异不会改变训练阶段语义

#### Scenario: 组合阶段不匹配或隔离角色数据
- **WHEN** 用户混入不同训练阶段的数据集，或含 diagnostic_dev/final_holdout 祖先的数据集
- **THEN** 组合被阻止并指出冲突的 dataset ID、阶段或隔离角色

### Requirement: Composition policy is explicit and reproducible
组合 MUST 使用用户可查看的拼接或加权采样策略；加权采样 MUST 记录每个父数据集的权重及抽样 seed，且所有权重有效并归一化。

#### Scenario: 拼接父数据集
- **WHEN** 用户选择拼接策略
- **THEN** 输出包含按声明顺序拼接的可用记录，并记录父级选择和输出记录数

#### Scenario: 加权采样父数据集
- **WHEN** 用户选择加权采样并给出目标预算
- **THEN** 系统按登记权重和 seed 产生确定的记录选择；相同父内容与配置得到相同输出内容

### Requirement: Composition output is a new immutable dataset
成功组合 MUST 创建独立 dataset ID 和内容摘要，登记组合配置/协议摘要、所有父 dataset ID、阶段、数量单位、构建 run 与质量报告；修改父内容或组合配置 MUST 产生新身份。

#### Scenario: 同一组合配置重复执行
- **WHEN** 父内容和组合配置完全相同
- **THEN** 派生数据内容身份可复现，但每次实际构建保留独立 run_id

#### Scenario: 组合配置或父内容改变
- **WHEN** 组合权重、采样 seed、预算、策略或任一父内容发生变化
- **THEN** 系统生成不同的派生数据集身份，不覆盖既有 dataset ID 或报告

### Requirement: The composed artifact is reviewed as a whole
系统 MUST 对实际组合内容执行目标阶段的版本化质量评估，并展示各父数据集贡献、组合后质量分母、重复/污染检查覆盖及未测项目；父级质量报告不能替代组合结果的质检。

#### Scenario: 父级质量状态不同
- **WHEN** 组合输入分别为通过、未测或探索性状态
- **THEN** 组合报告保留父级状态并对实际输出重新执行适用检查，不将父级状态折叠成统一 PASS

#### Scenario: 合格组合数据少于目标数量
- **WHEN** 组合产物非空但低于用户期望数量
- **THEN** 系统报告实际分母与数量差异；数量不足本身不替代质量结论或强制判失败

### Requirement: Training consumes one registered composition identity
训练入口 MUST 接收一个已登记的组合 dataset ID，并在预检与训练 lineage 中绑定该 ID；训练结果必须能从该数据集追溯组合规则和全部父级。组合编辑与登记属于数据模块，训练入口负责选择已登记版本或导航到数据创建。

#### Scenario: 使用组合数据启动训练
- **WHEN** 用户在阶段 Launcher 中选择有效组合数据集
- **THEN** 预检显示实际组合规模、策略和父数据集，训练 run 仅引用该 dataset ID，其 lineage 可递归追到各输入

#### Scenario: GRPO 输入 reward/reference 不兼容
- **WHEN** 组合 GRPO 数据集中的 reward/reference 契约不一致或不可执行
- **THEN** 系统阻止登记为可训练组合或阻止训练预检，并列出不匹配的契约证据

#### Scenario: 从训练入口准备新组合
- **WHEN** 用户从训练配置选择创建组合数据集
- **THEN** 系统带阶段和返回上下文进入数据模块，登记成功后预选新的 dataset ID 返回训练表单，不要求手工复制身份且不自动启动训练
