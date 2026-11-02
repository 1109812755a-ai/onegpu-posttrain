# Design

## Context

当前 SFT 生成准备把诊断报告、逐题预测、场景标签和 holdout 证据当作所有生成的前置条件，普通生成因此也会被定向补数的隔离要求阻塞。目标是让用户先生成、审核、登记并训练一批 SFT 数据，同时保留定向增强的证据标准。生成设置页还应默认使用本地已配置 provider，不将 endpoint、模型名和密钥输入变成普通用户的负担。

## Goals / Non-Goals

**Goals:**
- 提供“普通 SFT 生成”和“定向补数”两个清晰模式，分别执行适配的 readiness 检查。
- 普通模式以目标数量驱动执行，但不把数量当质量 gate，也不因凑数降低检查标准。
- 默认发现并选中凭据可用的本地 provider profile；浏览器不接收或保存密钥。
- 保持 canonical schema、适用的一致性/身份校验、人工审核、显式登记及 dataset/run/model lineage。
- UI 核心操作聚焦选择数据集/模板、确认目标数量、确认生成；预算和技术选项放在折叠区。

**Non-Goals:**
- 普通模式不声称能够定位模型薄弱能力，不依赖也不虚构诊断报告。
- 不自动判断未知样本语义正确；不因目标数量未达成而自动放宽质量规则。
- 不在打开页面、发现 profile、准备方案或刷新时调用远程 provider。
- 不在本变更中自动启动训练/评测，不引入数据库或远程凭据管理服务。

## Decisions

1. **普通与定向使用不同方案/执行记录，不伪造诊断证据。** 新增 `OrdinarySFTPlan` 与普通 SFT bundle/execution 身份，绑定普通 source dataset（及其 manifest/content 摘要）、profile、模板、协议、policy 和生成目标；普通计划不包含 report、scenario、role registry 或 protected holdout 字段。定向补数继续使用现有 `TargetedGenerationPlan` 与严格的角色/隔离验证。两条路径复用 provider request、候选 canonical 校验、质量报告、人工审核及 dataset registration 的底层实现，但各自保留清晰的模式身份和 lineage。

2. **优先使用稳定 ID，不接收路径或任意配置。** Prepare API 接受 mode、普通 dataset ID（或受支持的默认 source ID）、仅定向模式接受的 report/scenario ID、provider profile ID 和有界目标/预算；后台仅从 workspace registry 与固定默认目录解析文件。若唯一 provider 可用，则由 discovery 默认选中；用户可在多个已配置 profile 中选择。失败状态说明缺项，不接受用户提交任意路径、命令或密钥。

3. **provider 设置分为默认路径和高级维护。** 普通页面展示 provider 预设名称及凭据“已配置/缺失”状态；密钥仅从环境变量或 Git 忽略的本地配置读取。没有可用 profile 时给出短说明（复制变量名/本地配置指引）和支持的 provider 预设；不在浏览器输入 API key。endpoint、model ID、token、timeout 等高级字段默认折叠，正常生成不要求修改。

4. **目标数量是执行目标而不是 quality gate。** prepare 和 execution 记录请求目标数；provider 成功候选、格式拒绝、一致性拒绝、人工拒绝和最终登记数量分别统计。低于目标时可由用户决定是否补生成，系统不得暗中重试或降低校验门槛；用户选择较小数量仍可进行探索性审核、登记与训练，但产物需标注样本规模和实验状态。

5. **校验与 lineage 在两模式中一致且诚实。** 所有候选先校验 canonical 字段、记录身份、来源锚点和适用的答案/solution 一致性。只有存在可核验参考答案时才执行对应比对；缺少依据的语义正确性标为未验证并交人工审查。execution lineage 记录 mode、输入 dataset/template、provider profile 摘要、生成目标和结果；登记 dataset manifest 关联 execution；下游训练 run 引用 dataset ID/hash 并沿链追到来源。密钥不进入日志、bundle 或 lineage。

6. **页面只要求必要选择，危险/昂贵动作保持显式。** 默认主流程为：选择普通生成或定向补数 → 确认数据集/模板和目标数量 → 预检 → 显示成本/预算摘要 → 用户显式启动 → 查看候选和质量结果 → 人工审核 → 显式登记。高级预算/采样参数和自动解析身份摘要折叠；训练和评测是登记之后单独动作。页面打开与 prepare 始终零 provider 调用。

## Risks / Trade-offs

- [普通生成候选质量参差] → 所有候选经过格式/身份/适用的一致性检查，提供质量报告和人工审核；不把“生成成功”标成“语义正确”。
- [没有足够质量候选达到目标数] → 展示目标与实际数、拒绝原因，允许用户明确续生成；不自动放宽策略。
- [用户把普通生成误解为定向能力提升] → 模式名称和报告明确区分“常规补充”与“基于诊断的定向补数”；后一模式必须出示诊断证据。
- [本地 profile 或凭据不可用] → discovery 标明缺失的变量名和本地修复方法，且不能泄露变量值或启动调用。
- [降低设置复杂度掩盖实验参数] → 完整解析配置可折叠查看并写入 resolved config/bundle 摘要，保证可复现。

## Migration Plan

新增/调整 SFT 生成准备 UI/API 与 readiness 逻辑。既有 provider profile、workflow bundle、generation execution、dataset manifest、质量/审核流程和 CLI 保持可用；旧的定向路径继续执行原隔离校验。已有记录无需迁移，新增 execution 明确记录生成模式与目标数。若新 UI 停用，已登记 dataset 与 CLI 路径仍可继续使用。
