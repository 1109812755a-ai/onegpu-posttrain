# OneGPU-PostTrain 整体开发计划

本文是项目路线的中文入口；英文版[Development Plan](DEVELOPMENT_PLAN.md)保留更完整的版本契约和执行证据。愿景是先成为可复现的单卡后训练实验室，帮助用户理解 CPT、SFT、偏好优化和 RL 改变了什么；长期演进为可落地的平台级开源研究项目。

## 范围原则

- v0.1 以开源、CLI、本地文件和单张消费级 GPU 为核心；数据库、分布式调度、多用户、多卡和自动 HPO 不是 MVP 前置条件。
- 早期前端可选、本地优先，以只读 artifact 浏览和受限草稿编辑为主；不暗示页面在编排训练。
- provider 生成、训练和复测均须由用户显式执行；UI 不直接发起收费请求，不接收或保存 API 密钥值。
- v0.2 才评估受限本地 launcher；OVAT 确有批量需求后再加入串行队列；重型调度/部署在 v1.0 目标下按真实工作负载逐步设计。
- 所有数据与模型效果主张都必须能追到数据版本、处理/质量证据、run、parent model、评测协议、预算、seed 和环境。没有证据时显示未知或未评估。

## 版本阶段

| 阶段 | 目标与交付 | 当前状态 |
|---|---|---|
| 0–2：设计、Smoke 实现与 RTX 4060 执行 | 固化阶段边界，跑通 Base → CPT → SFT → DPO/SimPO/GRPO smoke，记录资源、指标和失败证据。 | 完成；Smoke 可行性不等同于质量提升。 |
| 3：Smoke Hardening 与基础协议 | 修复 smoke 暴露的偏好样本、GRPO 截断等问题；落地 lineage v1 与阶段数据质量语义。 | 完成但保留小样本与测量限制。 |
| 4：Mini | 建立 5M CPT、5K SFT、2K preference、2K GRPO 目标数据和完整模型/评测血缘。 | 缩量探索链路已跑通：5M CPT → 307 条/1024-token SFT → DPO/SimPO/GRPO 四分支及同协议小样本评测；评测与模型血缘已核验。仍非正式 Mini：Preference 仅 4 对、数据质量审核待完成，GRPO 仅 3 步且截断率 100%；2,048-token 完整 SFT pilot OOM。 |
| 5：必需 OVAT | CPT 数据量、SFT LR、DPO vs SimPO、GRPO reward design 的受控比较。 | 计划中；需满足数据、seed 和验收要求。 |
| 6：v0.1 发布 | 发布单卡基线、文档和可追溯实验结果。 | Mini、OVAT 和发布审查尚未完成。 |
| 7：v0.2 模型与实验扩展 | 评估选定 1B–3B profile、rank ablation 和受限本地 launcher。 | 规划中。 |
| 8：数据质量研究与 flywheel（v0.3/v0.5） | 匹配预算的数据清洗消融、混合/难度/来源实验、失败挖掘和 lineage-aware CI。 | 长期规划；当前定向合成能力不等于自主 flywheel。 |
| 9：Alignment / Agent / Tool RL | 扩展对齐算法和确定性工具环境。 | 规划中。 |
| 10：平台级 v1.0 | 任务生命周期/调度、资源观测、数据/模型/run/eval workflow、部署和运维。 | 最终目标，不提前引入平台基础设施成本。 |

## 提前交付的定向补数切片

`targeted-synthetic-data` 把评测发现薄弱场景后定向生成训练数据的能力提前纳入当前开发：场景逐题报告与分母、版本化计划、可选 provider/profile/prompt、候选验证与策略导出、成本/lineage、模型复测，以及本地 UI 配置草稿。首版范围是数学 SFT/Preference；fake 走离线 fixture，OpenAI-compatible 远程调用必须从 CLI 显式执行。

UI 只展示白名单摘要并导出未登记的 workbench 草稿；它不能代替正式 `TargetedGenerationPlan`，也不执行生成、训练或复测。答案验证不证明完整推理正确，生成数量不是能力提升；费用无证据时为 unknown，留出集隔离，no-improvement/退化都保留。详见[定向合成数据中文指南](../guides/TARGETED_SYNTHETIC_DATA_zh.md)和[前端路线图](../design/frontend/FRONTEND_ROADMAP_zh.md)。

## 项目管理与验收

改变行为、数据/lineage schema、评测语义、recipe 或用户流程的后续开发使用 OpenSpec；每个 OpenSpec task 以实现与测试证据为完成标准。版本发布时同步实现状态、milestone、README、changelog 和开源文档。完整开源发布计划见[中文开源计划](OPEN_SOURCE_PLAN_zh.md)，实现证据见[英文总计划](DEVELOPMENT_PLAN.md)。
