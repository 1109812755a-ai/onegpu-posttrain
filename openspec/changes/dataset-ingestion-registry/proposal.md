# Proposal

## Why

当前数据工作流以若干专用来源和构建脚本为中心，用户难以用同一视图理解“数据从哪里来、如何变成某阶段训练集、质量如何、哪个训练 Run 消费了它”。需要将产品提升为数据来源与数据集工作台，让来源类型和训练阶段成为可独立组合、可追溯的两条维度。

## What Changes

- 建立统一工作台主链路：来源（开源 / 本地导入 / 模型生成）→ 阶段化处理（CPT / SFT / Preference / GRPO）→ 质量评估 → 版本化 Dataset ID → 训练 Run 消费。
- 首个 MVP 打通四阶段的数据格式与登记链路：按项目 canonical JSONL 导入并注册，保留原文件摘要、可得来源/许可信息、格式校验、质量报告和完整数据血缘。
- 建立显式 source × stage 支持矩阵；不同来源只声明实际支持的阶段，不要求每种来源首版都覆盖所有阶段。
- 模型生成来源按阶段分步接入：先支持 SFT，再接 Preference（供 DPO/SimPO）；GRPO 生成须待可信、可执行的 reward/reference 方案，CPT 生成另行设计，不复用数学题生成流程。
- 内置开源来源、上传来源和生成来源最终汇聚到相同数据集身份/质量/血缘视图；不以格式通过冒充质量通过，也不编造未知来源或许可。
- 训练入口按 recipe 阶段选择兼容的已登记 dataset ID；保留现有 CLI 与显式执行语义，不把本变更扩成通用任务调度平台。

## Capabilities

### New Capabilities
- `dataset-ingestion`: 从多个来源准备各阶段数据集、评估质量并登记可供训练消费的版本化数据产物。

### Modified Capabilities

## Impact

- 数据层：canonical schema、来源描述、阶段处理、manifest/quality report 与数据/模型 lineage。
- 来源适配：复用已有开源数据构建器、本地 JSONL 导入，以及现有合成数据生成/验证能力；建立明确兼容矩阵，不重写各 producer。
- UI：从“定向补数”单一入口扩展为数据来源与数据集工作台，可按来源和训练阶段查看、导入/准备、检查质量并追踪训练消费关系。
- 训练：配方选择显式绑定 dataset ID、内容摘要和来源/许可使用确认；当前 Mini SFT Launcher 先接入兼容 SFT 数据，其他阶段通过对应 CLI 配方消费。
- v1 范围不包含 CSV/Parquet 字段猜测、远程对象存储、数据库 registry、调度器、GRPO reward 平台或 CPT 合成语料生成方案。
