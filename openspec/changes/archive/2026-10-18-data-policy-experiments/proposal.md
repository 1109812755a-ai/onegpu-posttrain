# Proposal

## Why

用户需要先用现有小规模数据完成单卡后训练流程，再调整数据规则并观察数据质量、覆盖和模型指标如何变化。当前 Mini 正式验收与探索性运行没有统一的产品表达，OpenR1 拒绝计数还混合了语义资格与长度限制，前端只能读取历史报告，难以解释低产出或组织可复现的策略对照。

## What Changes

- 新增版本化数据策略协议，明确可配置的质量规则、token 长度/数据量预算及固定的数据完整性要求；首个切片支持 SFT 与 Preference，CPT/GRPO 沿用既有路径并显示适用性边界。
- 提供小规模探索配置：按实际合格数据量运行并保留告警，不以正式 Mini 配额阻断探索；正式 Mini 仍按原 profile 验收，不将未验证样本或少量样本结果标成正式基线。
- 拆分 OpenR1 的正确性、未知标签、完整性、解析、长度及 pair 构造诊断；保留首拒绝计数，同时以明确分母报告其他诊断，不能把超长等同于低质量。
- 为 A/B 策略固定来源版本、题目分组和评测集；支持整条流水线效果与匹配预算的单变量策略对照，产出已有设计约定的 `data_effect_experiment.json` / `data_effect_report.json`。
- 在现有 Data 页面增加规则解释、模板、A/B 差异、有限本地预览、策略/实验保存、CLI 配置导出及结果比较。预览/保存采用显式本地可写模式；默认只读行为继续可用，训练仍由现有 CLI 显式执行。
- 明确预览只使用已配置本地原始来源、源缓存和本地 tokenizer；缺少输入时显示缺失原因，不触发下载，不用已筛选数据推算被删候选。
- 在设计和后续文档更新任务中串联下一变更 `targeted-synthetic-data`：评测识别薄弱场景 → 定向生成候选 → 质量策略 → 训练 → 固定评测复测。当前变更实现实验与数据协议基础，生成调用由后续独立变更交付。

## Capabilities

### New Capabilities

- `dataset-policy`: 版本化的数据资格/资源策略、可解释拒绝诊断、本地预览、探索性构建与固定题目分组。
- `data-policy-comparison`: 事先登记 A/B 控制条件，关联数据/run/model/evaluation，检查兼容性并报告数据、模型和成本变化。
- `data-policy-editor`: 前端的分阶段规则编辑、对照配置、明确范围的预览、可选本地保存与可复现实验导出。

### Modified Capabilities

无。当前主 `openspec/specs/` 为空；复用既有变更 `implement-lineage-v1`、`mini-data-readiness`、`cache-raw-dataset-sources` 中定义的身份、验收和来源契约，不复制这些通用要求。历史 builder 的默认语义保持既有协议；新策略路径使用新协议版本。

## Impact

- 数据层：`src/onegpu_posttrain/data/openr1.py`、`build.py`、`evidence.py`、`source_cache.py` 的调用边界，以及新增 policy/comparison 类型与共享执行服务；不增加数据库或重型质量框架。
- 执行与报告：`cli.py`、既有 lineage 和 reporting；现有训练/评测命令消费新 manifest，并保留探索用途与策略身份。
- UI：`src/onegpu_posttrain/ui/server.py` / `catalog.py`、`web/src/` 与 API/数据/前端交互测试；新增受限的策略保存和预览接口，不引入训练进程控制或任务调度。
- 文档：实施时更新中英文前端路线、ADR、数据处理效果设计、开发/开源计划和中文操作指南，显式记录用户同意提前交付的窄范围交互能力；冻结 v0.1.1 设计保留为历史协议参考，新增协议说明差异。
- 外部行为：本变更不下载数据、不调用付费模型、不声称已有训练收益；实测效果需要用户后续明确执行训练与评测。
