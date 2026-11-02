# 提案：多阶段普通生成与定向补数闭环

## Why

目前 SFT 已有较完整的普通模型生成入口，CPT 与 Preference 存在部分生成底层能力，GRPO 主要通过可信种子派生数据；不同阶段的 Web 能力声明、真实可用入口与文档不一致。用户需要在每个阶段都能完成普通生成或定向补数，并追踪到可审查数据、训练与评测，而不是把底层函数或模拟结果当作完整闭环。

## What Changes

- 为 CPT、SFT、Preference（供 DPO/SimPO 共用）和 GRPO 提供阶段化普通模型生成：配置主题或可选训练种子、Provider 和预算，显式确认调用，查看进度、候选、失败/partial、质量与来源，审核后登记并带入对应训练和评测。
- 定向补数通过同一阶段生成服务、任务、审核、登记及血缘链路执行；保留目标、证据、计划版本、训练和复测关系。定向数据也可通过导入、筛选、组合或可信种子派生，不强制模型生成。
- 各阶段分别约束 canonical 格式及证据：CPT 领域语料质量；SFT 指令/回答；Preference 同题候选与可审计排序；GRPO prompt 与可信 reference/可执行 reward 契约。缺乏可信证据保持未验证或阻断，不能由模型自评或格式通过冒充验证。
- 保持显式调用和预算、脱敏、失败与恢复、holdout 隔离、独立 dataset/run/model/evaluation 身份；多训练数据先登记组合数据集，训练仍消费准确 dataset ID/hash。真实 Provider/GPU 验收与离线模拟分开报告。
- 与现有 `targeted-data-stage-closure` 的关系：本变更负责跨阶段共用生成能力及普通/定向入口接入和端到端交接；既有变更继续负责定向计划历史、效果观察与其未完成的 6.2/6.4 验收。相关接口和测试共享证据，不复制实现或通过勾选推断其他变更完成。本提案不修改该变更的历史状态。

## Capabilities

### New Capabilities

无；复用现有数据生成、定向计划和质量/血缘能力。

### Modified Capabilities

- `synthetic-data-generation`：增加可从工作台显式运行的多阶段普通生成、阶段证据和生成到训练/评测交接。
- `targeted-data-planning`：定向计划复用相同阶段数据能力并贯穿独立生成或非生成来源、训练与可比复测。

## Impact

- 后端：`src/onegpu_posttrain/synthetic/`、`data/targeted_*`、`data/grpo_seed.py`、`ui/server.py`、阶段 capability、Launcher 和数据/模型 lineage。
- 前端：数据工作台生成入口、定向补数入口、执行状态与 Runs 交接；保留既有 SFT 使用与历史记录兼容。
- 测试及文档：阶段/模式矩阵、前后端交互、离线/真实验收证据和中文操作说明；真实付费调用及 GPU 执行必须另经明确确认。
