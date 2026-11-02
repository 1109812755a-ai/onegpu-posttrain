# Proposal

## Why

定向补数目前已有诊断与计划页面、数据生成和登记能力，但可执行闭环主要集中在 SFT；其余阶段尚未从目标缺口一路接到可训练数据、真实训练、评测与效果复查。需要把该流程按阶段打通，才能让用户根据模型未达标的能力补充相应数据，并用可追溯证据判断补数是否有效。

## What Changes

- 阶段训练支持默认/内置配方及受控 schema 定义的自定义配方，并支持组合多个同阶段已登记数据集；训练输入与生成种子分开记录，数据集 ID/hash、合并顺序/策略和解析后配方摘要进入不可变训练证据。

- 将定向补数扩展为阶段化闭环：CPT、SFT、Preference（供 DPO/SimPO）、GRPO 均可从目标/证据建立计划，生成或导入候选、执行阶段质量检查、登记不可变 dataset ID，并追踪后续训练、评测和前后效果。
- 按阶段定义数据语义与质量证据：CPT 面向领域语料/知识覆盖；SFT 面向指令与答案；Preference 要求同题候选及可审计排序依据；GRPO 要求 prompts 与可执行、可信的 reward/reference。不能用通用问答生成伪装成有效的 CPT、Preference 或 GRPO 数据。
- 在前端提供统一的计划、生成/导入、审核登记、训练/评测状态及效果查看流程；复用已有本地 provider、数据集注册、Launcher、Evaluation、Run 和 lineage 能力，支持 partial 产物，不把目标数量当质量 gate。
- 保留诊断集与最终 holdout 隔离、凭据不落盘、显式外部调用确认、预算/中断恢复、数据血缘和模型血缘；缺少阶段训练配方、reward、验证器或评测证据时显示明确阻塞/未测状态，不假报成功。

## Capabilities

### New Capabilities
- `targeted-stage-workflow`: 跨 CPT、SFT、Preference/DPO/SimPO、GRPO 的阶段化定向数据到训练、评测和效果复查工作流。

### Modified Capabilities
- `targeted-data-planning`: 计划支持各阶段不同的目标定义、数据角色、父模型、预算和效果复测条件。
- `synthetic-data-generation`: 为各阶段定义独立 canonical 输出、真实验证边界和可恢复生成/登记行为。
- `stage-quality-review`: 将阶段专属质量配置、实际覆盖和登记可用性贯穿生成数据流程，不以样本数替代质量。

## Impact

- 后端：`src/onegpu_posttrain/ui/server.py`、`src/onegpu_posttrain/synthetic/`、数据 canonical schema/registration、阶段训练与评测入口、lineage/report 读取器。
- 前端：`web/src/TargetedExperiments.tsx`、`DatasetWorkbench.tsx`、`Launcher.tsx`、`Runs.tsx` 及对应页面状态/API。
- 数据与证据：阶段化 canonical JSONL、quality report、dataset manifest、generation/build/train/evaluate run、模型产物及效果报告。
- 验证：fake provider 和 CPU fixture 覆盖每阶段数据闭环；真实模型训练与指标提升作为用户运行的实际证据，不由模拟测试宣称完成。
