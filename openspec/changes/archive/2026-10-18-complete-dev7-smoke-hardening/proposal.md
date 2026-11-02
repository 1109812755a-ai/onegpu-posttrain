# Proposal

## Why

真实 RTX 4060 Smoke 已证明训练链路能够运行，但偏好数据产出远低于目标、GRPO completion clipping 过高，且 DPO、SimPO 与两种 GRPO 模型尚未完成同协议任务评测。补齐这些证据后，Mini 才能建立在可解释、可复核的基线上，而不是把流程可运行误当成模型效果成立。

## What Changes

- 验证 OpenR1 preference train/eval 构建的目标驱动扫描、扫描上限/数据耗尽停止行为和拒绝原因统计；目标未满足时明确失败，不降低既有样本质量门槛。
- 对 GRPO completion 长度候选 512、768、1024、1536 进行受控校准，保存配置、截断率、奖励有效性、耗时及显存证据；据证据选择配置，或明确记录在当前硬件与预算下未达目标。
- 使用统一 GSM8K、MATH-500、internal-holdout 协议评测 M3-DPO、M3-SimPO、M3-GRPO-standard、M3-GRPO-lowmem，并生成以 M2 为共同基线的比较证据。
- 清理全仓 Ruff findings，并在清理后运行完整测试与 compileall 回归。
- 核对已完成的 Lineage v1 实现与项目状态文档，纠正过时的未完成状态；不回写或改写历史 Smoke 证据。

## Capabilities

### New Capabilities

- `data-build-evidence`: 目标驱动数据构建、停止原因、目标达成情况及样本拒绝诊断。
- `unified-model-evaluation`: 对不同阶段合并模型执行可比较的统一评测并记录模型、协议和覆盖情况。
- `grpo-calibration`: 受控比较 GRPO completion 长度候选并保存校准所需的运行与资源证据。

### Modified Capabilities

无。当前 OpenSpec 主 specs 目录为空。

## Impact

- 可能涉及 `src/onegpu_posttrain/data/openr1.py`、数据构建 CLI/报告、GRPO recipe 与运行/指标报告、统一模型评测 CLI/报告、对应测试，以及 `docs/project/IMPLEMENTATION_STATUS.md`、Smoke 运行报告和 dev7 里程碑文档。
- 预期不增加运行时依赖，不改变冻结的数据资格筛选规则、评测集/解析协议或 lineage schema；若证据表明必须改变这些语义，应另行提出版本化设计变更。
- GRPO 校准需要在 RTX 4060 8GB 参考环境实际运行候选配置；代码测试通过本身不代表校准门槛已通过。
- Mini 训练、必需 OVAT、完整前端、清洗效果因果实验均不属于本 change。
