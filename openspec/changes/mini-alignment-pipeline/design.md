# Design

## Context

参见 proposal 与 delta specs。可复用的父模型为 `runs/mini/sft-deepmath-1024-subset-full-v1/merged/M2-SFT`；本地 OpenR1 build-02 有 4 组 Preference pairs 与 2,000 条 GRPO prompts。质量报告未独立确认语义正确性，故只能开展 exploratory runs。现有 Smoke 配方针对 200 pairs/20 steps，不直接修改或冒充 Mini 配方。

## Goals / Non-Goals

**Goals:**
- 以独立 Mini 配方和新 run IDs 从同一 M2 实际运行 DPO、原始 SimPO、GRPO-standard、GRPO-lowmem。
- 数据量、暴露量、数据质量状态、父模型和模型输出均可追溯；成功与失败均形成真实运行证据。
- 对可产出的分支执行一致且有覆盖记录的评测。

**Non-Goals:**
- 不要求达到 2K Preference 或 GRPO 预算；不因此晋升 official Mini readiness。
- 不重试 2048-token SFT OOM，不下载数据/模型，不自动生成或放宽数据质量规则。
- 不据 4-pair 训练推断 DPO/SimPO 的有效性，也不把 GRPO reward 当作任务能力提升。

## Decisions

1. **运行输入用已有 immutable canonical 数据和当前成功 M2。** 先通过现有 adapter 转成本地训练器格式，并保留与 canonical manifest 的可验证关联；不得使用 Smoke fixture 或临时 JSONL 替代真实 Mini 数据。
2. **探索配方与 Smoke 分开。** DPO/SimPO 使用 4 对数据的一次有限遍历/最小可运行预算，明确极小数据与重复暴露限制；GRPO 两种 profile 使用 2,000 prompts，但设置有限 optimizer-step cap，避免把 20-step Smoke 误称 Mini 基线。原始 SimPO 固定 `cpo_alpha=0`。
3. **四个 branch 的 parent 和身份显式固定。** 每个训练输入都绑定相同 M2 ID/摘要；每次实际训练、合并、评测分别使用唯一 run_id，输出 adapter/merged model 只在确实存在时登记。
4. **质量状态只披露、不替代。** 现有 Preference/GRPO 的结构报告、未测语义状态及数据规模写入配置或运行/总结证据；运行成功不会改写原 manifest、quality report 或 Mini readiness。
5. **评测用同一 matrix，按完成度展示。** 固定样本 ID、协议和预算覆盖 M2 及成功分支；若单卡时间/内存不允许全部完成，记录明确失败/未运行状态，不把其余任务塞进不受控的临时配置。

## Risks / Trade-offs

- [4 对 pair 只适合验证软件和血缘链路，极易过拟合] → 限定预算并在所有产物标记 exploratory，不声称算法结论。
- [GRPO 生成可能耗时或 OOM] → 用已有 calibration/资源配置设置有限步数，失败保留 run evidence，不静默减配后宣称完成。
- [旧 manifest 或 adapter 映射不匹配] → 训练前验证数据 ID、摘要和行数；不匹配时停止该 branch。
- [完整 benchmark 评测较慢] → 评测协议与样本集一次固定；资源不足时保持未测状态，后续可单独继续。

## Migration Plan

新增配方、训练/评测 artifacts 与中文汇总，不覆盖 Smoke、build-02、旧 run 或 M2。验证通过后可归档本 change；官方 Mini readiness 仍由原有数据、质量审查和预算流程单独决定。
