# v0.1 Mini 数据预算 Profile

当前默认 Profile：[`configs/data/mini-v1.yaml`](../../../configs/data/mini-v1.yaml)，schema 为 `1.0`，profile ID 为 `mini-data-readiness`、版本为 `1`。

DeepMath Mini SFT pilot 使用 [`configs/data/mini-v2.yaml`](../../../configs/data/mini-v2.yaml)。v2 为每个 cohort 显式指定 manifest 路径，因此同一 stage 可并列评估 `openr1.sft_train` 与 `deepmath.sft_train`，不会因发现多个 SFT manifest 而混淆，也不会把两边数量相加。DeepMath 是本次独立 5K 目标 cohort；OpenR1 保留为可单独查看的历史 cohort。

| 数据集 | 预算 | 计数单位 | 说明 |
|---|---:|---|---|
| FineMath CPT train | 5,000,000 | 训练 tokens | 只计训练 split；CPT holdout 单独报告，不计入 5M |
| OpenR1 SFT train | 5,000 | 合格 examples | 使用当前已声明的完整性、可解析、验证标签和长度筛选规则 |
| OpenR1 Preference train | 2,000 | 完整 preference pairs | 每条记录必须包含 chosen/rejected；按训练对计数 |
| OpenR1 GRPO train | 2,000 | 可验证 prompts | 只计当前 verifier 支持的有效 prompt |

`preference_eval`、CPT holdout 和 internal holdout 都是独立留出数据。此 Profile 不把它们计入训练预算，也不为它们引入新的 Mini 数量门槛；它们仍须记录实际数量和适用的质量/血缘证据。偏好评测目标由单独的评测配置声明。

计数来自构建报告中按 dataset ID 关联的阶段输出，不从目录名、原始行数或平均长度推算。计数单位或来源缺失时，预算状态为不可用，不能判为达标。

## Profile 版本规则

改变阶段目标、计数单位/来源、资格协议引用或阻断性检查策略时，必须更新 Profile 版本。新 Profile 不会重写或重新解释历史 manifest、构建报告和质量报告；已有数据是否符合新 Profile 必须重新评估。

## 如何理解“就绪”

预算达成、质量证据、处理轨迹和血缘完整性是独立维度：

- 达到数量目标不代表语义质量已经通过；未测量或未校准的检查必须继续显示为未评估。
- 血缘校验通过只代表本地身份、引用和摘要可追溯，不代表数据质量或模型能力提高。
- 缺少受控下游比较时，清洗/构建对模型的影响为“未评估”，不能由保留率、训练 loss 或质量报告存在推断。
- 任何目标短缺都应保留 stop reason、已处理来源范围和可用拒绝原因；不得放松现有正确性、长度、污染或 split 规则来凑数。

阶段化质量指标及其状态语义见[数据质量设计](../../design/DATA_QUALITY_DESIGN_zh.md)，处理报告、manifest 和血缘校验见[数据证据实现说明](DATA_EVIDENCE.md)与[Lineage v1 设计](../../design/LINEAGE_DESIGN_zh.md)。
