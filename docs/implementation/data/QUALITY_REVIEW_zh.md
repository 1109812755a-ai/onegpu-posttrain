# 阶段数据质量审查

数据质量审查分为两层：`quality_report` 记录实际测量到的指标；`quality_review` 依据版本化 profile 形成审查结果。完整的指标清单、源数据标签或数量达标，都不等于质量批准。

## 审查 Profile 与结论边界

内置示例位于 [`mini-review-v1.yaml`](../../../configs/data/quality-review/mini-review-v1.yaml)。每个 stage 分别声明要审查的检查项、对应指标/方法、适用性和可选边界。profile 没有综合分数；新增或修改决策规则必须提高版本，并保留旧报告。

```yaml
schema_version: '1.0'
profile_id: mini-stage-quality-review
profile_version: '1'
description: Stage-specific quality review
stages:
  sft:
    checks:
      - check_id: canonical_schema
        metric_id: schema_valid_rate
        method: canonical_schema:v1
        minimum: 1.0
```

`minimum` / `maximum` 是明确单位指标的边界，不是跨指标质量分。内置 profile 会自动核对 canonical schema 和重复 ID；语义判断不会臆造阈值。需要人工判断的 stage 设有单独 `manual_review` 检查项。

上游正确性标签只作为来源标签证据；独立验证必须有可信 `metadata.reference_answer`，并记录 verifier、覆盖率和分母。当前数学答案 verifier 不证明推理过程完整。缺 reference、不可解析、不支持的 reference、verifier 缺失分别记为 `unverified`、`unparseable`、`unsupported`、`unverified`；它们都不会被记为错误答案。人工审核可以记录独立结论，但不能把不可用的自动验证伪装成已测量。

首次运行没有人工 decisions 时，stage review 保持 `pending`。要批准正式数据，决策文件至少应提供该数据集所有必需 `manual_review` 项的明确结论和理由。例如：

```json
{
  "reviewer_id": "reviewer-name",
  "decisions": {
    "openr1.sft_train": {
      "human_reasoning_review": {
        "decision": "pass",
        "rationale": "抽查 50 条；记录抽样范围、方法和观察到的缺陷。"
      }
    }
  }
}
```

Review 以新目录追加保存；修订时传入 `--supersedes` 引用先前 review bundle，不覆盖旧 review。

## 用途分类

- `baseline`：有明确协议和质量审查的正式候选；仍需通过 Mini 全部预算、血缘和评估验收。
- `exploratory`：用于试跑/策略实验，不能满足正式 Mini 门槛。
- `intentionally_noisy`：明确注入受控噪声，只用于噪声对照，不能提升为官方数据。
- `legacy_unknown`：历史产物没有用途标签；保持可读，但不会默认视为正式候选。

历史产物缺少用途字段时只会读为 `legacy_unknown`，不回写 manifest。`exploratory`、`intentionally_noisy` 和 `legacy_unknown` 可以用于研究/试跑，但即使预算达标也不能满足官方 Mini readiness；正式输入必须显式标记 `baseline`，并有匹配的 approved review。

复评和 review 产物追加写入独立目录并绑定 dataset ID、stage、canonical 内容 SHA256、质量报告 ID/哈希、profile 和检查器版本，不修改原始 JSONL、manifest 或旧报告。评测指标、数据质量测量、人工审核和模型效果是不同证据层；review 不证明训练提升，也不会改变冻结 Mini 预算。

## CLI

以下命令只读取 artifact root 中的本地数据。若已有 quality assessment bundle，可以直接从第二步开始：

```powershell
$root = "data/processed/mini/2026-10-17-build-02"
$qualityAssessment = "$root/reports/mini-quality-assessments/example-v2"

# 第一步（可选）：产生/复用原始质量测量 bundle；这是测量，不是审批。
onegpu-posttrain assess-mini-data-quality `
  --root $root `
  --profile configs/data/mini-v1.yaml `
  --output $qualityAssessment

# 第二步：依 stage profile 生成新的、离线、digest-bound review bundle。
onegpu-posttrain review-mini-data-quality `
  --root $root `
  --mini-profile configs/data/mini-v1.yaml `
  --quality-assessment $qualityAssessment `
  --review-profile configs/data/quality-review/mini-review-v1.yaml `
  --output "$root/reports/mini-quality-reviews/example-v2"

# 有 reviewer decisions 时，可追加 --decisions decisions.json；
# 重新审核时可追加 --supersedes <旧 review bundle>。

# 第三步：同时指定原质量报告和 review bundle，检查官方 Mini readiness。
onegpu-posttrain validate-mini-data `
  --root $root `
  --profile configs/data/mini-v1.yaml `
  --quality-assessment $qualityAssessment `
  --quality-review "$root/reports/mini-quality-reviews/example-v2"
```

三个命令均为本地离线操作：不会下载数据/依赖、调用 LLM provider、启动训练、运行 GPU 或使用外部质量服务。第二步的输出目录必须是新路径；review/readiness 输入路径都做哈希、身份和目录包含性检查。
