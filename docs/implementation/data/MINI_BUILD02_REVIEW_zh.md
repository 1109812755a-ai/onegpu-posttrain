# Mini build-02 阶段质量审查记录

本记录对应 2026-10-19 在本地对保留的 `data/processed/mini/2026-10-17-build-02` 执行的离线审查。过程只新增报告目录，不改数据集、manifest、原质量报告、旧 readiness 报告或冻结的 `mini-v1` 预算。

## 输出与结论

- 原始质量测量 bundle：`reports/mini-quality-assessments/local-check-v1`，assessment `mini-quality-66d4711051bc46889d55e20708234248`。
- 阶段 review bundle：`reports/mini-quality-reviews/local-check-v1`，assessment `mini-review-bf76708674c94cbe8da9f8166faff3d0`。
- readiness：`reports/mini-readiness/local-check-v1/mini_readiness.json`，assessment `mini-check-3cf11d3baf1e4ba5a8e34963b89ff1b7`，结论 `not_ready`，23 个 blocker。
- 七个 stage 都判为 `legacy_unknown`，七个 review 均为 `pending`。旧 manifest 没有显式 usage 分类；未将其自动提升为 `baseline`。
- SFT 独立答案核验覆盖率为 `0/375`；Preference train 为 `0/8` 个回答（4 对），Preference eval 为 `0/2` 个回答（1 对）。对应样本都记为 `unverified`，原因是没有可信的 `metadata.reference_answer`，不是判成答错。源 `chosen_correct` / `rejected_correct` 标签没有被当作独立核验。
- 结构检查方面，七个数据集的 canonical schema 和 record ID 检查均通过；语义/奖励/评测协议检查仍为 `not_measured` 或等待人工审查。答案匹配本身也不等于推理过程正确。

| 数据集 | 用途 | 当前预算 | Mini 目标 | 主要未决项 |
|---|---|---:|---:|---|
| FineMath CPT train | legacy_unknown | 5,000,368 tokens | 5,000,000 | 语义审查待定 |
| FineMath CPT holdout | legacy_unknown | 50,078 tokens | 不计训练预算 | 语义审查待定 |
| OpenR1 SFT train | legacy_unknown | 375 examples | 5,000 | 短缺 4,625；无可信 reference；人工推理审查待定 |
| OpenR1 Preference train | legacy_unknown | 4 pairs | 2,000 | 短缺 1,996；无可信 reference；人工 pair 审查待定 |
| OpenR1 Preference eval | legacy_unknown | 1 pair | 支持集，不计训练预算 | 无可信 reference；人工 pair 审查待定 |
| OpenR1 GRPO train | legacy_unknown | 2,000 prompts | 2,000 | reward/verifier 支持和人工语义审查待定 |
| OpenR1 internal holdout | legacy_unknown | 500 examples | 支持集，不计训练预算 | scoring protocol 与人工审查待定 |

因此 build-02 可用于继续做明确标注的探索性分析，但不能作为官方 Mini-ready 的数据集。预算数量达标不等于用途、语义质量或审核达标。

## 完整性与可复现性

生成 review 前后，对 build-02 下 59 个已存在文件逐一比较 SHA256；除新建 review 输出目录外均未变化。readiness 在同时加载 quality assessment 和 review bundle 时验证 review 的数据集 ID、stage、canonical digest、质量报告 ID/哈希、用途和 profile 身份；结果正常落为 `not_ready`，没有将 pending 或 legacy 解释为通过。

所有操作都在本地完成：没有网络访问、数据下载、provider 调用、GPU 或训练。可用以下命令复核：

```powershell
$root = "data/processed/mini/2026-10-17-build-02"
onegpu-posttrain validate-mini-data `
  --root $root `
  --profile configs/data/mini-v1.yaml `
  --quality-assessment "$root/reports/mini-quality-assessments/local-check-v1" `
  --quality-review "$root/reports/mini-quality-reviews/local-check-v1" `
  --output "$root/reports/mini-readiness/recheck-again-v2"
```

## 代码验证

- 定向回归：85 passed（review/readiness、policy、synthetic policy export、targeted e2e、verifier）。
- Ruff：`python -m ruff check src tests`，All checks passed。
- 编译：`python -m compileall -q src/onegpu_posttrain`，通过。
- OpenSpec：`openspec validate mini-data-remediation --strict`，valid；apply task 12/12 完成。
- CPU fixture 中的 targeted generation 使用 `FakeProvider`，并断言 `network_used=false`；build-02 review/readiness 本身不实例化 provider。
