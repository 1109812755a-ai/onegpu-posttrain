# 数据与模型血缘设计

**版本：** Lineage v1  
**状态：** Lineage v1 已实现；真实 Mini/OVAT 运行验收仍待后续里程碑
**范围：** 本地文件中的数据、运行、模型、质量报告和评测来源关系

英文设计见 [LINEAGE_DESIGN.md](LINEAGE_DESIGN.md)，数据处理效果比较另见 [数据处理、质量变化与模型效果设计](DATA_PROCESSING_EFFECT_DESIGN_zh.md)。

## 三层身份

| 字段 | 含义 | 是否唯一 | 示例 |
|---|---|---:|---|
| `experiment_id` | 一组受控实验的分组 ID，可关联多个运行 | 否 | `exp-cleaning-ablation-v1` |
| `run_label` | 面向人的配方／变体／seed 标签 | 否 | `dpo-beta01-s42` |
| `run_id` | 每次实际操作的唯一 ID，是血缘图里的运行主键 | 是 | `run-dpo-beta01-s42-a1b2c3d4e5f6` |

Run 类型为 `data_build`、`quality_assess`、`train`、`merge`、`evaluate`。重复运行相同配方会保留相同 `run_label`，但生成新的 `run_id`。dry-run 是计划记录，不冒充实际 Run。

历史 Smoke 文件中的 `run_id` 曾用于配方标签。新读取器会把没有 `lineage_schema_version` 的历史运行显示为 `legacy`，保留原字段供查看，不会将旧标签改解释为唯一运行 ID。新运行将 `run_id` 同时写入 `run.json` 与 `lineage.json`。

## 证据链

```text
源数据 → 数据构建 Run → 清洗后 Dataset + 处理报告
                              ↓
                      质量评估 Run → QualityReport
                              ↓
父模型 + Dataset → 训练 Run → Adapter
父模型 + Adapter → 合并 Run → Model
Model + 评测数据/协议 → 评测 Run → Evaluation + 指标
```

M0 Base → M1 CPT → M2 SFT 后，DPO、SimPO、GRPO-standard、GRPO-lowmem 是从同一 M2 身份和摘要出发的并行训练 Run；各自 adapter 再由独立 `merge` Run 合并成模型产物。

每个节点只记录实际存在并可核实的产物。缺失的来源修订、模型摘要或报告必须标为不可得；不能用路径或展示名代替身份。普通 JSONL 可以继续训练，但没有可验证 manifest 时，血缘完整度标为不完整。

当上游提供稳定 UUID／generation 序号时，规范化记录会保留 `source_record_id`、`problem_id` 和父 generation 的 `parent_record_ids`，偏好对包含 chosen 与 rejected 两个父记录。上游没有可靠记录键时只标 `dataset_only`，不得用行号冒充来源身份。

## 示例

```json
{
  "lineage_schema_version": "1.0",
  "run_id": "run-sft-mini-s42-a1b2c3d4e5f6",
  "run_type": "train",
  "run_label": "sft-mini-s42",
  "experiment_id": "exp-cleaning-ablation-v1",
  "status": "completed",
  "started_at": "2026-10-17T00:00:00Z",
  "inputs": [
    {"id": "model-m1-cpt", "kind": "model", "role": "parent"},
    {"id": "ds-openr1-sft-clean-v1", "kind": "dataset", "role": "train"}
  ],
  "outputs": [
    {"id": "model-sft-adapter-a1b2", "kind": "model", "role": "adapter"}
  ]
}
```

## 清洗与模型指标

血缘可让人从评测指标回溯到模型、训练 Run、数据集、处理步骤和质量报告。它提供可核查的关联证据，不足以单独证明清洗导致了指标变化。因果比较需后续受控实验固定父模型、样本池、训练预算、seed 和评测协议，再报告差异与限制。

身份、摘要、状态及本地路径校验由 [`onegpu_posttrain.lineage`](../../src/onegpu_posttrain/lineage.py) 实现；协议详情以 OpenSpec `implement-lineage-v1` 为准。

## 评测回溯与证据状态

实际评测写出 `evaluation_lineage.json` 与 `evaluation_run.json`：`eval_id` 指向指标产物，评测 `run_id` 指向这次执行；其输入包含模型身份/本地摘要、benchmark ID、可解析的 holdout 数据集引用、模型 revision、评测协议版本和配置摘要。指标保留可取得的样本分母。评测可继续追到模型产物的 `produced_by_run_id`，再沿训练／合并 Run 找到 parent model、adapter、数据 manifest、处理报告和质量报告。

```json
{
  "lineage_schema_version": "1.0",
  "eval_id": "eval-79b0fbcafad34fe8a31d3ca66ad822a7",
  "produced_by_run_id": "run-sft-mini-s42-a1b2c3d4e5f6",
  "model_id": "model-sft-mini-9c44c15e1a2b",
  "dataset_ids": ["benchmark-gsm8k", "ds-math-holdout-v1"],
  "protocol_version": "math-eval-v1",
  "status": "completed",
  "metrics": {
    "evalscope.gsm8k.accuracy": {"value": 0.42, "denominator": 50}
  }
}
```

- `legacy`：没有 v1 schema 的历史记录。保留原值以便查看，但不把旧 `run_id` 重解释为执行 ID，也不声称已验证血缘。
- `unavailable`：证据本来就不在本地或无法取得，例如外部 benchmark 内容摘要；这是显式未知，不等同于错误或零值。
- `incomplete`：记录可以使用，但存在缺失 manifest／外部来源等可复现证据。普通 JSONL 仍可训练；Mini/OVAT 的完整血缘门禁应使用 `--require-complete`。

对一个 artifact 根目录运行只读校验：

```bash
python -m onegpu_posttrain validate-lineage --root ./
python -m onegpu_posttrain validate-lineage --root ./ --require-complete
```

默认发现重复 ID、失效本地引用、摘要不匹配和血缘环等完整性错误时返回非零；普通模式把不完整／外部不可得作为警告。`--require-complete` 会将不完整证据提升为错误。校验只验证身份、引用和内容一致性，不评判数据质量、训练有效性或模型能力，也不建立清洗对模型指标的因果结论。
