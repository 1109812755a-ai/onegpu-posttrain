# M2/M3 统一评测矩阵

矩阵协议配置：[`dev7-m3-matrix.yaml`](../../../configs/evaluation/dev7-m3-matrix.yaml)。它覆盖 M2-SFT 基线和 M3-DPO、M3-SimPO、M3-GRPO-standard、M3-GRPO-lowmem，并将每个模型的 `eval.json`、`evaluation_lineage.json` 与 `evaluation_run.json` 纳入同一比较。

冻结参数：EvalScope `math-eval-v1`、GSM8K limit=5、MATH-500 每难度级 limit=5（总 25）、internal holdout limit=20、domain PPL limit=20、max new tokens=512、seed=42、float16。五个模型使用同一内部 holdout 文件和 `recipes/smoke/00-base-eval.yaml`。旧的 Base/M1/M2 smoke 结果不被覆盖；dev7 在 `runs/dev7/unified-m3-matrix-final/<model-id>/` 下生成新的评测执行及 lineage。

可对单个模型运行统一评测（示例）：

```powershell
onegpu-posttrain run-model-eval `
  --config recipes/smoke/00-base-eval.yaml `
  --run-label dev7-m2-unified-s42 `
  --stage sft `
  --model runs/smoke/sft-smoke-s42/merged/M2-SFT `
  --revision "" `
  --limit 5 `
  --internal-holdout data/processed/smoke/openr1/canonical/internal_holdout.jsonl `
  --internal-limit 20 `
  --domain-holdout data/processed/smoke/finemath/canonical/cpt_holdout.jsonl `
  --domain-limit 20 `
  --output runs/dev7/unified-m3-matrix-final/M2-SFT
```

对其余四个模型重复相同参数，仅替换 model path、stage/run label 和输出目录。全部执行完后汇总：

```powershell
onegpu-posttrain compare-evaluation-matrix `
  --config configs/evaluation/dev7-m3-matrix.yaml `
  --output runs/dev7/unified-m3-matrix-final/matrix
```

矩阵显式区分 `completed`、`partial`、`failed`、`missing`；缺失指标保留 null/N/A。每行包含样本量与 evaluation lineage 引用。`provenance.status` 另标记来源 `verified`、`incomplete` 或 `mismatch`，并用 `provenance.issues` 说明缺少或冲突的证据。矩阵会交叉核验 eval payload、evaluation run、evaluation lineage 的 run/eval/model 身份、终态以及 Run 声明的指标输出；旧版/缺字段评测仍展示可用数字，但不当作完整成功评测。只有所有模型评测和来源均完成核验，且 protocol version、评测 config hash 和 dataset IDs 一致时，整体可比性才标为 `comparable`。否则整体标为 `partial_or_inconsistent`，不填补或推断缺失指标。

Mini 真实对齐矩阵的来源核验结果保存在 `runs/mini/alignment-existing-data-v1/evaluation-matrix-v2/`。该目录是新增的可追溯矩阵，不覆盖原矩阵；GRPO 成功评测实际位于 `runs/e/grpo-s/`、`runs/e/grpo-l/`，必须按其稳定 run/eval/model ID 关联，不能从失败重试目录或模型显示名称推断来源。

## dev7 RTX 4060 实测（Smoke）

运行产物位于 [`runs/dev7/unified-m3-matrix-final/`](../../../runs/dev7/unified-m3-matrix-final/)；五个模型均为 `completed`，实际样本量均为 GSM8K 5、MATH-500 25、internal 20、domain PPL 20，协议版本、配置 SHA256 和 dataset IDs 一致。汇总文件：`matrix/evaluation_matrix.json` 与 `matrix/evaluation_matrix.md`。

| 模型 | GSM8K Acc (n=5) | MATH-500 Acc (n=25) | Internal Acc (n=20) | Domain PPL (n=20) |
| --- | ---: | ---: | ---: | ---: |
| M2-SFT | 0.60 | 0.40 | 0.20 | 6.6324 |
| M3-DPO | 0.40 | 0.48 | 0.15 | 6.7560 |
| M3-SimPO | 0.40 | 0.28 | 0.00 | 6.9502 |
| M3-GRPO-standard | 0.60 | 0.36 | 0.15 | 6.6319 |
| M3-GRPO-lowmem | 0.60 | 0.40 | 0.10 | 6.6323 |

这只是单 seed、小样本 smoke 验证，用于确认评测、数据和 lineage 流程并观察方向；不构成稳定的算法优劣结论或 benchmark/leaderboard 声明。EvalScope 在 Windows 上会输出 multiprocessing `WinError 6` worker 告警，但本轮五次 CLI 均以退出码 0 完成，且 artifacts、样本数和 lineage 完整；该环境告警应后续单独清理。
