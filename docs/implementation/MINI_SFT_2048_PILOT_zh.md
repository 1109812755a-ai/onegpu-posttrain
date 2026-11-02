# Mini CPT → DeepMath SFT 2048 Pilot

Mini 已进入真实训练阶段：5M-token CPT 与 M1 合并、M1 统一小样本评测已完成；新 M1 上的 2,048-token SFT 探测通过，但完整 SFT 在第 19/624 步 OOM，尚无 M2。DeepMath 数据来自 `zwhe99/DeepMath-103K@5cf055d1fe3d7a2eb19719ac020211469736ae44`，本地 pinned card 声明 MIT；源快照和每个 Parquet shard 的 SHA256 写入 processing report。全流程可离线运行，不会隐式下载数据或替换模型。

## 数据构建与证据

实际构建产物位于 `data/processed/mini/2026-10-19-mini-sft-2048-pilot/deepmath-sft/`：

- `canonical/sft.jsonl` 与 `native/sft.jsonl`：规范数据与 ms-swift 输入。
- `manifests/sft-train-v1.yaml`：固定 source/tokenizer、review usage、内容摘要与 lineage。
- `reports/processing_report.json`、`deepmath_sft_build.json`：顺序处理步骤、分母、拒绝原因、source 文件摘要、split 与选择信息。
- `reports/deepmath_sft_quality.json`：长度、答案核验方法/覆盖、分布、污染检查覆盖和限制。
- `reports/contamination.jsonl`：命中的本地污染题目。

构建结果：103,022 source rows；去重后 101,852 道题，train 96,884 / eval 4,968；训练分区命中本地污染索引 4 道；14,208 条通过 exact-answer 与完整 chat-template 2048-token gate，随后按稳定 hash 选择 5,000 条。超长样本拒绝、不截断。污染索引当前只覆盖项目 benchmark 样例和 Mini internal holdout，属于 partial，不代表整个 GSM8K/MATH-500 全集无污染。

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
onegpu-posttrain build-deepmath-mini-sft `
  --source "$env:USERPROFILE\.cache\huggingface\hub\datasets--zwhe99--DeepMath-103K\snapshots\5cf055d1fe3d7a2eb19719ac020211469736ae44" `
  --output data/processed/mini/<new-build-id>/deepmath-sft `
  --contamination-index data/examples/policy/benchmark.jsonl `
  --contamination-index data/processed/mini/2026-10-17-build-02/openr1/canonical/internal_holdout.jsonl `
  --seed 42 --max-length 2048 --target 5000
```

`--output` 必须是新目录；如需重建，换用新的 build-id，不覆盖既有 cohort。筛选通过不等于推理正确：5,000 条的 exact final-answer agreement 与 independent answer check 均不能代替人工 reasoning review。

## 来源分开验收

使用 `configs/data/mini-v2.yaml`，并以 `data/processed/mini` 为 readiness root。`openr1.sft_train` 与 `deepmath.sft_train` 通过各自 `manifest_path` 分开列示，不相加。当前 OpenR1 为 375 条，仅供独立比较；DeepMath 达到自己的 5,000 条数量目标，但 usage 为 `exploratory`、human reasoning review pending，整体 Mini readiness 仍为 NOT READY。

```powershell
onegpu-posttrain assess-mini-data-quality `
  --root data/processed/mini --profile configs/data/mini-v2.yaml `
  --output <new-assessment-directory>

onegpu-posttrain review-mini-data-quality `
  --root data/processed/mini --mini-profile configs/data/mini-v2.yaml `
  --quality-assessment <assessment-directory>/index.json `
  --review-profile configs/data/quality-review/mini-review-v1.yaml `
  --output <new-review-directory>

onegpu-posttrain validate-mini-data `
  --root data/processed/mini --profile configs/data/mini-v2.yaml `
  --quality-assessment <assessment-directory>/index.json `
  --quality-review <review-directory>/index.json `
  --output <new-readiness-directory>
```

自动 review 会保留 `human_reasoning_review=pending`，必须由维护者针对完全匹配的数据 digest 完成审阅；不得将当前输出标成 official/baseline。

人工审核至少应抽查不同 topic/难度段（当前源字段缺失会如实标为 `<missing>`）、推理步骤完整性、最终答案与题意的一致性、是否存在模板/泄漏问题，并查看污染覆盖限制。审阅人应在 digest-bound decisions 文件中给出 reviewer identity、决定和理由；只确认 5K 数量或 exact answer match 不足以批准。详细操作见 [`QUALITY_REVIEW_zh.md`](data/QUALITY_REVIEW_zh.md)。

## 训练配方与当前阻塞

- `recipes/mini/10-peft-cpt.yaml`：FineMath 5,000,368-token CPT，一轮完整 pass，Qwen3-0.6B-Base、LoRA rank 8、micro-batch 1、gradient checkpointing。
- `recipes/mini/20-sft.yaml`：新 Mini CPT 的 `M1-CPT` 为 parent，DeepMath 5K、2048 max length、一轮完整 pass、LoRA rank 8、micro-batch 1、gradient checkpointing。
- Smoke SFT 的 1024 配方不变。

Pinned Base 权重现已缓存并离线使用。Mini CPT run `run-mini-cpt-finemath-5m-s42-1109940fd5af` 完成 413/413 updates，处理 5,000,368 tokens，峰值显存采样 7,055 MiB，训练耗时 1,178 秒；merge 产物为 `runs/mini/cpt-finemath-5m-s42-v3/merged/M1-CPT`（merge run `run-m1-cpt-merge-d679a3f662ff`）。此前缺权重及错误 Python 环境的失败尝试保留为历史记录，不覆盖成功 run。

精确 parent/config 的 2,048-token probe 为 `run-mini-sft-deepmath-2048-probe-s42-9d2facaaa886`，3/3 steps PASS，采样峰值 7,945 / 8,188 MiB，说明 8GB 显存余量很小。完整单轮训练仅在这个原始配置探测通过后启动；不缩短长度、不量化、不换模型。

完整 SFT 的离线失败证据保存在 `runs/mini/sft-deepmath-5k-s42-v1/`：run `run-mini-sft-deepmath-5k-2048-s42-e3ee24cc8136` 在第 19/624 步发生 CUDA OOM，未产生 checkpoint/M2。当前完整配方不能在 8GB RTX 4060 上安全完成；不可直接重跑或无记录地降低长度/量化。

```powershell
onegpu-posttrain run-sft `
  --config recipes/mini/20-sft.yaml `
  --model runs/mini/cpt-finemath-5m-s42-v3/merged/M1-CPT `
  --dataset data/processed/mini/2026-10-19-mini-sft-2048-pilot/deepmath-sft/native/sft.jsonl `
  --output runs/mini/sft-deepmath-5k-s42-v1 --merge
```

M1 已在 `runs/mini/eval-m1-cpt-v1/` 完成同一 `math-eval-v1` 小样本评测；因 M2 缺失，M1/M2 对照仍未完成。数据 review pending 时，两阶段训练只能作为显式 exploratory research，不是官方 Mini baseline。完成 Mini 验收仍需先解决 2048 SFT 的真实 OOM，再产出 M2、补齐 M2 统一评测，并重新核验 lineage/readiness/UI。

运行 lineage 校验：

```powershell
onegpu-posttrain validate-lineage --root data/processed/mini/2026-10-19-mini-sft-2048-pilot/deepmath-sft
```
