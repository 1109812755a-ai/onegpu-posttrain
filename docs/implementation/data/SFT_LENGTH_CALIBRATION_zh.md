# SFT 长度校准（探索性）

本工具用于观察 1,024 / 2,048 / 4,096 个完整渲染 token 上限下，DeepMath 本地样本的长度留存和单卡短程可运行性。它不修改 `recipes/smoke/20-sft.yaml`，不会下载数据，也不会把候选产物登记为官方 Mini 数据。

## 1. 本地数据准备

需要用户已经缓存的 DeepMath-103K Parquet 快照。工具只读取本地文件，必需列为 `question`、`final_answer`、`r1_solution_1`、`r1_solution_2`、`r1_solution_3`。PowerShell 中可以先定位：

```powershell
$source = Join-Path $env:USERPROFILE '.cache\huggingface\hub\datasets--zwhe99--DeepMath-103K\snapshots\<本地快照revision>\data'
Get-ChildItem $source -Filter '*.parquet'
```

若路径或 Parquet 列不符合要求，命令会报错退出；不会回退到 Hugging Face 在线下载。使用项目 `data` extra（含 Transformers 和 Parquet 读取依赖），并确保固定 Qwen tokenizer 已在本机缓存：

```powershell
pip install -e '.[data,dev]'
```

## 2. CPU 离线长度校准

先运行确定性、分片无偏的 hash 抽样；默认抽 2,000 道源题，seed 为 42，并比较 1,024、2,048、4,096：

```powershell
onegpu-posttrain calibrate-sft-length `
  --source $source `
  --output artifacts/calibrations/sft-length/deepmath-s42 `
  --sample-limit 2000 `
  --seed 42
```

输出路径必须不存在。工具会生成每个长度候选的 `canonical/sft.jsonl`、`native/sft.jsonl` 和 `manifest.json`，以及根目录 `calibration_report.json`。manifest 记录本地源文件摘要、快照、抽样 IDs 摘要、tokenizer/template、长度分母、内容 hash 和质量证据状态。

长度按固定 Qwen tokenizer 对最终 user/assistant chat-template 渲染结果计数，超过上限的 trace 会被拒绝，不截断。报告同时给出 trace 级长度留存和每题最多一条、最短 exact-normalized-answer-match 的训练候选。该答案一致性仅为保守字符串筛选，**不等于 math_verify 语义验证**；语义正确性、整体质量批准和模型效果均明确标为未测/未评估。

### 可选：语义最终答案验证

默认 `--answer-verification exact`（也就是不传该参数）保持原有的规范化字符串精确匹配，不启动语义校验 worker。用户可显式选择 `semantic`，让工具对抽取出的最终答案与参考答案运行本地、受限批次的 `math-verify`：

```powershell
onegpu-posttrain calibrate-sft-length `
  --source $source `
  --output artifacts/calibrations/sft-length/deepmath-s42-semantic `
  --sample-limit 2000 `
  --seed 42 `
  --length 1536 `
  --answer-verification semantic
```

语义模式需要已安装项目可选的 `data` 依赖（包括 `math-verify`）；不会下载模型或数据，也不会调用远程 API。验证以最多 32 对为一批，由父进程对每批设置 60 秒截止时间；过长答案（超过 4,096 字符）、缺失依赖、无法解析、超时和 worker 错误均保留各自状态，不会算作 `incorrect`，且只有状态为 `correct` 的 trace 可进入该候选训练集。报告/manifest 会记录模式、协议、verifier 及版本、分母、逐状态数量、判定覆盖率、输入与候选摘要。若没有可判定结果，coverage 为 0，而不是把未验证样本当成负例。

此验证只针对**抽取到的最终答案等价性**，不检查推理过程是否有效、完整或忠实，也不代表数据质量获批或模型能力提升。它是探索性筛选信号；不要据此直接覆盖或批准官方 Mini 数据。精确模式与语义模式使用不同的校准/数据集身份，必须写入新的输出目录，不会覆盖旧产物。

## 3. 可选短程 GPU 可行性探测

只有明确添加 `--run-probes` 并提供本地 parent checkpoint 才会启动训练。Hub 模型 ID 不接受，避免训练阶段隐式下载模型。每个长度候选最多运行 5 个 optimizer steps（默认 3），共用 parent、seed 和训练控制，不合并 checkpoint：

```powershell
onegpu-posttrain calibrate-sft-length `
  --source $source `
  --output artifacts/calibrations/sft-length/deepmath-s42-probe `
  --sample-limit 2000 `
  --seed 42 `
  --run-probes `
  --model runs/smoke/cpt-peft-smoke-s42/merged/M1-CPT `
  --probe-steps 3
```

每个 probe 会保留标准 `run.json`、`lineage.json`、环境和日志；OOM 或失败也保留在该候选记录里。短程 probe 关闭 TensorBoard event 写入以避免 Windows 深路径兼容问题，runner 的 stdout/stderr 与运行记录仍保留。报告中的显存是现有监控器用 `nvidia-smi` 采样到的 GPU 0 系统级显存，不是进程独占峰值。短程 probe 只回答这套配置是否能跑及其资源成本，**不能比较模型能力、训练收益或数据清洗因果效果**。不同长度门槛会产生不同 eligible 数据子集，因此不应把 loss/训练结果当作只改变 max length 的模型效果对照。

## 4. 边界

- 校准样本和候选数据是探索性产物，不满足 Mini 数据质量、血缘或 readiness 验收。
- 不为达到某个数量而放宽答案筛选或完整长度规则。
- 不把校准结果自动写回正式 recipe。若要改变 Smoke/Mini 的默认长度，需要单独审查证据并通过另一项 OpenSpec 变更。

## 5. 本机 RTX 4060 探索性结果（2026-10-19）

数据快照 `5cf055d1fe3d7a2eb19719ac020211469736ae44` 共 103,022 行；按 seed 42 对全量源题做稳定 hash 抽样 2,000 题，合计 6,000 个非空 trace。固定 Qwen3-0.6B tokenizer revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，answer-match 仅是规范化字符串相等，不是 math_verify。

| max length | 长度合格 trace | 留存率 | 有长度合格 trace 的题 | exact-answer-match 训练候选 | 3-step probe | wall time | 采样峰值显存 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 1,024 | 6 / 6,000 | 0.10% | 6 | 5 | PASS | 91.9 s | 6,509 MiB |
| 2,048 | 557 / 6,000 | 9.28% | 312 | 298 | PASS | 105.1 s | 7,931 MiB |
| 4,096 | 2,751 / 6,000 | 45.85% | 1,194 | 1,171 | OOM | 96.7 s | 7,947 MiB |

执行环境为 NVIDIA RTX 4060 Laptop GPU 8 GiB、ms-swift 4.5.3、PyTorch 2.14.1+cu126。wall time 包含模型加载和训练；显存是 `nvidia-smi` 系统级采样。4,096 档在该短程设置中 OOM；不能据此断言所有 4,096-token 训练配置都不可用。减少 batch、改变量化/checkpointing、不同数据长度分布等都可能改变结果，但都需要新的受控校准。2,048 也已接近显存上限。

完整逐档 run、OOM 和 lineage 记录见 [本次 GPU 校准报告](../../../artifacts/calibrations/sft-length/deepmath-probe-reportnone-20261019-s42/calibration_report.json)。该结果只反映当前固定快照、抽样、M1 parent、seed 42、LoRA rank 8 和 3-step 设置的资源可行性；不构成模型能力、Mini 合格或数据质量通过结论。
