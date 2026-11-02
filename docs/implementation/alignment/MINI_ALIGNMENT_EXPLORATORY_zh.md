# Mini 对齐阶段探索运行（中文）

本流程先验证已有 Mini M2 能否继续进入 DPO、原始 SimPO 和 GRPO，不等待达到正式数据目标。它是工程链路/方法探索，不是算法效果结论或 official Mini 完成标志。

## 当前输入与限制

| 阶段 | 当前输入 | 质量状态 | 本次解释 |
| --- | --- | --- | --- |
| DPO / SimPO | OpenR1 build-02 Preference：4 组完整 pair | `NOT_EVALUATED`；结构检查通过，但无独立语义正确性验证 | 仅验证训练与 lineage，4 对不能支持效果判断 |
| GRPO | OpenR1 build-02：2,000 prompts | `NOT_EVALUATED`；静态结构证据不等于 rollout/reward 校准 | 3 个 optimizer steps 的短程链路验证 |
| Parent | DeepMath 1024-token exploratory SFT 的 M2-SFT | M2 从现有 CPT M1 派生 | 四支必须绑定相同 M2 路径和计算出的内容摘要 |

数据数量不是探索运行启动门槛，但质量审查和 lineage 必须如实记录。此运行不会修改原始 manifests、quality reports、数据或 M2，也不会改变 Mini readiness。

## 执行命令

在仓库根目录 PowerShell 执行。先准备原 canonical 数据到 ms-swift 格式；adapter sidecar 记录输入/输出 SHA256，训练 lineage 会据此回链 canonical manifest：

```powershell
$cli = ".\.venv\Scripts\onegpu-posttrain.exe"
$m2 = "runs/mini/sft-deepmath-1024-subset-full-v1/merged/M2-SFT"
$raw = "data/processed/mini/2026-10-17-build-02/openr1/canonical"
$native = "artifacts/native/ms-swift/mini-alignment-existing-data-v1"

& $cli adapt-ms-swift --stage preference --input "$raw/preference.jsonl" --output "$native/preference.jsonl"
& $cli adapt-ms-swift --stage grpo --input "$raw/grpo.jsonl" --output "$native/grpo.jsonl"
```

完成预检后按分支运行；每个 output 必须是新目录，CLI 会拒绝覆盖已有真实运行：

```powershell
& $cli run-dpo --config recipes/mini/30-dpo-exploratory-4pairs.yaml --model $m2 --dataset "$native/preference.jsonl" --output runs/mini/alignment-existing-data-v1/dpo --merge
& $cli run-simpo --config recipes/mini/31-simpo-exploratory-4pairs.yaml --model $m2 --dataset "$native/preference.jsonl" --output runs/mini/alignment-existing-data-v1/simpo --merge
& $cli run-grpo --config recipes/mini/40-grpo-standard-exploratory.yaml --model $m2 --dataset "$native/grpo.jsonl" --output runs/mini/alignment-existing-data-v1/grpo-standard --merge
& $cli run-grpo --config recipes/mini/41-grpo-lowmem-exploratory.yaml --model $m2 --dataset "$native/grpo.jsonl" --output runs/mini/alignment-existing-data-v1/grpo-lowmem --merge
```

建议一次只运行一个 GPU 任务。任何分支失败时保留其日志、`run.json` 和 `lineage.json`；不要复用失败目录，不要自动换模型、改数据或将 OOM 重写为通过。完成后用 CLI 的 `validate-lineage --help` 选择覆盖这些 artifacts 的验证根目录，并检查 run 的父模型摘要、数据 manifest、adapter 和 merge lineage。

## 解释边界

- 运行记录中的实际 count、unit、quality status 与 `official_mini_acceptance=not_assessed` 是独立事实；成功不等于质量通过。
- 4-pair DPO/SimPO 可能记忆全部训练样本；loss 下降或 pair 指标上升不能说明泛化能力提升。
- GRPO reward 是训练信号，不等于 GSM8K/MATH 能力改善；只有统一协议评测才能作有限比较。
- 失败、未运行和未评测都需明确保留，不能补成 0 分或 PASS。

## 本次实际运行记录（2026-10-20）

四个分支均从同一个 M2-SFT 运行并成功合并：DPO/SimPO 各用 4 对数据、4 步；GRPO-standard/lowmem 各用 2,000 prompts、3 步。Preference/GRPO 的语义质量状态仍为 `NOT_EVALUATED`。DPO、SimPO 峰值显存采样约 7,947 MiB；GRPO 两支约 3,203 MiB。两种 GRPO 的 completion 均达到 512 token 上限（clipped ratio=1.0），reward std 接近/等于 0；因此只是证明训练链路可执行，不代表有效学习信号。

M2 与四个 M3 使用 `math-eval-v1`、seed 42、相同样本预算与配置哈希完成矩阵评测：GSM8K 每模型 2 条、MATH-500 每级 2 条共 10 条、内部 holdout 5 条、Domain PPL 5 条（其中 3 条按 512 token 评测上限截断）。矩阵比较状态为 `comparable`，结果见 `runs/mini/alignment-existing-data-v1/evaluation-matrix/`；完整人读/机器读总结见该目录上级的 `mini_alignment_summary_zh.md` 与 `.json`。此评测规模远小于正式 benchmark，特别是 SimPO GSM8K 的 2/2 不能作为能力提升证据。

GRPO-standard 在长目录下曾遇到 EvalScope Windows 路径过长；失败尝试的 run 记录保留。之后改用 `runs/e/grpo-s` 与 `runs/e/grpo-l` 短输出目录，在不改变评测协议的前提下成功完成；矩阵配置指向成功结果。短目录是评测产物路径，不改变模型 lineage 中的共同 M2、canonical 数据集 ID/hash 或模型身份。

复核命令：

```powershell
& $cli compare-evaluation-matrix --config configs/evaluation/mini-alignment-exploratory-v1.yaml --output runs/mini/alignment-existing-data-v1/evaluation-matrix
& $cli validate-lineage --root runs/mini/alignment-existing-data-v1 --require-complete
```

lineage 校验根目录应覆盖共同 M2、canonical 数据和四个分支；单独以 alignment 输出目录作为根时，父模型/源数据会被报告为根目录外引用。项目级全量校验还会扫描 `.pytest-tmp-full` 中的测试夹具，不能把那些 fixture 错误归因于本次实验。
