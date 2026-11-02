# GRPO Completion 长度校准

状态：候选清单、汇总工具和 RTX 4060 顺序实测已完成。完整报告：[`grpo_calibration_report.json`](../../../runs/dev7/grpo-completion-calibration-attempt3/grpo_calibration_report.json) / [`grpo_calibration_report.md`](../../../runs/dev7/grpo-completion-calibration-attempt3/grpo_calibration_report.md)。

## 冻结控制项

配置：[`grpo_completion_calibration.yaml`](../../../configs/experiments/grpo_completion_calibration.yaml)。它从现有 low-memory GRPO recipe 派生 512、768、1024、1536 四个 completion 长度候选，固定以下条件：

- M2-SFT parent：`runs/smoke/sft-smoke-s42/merged/M2-SFT`
- canonical prompt dataset：`artifacts/native/ms-swift/smoke/grpo.jsonl`，固定 200 个 prompt ID，并在 manifest 中保存 ID 列表和 SHA256
- seed 42、lowmem profile（`beta=0`、无 reference model）、reward、生成参数、20 optimizer steps
- 训练预算：80 个 completion instances；候选之间只改变 `max_completion_length_candidate`

准备会写出四份 recipe 和 `calibration_manifest.json`，记录 experiment ID、每个 run label、recipe hash、数据指纹、固定训练配置和逐候选命令：

```powershell
onegpu-posttrain prepare-grpo-calibration --config configs/experiments/grpo_completion_calibration.yaml
```

随后按 manifest 的 `candidates` 顺序逐个执行各自的 `command`。不要并行启动 GPU 候选。每个候选共享 `experiment_id`，`run-grpo` 在线生成独立 `run_id`；保留 OOM/FAILED 的 `run.json` 和 lineage，不要删掉失败记录。完成或中断后汇总：

```powershell
onegpu-posttrain summarize-grpo-calibration --manifest runs/dev7/grpo-completion-calibration/calibration_manifest.json
```

汇总 JSON/Markdown 为 `grpo_calibration_report.json` 和 `grpo_calibration_report.md`，包含 status、实际平均生成长度、clipping ratio、reward/accuracy reward、耗时、采样峰值显存；没有产出的指标标为 unavailable。clipping target 为严格小于 30%，若无候选达标，报告只选当前成功候选中 clipping 最低者，并明确 `target_met=false`。

## RTX 4060 实测结果（2026-10-17）

执行环境：RTX 4060 Laptop 8 GiB、ms-swift 4.5.3、PyTorch 2.14.1+cu126。四个 run 全部 `PASS`，各 20 steps / 80 planned completion instances；共享 `experiment_id=exp-grpo-completion-calibration-smoke-s42`，各有独立 `run_id`。prompt ID hash 为 `fb4633b1672520906b35ec32bce5514f52e9c3c056bda5e5cde8fb6ee4e63b98`，dataset SHA256 为 `3262b80fe422e25c9e5fbe56f0434653460a44975f2f5b2632d004d4c27d5a2a`。

| Completion length | Status | Mean generated tokens | Clipping | Mean reward | Accuracy reward | Wall time | Peak sampled VRAM |
|---:|---|---:|---:|---:|---:|---:|---:|
| 512 | PASS | 495.4 | 91.25% | 0.01875 | 0.0125 | 624 s | 3,141 MiB |
| 768 | PASS | 694.5 | 81.25% | 0.04250 | 0.0125 | 914 s | 4,043 MiB |
| 1024 | PASS | 935.1 | 72.50% | 0.05625 | 0.0125 | 1,200 s | 6,329 MiB |
| 1536 | PASS | 1,239.0 | 55.00% | 0.06250 | 0.0000 | 1,746 s | 7,417 MiB |

没有候选达到 clipping <30%；1536 是本轮最低 clipping 候选，但其 accuracy reward 均值为 0，较高的总 reward 由 format reward 贡献，不能解释为数学正确性变好。1536 在当前 8 GiB 机器、80-completion smoke 预算下成功运行，峰值约 7.4 GiB、耗时约 29 分钟。结论仅适用于这组 M2、200 prompts、lowmem profile、seed 42 和短预算。

因此保留低于 30% clipping 的门槛、明确报告“未达标”，并且**不改写** `recipes/smoke/41-grpo-lowmem.yaml` 默认值。可把 1536 作为本设置下后续扩大评估的候选，而非已验证的默认最佳配置。此校准是 smoke feasibility / clipping 证据，不是算法效果结论；不要据此改写历史 GRPO smoke 报告。

首次在原长工作区尝试的两个 512 run 因 Windows ModelScope cache lock 的路径/权限问题而在训练前失败，均在独立 attempt 目录保留。最终正式矩阵在临时短盘符 `X:` 下运行并把 cache 定向至工作区；失败 attempt 不纳入正式矩阵报告。
