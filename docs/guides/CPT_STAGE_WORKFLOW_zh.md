# CPT 阶段训练工作流

CPT 是从 Base 开始的连续语料训练，不使用 SFT 的 prompt/answer 格式。当前批准 profile 是 `cpt-smoke-v1`（LoRA PEFT-CPT、pinned Base 配置、固定 `recipes/smoke/10-peft-cpt.yaml`）。模型缺少本地配置或权重时预检会阻断；本流程不会联网下载。

## 输入与预检

训练输入必须是已登记 `stage=cpt` 数据集，canonical 记录必须有 `metadata.data_role=train_seed`。`diagnostic_dev` 与 `final_holdout` 不允许进入训练；CPT canonical schema、manifest、processing/quality report 摘要、内容 SHA-256 与 Base 权重都在预检中核对。

UI/API 只读预检接口：

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8765/api/training/cpt/preflight `
  -ContentType 'application/json' -Body (@{ dataset_id = '<已登记的 CPT dataset_id>' } | ConvertTo-Json)
```

返回 `ready_for_cli` 只表示本机数据、固定配方和 Base 模型可用；质量或许可证据不足会提示但不因样本数量自动拒绝探索。该 API 不执行训练，也不签发通用阶段确认 token。CPT 多阶段 Launcher UI 与统一 Run 启动仍在后续任务开发中。

## 受控 CLI 执行

手动运行已登记 CPT CLI 时，manifest 必须位于 workspace 内，输出目录必须是新的 workspace 内路径：

```powershell
.\.venv\Scripts\python.exe -m onegpu_posttrain run-registered-cpt `
  --workspace . `
  --manifest data/processed/<dataset-artifact>/manifests/cpt-user-upload-v1.yaml `
  --config recipes/smoke/10-peft-cpt.yaml `
  --output runs/cpt/manual-cpt-execution
```

命令会重复执行 fail-closed preflight、生成不可变的 ms-swift native 数据视图，再委托现有受控 PEFT-CPT runner；默认完成后执行 LoRA merge。不要传任意 shell command 或改变已批准 recipe。要中止/调试时保留真实日志和 `run.json`，不要覆盖历史输出。

## 结果与血缘

实际成功才会登记 adapter，merge 目录确实存在才会登记 merged M1-CPT。训练状态和模型产物以 `lineage.json` 与 `model_artifacts/*.json` 为准；失败/OOM 无 checkpoint 时 outputs 为空。CPT lineage 使用规范 stage `cpt`，即使旧 CLI `run.json` 仍保留执行 flavor `peft_cpt`。校验：

```powershell
.\.venv\Scripts\python.exe -m onegpu_posttrain validate-lineage --root runs/cpt/manual-cpt-execution --require-complete
```

CPU/fake CLI 用例用于验证契约与失败语义，不能替代 RTX 4060 上的真实 CPT smoke。
