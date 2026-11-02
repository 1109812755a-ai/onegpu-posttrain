# 修复旧 CPT 数据角色

新版 FineMath 构建会在每条记录写入训练/留出角色。旧版 `finemath-build-v2` CPT 训练数据可能没有 `metadata.data_role`，训练预检会拒绝它。可以使用本命令检查并派生一个新的、带 `train_seed` 标签的数据集；原数据集不会被改写。

## 先只读预览

在项目根目录的 PowerShell 执行，把示例 manifest 路径换成旧 FineMath CPT 训练数据对应的 manifest：

```powershell
.\.venv\Scripts\python.exe -m onegpu_posttrain repair-cpt-roles `
  --workspace . `
  --manifest "data\processed\finemath\manifests\cpt-smoke-v1.yaml"
```

预览会核对支持的旧构建协议、canonical SHA-256、处理报告及其 build-run 输出，并显示将分配的角色和记录数。默认操作不写文件、不改 registry、不下载模型、不启动训练。只有 `status` 为 `eligible` 时才可继续。

## 明确确认后创建新数据集

确认预览中的父数据集 ID、摘要及 `train_seed` 角色映射后，再执行同一命令并加 `--apply`：

```powershell
.\.venv\Scripts\python.exe -m onegpu_posttrain repair-cpt-roles `
  --workspace . `
  --manifest "data\processed\finemath\manifests\cpt-smoke-v1.yaml" `
  --apply
```

新数据集登记在 `data/processed/cpt-role-repair/`，输出会给出新的 dataset ID 和 manifest 路径。训练时选择这个新 ID；原数据集及历史报告保留不动。之后仍需准备项目训练依赖和 recipe 指定的本地 Base 模型权重，并重新运行训练预检。

## 适用边界

- 仅支持证据完整且协议为 `finemath-build-v2` 的旧 CPT 训练 split。
- 普通上传、模型生成数据、角色字段已有值、report/hash 不匹配、证据缺失或留出 split 都不会被自动修复。请重新导入并明确逐条角色，或用新版构建流程重建。
- 命令不判断语料许可是否合法，也不验证语义事实正确性；这些仍由用户确认和数据质量流程负责。
