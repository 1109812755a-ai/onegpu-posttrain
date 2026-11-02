# Proposal

## Why

当前 Web 工作台可以查看实验结果并编辑/导出数据策略，但仍须回到终端启动训练，无法在页面连续查看训练状态。项目已有经过实际 RTX 4060 运行的 1,024-token Mini SFT 配方及完整数据/模型 lineage，适合作为范围可控的首个本地 Launcher 纵向切片。

## What Changes

- 首版只允许从 UI 启动一个已验证的 Mini SFT 1,024-token recipe：`recipes/mini/23-sft-1024-fallback.yaml`，其固定输入为已登记的 Mini M1-CPT 与 307 条 exploratory SFT 子集；缺失或摘要不匹配时拒绝启动。
- 不提供任意路径、自由参数编辑、脚本或命令行输入；用户先查看精确 resolved config、数据/模型摘要、质量/探索性提示和 best-effort GPU 状态，再显式确认启动。
- 用户确认后调用现有 `run-sft` CLI；为本次执行分配唯一 `run_id`，不覆盖历史运行。前端显示运行状态、日志、可得训练进度和真实产物/lineage。
- 首版最多允许一个 UI 启动的 SFT 任务，不排队、不自动重试、不自动下载、不静默更改配方或参数。关闭浏览器不停止服务端任务；停止 UI 服务会请求训练进程组优雅终止并记录 `interrupted`，不得把中断误报为成功。
- UI 评测仍使用现有 CLI；训练完成后明确显示“评测待执行”。CLI 继续独立可用；后续阶段、参数编辑和评测启动不纳入本次实现。

## Capabilities

### New Capabilities

- `local-experiment-launcher`: 通过本地 UI 安全预检、显式启动和观察唯一允许的 Mini SFT 训练 run。

### Modified Capabilities

无。数据策略编辑器仍只负责规则/计划的编辑、预览与 CLI 导出；训练启动由独立 Launcher 页面承担。

## Impact

- 影响 `src/onegpu_posttrain/ui/server.py`、SFT CLI 的 run identity 接口、受限的本地任务生命周期服务、`web/src` 的 Launcher 页面、相关 API/前端测试及中英文启动/路线图文档。
- 复用固定 Mini SFT recipe、现有 `run-sft`、记录格式和 lineage；不引入数据库、队列、远程执行服务或新训练框架。
- 本地训练启动默认关闭，仅 loopback 可用；通过显式启用、来源摘要校验、单任务锁、输出保护和用户确认限制高资源本地进程控制能力，不修改系统防火墙。
