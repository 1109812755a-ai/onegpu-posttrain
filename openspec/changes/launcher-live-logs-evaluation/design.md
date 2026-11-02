# Design

## Context

参见 [proposal.md](proposal.md) 了解问题与范围。现有 Launcher 每 2.5 秒轮询 `/api/launcher/run`，但 API 返回的是 supervisor 收集的 `launcher.log`；实际训练指标和 step 写入 `runs/.../logs/train/stdout.log`，且记录形如 `global_step/max_steps: 10/38`，与现有解析模式不同。现有 run 页面仅在训练结束时扫描模型与 lineage。项目 CLI 已有 `run-model-eval`，使用 `configs/evaluation/math-v1.yaml`（GSM8K、MATH-500）并创建独立评测 run/lineage；评测子命令当前通过捕获整个 EvalScope 输出后才返回，需确保监督器可收到增量日志。Launcher 保持 loopback、本机单 GPU、单活动子进程、固定输入及显式用户确认。

## Goals / Non-Goals

**Goals:**

- 让训练和评测日志在进程执行时近实时可见，进度仅基于已验证的日志格式。
- 将 SFT 完成、评测预检/确认、评测结果放进同一页面流程，但保持训练与评测身份及证据分离。
- 复用 CLI 创建训练和评测产物，使用现有评测血缘与 run/model lineage 语义。
- 改善桌面信息层级，并让关键状态、按钮、日志、指标在窄屏和键盘操作下仍可用。

**Non-Goals:**

- 任意 recipe/模型/数据/评测参数编辑、其他训练阶段、评测矩阵、队列、并发或跨机器调度。
- 训练完成后自动评测、自动续跑或把 exploratory 子集结果标成正式 Mini 验收。
- 在该变更中设计通用日志采集平台或数据库。

## Decisions

### 1. 保留短间隔轮询，改为读取阶段日志

继续由页面定时读取有界的 run 状态与日志尾部，不引入 SSE/WebSocket。现有 2.5 秒轮询已能在页面 reload 后恢复持久化状态，适合本地单用户；把间隔调整为约 1.5–2 秒，并让 supervisor 为活动阶段选择真实日志源：训练阶段优先读取 `logs/train/stdout.log`，合并阶段读取合并日志，评测阶段读取增量输出的评测日志；supervisor 自身 `launcher.log` 仅作为阶段事件/兜底日志，完整 JSON 摘要不作为训练日志显示。文件读取须限定在当前 workspace/run 输出目录、拒绝符号链接并返回有界尾部。

替代方案是 SSE/WebSocket 推送。它在本机 UI 中收益有限，却增加连接生命周期、浏览器重连和刷新后状态恢复的处理；轮询与现有架构更一致。日志每次响应仍需有最大字节数，前端支持自动跟随最新行、用户滚动上翻时暂停自动滚动并提示有新日志。

### 2. 进度解析只针对已观察到的训练格式

从实际 JSON 风格行读取 `global_step/max_steps` 和有限指标（loss、learning rate、token accuracy、elapsed time）；EvalScope 行仅展示日志，不推断百分比。解析失败不阻塞任务，也不修改原始日志，状态保持 unknown。测试样例必须基于本次已观察到的 ms-swift 行格式及格式变化/乱码输入。

### 3. 评测复用现有 CLI，并增量转发子进程输出

新增固定的 Launcher 评测预检和确认路径，模型只能由成功 SFT run 中已核实的 M2 merged model 解析，评测配置绑定 `configs/evaluation/math-v1.yaml`，阶段标签为 SFT。首个可用版本使用有界 smoke 覆盖：GSM8K 与 MATH-500 各 5 条样本；页面显著标记为 exploratory，不作为 Mini 正式验收或充分统计结论。使用 `run-model-eval` 产生正式 eval.json、evaluation lineage 和独立评测 run；不从前端自由传参。评测前只读检查 evalscope 命令、协议、模型摘要、benchmark 运行条件和无冲突输出目录；任何数据/模型获取可能性均在确认前提示，不在预检中启动下载。

评测子进程的 stdout/stderr 必须逐行/分块转发到监督器可读的有界日志，同时保留 CLI 的最终结果与 lineage 写入职责。避免由服务端另写一套评测指标或直接调用 EvalScope 绕过 CLI。训练和评测共用一个本地活动进程锁，评测失败/部分完成仍保留各自终态与已有 evidence。

### 4. 每次评测有独立执行身份，并关联原模型

评测 `run_id` 与训练 `run_id` 不复用；eval 产物有独立 `eval_id`。目标模型通过已登记的 `model_id` 与 digest 锚定，并由模型 registry/lineage 追溯到 SFT 训练与合并。输出放到训练输出目录下独立、不可覆盖的 evaluation 子目录，运行状态文件保存在 launcher state store。API 仅接受一次性预检确认 token 和目标训练 run 身份，不接受路径、命令、环境或参数覆盖。

这延续项目的 run/evaluation lineage，不用 UI 的短期状态 JSON 代替 CLI 权威产物。报告只展示实际生成指标与 denominator；缺失、失败或 partial 保持其状态。

### 5. 页面采用阶段式工作区，而非继续纵向堆卡片

顶部显示固定配方与当前 run 的阶段状态条：配置/训练/评测/结果。主区域按当前阶段突出一张操作卡：预检与确认、运行状态与日志，或评测结果。配置摘要、完整 resolved config、warning、原始 CLI JSON 等低频内容折叠；主内容以分区/标签查看“运行概览”“日志”“评测结果”“产物与血缘”。训练和评测日志分别标识当前阶段，避免两个 run 的日志混在同一视窗。手机宽度降为单列，固定主操作可达；日志表格可在独立容器内滚动，不造成整页横向溢出。

## Risks / Trade-offs

- [训练工具可能将日志写到不同位置或延迟 flush] → 以实际存在的阶段日志文件为优先，保留 supervisor 事件/兜底信息，展示最近更新时间与等待状态，集成测试覆盖日志晚到。
- [评测结果可能因 benchmark 下载或网络不可用而失败] → 预检只读且不承诺网络可用；显式确认前说明固定 benchmark 与潜在资源/网络开销；失败保留日志和 failed/partial 状态。
- [小样本 smoke 分数波动大] → 固定协议和样本上限，标注 exploratory 与分母，不作为正式 Mini gate 或因果证据。
- [日志包含路径或第三方原始输出] → 不记录密钥/环境值；按 workspace 约束路径读取、限制响应字节、在 UI 以文本转义显示。
- [评测继续占用单卡显存] → 训练/评测共用单活动锁，评测前显示资源提示并由用户再次确认。

## Migration Plan

1. 旧训练状态与 run.json/lineage.json 保持读取兼容；对历史已完成 run，从其 `run.json` 和输出树解析可用日志/模型，不伪造缺失状态。
2. 训练和评测状态采用可区分的 stage/type 字段；旧 state 无该字段时按训练状态兼容显示。
3. 新页面部署后，已有训练结果仍可查看但不会自动补跑评测；用户显式点击评测并通过预检后才创建新评测 run。
4. 回滚时可隐藏 UI 评测入口并恢复 CLI 工作流；训练、评测 CLI 产物保持不变且仍可独立使用。
