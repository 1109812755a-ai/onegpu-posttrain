# Spec Delta

## Purpose

让用户从本地 Web 工作台启动并观察一个明确受限的 Mini SFT 训练 run，同时保留 CLI 为稳定执行边界。首版用于快速验证前端训练启动链路，不提供通用训练平台或任务调度。

## ADDED Requirements

### Requirement: Launcher is explicit and local-only
SFT 启动能力 MUST 默认关闭，只能由维护者显式启用；启用后启动接口 MUST 只接受 loopback 本机请求，且不能随远程 UI 模式开放。

#### Scenario: Launcher is not enabled
- **WHEN** UI 服务未启用 SFT Launcher 时收到预检或启动请求
- **THEN** 服务拒绝请求，页面说明当前为只读模式，且不创建训练进程或 run 产物

#### Scenario: Remote access is requested with the launcher
- **WHEN** 服务配置同时启用 SFT Launcher 与非 loopback/远程监听
- **THEN** 服务启动失败并说明 Launcher 仅支持本机访问，且不修改系统防火墙

### Requirement: Only the fixed Mini SFT recipe is launchable
首版 MUST 只允许用户选择版本化 Mini SFT 1,024-token 配方及其绑定的本地父模型和数据集；浏览器不得提交任意 recipe、模型/数据路径、脚本、命令、环境变量或训练参数覆盖。

#### Scenario: User requests the supported SFT recipe
- **WHEN** 用户选择 `recipes/mini/23-sft-1024-fallback.yaml`
- **THEN** Launcher 展示其固定模型、数据、seed、预算、训练配置和 exploratory 用途供预检

#### Scenario: User requests another stage or modifies a fixed input
- **WHEN** 用户提交 CPT、DPO、SimPO、GRPO、未知 recipe、额外参数或与配方绑定值不符的输入
- **THEN** 服务拒绝请求，不启动进程且不修改配方或历史 artifact

### Requirement: Preflight is visible and has no training side effects
每次启动 MUST 先对固定 recipe、模型、数据和输出位置执行只读预检，并展示配置摘要、数据/模型 ID 与可得摘要、数据量、exploratory/质量提示、输出位置及 best-effort GPU 状态；预检 MUST NOT 下载或修改输入，也不得启动训练。

#### Scenario: Required source or integrity check fails
- **WHEN** 固定模型/数据缺失、manifest 或摘要不匹配、输出冲突或必需环境检查失败
- **THEN** 页面列出阻断原因并提供下一步，且不创建训练 run

#### Scenario: Exploratory input passes structural preflight
- **WHEN** 固定输入完整且通过结构/血缘预检，但数据仍标记为 exploratory 或语义质量未审
- **THEN** 页面明确显示该限制并允许用户显式决定是否继续，不把它伪装成 official readiness

### Requirement: Launch requires confirmation of the current preflight
Launcher MUST 只在用户确认当前未过期预检的确切配置摘要后启动 SFT，并为该次实际执行使用唯一 `run_id`；确认不可复用，输入或配置变化时 MUST 重新预检。

#### Scenario: User confirms the displayed configuration
- **WHEN** 用户显式确认当前通过的预检
- **THEN** Launcher 使用既有 `run-sft` CLI 执行，并将同一唯一 `run_id` 写入运行记录、状态与 lineage

#### Scenario: Preflight is stale, expired, or already consumed
- **WHEN** 请求使用过期/已用预检令牌，或配置摘要与当前绑定的模型/数据不一致
- **THEN** 服务拒绝启动并要求重新预检，不产生重复训练 run

### Requirement: At most one UI-launched SFT task is active
首版 MUST 最多运行一个 UI 启动的 SFT 任务；不得排队、并发、覆盖既有输出或自动重试。

#### Scenario: A UI-launched SFT is already active
- **WHEN** 用户在另一个 UI SFT 仍运行时再次启动
- **THEN** 服务返回当前 run_id 与状态并拒绝第二项任务

#### Scenario: A browser page is closed during training
- **WHEN** 用户关闭或离开浏览器页面但本地 UI 服务仍运行
- **THEN** SFT 进程继续运行，用户重新打开页面后可查询同一 run 状态和日志

### Requirement: Run status and bounded logs remain truthful
页面 MUST 展示实际 `run_id`、SFT 状态、开始时间、运行时长、受限日志尾部及可获得的训练进度；未知进度不得显示成零或成功，失败/OOM/中断 MUST 保留实际终态与已产生证据。

#### Scenario: SFT reaches a terminal state
- **WHEN** SFT 成功完成、失败或 OOM
- **THEN** 页面显示真实终态、退出信息、可用日志与真实 adapter/merged-model/lineage 引用，不虚构缺失产物

#### Scenario: Progress is not available yet
- **WHEN** CLI 尚未输出可解析的 step/epoch 进度
- **THEN** 页面显示等待/未知进度并提供最新日志，不显示 0% 或假进度

### Requirement: Stopping the UI service is explicit in task semantics
UI Launcher MUST 将本地服务作为 SFT 进程的生命周期监督者；浏览器关闭不得停止 run，服务关闭时 MUST 请求训练进程组优雅终止并将未完成 run 标为 interrupted，而非成功。

#### Scenario: UI service shuts down during active SFT
- **WHEN** 本地 UI 服务退出时仍有活动 SFT 进程
- **THEN** 服务请求进程组停止、保留已有日志/检查点证据，并把非终态记录标记为 interrupted；重启后不得自动续训

### Requirement: CLI and evaluation remain independently usable
Launcher MUST 复用现有 `run-sft` CLI；默认 UI 与现有 CLI 行为保持不变，且首版评测继续由 CLI 显式执行。

#### Scenario: User runs SFT from the terminal
- **WHEN** 用户直接使用现有 CLI 执行 SFT
- **THEN** 不需要启动 Web 服务，CLI 自动生成唯一 run_id 的既有行为继续有效

#### Scenario: UI-launched SFT is complete but not evaluated
- **WHEN** 前端启动的 SFT 完成且没有评测 run
- **THEN** 页面显示模型训练已完成、统一评测待执行，并提供现有评测 CLI 的后续说明，不显示模型能力提升结论
