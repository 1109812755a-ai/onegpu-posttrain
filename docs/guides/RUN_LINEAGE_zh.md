# Training Run 与模型血缘

## 在 Runs 页面阅读记录

列表和详情首先展示可读的运行名称、阶段、真实状态、输入/产物证据数量和可执行的下一步；失败、部分完成或缺失证据不会被标为成功或可交接。Run ID、实验 ID、完整路径、原始配置与逐项血缘位于“查看…”折叠区，仍可按需核验。日志内容可直接查看，日志文件路径单独折叠；日志或产物未记录时页面明确说明，不以空白代替。

评测先做只读预检，再显式确认启动。确认请求成功后打开精确评测 Run；若网络超时，先查询本次预检对应的 Run，不自动重提启动请求。服务端拒绝或令牌失效则重新预检。修改模型、方案或评测集会清除旧确认，已启动 Run 不受草稿修改影响。模型交接同样只打开带有已核验父模型身份的工作台草稿，不自动选数据或训练。

训练 Run、合并 Run、评测 Run 是不同的执行，不因页面入口不同而复用身份。`run_id` 标识一次实际执行；同一 profile/data/model 的重跑必须有新的 `run_id`。阶段算法和参数方案由 `experiment_id`/profile 表达，不能代替执行身份。

## 标准训练产物

使用项目 CLI 训练时，训练输出目录包含：

- `run.json`：CLI 执行结果和真实终态；
- `lineage.json`：schema version、`run_id`、stage、recipe/config 摘要、父模型和数据集输入、真实产物引用；
- `resolved_config.json`：解析后的实际训练参数；
- `model_artifacts/<model_id>.json`：只有磁盘上确实存在的 adapter/checkpoint 才会登记，并带 SHA-256 与 parent model 关系。

LoRA merge 是独立 `merge` Run：训练 Run 中记录 adapter，merge Run 将 parent model 与 adapter 作为输入，只有 merge 命令成功且合并目录实际存在时才产生 merged model artifact。训练失败、OOM、找不到 checkpoint 或 merge 失败时，lineage 保留真实终态与已发生的输入，`outputs` 为空或只列出此前实际生成的 adapter；不会伪造 merged model。

数据输入通过已登记 `dataset_id`/manifest 与 canonical digest 关联；缺 manifest 或摘要时会标记 lineage incomplete/warning，不伪装成可复现的完整链。被定向补数触发时，训练 lineage 会保存精确的 `targeted_plan_id + version`，但方案 ID 不取代 dataset/model/run 身份。

## 验证

```powershell
 .\.venv\Scripts\python.exe -m onegpu_posttrain validate-lineage --root runs/<stage-run> --require-complete
```

验证会检查 manifest 与 canonical hash、数据集构建 Run 的输出 ID/摘要、训练 Run 的数据 role/hash、父模型 ID/hash、recipe/config 指纹、模型 producing Run/parent/output 摘要、评测 Run 的模型/数据输入及 lineage 身份，并检查重复 ID、断链和图环。`--require-complete` 会把关键缺失证据升级为错误；历史记录或外部 benchmark/Base 仅在证据确实不可本地取得时作为 warning，绝不按显示名称猜关系。失败 Run、部分产物会保持原状态；来自失败 merge 的模型不具备成功交接资格。通过只证明本地证据可追溯，不证明数据质量合格、训练可复现、评测 protocol 相同或模型能力提升。

训练 Run 详情同时区分“本次 Run 直接产物”和“可交接模型”。例如 adapter 经独立 merge Run 生成 merged model 时，系统必须核对训练 Run 输出的 adapter ID/hash、merge Run 对同一 adapter 的输入 ID/hash、merge Run 的成功状态及其 merged 输出 ID/hash，再将该 merged model 作为下阶段候选；不会把 merge 产物伪装成本次训练 Run 的直接输出。

成功 Run 的模型交接按批准 profile 的 parent 规则筛选：CPT → 固定父模型要求的 SFT；SFT merged model → DPO、SimPO、GRPO 分支。父模型选择器和 Run 详情显示阶段、artifact 类型、短 ID、producing Run 与 lineage 链接。交接 URL 只传精确 `model_id`、profile/stage 与来源 `run_id`；训练数据不会自动选择，用户必须返回工作台选数据、重新预检并显式启动。刷新后会恢复相同候选身份；失败 Run、缺失/重复登记、摘要不匹配或不兼容 profile 均没有可执行 CTA。

## Launcher 兼容层

Mini SFT UI 仍保留旧 Launcher 状态/API 与 URL，其训练期间也把 `run_id`、阶段、profile、进度、有限日志及终态镜像到 `.onegpu-posttrain/supervisor/`。该 sidecar 生命周期状态不取代 CLI 写出的 `lineage.json`/模型 artifact；成功与模型评测入口仍以 CLI 真实 lineage 核验为准。

Launcher 的“在实验 Runs 查看完整记录”和历史 Runs 列表都通过同一个 `run_id` 打开 `/api/runs/{run_id}` 的规范详情，不创建第二份执行或前端身份。Runs 详情在训练或关联评测处于 queued/starting/running 时每 1.5 秒重新读取同一 Run 聚合视图，展示状态、日志尾部、真实产物与 lineage；任务终态后停止轮询。成功训练 Run 可从其详情选择适用的阶段评测 profile；数据型 profile 还需选已登记 holdout。服务端重新核验训练 lineage、merged model 内容摘要及评测数据摘要，签发短时一次性确认。评测通过独立子 Run 执行并回链到训练 Run；训练 Run 的状态和 lineage 不会被评测结果覆盖。失败、OOM、中断或缺模型均不能进入评测启动流程。

阶段评测 API 仅接受 `training_run_id`、批准的 `profile_id`、可选的登记 `dataset_id` 和一次性确认令牌；不接受模型路径、recipe、shell 或参数覆盖。评测子 Run 的 `evaluation_run.json`、`evaluation_lineage.json` 和 `eval.json` 保存真实状态、协议、模型/数据身份、指标状态及实际分母。Runs 详情的父/子链接可在训练 Run 与评测 Run 之间往返；未知分数保持 `not_measured`，不会补零。
