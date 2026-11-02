# 定向合成数据：离线诊断与计划登记

本指南介绍如何从评测中的薄弱场景创建一个**尚未执行**的定向数据计划。当前示例只使用仓库内的虚构小样本和 fake/local 身份，不下载数据、不访问网络、不调用模型、不训练，也不声称模型能力得到提升。

## 工作链路与证据边界

```text
diagnostic_dev 逐题结果
  → 按固定场景标签聚合并检查样本分母
  → 从 train_seed 中选择未命中保护索引的训练题
  → 登记生成计划（本示例止于此）
  → 可由用户显式执行生成 CLI、候选验证/策略导出，再单独训练和 final_holdout 复测
```

- `train_seed` 是允许用于训练/生成上下文的项目自有训练种子；不得把诊断题正文、答案或最终留出题提供给生成器。
- `diagnostic_dev` 用于发现问题和形成训练目标，不是最终效果证明。
- `final_holdout` 必须保持隔离，只能在完成训练后用于最终评测；它与训练种子及其祖先不得重叠。
- 场景指标按逐题 prediction 与稳定 case/problem ID 关联。只有总体分数时，不会推算或伪造场景分数。
- “available / insufficient_evidence”表示证据是否达到本次计划登记的最低样本数，不等于数据质量结论、统计显著性或能力达标。
- 示例中的阈值 `0.66`、最低样本数 `2`、请求预算和训练预算仅用于演示身份绑定及离线登记，**不是推荐阈值、通用质量门槛或准确率目标**。真实项目应按评测协议、场景风险和统计功效确定并记录阈值来源。
- 选择训练种子会检查项目已有 BenchmarkIndex；这只能说明与当前索引的比对结果，不能证明不存在未知污染或语义近重复。

## 运行离线示例

在仓库根目录执行：

```powershell
.venv/Scripts/python.exe scripts/targeted_data_offline_example.py
```

示例会在系统临时目录创建临时报告、角色登记和计划文件，随后删除临时目录。输出应包含：

- `evidence_kind: fixture`
- 一个场景为 `available`、另一个因分母不足为 `insufficient_evidence`
- `training_seed_shortfall`（因为仓库示例种子数故意小于目标数）
- `execution_status: not_generated_not_trained_not_reevaluated`
- `provider_calls: 0` 和 `network_used: false`

计划登记只固定输入身份与配置摘要，不会启动 provider。即使离线演示通过，也不代表真实模型生成、数据质量或训练收益已经验证。

## 本地前端：配置草稿与报告追踪

默认启动本地 UI（仅 loopback）：

```powershell
onegpu-posttrain serve-ui --workspace . --port 8766 --enable-policy-editor
```

打开 `http://127.0.0.1:8766/?page=targeted`，选择“定向补数”。页面可查看当前本地白名单报告中的场景分母、来源 eval/model/dataset/protocol、已登记计划、生成/验证与复测摘要；也可编辑 provider profile、非敏感 endpoint/model/凭据环境变量名、提示词、输出字段和调用预算并导出 `targeted-synthetic-workbench-draft.json`。

若需要临时从同一可信局域网的其他设备访问，可显式加上 `--host 0.0.0.0 --allow-remote`，再使用运行服务电脑的局域网 IP 打开页面。此模式会向可连接的客户端暴露本地报告/数据接口；若同时加上 `--enable-policy-editor`，策略预览和草稿保存接口也可被访问。它不是带认证的远程服务，不会修改防火墙；请仅在可信网络短时使用，并在完成后停止服务。

此 JSON 是**工作台草稿**，不是 `TargetedGenerationPlan`，不能直接传给执行 CLI。仍需补齐 train_seed/角色登记/污染索引/起点模型/final_holdout/预算等真实身份后，按 CLI schema 登记计划。页面不读取凭据环境变量、不接受 API 密钥值、不发出模型请求、不启动训练。若本地 UI 端口拒绝连接，请在同一仓库终端确认命令仍在运行、端口未被占用，再刷新页面。

页面会区分分母缺失与 0、样本不足、fake 与远程来源、partial/failed、费用 unknown/estimate 和待生成/待训练/待复测。配置变化或诊断报告身份变化后，最近导出的草稿会标为过期；报告读取失败不会清空当前编辑状态。

配置草稿区仍然只是导出草稿，不会执行 provider；同页新增的“SFT 生成 → 审核 → 登记为数据集”是另一条显式执行路径，只有以 `--enable-synthetic-generation` 启动本地 UI 后才可用。它要求此前通过 `register-sft-workflow` 登记完整 bundle，预检不会发请求，只有用户二次确认才会执行 provider。完整前置条件、命令、预算/费用、审核和数据血缘见 [本地 SFT 生成、审核与训练闭环](SFT_GENERATION_WORKFLOW_zh.md)。响应式与键盘检查记录见[定向补数 UI 交互验收走查](TARGETED_DATA_UI_WALKTHROUGH_zh.md)。

从评测结果选择“分析 / 补充数据”会把真实 `eval_id`、可解析的 `model_id`、held-out `dataset_id` 和阶段带入薄弱场景分析。继续到定向 SFT 生成时，这些上下文沿返回链接保留；数据登记完成后可用“返回原薄弱场景诊断”回到原分析身份。该回跳只导航，不自动触发生成、训练或评测。

## Provider 配置、dry-run 与显式生成

仓库提供 fake 与 OpenAI-compatible 配置样例：

- [本地 fake profile](../../configs/synthetic/examples/provider.fake.yaml)
- [OpenAI-compatible profile 模板](../../configs/synthetic/examples/provider.openai-compatible.example.yaml)
- [DashScope Qwen3.6-Plus profile 示例](../../configs/synthetic/examples/provider.dashscope-qwen3.6-plus.example.yaml)
- [SFT 提示词模板](../../configs/synthetic/examples/prompt.sft-math.yaml)
- [SFT 候选输出协议](../../configs/synthetic/examples/output.sft-candidate.yaml)

保存计划前，计划中的 `provider_profile_id` 必须等于 profile 的 `profile_id`，`prompt_template_sha256` 必须由提示词模板正文计算得到。修改任一配置后，应重新登记计划；不能覆盖已有计划或沿用旧摘要。

先用 fake 配置验证全套输入，不需要账号或密钥：

```powershell
$planId = "<registered-plan-id>"
$planDir = "artifacts/synthetic/$planId"
$inputs = @(
  "--plan", "$planDir/plan.json",
  "--profile", "configs/synthetic/examples/provider.fake.yaml",
  "--template", "configs/synthetic/examples/prompt.sft-math.yaml",
  "--output-protocol", "configs/synthetic/examples/output.sft-candidate.yaml",
  "--definition", "<scenario-definition.yaml>",
  "--report", "<scenario-report.json>",
  "--role-registry", "<role-registry.json>",
  "--train-seeds", "<train-seed.jsonl>",
  "--protected-jsonl", "<diagnostic-dev.jsonl>",
  "--protected-jsonl", "<final-holdout.jsonl>"
)
onegpu-posttrain dry-run-targeted-generation @inputs
```

dry-run 检查计划、报告、scenario definition、角色登记、训练集摘要和污染索引，输出候选数/短缺与预算；它不会读取凭据环境变量、构造 provider 或发请求。显式 fake 执行只生成本地模拟响应：

```powershell
onegpu-posttrain generate-targeted-data @inputs --execute `
  --output "$planDir/private/fake-run-001.jsonl"
```

真实 provider 需要先复制并修改 OpenAI-compatible profile 模板，让 profile ID 与计划一致，并填写服务 endpoint、模型名及服务支持参数。若服务提供不可变模型 revision，应填写该 revision；若不提供则保留 `null`，并接受复现边界只能绑定到用户填写的模型名和当时的服务行为，不能声称精确复现。修改 profile 后应重新登记计划。

凭据可以通过环境变量提供，也可以放在仓库根目录的本地 `.env.local` 文件。先复制安全模板并编辑：

```powershell
Copy-Item configs/synthetic/examples/env.local.example .env.local
notepad .env.local
```

把占位值替换为**已轮换的新密钥**。该文件被 `.gitignore` 忽略；不要提交、截图或分享。当前支持 `KEY=value`、`export KEY=value`、单/双引号包裹值和 `#` 注释行；不执行 shell、不支持变量插值。只放当前 profile 指定的凭据变量。当前终端已有的非空环境变量优先于文件值。密钥曾在聊天中暴露，应先在供应方控制台撤销并轮换，不能继续使用旧值。

之后在仓库根目录运行；只有带 `--execute` 的真实生成命令会读取 `.env.local`：

```powershell
onegpu-posttrain dry-run-targeted-generation @inputs
onegpu-posttrain generate-targeted-data @inputs --execute `
  --output "$planDir/private/run-001.jsonl"
```

第二条命令中的 `--execute` 是唯一启动 provider 请求的 CLI 入口；dry-run、UI 和配置检查均不读取该文件。真实响应写在 plan 的 `private/` 目录；请勿把该目录提交或通过 UI/公开报告分享。端点必须是无认证信息的 HTTP(S) URL，`api_key_env` 只保存环境变量**名称**。提示词只接收已登记 `train_seed` 记录的题目和场景标签，不接收 reference/gold、diagnostic 或 final-holdout 正文。

### 预算、费用与停止/恢复

- `max_requests` 约束网络尝试总数，包含失败和重试；每次 `max_output_tokens` 作为上限参数传给服务，timeout 受计划控制。单计划顺序执行，不并发补满目标。请求账本位于 `artifacts/synthetic/<plan_id>/requests.jsonl`，重复执行不会重置已消费预算。
- 若服务返回输入/输出 token usage 且配置了用户声明费率，报告的是**估算费用**，不是账单或硬费用上限；费率或 usage 缺失时费用为 `unknown`，不按零处理。服务可能忽略 token 参数或在超时后仍计费。
- 成功响应先原子写入带 hash 的本地缓存，再记为 completed；同配置再次执行会校验并复用缓存，不重复请求。配置变化产生不同 request key。
- HTTP 429 等明确可重试失败按 `max_retries` 与剩余总请求预算重试。超时/传输中断标为 `uncertain`，可能已被服务受理；默认不自动重发。
- 正常情况下用新 `--output` 路径重新运行即可续做。若进程被强制结束，可能残留 `.execution.lock`：先读取其中 PID 并确认进程已退出，**只有确认不再运行后**才手动删除锁；不要在任务仍运行时删除。随后重新执行可恢复已完成缓存。若账本含 `pending/uncertain` 且没有完整缓存，只有在接受潜在重复费用后才显式加 `--retry-uncertain`。

```powershell
$lockPath = "$planDir/.execution.lock"
Get-Content $lockPath | ConvertFrom-Json
$ownerPid = (Get-Content $lockPath | ConvertFrom-Json).pid
Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
# 仅在确认没有对应运行进程后：
Remove-Item -LiteralPath $lockPath

onegpu-posttrain generate-targeted-data @inputs --execute `
  --retry-uncertain --output "$planDir/private/recovery-001.jsonl"
```

fake 示例不会产生远程费用。真实 API 的价格和服务条款由用户/供应商决定；本项目只记录 usage 与用户提供费率的估算，不自动推断价格。

## 候选验证、训练数据导出与离线端到端示例

候选分成三层状态，避免把“格式正确”误读为“答案正确”：

1. 结构/身份/完成原因/目标 tokenizer 渲染长度校验。通过仅表示 `candidate_ready_for_quality_review`，不会设置正确标签。
2. 使用训练种子内保留的可信 reference 做最终答案检查；缺 reference、无法解析、不支持的 reference 类型或缺数学等价验证器时，分别保留 `unverified` / `unparseable` / `unsupported`，不能把未知当错误。
3. 通过现有数据 policy 选择 SFT 或同题 Preference。Preference 必须具有同题已验证的 correct 与 incorrect 两侧；缺一侧、全正确、全错误、未知标签和长度比失败都会报告缺额，不复制候选补数。

这类答案验证只验证最终答案，不证明推理过程完整、无错误或具有泛化能力。静态数据质量报告也不是模型效果结论。
候选和训练数据 manifest 会保留 train-seed parent、policy、provider adapter/model、提示词摘要及 revision 状态；若 provider 不提供固定 revision，血缘明确记录 `unavailable`，不宣称远端行为可精确复现。原始响应与候选/训练正文仅在本地 private 目录，UI artifact catalog 不允许读取该目录。

生成登记还会记录 `provider_identity_status=profile_declared_not_independently_verified`。这表示模型名/revision 来自操作者选定的 Provider profile，不等于服务端已证明实际加载的权重。当前 OpenAI-compatible Provider 契约没有可信的 loaded-model attestation，因此即使远端模型名恰好与已登记模型相同，也不会填写 `registered_model_id` 或权重 hash；缺少 revision 时明确记为 `unavailable`。只有将来接入可验证加载身份的本地推理服务，才能关联项目登记模型 ID/hash。

下面的离线示例用 fake provider、程序化构造且 reference 独立计算的新题走过候选校验、SFT policy 导出、quality report 与标准数据血缘；它不会联网、计费、训练或复测模型：

```powershell
.venv/Scripts/python.exe scripts/targeted_synthetic_pipeline_example.py
```

预期为 `evidence_kind: fixture`、`network_used: false`、`lineage_errors: []`、`quality_gate_status: NOT_EVALUATED` 和 `model_effect: not_evaluated`。示例输出中的候选/训练文本只写到自动清理的临时 private 目录。

## SFT 最终整数答案噪声实验

项目另提供可复现的 `sft_final_integer_answer_noise_v1` 研究变换，仅支持有可信整数 reference 且解答里存在明确整数最终答案的 SFT 样本。该变换按 `seed + record_id` 稳定挑选，改变最终整数答案，并记录 parent、目标/实际比例、transform 版本及整数不等价证据；无 reference、非整数格式或不支持答案会跳过，不复制样本填满比例。它不修改 Preference chosen/rejected，也不会自动启动训练或复测。

## 仓库示例数据

`tests/fixtures/targeted_synthetic/` 包含项目自有的微型 fixture：

- `train_seed.jsonl`：两条合成训练种子记录。
- `diagnostic_dev.jsonl`：三条诊断评测记录，带固定场景 metadata。
- `diagnostic_predictions.jsonl`：与诊断记录按 ID 对齐的逐题结果。
- `final_holdout.jsonl`：两条独立的留出 fixture，只用于演示隔离和身份登记。

这些 fixture 很小，只供 schema、血缘和离线命令测试；不得将其指标解释为模型效果或数据集代表性证据。
