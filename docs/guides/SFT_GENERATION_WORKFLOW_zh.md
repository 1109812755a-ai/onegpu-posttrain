# 本地 SFT 生成、审核与训练闭环

本指南说明如何在数据集工作台完成普通或定向 SFT 数据生成；生成结果都使用同一套候选预览、人工审核和数据集登记流程，再单独交给 Launcher 训练和评测。定向诊断与方案准备仍在“分析与改进”模块，点击继续后场景、来源评测、模型和已登记方案会自动带入数据工作台，无需手填 plan/bundle ID。浏览、准备、预检和刷新都不会调用 provider；只有明确确认启动后才会发请求。

## 前置条件

- “参考已有数据集”模式需要一个已登记且摘要有效的 SFT 数据集；它从样本提取问题，让模型补充解答。
- “按主题生成”模式不需要种子数据集，只需主题、能力点、难度和可选约束；提示词与 JSON 输出协议由项目固定，目标最多 1,000 条、请求预算最多为目标的两倍。
- 两种普通模式都需要凭据已在本机配置的 Provider profile，不需要诊断报告、场景定义、角色登记或 holdout。
- 定向补数仍需匹配的逐题诊断、场景定义、训练 seed、角色登记及隔离数据。最终 holdout 只用于隔离检查，不会作为生成样本或提示词上下文。
- 所有输入由服务端通过 dataset/profile ID 在 workspace 内解析；浏览器不接收密钥、文件路径或任意命令。
- 先用 fake provider 做离线验证；真实 provider 可能计费，且结构正确不代表回答语义正确。

## 1. 登记完整 SFT workflow bundle

从仓库根目录运行。每个路径都相对 workspace（默认当前目录），命令只校验并写入不可变 bundle，不会读取密钥或发起网络请求：

```powershell
\.venv\Scripts\python.exe -m onegpu_posttrain register-sft-workflow `
  --plan ".\path\to\plan.yaml" `
  --profile ".\path\to\provider-profile.yaml" `
  --template ".\path\to\prompt-template.yaml" `
  --output-protocol ".\path\to\candidate-protocol.yaml" `
  --definition ".\path\to\scenario-definition.yaml" `
  --report ".\path\to\scenario-report.json" `
  --role-registry ".\path\to\role-registry.json" `
  --train-seeds ".\path\to\sft-train-seeds.jsonl" `
  --policy ".\path\to\sft-data-policy.yaml" `
  --diagnostic ".\path\to\diagnostic-dev.jsonl" `
  --holdout ".\path\to\final-holdout.jsonl"
```

命令会输出 `bundle_id`、配置摘要及每个输入文件的 SHA256，并在 `artifacts/synthetic/<bundle_id>/workflow_bundle.json` 留存身份。相同 bundle 不会覆盖；修改任一输入后应创建新计划身份。当前 UI 执行路径只接受 `stage=sft` 的 bundle。

## 2. 启动本地 UI

需要开启模型生成和 Mini Launcher 时，在项目环境中运行：

```powershell
\.venv\Scripts\python.exe -m onegpu_posttrain serve-ui `
  --workspace . --host 127.0.0.1 --port 8765 `
  --enable-synthetic-generation --enable-experiment-launcher
```

打开 `http://127.0.0.1:8765/?page=workbench`。在“数据集工作台”的“SFT 数据生成”卡片选择“参考已有数据集”或“按主题生成”。定向流程从“分析与改进”的薄弱场景诊断进入，完成诊断和方案准备后点击“在数据模块继续生成与审核”；后续执行、候选审核和登记都在数据工作台完成。`--enable-synthetic-generation` 是单独 opt-in；远程 bind 不允许启用该能力。端口已被占用时不要再启动第二个相同端口服务；确认当前服务是否仍运行，必要时由用户停止旧服务后重启。

启用后，UI 会提供只读准备状态接口 `GET /api/synthetic/sft/readiness`。它返回 SFT 数据集 ID/摘要、普通/定向资格、Provider 凭据是否已配置及缺项原因；密钥值不会被读取或返回。普通生成按“准备方案 → 预检 → 确认并生成”操作。准备与预检都不调用 Provider。

## 3. 预检、预算确认、生成

1. 选择生成方式。参考模式选择一个 SFT 数据集；按主题模式填写主题、能力点、难度和可选约束，不需要导入任何种子数据。确认目标数量与本地 Provider profile。只有一个凭据可用的 profile 时会自动选中；存在多个时需要选择。
2. 点击“准备生成方案”，再点击“预检方案”。两步只绑定并核对 immutable bundle，不会联网调用模型。
3. 预检通过后点击“确认并生成”，再次确认模型与最大请求数。费用无法准确预估时以 Provider 控制台为准。
4. 页面自动轮询进度；刷新后会依据浏览器保存的 execution ID 恢复状态。切换普通生成方式只会更改草稿，不会清除或重启原任务；有任务运行时可编辑下一份草稿，但需等当前任务结束后再预检/启动。跨模块后可通过页面顶部“活动任务”返回生成或训练/评测 Run。UI 服务重启不会重放请求。
5. 目标数量是停止目标而非质量门槛；若被过滤后数量不足，会保留实际结果和拒绝统计，不会自动放宽规则。

真实 provider 凭据通过已登记 profile 指定的环境变量读取，或放在 workspace 根目录 `.env.local`。只在该本地文件中放密钥，不要写进 plan/profile、页面、命令行参数、截图或日志；`.env.local` 不应提交。UI 不接受或回显密钥值，异常也只记录错误类型。真实 provider profile 必须在 workspace 内并随 workflow bundle 固定摘要；更改 provider、模型、提示词、策略或输入后重新登记新 bundle。

## 4. 人工审核并登记数据

任务完成或部分完成且有可用样本后，普通和定向流程都会显示同一个“审核候选样本”步骤。预览最多显示 5 条 canonical 样本；检查内容是否符合目标、是否包含敏感信息和是否适合训练。定向模式会同时展示场景、来源评测和模型身份供追溯，但这些诊断样本不会进入生成提示词。格式/摘要通过不等于答案正确，系统会标明语义正确性未独立验证。

人工确认审核后点击“审核通过并登记数据集”。普通与定向生成共用预览、人工确认和登记组件，但输入契约及血缘不同：普通模式使用种子数据或主题；定向模式自动绑定诊断场景和不可变 bundle。登记会重新验证 canonical SFT schema 和数量，基于内容 SHA256 创建新的稳定 dataset ID，不改写生成中间产物或源 seed。空数据、schema 错误、摘要不匹配或数量不一致都会拒绝登记。已存在的完全相同内容会复用原不可变数据集，不会覆盖 manifest。

新数据集位于：

```text
data/processed/model-generated/<dataset_id>/
├── canonical/sft.jsonl
├── manifests/sft-generated-v1.yaml
├── reports/processing_report.json
├── reports/quality/sft.json
└── runs/<run_id>/lineage.json
```

`origin.kind=model_generated`；模型由 provider profile 声明。没有可靠 license 信息时为 `not_provided`，不会臆测为已核验。质量状态为 `report_only`：schema、重复 ID、长度/来源统计等报告不能替代独立语义评估。生成来源、bundle、execution ID、真实样本数及摘要写入 dataset manifest 和 run lineage。参考模式记录 seed dataset 作为父节点；按主题模式是血缘根节点 `parents=[]`，只记录 prompt/template/protocol/provider 的摘要和配置证据，不伪造数据集父节点。按主题生成会标记 `semantic_correctness=not_independently_verified`；格式校验不代表数学答案正确。

## 5. 用数据集训练与评测

登记成功后，按钮会打开 SFT 工作台，并用 `dataset_id` 预选新数据集。选择 `sft-registered-v1` 可从已登记且本机可用的 Base 或合并 CPT 模型开始训练；预检会核对权重、摘要、Transformers 配置、tokenizer、SFT 数据与本机训练依赖。任一模型不兼容或权重缺失时会显示阻断原因，不自动下载/替换。`sft-mini-1024-v1` 仍使用固定 Mini M1-CPT 配方。所有训练均须单独预检并由用户确认；Run 记录父模型 ID/hash、数据集身份和有效配置。

整体血缘：

```text
参考数据集（仅参考模式）或主题/能力点（按主题模式）
  → SFT 生成 execution（provider/profile/prompt/policy）
  → generated intermediate dataset
  → 人工审核与 canonical 注册（新 dataset ID + SHA256 + quality report）
  → SFT training run（引用 registered dataset ID/hash 与 parent model）
  → merge/model artifact
  → evaluation run（引用模型、训练 run、数据及评测协议）
```

按主题模式的离线 fake-provider walkthrough：使用测试 Provider 返回 `{ "question": "Compute 2+3.", "solution": "2+3=5.", "final_answer": "5" }`，完成生成、候选预览与显式登记；检查新 manifest 的 `lineage.parents` 为空，并检查 `semantic_correctness` 为 `not_independently_verified`。达到 Mini 完整闭环还需要本机模型、训练依赖和评测资源可用；离线 fake-provider 测试只验证软件与血缘，不证明真实生成质量、训练有效性或 benchmark 提升。
