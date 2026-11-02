# 本地实验 Launcher 使用说明

可选的 Launcher 提供一条范围受控的 Mini SFT 与探索性评测流程。它不是通用任务调度器，不接受任意 recipe、路径或命令参数。

## 启动前端

在仓库根目录将项目 `train` extra 安装到 `.venv` 后运行：

```powershell
.\.venv\Scripts\python.exe -m onegpu_posttrain serve-ui --workspace . --enable-experiment-launcher
```

打开 `http://127.0.0.1:8765/?page=launcher`。如果 UI 服务已经运行，请复用该服务，不要在同一端口重复启动。Launcher 只接受 loopback 本机及同源请求。

## 浏览历史 Runs

点击左侧“实验 Runs”，或打开 `http://127.0.0.1:8765/?page=runs`。列表默认只读读取本地 `runs/**/run.json`、`runs/**/evaluation_run.json` 和 `.onegpu-posttrain/launcher/runs/*.json`；支持阶段、状态和关键词筛选，筛选条件及选中 `run_id` 保存在 URL 中，刷新和浏览器前进/后退会恢复当前视图。

打开一条记录可查看其来源、状态、时间、真实评测指标及分母/状态、受限日志尾部、配置、输出文件和已记录的 lineage。失败/部分完成/中断、历史缺字段、父模型或父运行不可解析等情况会单独显示；没有分数时不会补 0。活动 Run 详情每 1.5 秒自动刷新。Launcher 完成页的“在实验 Runs 查看完整记录”会打开对应训练 run，训练 Run 的 dataset handoff 会保留到返回工作区的链接。Runs 详情可发起适用的独立评测；评测详情提供“分析 / 补充数据”入口，带入 eval/model/dataset/stage 身份进入薄弱场景工作区。导航只传递已记录 ID，不会自动启动生成、训练或评测。评测是单独的 `run_id`，不会改写训练 Run 的终态；训练详情可打开其关联评测详情。

成功训练 Run 详情还会按 profile parent 规则列出可交接模型。若目标模型由独立 merge Run 产出，系统核对训练 Run 的 adapter 输出、merge Run 对同一 adapter 的 ID/hash 输入、成功状态及 merged 输出摘要后，才显示“用作下一阶段父模型”。点击会准确预选阶段/profile/model ID 并显示来源 Run；不会带入训练数据或预检令牌。用户仍需另选数据、重新预检并显式启动。失败 Run、缺 lineage、重复模型登记、摘要不匹配或无兼容 profile 时不提供交接按钮并显示原因。

Runs API 是只读的：`GET /api/runs` 支持 `stage`、`status`、`q`、`limit`、`offset`；`GET /api/runs/{run_id}` 返回白名单详情；`GET /api/runs/{run_id}/artifacts/{filename}` 仅允许详情中发现的固定 artifact 文件。读取器限制扫描记录数、单文件大小和日志尾部，并拒绝符号链接、越出工作区的路径和未知文件名；不会执行命令、读取模型权重或改写历史记录。

## 训练流程

### 页面操作与禁用原因

训练与评测操作区固定在各自工作流卡片底部附近，滚动页面时保持可见；每一步只突出当前主要动作。先检查配置/预检，核对摘要后再启动，不会因预检通过而自动启动。

- 未选择训练数据或兼容父模型时，操作区会说明缺项，并提供跳转到对应控件的入口；没有合适数据时可直接返回“数据”模块登记。
- 评测启动前需先选当前阶段适用的协议；协议要求 holdout 时，还要选择对应已登记数据集。缺项会在预检按钮旁明确提示并提供跳转入口。
- 数据许可状态未独立核验时，只有勾选页面上的来源/许可确认框才可启动；此操作只记录用户确认，不会更改数据许可状态。
- 本机 GPU 已被训练或评测占用时，启动按钮会显示忙碌状态，并提供“查看当前 Run”入口。训练与评测共用本地 GPU 任务锁，不允许重复启动。
- 键盘操作可用 Tab 聚焦主操作和补救入口；窄屏（含约 390 CSS 像素）下按钮改为纵向排列，页面内滚动即可到达，不遮住表单内容。

1. SFT profile 固定为 `sft-mini-1024-v1`，父模型选择规则也固定为批准的 Mini CPT M1；页面可在已登记 SFT train 数据集中选择 `dataset_id`，不允许提交模型路径或 recipe 覆盖。`GET /api/launcher/capabilities` 列出可选登记数据集。
2. 在阶段工作台选择训练方案、父模型和一个或多个同阶段已登记数据集，然后点击“检查并预检”。当前真实 Web 训练入口是 Mini SFT，底层固定配方消费单个 dataset ID：如多选，先在下方“组合所选数据集”中执行预检，确认拼接或加权抽样策略并登记 mixture；Launcher 会把选择替换成新 mixture ID，再按该 ID 进行训练预检。组合动作不会自动启动训练。切换阶段、方案、父模型、数据集或组合策略会使此前预检摘要失效。
3. 检查解析后的配置和风险提示，再确认一次性令牌以启动训练。来源/许可尚未核验的数据需要明确确认 dataset ID；确认只记录用户选择，不会把许可升级成 verified。
4. 运行面板自动轮询真实训练/合并日志与指标。向上滚动会暂停自动跟随；点击“有新日志 · 回到最新”恢复。浏览器刷新后会从本地状态恢复最近一次训练。
5. 只有 CLI 的 run 与 lineage 记录确认成功、且 merged M2 模型已登记，训练才显示为成功。

### 训练期间验证数据

阶段工作台默认选择“不使用验证”，兼容现有训练配置。需要训练期验证时，可选择已登记的独立验证数据集，或将当前训练集按固定随机种子切分：

- 独立验证数据必须通过当前阶段 schema、manifest、canonical hash 与样本角色校验；SFT/Preference 等使用 `validation` 或 `diagnostic_dev` 角色，CPT 只允许 `diagnostic_dev`。`final_holdout` 仅供训练后的独立评测，不能作为训练期验证输入。
- 随机切分比例限制为 5%–40%，并记录种子与算法版本 `onegpu-group-sha256-v1`。相同登记数据摘要、比例和种子会得到相同成员；重复题目/来源组以及 Preference pair 保持在同一分区。原登记数据不会被改写，训练和验证输入及包含逐条成员摘要的 `split_manifest.json` 会写入当前 Run 的独立派生文件。
- 预检摘要、一次性确认、resolved config、Run 和 lineage 会记录验证来源/成员摘要及计数。训练命令使用 ms-swift 的 `--val_dataset` 与阶段配置的评测策略，不会同时打开其隐式 `--split_dataset_ratio`；验证指标是训练期指标，不替代正式阶段评测。
- 更改验证模式、数据集、比例或种子会清除旧预检和确认，必须重新检查后启动。随机切分会减少用于梯度更新的训练样本，并不能替代去重或正式 holdout 评测。

Mini SFT 的旧 Launcher 状态/API 保持兼容，同时将统一生命周期镜像写入 `.onegpu-posttrain/supervisor/runs/`，有界通用日志写入同目录的 `logs/`。这不会改变现有 Runs 展示或 `run_id`；后续各阶段可以复用统一状态协议。

### 阶段 Profile 与只读输入预检 API

- `GET /api/training/stage-profiles` 返回 CPT、SFT、DPO、SimPO、GRPO 的批准 profile 清单。DPO 与 SimPO 是独立算法/profile；Preference 是它们共同消费的数据阶段。
- `POST /api/training/preflight` 接受 `stage`、`profile_id`、`dataset_ids` 和 `parent_model_id`。请求不接受路径、recipe、命令、环境变量或参数覆盖；数据与模型只能通过本地已登记 ID 定位。
- 预检会核对 profile/recipe、dataset manifest 与关联报告、canonical SHA-256、阶段 schema/训练角色、父模型摘要和 producing-run lineage，以及当前 Launcher/`swift` 可用性。结果包括逐项 checks、blockers、warnings、数据/模型摘要和硬件观察值。
- 固定 Mini SFT 与通用阶段预检使用同一 Python Scripts 搜索路径检查 `swift`，也使用同一父模型 producing-run 校验；合并 Run 可位于模型目录下的 `merge_runs/<run_id>/lineage.json`。
- 固定 Mini SFT 的默认数据由批准 recipe 显式绑定 canonical 文件；其他已登记 SFT 数据必须在 manifest `metadata.canonical_path` 中声明相对于数据集 artifact 根目录的路径，且文件摘要与 `canonical_records_sha256` 一致。缺少路径时预检会阻断，不会猜测或改用其他文件。
- 缺少质量报告或许可证明会如实显示为 warning，不按样本数量自动拒绝探索；证据摘要不匹配、阶段/角色冲突、父模型无效则阻断。GPU 状态是观察值，不代表显存保证。
- 基础阶段预检接口只读；`POST /api/training/preflight/confirm` 会签发绑定当前证据、300 秒有效且仅内存保存的一次性令牌，但不创建训练 Run、不启动进程、不下载模型或数据。网页训练目前仅固定 Mini SFT 可启动，内部仍使用兼容的 `/api/launcher/*` 受控确认协议；它会与通用阶段预检进行身份摘要一致性核对。CPT、DPO、SimPO、GRPO 仍显示 blocker，不会创建训练 Run。独立阶段评测使用 Runs 详情中的单独短时确认令牌。详见[阶段预检确认安全边界](STAGE_PREFLIGHT_SECURITY_zh.md)。

## 探索性评测

SFT 成功且登记的 merged M2-SFT 已核实后，打开“评测”并执行只读预检。Launcher 当前仅支持固定的 `math-eval-v1`：

- 配方：`recipes/mini/30-eval-math-v1.yaml`
- 协议：`configs/evaluation/math-v1-launcher-mini.yaml`
- 探索性抽样：GSM8K 5 条；MATH-500 从 Level 1–5 各取 1 条，共 5 条。总计 10 条；不会把 EvalScope 的 `--limit` 误当成跨子集总数。
- 基准：GSM8K 与 MATH-500，各最多 5 条，seed 42

预检不会创建运行产物、下载基准数据或加载模型。用户明确确认后，系统会创建独立的评测 `run_id`，新评测产物放在较浅的 `runs/evaluations/<evaluation_run_id>/`，减少 Windows 深层路径失败；历史评测仍从 state 记录的原嵌套目录读取，不会搬迁。路径预检会按 EvalScope review 缓存文件名估算 Windows 路径预算，超限时在启动前给出阻断提示，不要求开启全局长路径设置。训练与评测共用一个本地 GPU 任务锁。刷新浏览器会恢复状态和日志；关闭 UI 服务会中断当前任务并记录中断状态。

若 EvalScope 所需 benchmark 尚未缓存，确认启动后的评测执行期间可能下载数据；启动前会明确提示。该结果仅为探索性结果，不能当作完整 GSM8K 或 MATH-500 基准分数。当前固定配置总计最多 10 条（GSM8K 5 条、MATH-500 五个难度级别各 1 条）。每个展示的分数都带 EvalScope 实际报告的样本数；基准报告缺失时保留缺失并将评测标为 partial，不会伪造 0 分。

评测 lineage 记录精确的 merged `model_id` 与摘要、登记的模型生产/合并 run、评测 `run_id` 与 `eval_id`、协议、benchmark 数据集 ID 和可用指标。可以在评测页查看 `eval.json`、`evaluation_lineage.json` 和 `evaluation_run.json`。

### 从 Runs 发起各阶段评测

在训练工作台打开一个成功的阶段 Run，评测工作台会从阶段 profile 目录给出适用方案。CPT 选择 `cpt-math-v1` 时还需登记的 CPT holdout；DPO/SimPO 的 `preference-heldout-v1` 需登记的 `preference_eval` 数据集；共享数学 profile 和 GRPO profile 不接受额外数据集。对于任何有已登记 merged model 的阶段，还可以选择 `choice-accuracy-v1` 与已登记的 `evaluation_only` 客观题 suite。客观题预检核对数据角色、处理证据、canonical SHA 和训练 Run 中的训练数据 ID/hash；重合即阻断。预检摘要展示 suite ID、题数、摘要与 comparability identity。结果逐题型展示正确数/分母、准确率和解析失败数；只有相同 suite 内容摘要、样本覆盖和协议版本的结果才可直接比较。

客观题评分策略默认为“精确匹配”，也可选“阿里百炼大模型评判”。百炼方式使用本地固定 profile `dashscope-qwen3.6-plus-local`（模型 `qwen3.6-plus`），凭据从 `.env.local` 或服务进程环境读取 `ONEGPU_SYNTHETIC_API_KEY`。预检会显示模型、预计请求数以及可能产生百炼 API 费用的提示，并说明题目、选项、标准答案及被测模型回答会发送到阿里云；点击“确认并启动”才会发出 API 请求。无法判定和 API 错误不算答错；汇总分开显示判定准确率与判定覆盖率。该协议 `choice-llm-judge-v1` 与精确匹配分数不可直接比较。

点击“确认并启动独立评测 Run”后，Runs 页面打开新评测 Run 并在运行期间每 1.5 秒刷新状态与日志。评测使用与训练相同的单任务 GPU 锁；训练 Run 的终态不会被评测覆盖，详情中的父/子 Run 链接可往返追溯。当前训练 UI 仍只启动 Mini SFT；其他阶段若已有符合 lineage 的 CLI 训练产物，也可从 Runs 发起适用评测。

### CPT / SFT 指标口径

阶段评测 profile 由服务端白名单提供，常规数学协议使用 `math-eval-v1`。SFT 的共享任务指标来自 GSM8K、MATH-500（可选 internal holdout）；CPT profile 还声明 `domain_ppl.perplexity`，只有提供独立 canonical `cpt_holdout` 并实际运行 evaluator 才会有值。CPT train 数据不能代替 holdout。用户客观题 suite 使用独立的 `choice-accuracy-v1`，不依赖 EvalScope；单选/判断精确匹配、多选按答案集合完全匹配，无法解析的输出计入分母并单列，不代表答案正确性或数据质量已经独立验证。

Runs 列表中的“评测已登记的 Base 模型”可对通过本地登记、路径和 SHA-256 核验的 base 模型单独启动数学基准或 `choice-accuracy-v1` 客观题评测。客观题必须选择已登记的 `stage=eval`、`role=evaluation_only` 数据集。Base 模型是外部 lineage 根节点，没有训练 Run；评测仍创建独立 Run 并记录模型/数据身份。没有出现在选择器中的模型需先登记到本地模型目录。

打开客观题评测 Run 后，可在 Runs 详情的“逐题测评报告”中查看每题题干、选项、标准答案、模型回答和判定。百炼评判报告还显示判定理由、Provider 请求 ID、无法判断和 API 错误状态；未完成/未决记录不会算作答错。报告只在 Run 的评测协议、dataset ID、canonical SHA-256 和题目 ID 均与本地登记证据一致时显示；证据缺失或不匹配时会说明报告不可用，不会按题目行号猜测答案对应关系。

- GSM8K / MATH-500 / internal accuracy 的分母是对应报告实际完成并判分的样本数，不是请求上限；缺报告、没有样本或评测失败时，`eval.json.metric_statuses` 标为 `not_measured`，不会填 0。
- Domain PPL 是 token-weighted NLL 的指数，分母是 `predicted_tokens`；同时记录处理样本数与被截断样本数。没有 holdout、PPL 执行失败或没有可计分 token 时显示 `not_measured`。
- `eval.json.resolved_evaluation` 记录 profile ID、协议、数据输入、样本上限和解码配置；实际样本分母及指标状态记录在结果/lineage 中。不同协议或不同 benchmark 覆盖不能直接横向比较。
- 这些 profile 声明“可测什么、如何计数”，不代表本机已完成真实模型评测；没有实际运行证据的指标保持未测。

## 安全边界与适用范围

- 所有启动操作仅限本机、受令牌保护，并要求显式确认。
- 训练和评测在同一张 GPU 上顺序运行；当前没有队列、远程执行、用户 shell 命令或任意模型/数据路径入口。
- 预检只读；已确认的评测可能使用 EvalScope 管理的联网/缓存行为。
- Mini SFT 数据和每项 5 条的评测仅用于探索，不能作为正式发布验收。

### 阶段训练参数与自定义配置

训练工作台默认直接使用所选 profile 的批准配置，不要求用户填写参数。需要调整时，可展开“训练参数”；页面只显示 profile 明确开放的字段、默认值和范围。当前 SFT Mini profile 开放最大序列长度（默认 1024 tokens，可输入 512–4047 范围内的整数）、训练轮数（1–3）、学习率（0.000001–0.0005）和梯度累积步数（1–32）。保留默认值时无需另行设置。这里的“自定义”是受控参数覆盖，不代表可上传任意 recipe、脚本或路径；任何改动都要求重新预检。更长序列会显著增加显存占用，超出设备承载能力时训练仍可能 OOM；profile 上限不是设备可运行保证。CPT、DPO、SimPO 的最大序列长度同样允许在各自 profile 声明范围内自定义。GRPO 则分别配置最大提示长度和最大回答长度。CPT、DPO、SimPO、GRPO 的 Web 执行 adapter 仍未全部接通，阶段预检不能当成训练已启动。

任何参数变化都会清除现有预检摘要，必须重新预检。服务端将按批准 recipe 合成每次运行的 resolved_recipe.yaml，预检与启动共同绑定其有效配置 SHA-256；启动前会重新计算摘要，不匹配则拒绝启动。Run 目录保存解析配置，run.json 记录覆盖值和摘要；CLI lineage 的 resolved-config hash 指向实际执行配置。其他阶段即使显示参数范围，在网页 adapter 接通前仍不能由网页启动。

## 评测启动状态说明（工作区体验统一）

在阶段评测页选择协议及可选的已登记评测集后，先执行只读预检。确认启动按钮仅在预检通过时出现；点击后“提交中”只表示 HTTP 请求仍未返回。收到评测 Run ID 后，页面以服务端 Run 状态显示后台运行，并提供指向该**评测 Run** 的入口；训练 Run 与评测 Run 不应混淆。

若服务端明确拒绝（包括令牌过期或 409），旧确认作废，应重新预检。若网络超时，页面先按预检给出的评测 Run ID 只读核查：已创建则恢复该 Run；暂未查到则保持“启动结果待核实”，让用户再次核查。任何情况下都不会自动重复提交训练或评测启动请求。

