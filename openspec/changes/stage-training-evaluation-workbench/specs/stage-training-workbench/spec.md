# Spec Delta

## Purpose

让用户在本地工作台按 CPT、SFT、DPO、SimPO、GRPO 阶段选择已登记训练数据、执行受控训练，并在训练模型核验成功后对该阶段产物启动独立评测，以单卡、可复现方式走通后训练闭环。

## ADDED Requirements

### Requirement: Training data is selected independently by stage
工作台 MUST 按训练阶段列出已登记且用途匹配的数据集，并在提交启动前验证阶段 schema、训练角色、质量/许可证据及父模型要求。

#### Scenario: Select one SFT dataset
- **WHEN** 用户在 SFT 阶段选择一个已登记 SFT train 数据集
- **THEN** 用户选择与当前 SFT profile 兼容的已登记父模型和配方，预检展示 dataset ID、内容摘要、记录规模、质量与许可状态及父模型身份

#### Scenario: Dataset stage or role is incompatible
- **WHEN** 用户选择非当前阶段、holdout/diagnostic 或不可用于训练的数据集
- **THEN** 工作台阻止训练，并指出 dataset ID 与具体不兼容项

#### Scenario: Select stage-specific training data
- **WHEN** 用户选择 CPT、Preference（DPO/SimPO）或 GRPO 数据
- **THEN** CPT 按 token 语料、Preference 按不可拆分 pair、GRPO 按 prompt 与 reward/reference 契约校验，且仅展示对应阶段可训练的数据集

#### Scenario: Prepare a registered GRPO training dataset from trusted seeds
- **WHEN** 用户在数据模块选择 GRPO 数据准备，并指定有已登记来源与可信参考答案的 train seed 及批准的 reward contract
- **THEN** 系统通过不依赖定向方案的共享确定性构建核心生成 GRPO prompt 数据，逐条保留 seed record/dataset 身份、参考来源和契约版本/摘要，执行 schema、reference、reward 可执行性、内容摘要及血缘校验后再登记；首版仅接受批准来源协议（如 OpenR1 固定 build protocol/revision）及逐条 correctness label/source record 证据，报告明确该标签属于上游证据而非本系统重新独立验证；未经可信来源验证的上传内容不得仅凭用户勾选提升为可信参考答案

#### Scenario: Extend ordinary GRPO seed derivation to a targeted plan
- **WHEN** 后续定向补数为同一 GRPO 构建核心提供已注册 targeted plan 上下文
- **THEN** 构建核心保持相同 seed/reference/reward/schema 校验，并额外记录 plan/version、诊断来源和 train-seed ancestry；diagnostic/final-holdout 数据不得作为训练 seed

### Requirement: Dataset builders persist a verifiable canonical path
所有创建或登记 dataset manifest 的构建器 MUST 写入以 dataset artifact root 为基准的相对 `metadata.canonical_path`。创建 manifest 时 MUST 验证路径安全地位于 artifact root 内、目标 canonical 文件存在且其 SHA-256 与 manifest hash/lineage 一致。旧 manifest 可继续读取以展示历史信息，但缺少或无法验证 canonical 路径时 MUST 保持不可训练；不得通过路径猜测绕过预检。

#### Scenario: Build a dataset artifact
- **WHEN** 任一阶段的数据构建器、导入器或生成登记器创建 manifest
- **THEN** manifest 记录正确的 canonical 相对路径，创建流程核验文件 containment、存在性与内容摘要

#### Scenario: Read a legacy manifest without canonical path
- **WHEN** 目录读取到早期 manifest 且缺少 canonical_path 或引用路径/摘要不匹配
- **THEN** 历史目录仍可展示，但训练预检指出具体 manifest 证据问题并阻断启动

#### Scenario: Repair a legacy manifest
- **WHEN** 本地迁移为旧 manifest 补充 canonical_path
- **THEN** 仅在唯一候选 canonical 文件存在于 artifact root 内且字节摘要与原登记 hash/lineage 完全一致时写入；否则保留原 manifest 并报告无法安全修复

### Requirement: Stage training uses an approved bounded profile
每个可启动阶段 MUST 基于项目批准并版本化的训练 profile 运行。用户 MUST 能直接使用默认参数，也 MUST 能在服务端允许的阶段参数白名单及取值范围内自定义本次运行；不得从浏览器提交任意 recipe、模型或数据路径、shell 命令、环境变量或白名单外参数。自定义值不得绕过父模型、数据集、资源及预检约束。

SFT MUST 支持以已登记且通过兼容性预检的 Base 模型或 CPT 模型作为父模型。父模型选择 MUST 来自服务端登记的模型目录；Base 模型 MUST 有可解析且摘要匹配的本地权重，模型架构/分词器须受当前训练环境支持，且其输入格式须能与所选 SFT 数据及批准配方兼容。无法满足条件的已登记 Base 模型 MUST 在预检中给出具体 blocker，不得通过浏览器提交任意路径绕过。Base 模型是外部 lineage 根节点时 MUST 记录精确 model ID 与内容摘要，不得伪造 producing Run。现有 CPT→SFT 配方及固定 Mini SFT 行为 MUST 保持可用。

#### Scenario: Train SFT directly from a registered Base model
- **WHEN** 用户选择已登记的 Base 模型、兼容的已批准 SFT profile 和有效 SFT train 数据集
- **THEN** 系统核验 Base 模型登记、权重可用性、内容摘要、训练框架支持及数据输入兼容性，并将精确父模型 ID 与摘要绑定到预检、训练 Run、模型产物和 lineage 后等待用户显式确认

#### Scenario: A registered Base model is not locally trainable
- **WHEN** 已登记 Base 模型的权重缺失、摘要不匹配、架构/分词器不受支持或与 profile/数据输入不兼容
- **THEN** SFT 预检阻止启动并指出具体原因，不下载或替换模型，不接受用户传入路径，也不创建训练 Run

#### Scenario: Keep CPT-to-SFT and Mini SFT available
- **WHEN** 用户选择兼容的已登记 CPT 模型或现有 Mini SFT profile
- **THEN** 工作台保留原有 CPT→SFT 操作、批准配方和默认参数语义

#### Scenario: Preflight a supported stage profile
- **WHEN** 阶段、父模型、已登记数据集和环境均可用
- **THEN** 工作台展示应用默认值后的有效训练配置、资源提示、探索性/质量限制及输出位置，并等待用户显式确认

#### Scenario: Use the default training configuration
- **WHEN** 用户保持某阶段 profile 的默认训练参数
- **THEN** 工作台无需用户逐项配置即可预检；预检及启动均使用该 profile 当前版本的默认值

#### Scenario: Customize supported training parameters
- **WHEN** 用户修改当前阶段开放的一个或多个参数，且其值处于服务端为该 profile 声明的允许范围内
- **THEN** 工作台以覆盖默认值后的有效配置执行资源预检，并明确展示修改项、最终值及资源提示

#### Scenario: Reject unsupported parameter overrides
- **WHEN** 用户提交未知参数、阶段不支持的参数或超出允许范围的值
- **THEN** 服务端拒绝预检或启动并指出具体字段及约束，不忽略、不钳制，也不创建 Run

#### Scenario: A stage profile or required dependency is unavailable
- **WHEN** 阶段没有批准的 profile，或父模型、数据、依赖、GPU/输出条件不满足
- **THEN** 阶段入口显示具体 blocker，不创建训练进程或成功 Run

#### Scenario: Preflight and launch share one stage configuration
- **WHEN** 用户选择阶段 profile、父模型和训练数据，设置可选参数并完成预检
- **THEN** 启动操作 MUST 使用完全相同的 stage/profile 版本、解析后的有效参数及其摘要、父模型 ID、dataset ID 与内容摘要；不得要求用户在另一个入口重复选择，任何输入变化都 MUST 使旧预检失效

#### Scenario: Mini SFT is a recipe within SFT
- **WHEN** 用户在 SFT 阶段选择训练配方
- **THEN** Mini SFT MUST 作为一个有明确固定参数的 SFT profile 呈现，而不是与阶段工作台并列的第二个训练入口

#### Scenario: Stage workflow is consistent while inputs remain stage-specific
- **WHEN** 用户切换到 CPT、SFT、DPO、SimPO 或 GRPO
- **THEN** 页面沿用阶段、配方、所需父模型/数据、预检、启动、阶段评测的同一操作路径，并只呈现该阶段实际需要的输入与检查规则

### Requirement: Training-time validation is optional, explicit, and isolated from final testing
阶段训练 MUST 将训练数据与训练时验证数据分开处理。用户可选择一个已登记且角色/阶段兼容的独立验证集、从当前训练集按批准比例和固定随机种子可复现地切分验证子集，或明确不使用训练时验证；默认 MUST 保持不切分以兼容现有配方并保留全部训练样本。独立验证集和随机切分 MUST 互斥。有效配置、Run 和 lineage MUST 记录验证来源、数据身份/摘要；随机切分还 MUST 记录算法版本、比例、随机种子、训练/验证样本数及切分身份。验证样本不得参与梯度更新。`final_holdout`/最终测试集 MUST 不得输入训练、验证、checkpoint 选择或早停；只能用于训练完成后的独立评测。

#### Scenario: Use a separately registered validation dataset
- **WHEN** 用户为当前阶段选择已登记的独立验证数据集
- **THEN** 预检验证阶段 schema、验证角色、数据摘要及其与训练数据/最终测试集的样本隔离；训练命令只将其传给验证流程，不用于梯度更新，Run 记录精确验证集身份

#### Scenario: Split a reproducible validation subset from the training dataset
- **WHEN** 用户未选择独立验证集而选择“从训练集随机切分”，并提供允许范围内的比例和随机种子
- **THEN** 系统基于当前训练数据 ID/hash、固定切分算法版本和种子确定验证子集，展示切分后训练/验证规模并将完整切分配置绑定到预检、Run 与 lineage；Preference pair 等复合记录不得拆开，存在稳定问题/来源分组键时同组样本不得跨训练与验证

#### Scenario: Keep all records when no training-time validation is selected
- **WHEN** 用户选择不使用训练时验证
- **THEN** 所有合格记录均保留在训练集，训练期间不计算或伪造验证指标，训练完成后仍可独立执行阶段评测

#### Scenario: Reject invalid or leaking validation/test data
- **WHEN** 验证集角色/格式错误、随机切分后任一侧为空或不足 profile 最小样本要求，或最终测试样本与训练/验证样本存在重叠
- **THEN** 预检指出具体冲突并阻止启动；最终测试数据不因用户确认而升级为训练或验证输入

#### Scenario: Stage has no web execution adapter
- **WHEN** 当前 profile 可读取或可预检，但其网页训练启动 adapter 尚未接通
- **THEN** 页面 MUST 明确显示网页启动不可用及具体原因，不得呈现为预检通过后即可训练

### Requirement: Checkpoint selection defaults to final and optionally supports validation best
每个阶段训练 MUST 以 `final` 作为默认 checkpoint selection policy，并允许用户显式选择 `best`。`final` 指训练完成后步数最大的有效已保存 checkpoint；`best` MUST 仅依据训练期验证结果及该阶段声明的指标和优化方向选择。选择 `best` 时若未启用有效训练期验证、指标缺失或无法比较，预检 MUST 阻止启动，不得静默回退到 `final`。最终测试/holdout MUST 不得参与 checkpoint 选择。系统 MUST 保留可用于续训的 latest checkpoint，并分别保留 final 与实际选中的模型 checkpoint（若二者相同可共享同一份文件），Run/lineage 记录 policy、指标名/方向、所选 step/path/hash 及验证数据身份；Run 的实际模型产物按所选 policy 指向。

#### Scenario: Use default final checkpoint selection
- **WHEN** 用户未修改 checkpoint selection policy
- **THEN** 预检和训练使用 `final`，成功后选取步数最大的有效已保存 checkpoint 作为阶段模型，并记录选择结果

#### Scenario: Select best checkpoint using validation
- **WHEN** 用户选择 `best` 且该阶段启用有效训练期验证并声明可计算的选择指标和方向
- **THEN** 训练按 profile 协调验证与保存频率，阶段模型取验证指标最优且已保存的 checkpoint；latest checkpoint 仍保留用于续训，Run/lineage 可追溯 best 与 final 的身份

#### Scenario: Reject best selection without a usable validation metric
- **WHEN** 用户选择 `best` 但没有有效训练期验证、指标未声明/未产出，或验证数据不满足阶段契约
- **THEN** 预检指出具体原因并阻止启动，不得使用 final holdout、训练指标或自动回退来伪造 best 选择

#### Scenario: Preserve distinct final and best model identities
- **WHEN** 验证最优 checkpoint 与训练最后 checkpoint 不同
- **THEN** 系统保留两者并记录各自 step、路径和摘要；阶段输出模型按用户选择的 policy 绑定到对应 checkpoint

### Requirement: Fixed and configurable SFT entry points share evidence validation
固定 Mini SFT 与通用阶段工作台 MUST 使用一致的训练环境发现、数据摘要和父模型 lineage 校验语义；固定配方可保留白名单路径和默认数据，但不得因入口不同而绕过错误或误拒绝有效证据。

#### Scenario: Resolve a merged parent model produced by a nested merge run
- **WHEN** 父模型登记指向 `merge_runs/<run_id>/lineage.json` 且该 run 已完成并声明了同一模型输出
- **THEN** 两种 SFT 入口都接受该父模型；不能因只查父目录根部的 `lineage.json` 而误报

#### Scenario: Resolve the training command from the active Python environment
- **WHEN** `swift` console script 位于运行 UI 的 Python 环境 Scripts 目录，但该目录未预先加入父进程 PATH
- **THEN** 固定与通用 SFT 预检按同一 child-process PATH 规则发现命令

### Requirement: Training runs have truthful lifecycle and lineage
每次确认的阶段训练 MUST 具有唯一 `run_id`，记录阶段、实际 dataset ID、父模型、profile 版本、最终生效参数及 config 摘要和真实状态；页面 MUST 自动更新有界日志，并只展示已核验的实际模型产物。

#### Scenario: Stage training completes
- **WHEN** CLI 成功完成训练及该 profile 要求的模型合并
- **THEN** 页面展示训练终态、训练 run、核验后的 adapter/merged model 与 lineage，并开放适用的评测入口

#### Scenario: Training fails, OOMs, or is interrupted
- **WHEN** 训练失败、OOM、服务中断或模型合并失败
- **THEN** 页面保留实际终态与已产生证据，不开放依赖缺失模型的评测，也不伪造成功产物

### Requirement: Data and model lineage is verifiable end to end
数据产物、阶段训练、模型产物及评测 MUST 通过稳定对象 ID、内容摘要和显式关系形成可验证的完整链路。每个 Run MUST 记录其实际消费的 dataset ID/hash/role、父模型 ID/hash、profile/config 摘要和实际产物；模型产物 MUST 指回生成它的 Run 及其 parent；评测 MUST 指向精确的模型产物、训练 Run、评测数据/协议与评测 Run。界面 MUST 能从数据集或模型沿关系追溯上下游，不得仅凭名称、目录或页面当前选择推断关系。

#### Scenario: Trace a dataset through training and evaluation
- **WHEN** 用户从数据集详情、训练 Run、模型详情或评测结果打开 lineage
- **THEN** 可沿真实登记关系追溯数据来源/父数据与内容摘要、被哪些 Run 消费、Run 的父模型与 profile、产出的 adapter/merged model，以及模型对应的评测 Run、数据和协议

#### Scenario: Verify a multi-stage model chain
- **WHEN** CPT 或 SFT 模型被后续阶段作为 parent，或同一个 SFT parent 分支进入 DPO、SimPO、GRPO
- **THEN** 每个子 Run/模型都保留准确 parent model ID/hash 与独立 dataset/run 身份，分支关系可区分，不得把并行实验串成单一链

### Requirement: Completed stage models are easy to hand off to compatible next stages
成功 Run 的已登记模型产物 MUST 提供清晰的下游阶段交接；模型选择器 MUST 展示足以区分模型的来源信息，并从阶段 profile 解析兼容关系。交接只传递已登记的精确 `model_id` 和来源上下文，不自动启动训练；模型摘要、producing Run 与 lineage 仍须由预检重新核验。

#### Scenario: Continue from a verified completed training Run
- **WHEN** 用户查看成功完成且包含已登记模型产物的 Run
- **THEN** 页面仅为该产物显示由批准 profile 决定的兼容下游阶段操作（例如 CPT 到 SFT、SFT 到 DPO/SimPO/GRPO），并允许用户选择目标阶段

#### Scenario: Hand off a model to the next stage
- **WHEN** 用户选择“用作下一阶段父模型”
- **THEN** 训练工作台打开对应阶段/profile 并预选精确父模型 ID，显示其模型阶段/类型和 producing Run；不自动选训练数据、不自动启动，用户仍需完成数据选择和预检

#### Scenario: Distinguish candidate parent models
- **WHEN** 当前阶段有多个兼容父模型可选
- **THEN** 选择器以阶段、artifact 类型、producing Run 和可读的短 ID 区分候选，并提供查看完整 lineage 的入口；不要求用户凭长 hash 猜选

#### Scenario: Parent model is not eligible for handoff
- **WHEN** 模型产物缺失、producing Run 失败/不匹配、摘要或 lineage 未通过校验，或没有兼容下游 profile
- **THEN** 系统不提供可执行的交接操作，并说明该模型不可用的具体原因；不得只凭文件夹或模型名称推断可用性

#### Scenario: Lineage evidence is absent or inconsistent
- **WHEN** manifest、dataset hash、parent model、Run、产物或 evaluation 之间缺少关系或摘要不匹配
- **THEN** 验证器明确报告断链/不一致对象并阻断依赖该证据的启动或评测；历史对象仍可查看，但不得补造或静默修正 lineage

### Requirement: Evaluation is a separate explicit post-training action
阶段评测 MUST 仅在对应训练成功且目标模型身份与摘要核验后可启动；评测使用独立、版本化、阶段匹配的 profile 和非训练用途数据，不自动执行。

#### Scenario: Start evaluation for a completed stage run
- **WHEN** 用户在已成功的训练 Run 上选择适用评测并确认预检
- **THEN** 系统为精确模型创建独立评测 run/eval identity，记录评测数据集或 benchmark、协议、覆盖范围和与训练/合并 run 的关系

#### Scenario: Evaluation would reuse training data or holdout is unavailable
- **WHEN** 评测选择与训练数据身份/内容冲突，或所需 benchmark/holdout 不可用
- **THEN** 工作台阻止或明确标为不可运行，并报告冲突或准备 blocker，不把训练数据结果称为泛化评测

### Requirement: Stage evaluation reports distinguish shared and stage-specific evidence
阶段评测 MUST 如实显示固定 profile 规定的通用任务指标与可用阶段专属指标、分母、协议和状态；失败、缺失、部分、未测或不适用 MUST 保持区分。

#### Scenario: CPT and SFT model evaluation
- **WHEN** CPT 或 SFT 评测完成
- **THEN** 结果按 profile 展示共享任务 benchmark，并展示适用的领域 PPL/数据分母或明确其未测状态

#### Scenario: Preference optimization model evaluation
- **WHEN** DPO 或 SimPO 模型评测完成
- **THEN** 结果分别展示共享任务 benchmark 与可用的 held-out preference 指标，不把训练 preference pairs 计作评测样本

#### Scenario: GRPO model evaluation
- **WHEN** GRPO 模型评测完成
- **THEN** 结果展示共享任务 benchmark 和可用 reward/rollout 诊断；无真实 rollout 证据的指标标为未测而非推断

### Requirement: Only one local GPU run is active
本地工作台 MUST 在所有阶段训练和评测之间共享单活动 GPU 任务约束，不提供队列、并行执行、自动重试或远程调度。

#### Scenario: Another stage run is active
- **WHEN** 用户尝试启动第二个训练或评测而本地任务仍活动
- **THEN** 系统拒绝新任务并提供当前 run ID 和状态
