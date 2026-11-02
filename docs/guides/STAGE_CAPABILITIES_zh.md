# 阶段训练能力与 Profile

## 前端阶段工作台（当前）

训练与评测页提供 CPT、SFT、DPO、SimPO、GRPO 阶段导航。切换阶段会清除旧的父模型、数据集选择及预检摘要，避免把上一阶段的确认误用于当前阶段。阶段、批准 profile、父模型和数据集在同一工作台中选择；点击“检查并预检”会重新验证 dataset manifest/content hash、角色、父模型阶段与模型血缘、本机依赖及执行能力。SFT 提供固定 Mini 配方（父模型锁定为 Mini M1-CPT）和通用登记模型配方（可选已登记 Base 或合并 CPT）；通用配方会核验本地 Transformers 配置/权重、tokenizer、登记摘要及安全路径，未通过的模型会在选择器中禁用并显示原因。阶段预检会将精确父模型 ID/hash、profile 与数据身份绑定到启动确认。只有无 blocker 才能显式启动；切换任一输入都会清除旧预检与启动令牌。

浏览器内阶段工作台通过服务端一次性预检/确认能力支持 CPT、Mini SFT、DPO、SimPO 与两个 GRPO profile 的受控启动。支持能力不代表当前配置必然可运行：数据登记、数据角色与 reward/reference 契约、父模型/本地权重、许可确认、ms-swift 和 GPU 等任一检查失败都会阻断启动。只有显示“确认并启动训练”并由用户显式确认后才创建 Run；只读的配置摘要或“检查配置并预检”不代表 Run 已创建。“实验 Runs”也可对具有成功训练 lineage 和已登记 merged model 的阶段 Run 发起适用的独立评测。

阶段工作台使用服务端维护的批准配置目录描述“每个阶段需要什么、当前哪些能力可用”。目录是显式 allowlist，不会从用户输入拼接任意 recipe 或 shell 命令。

## 阶段与 Profile

| 阶段 / 算法 | Profile | 父模型 | 训练数据 | 当前说明 |
|---|---|---|---|---|
| CPT | `cpt-smoke-v1` | Base | CPT token corpus，`train` | Web 受控 adapter 已接通；本地语料角色、Base 权重与 GPU 检查仍可能阻断 |
| SFT | `sft-mini-1024-v1` | 固定 Mini M1-CPT | SFT canonical，`train` | Mini 配方固定父模型；保持单数据集 ID |
| SFT | `sft-registered-v1` | 已登记 Base 或合并 CPT | SFT canonical，`train` | 直接 Base→SFT 或 CPT→SFT；仅允许本机可用且通过 profile/摘要/tokenizer 预检的模型 |
| DPO | `dpo-smoke-v1` | SFT merged | Preference pairs，`train_pairs` | Web 受控 adapter 已接通；数据与父模型需通过预检 |
| SimPO | `simpo-smoke-v1` | SFT merged | Preference pairs，`train_pairs` | Web 受控 adapter 已接通；独立于 DPO |
| GRPO | `grpo-standard-smoke-v1`、`grpo-lowmem-smoke-v1` | SFT merged | GRPO prompts，`train_prompts` | Web 受控 adapter 已接通；必须通过逐条 reference/current reward-contract 预检 |

Preference 是数据阶段，DPO 和 SimPO 是两种独立训练算法。旧的 `/api/targeted/stages` 为兼容既有数据工作台，仍将两种算法归到 `preference` 项下，并在 `profiles` 字段列出各自 profile；训练工作流应使用新的 profile ID，而不是把 DPO/SimPO 合并成一个 Run。

## 数据生成能力（与训练能力分开）

| 数据阶段 | 普通生成 | 定向补数 | 登记/训练交接 |
|---|---|---|---|
| CPT | 数据工作台可用：主题生成、预检、显式启动、候选审核和登记 | 可用：绑定 CPT 计划 ID/version 与精确 `train_seed` 父数据身份 | 已登记 CPT dataset ID 可进入 CPT Launcher；生成本身不启动训练 |
| SFT | 数据工作台可用：按主题或参考 SFT seed 生成 | 可用：通过定向方案/共享任务，并复用审核登记路径 | 生成后显式登记，再进入 Mini SFT Launcher；正确性仍按对应证据报告 |
| Preference | 数据工作台可用：用户提交同题目、生成未排序候选、逐题人工选择 chosen/rejected 后显式登记 | 可用：绑定已登记 Preference 计划与精确 `train_seed` dataset ID/hash，只生成候选并独立校验最终答案 | 登记后可由 DPO 或 SimPO 独立消费；二者使用相同精确 dataset ID/hash 时仍创建独立 Run |
| GRPO | 当前仅有 prompt-only 候选适配器；未接工作台/API、审核或登记 | 同样仅是受 `train_seed` 和精确 reward contract 约束的候选适配器；尚未接 Web 闭环 | 候选始终 `trainable=false`；要训练必须另行具备可信 reference、版本匹配的 reward contract 并通过现有预检 |

"模型生成”只用于确实调用 Provider 产出候选的路径；OpenR1 GRPO seed 派生是“派生”，上传是“导入”。Preference 的候选需经人工排序（普通模式）或独立最终答案校验（定向模式）后才可登记。工作台能力摘要中的 `generation_modes` 是能力与阻断说明，不代表已准备具体方案或获得 Provider 调用授权。

DPO/SimPO 的 profile 都要求已登记、已合并且具有成功训练 lineage 的 SFT 模型；具体训练 Run 绑定用户选择的精确 `parent_model_id`，因此两个算法可以从同一个 SFT 模型平行分叉。预检会验证 Preference canonical pair 的 chosen/rejected 非空且不同，并检查排序依据元数据。上传数据没有可识别排序证据时会显示 `unverified` 警告，不把未知质量伪装成已验证，也不因缺少这类可选来源证据直接阻止探索性训练。

## Profile API

`GET /api/training/stage-profiles` 返回 schema version `1.0` 的类型化 profile 目录。每个 profile 固定描述：

- 阶段、算法和批准的 recipe 路径；
- 允许的父模型阶段与训练数据阶段/角色；
- 阶段 adapter、预期产物类型及产物核验方式；
- 可使用的评测 profile、协议、数据角色和指标；
- 当前是否接通执行，以及阻断原因；
- GRPO 使用的 reward contract（仅 GRPO profile）。

服务端会验证 recipe 与 reward contract 是 workspace 内的普通文件；路径缺失、符号链接或越出 workspace 时，Profile 保留在目录中但标记为不可执行，并给出 blocker。目录查询只读，不会启动模型提供方、训练或评测。

profile 同时声明批准 recipe 的训练默认值，以及服务端逐项允许自定义的参数类型、范围与离散选项。阶段训练页默认使用批准值；展开“训练参数”后只显示当前 profile 的白名单字段。覆盖参数会在阶段预检与实际启动预检中共同提交并由服务端重新校验；recipe、路径、命令、环境及未声明字段不可由浏览器指定。对应 profile 的 `execution_supported` 仅表示 adapter 已接通；环境和实际数据检查仍以每次预检结果为准。

DPO/SimPO 可从 CLI 受控执行：`run-registered-preference` 仅接受已登记 Preference train dataset ID、profile ID、已登记 merged SFT model ID、run ID 和由 profile 推导的输出位置；自动复核 manifest/摘要和 parent lineage、生成 trainer-native adapter 数据，再调用固定 DPO 或 SimPO recipe。训练 Run 与 LoRA merge Run 分开登记；失败时保留失败状态和实际 adapter，不伪造 merged model。

DPO/SimPO 模型可使用独立的 Preference held-out profile `preference-heldout-v1` 做偏好评测。Runs 详情要求选择已登记的 `preference_eval` dataset；预检会复核目标 merged model、生产训练 Run、训练 dataset ID/digest、held-out manifest 与内容摘要，样本需带 `problem_id` 和 `evaluation_only=true`，且 train/held-out 的 `problem_id` 无交集。不能把 `preference` 训练数据、任意路径或同一内容副本当作评测集。评分协议为 `response-mean-logprob-v1`，`preference.accuracy` 是 chosen 均值 log-prob 高于 rejected 的 pairs 占比，`preference.margin` 是 chosen-rejected 均值 log-prob 差的 pair 平均；分母记为实际完成评分的 pair 数。实际 pair 少于登记数量时 Run/指标标记 `partial`；模型加载或评分失败则是 `failed`，偏好指标标为 `not_measured`。GSM8K/MATH 等共享任务 benchmark 属于另一评测 profile，不混进偏好指标。

GRPO 的训练预检还要求登记数据 manifest 与每条 prompt 都绑定当前 `ExecutableRewardContract` 的 ID、protocol SHA-256 和 implementation SHA-256；每条记录必须带非空问题、可执行的 prompt、非空 reference answer、`problem_id`、`data_role=train_seed` 与 `reference_source=registered_train_seed`。预检会运行一次固定 reward smoke check，并检查 workspace 内的批准插件包含 `onegpu_accuracy` / `onegpu_boxed_format` 两个注册入口，返回 contract 摘要、插件摘要和参考答案覆盖行数。缺 contract、contract 版本/摘要不匹配、reference 缺失或插件不可用都会阻断预检；这只是结构和代码可执行性检查，不代表 reference 的语义正确性已独立复核。

GRPO 可使用 `run-registered-grpo` 通过 CLI 启动；它仅接受 `grpo-standard-smoke-v1` / `grpo-lowmem-smoke-v1`、登记 dataset ID、登记 merged SFT model ID 和 run ID，recipe、reward 配置、插件与输出目录都由 profile 固定。训练先生成带输入摘要和 provenance sidecar 的 trainer-native adapter，再调用现有单卡 supervisor；标准方案 beta=0.04 并启用 reference，lowmem beta=0 并禁用 reference。训练/合并 lineage 分开记录，Run 输入回指原始 dataset ID 与精确 parent model ID。训练状态为 OOM/FAILED 时不尝试合并，并将 reward/rollout 指标记为 `not_measured`；训练成功但缺少 trainer_state 时仍如实完成训练 Run，但 rollout 诊断保持 `not_measured`。现有 trainer_state 字段逐项记录状态与日志点分母，beta=0 的 KL 标为 `not_applicable`，未知 reward 不从 loss 或配置推导。

成功的 GRPO merged model 可用 `run-registered-grpo-eval` 运行固定 `grpo-math-eval-v1` 共享数学 smoke benchmark。该评测独立创建 evaluation Run；`gsm8k/math_500/internal` 使用各自实际判分样本数。reward、KL、completion 长度与截断诊断不在评测时重算，而从该模型生产训练 Run 的 `grpo_metrics.json` 只读附入，并同时记录训练 Run ID、merge Run ID、metrics 文件 digest 与 trainer-log-point 分母。缺少真实 rollout 字段时显示 `not_measured`，beta=0 的 KL 显示 `not_applicable`；不会用 benchmark accuracy、loss 或配置推导这些值。Runs 详情提供 profile 预检和独立评测 Run 观察。

`GET /api/targeted/stages` 是数据工作台使用的能力摘要 API。它额外提供本机硬件、生成/Launcher 开关和阶段级 readiness；这些字段不是 profile 本身的批准状态。

## 评测 Profile 与数据隔离

当前目录声明共享数学任务、CPT domain PPL 和 Preference held-out 评测。Profile 的 `metric_sources` 与 `denominator_semantics` 描述指标证据来源和分母：benchmark accuracy 使用实际判分样本数；domain PPL 使用预测 token 数，并保留样本/截断数量。训练角色（例如 `train_pairs`）不等同于评测角色（`preference_eval`）。最终 holdout 可出现在评测声明中，但不得作为数据生成或训练输入。未实现或未实际运行的阶段能力必须显示为不可用/not measured，不能用配置存在推断训练或评测已经完成。

## 边界

阶段 Profile 是阶段训练/评测的稳定契约。CPT、Mini 与登记模型 SFT、DPO、SimPO、GRPO 均有受控 Web 启动路径，但每个 Run 仍必须单独通过权威预检并接受明确确认。Base 作为外部根模型写入精确 ID/hash，不创建虚构的训练来源。Runs 评测入口只对成功训练、模型身份/摘要核验通过且有已批准 profile 的 Run 开放，数据型 profile 必须再选对应登记 holdout；确认后创建独立评测 Run，训练 Run 终态保持不变。
