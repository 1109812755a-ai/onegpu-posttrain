# Design

## Context

动机与范围见 [proposal.md](proposal.md)。本变更以用户确认的“评测发现薄弱场景后定向生成训练数据”为主用例，同时支持验证数据质量干预与模型结果之间的关系。

本变更依赖 [data-policy-experiments](../data-policy-experiments/proposal.md) 的策略、固定问题池、探索用途、实验比较和受限前端编辑。当前该依赖只有规划，实施本变更前必须先完成其相关协议与测试。

已核对的现状：

- `evaluation/internal_math.py` 已保存逐题预测、problem_id、解析结果和正确性；场景标签没有直接输出，可通过固定数据集的 `EvalRecord.metadata` 或显式 sidecar 按 ID 关联。
- `evaluation/evalscope_reports.py` 当前主要采集总分，没有现成的通用场景切片服务；不能从旧总分倒推出场景分数。
- `evaluation/answers.py` 提供抽取、规范化和可选 math-verify，但布尔相等结果不足以表达 reference 缺失、可选依赖缺失、验证器不支持与真实不等价。新增验证结果需要显式区分这些状态。
- lineage v1 支持 `data_build`、`quality_assess`、`train`、`merge`、`evaluate` 及 file refs，足以通过显式生成 transform 接入第一版合成数据，不必先引入元数据服务。
- 核心依赖只有 Pydantic/PyYAML；HTTPX 目前仅在可选 UI 测试依赖中。外部模型调用应成为可选 extra，原始开源数据流程继续独立工作。

## Goals / Non-Goals

**Goals:**

- 用户可从真实诊断证据登记一个可解释的薄弱场景，配置模型与提示词，生成/验证数据并显式训练后复测。
- 用同一质量/血缘/实验体系比较未经验证、答案经验证及受控噪声数据；数据变化和模型结果同时可见。
- 首版可用本地 fixture、fake provider 和已有数据进行 CPU 契约验收，真实 API/GPU 结果作为后续显式执行的独立证据。

**Non-Goals:**

- 不自动追逐指标启动生成/训练，不建设 agent、自主数据飞轮、分布式合成或调度平台。
- 不承诺任意场景、任意新题能自动获得正确 reference；不把模型自评或答案验证当成完整推理正确性的证明。
- 首版生成适用于 SFT/Preference 的数学答案候选。任意全新问题合成、CPT 长语料合成、GRPO 全新任务/reference 构造及多供应商专用 SDK 后续独立扩展；支持导入有独立 reference 的程序化新题。

## Decisions

### 1. 以诊断证据和训练种子组织人工控制的迭代

流程为：

```text
diagnostic eval + scenario definitions
                 |
                 v
scenario report --> generation plan <-- isolated training seeds
                          |
                          v
                  explicit generation
                          |
                          v
               candidates + verification
                          |
                          v
             versioned datasets + quality
                          |
                          v
                  explicit training
                          |
                          v
              fixed re-eval + guardrails
                          |
                          v
              remediation effect report
```

每轮由用户保存新计划、明确执行。没有数据/结果时对应环节保持未执行。相较于先做“自动发现问题并不断训练”，此方式保留假设、预算和失败证据，适合单卡与开源贡献者复现。

### 2. 场景定义是版本化数据，不是自由脚本或自动评分意见

新增 `scenario_definition.json/yaml`，包含 taxonomy/version、明确标签映射和适用数据集。首版支持 metadata 中已存在的 topic/problem_type/difficulty 标签，或项目自有的 case-ID → 标签 sidecar；没有标签的记录进入 unknown，不自动使用 LLM 猜标签。

`scenario_report.json` 关联 eval/model/dataset/protocol 和逐题结果摘要，保存各场景的总量、正确/错误、解析失败、评分不支持、排除项及分母。空场景为未评估，不能显示 0% 正确率。一个题目可属于多个场景，重叠需说明，场景数不可直接相加为题目总数。

诊断阈值、指标方向和最小有效样本要求由场景协议/用户配置事先提供；不预置“所有场景低于 80% 都不达标”之类通用标准。仅有 EvalScope 总分时报告缺少逐题/标签证据，后续 adapter 另行增加，不能回填历史标签。

### 3. 定向计划同时固定能力目标、生成条件和复测约定

`generation_plan.json/yaml` 保存：

| 字段组 | 内容 |
|---|---|
| 诊断 | plan_id/version、scenario report/eval/model refs、taxonomy、目标指标/方向/阈值来源、证据样本要求 |
| 数据 | stage、train_seed dataset/content hash、problem IDs/split snapshot、topic 选择策略、candidate/pair 目标、质量 policy ref |
| 生成 | provider profile、model/revision 可用性、prompt template/hash、输出 schema、temperature/支持的 seed、调用与输出预算 |
| 训练 | starting_model、recipe/hash、新增数据混合方式及比例、example/token/step 预算、seeds |
| 复测 | comparison_design、diagnostic 与 final datasets/roles、评测 protocol、主指标/保护指标、停止条件、历史迭代 refs |

计划可以包含多个场景配额，但不自动超额调用来填满；每个场景实际候选、验证、拒绝和缺额分别报告。首个切片采用确定性的训练种子标签选择；没有足够匹配种子时明确报告短缺，可导入带可靠 reference 的独立新题，而不是从评测题生成改写。

模型生成提示词默认使用训练 question、抽象 skill/topic、输出格式和目标长度，不注入 diagnostic/final 的题目、gold 或预测正文。可信训练 reference 默认保留在本地验证器中；用户若明确选择训练 reference-guided 模板，必须记录该生成模式，且仍不能证明推理过程正确。

### 4. 将诊断集、训练来源和最终留出集分开登记

数据角色为 `train_seed`、`diagnostic_dev`、`final_holdout`，同一问题及其派生候选继承祖先角色，不因改写内容或新 ID 消失。诊断题用于识别技能与监控进展；训练候选只从训练种子或独立构造的新题派生。最终留出集的正文/答案既不发给生成器，也不进入训练输入。

实际已反复用于数据/参数选择的 benchmark 必须如实登记为诊断用途；若某历史结果来自这样的来源，仍可保存，但不能作为“未用于调优的最终测试”。使用者需要为正式最终评估固定新的独立留出集。角色修改生成新的登记版本，不能重写历史。

新候选的 exact/problem-ID/祖先检查及已有 near-duplicate 方法共同形成污染证据；方法、索引版本和未覆盖情况写入报告。无法证明无近重复时报告局限，不把未知当零污染；缺少必要的评测隔离索引时不得导出宣称通过防护的训练数据。

### 5. 可选 provider adapter，不将账户依赖带入核心

新增可选 `synthetic` extra，复用项目当前兼容的 HTTP 客户端版本策略，数学 verifier 采用已有可选依赖。首版实现一个 OpenAI-compatible chat-completion adapter 和 fake adapter，适配能力需显式声明：支持的 endpoint、输出/seed 参数、usage、幂等与 revision 证据。不静默换供应方、模型或降级参数。

provider profile 保存非秘密的 ID、endpoint、model 和 `api_key_env` 等凭据变量引用；执行时才读取秘密。禁止内嵌认证 URL、敏感查询值和模板中的环境变量插值；共享脱敏边界处理请求日志、供应方异常与报告。UI 只展示已配置 profile 的就绪状态，不能查询任意环境变量。

首个配置模板和提示词提供 SFT 完整短解答以及同题多候选格式，包含明确输出 schema；提示词是用户可编辑的版本化文本。默认输出包含 problem_id、solution、final_answer，验证 final_answer 与 solution 的最终答案一致。用户自定义模板必须保留输入身份及输出解析契约。

### 6. 预算、缓存与恢复优先保证实际调用可审计

生成串行执行，适合单卡实验室和请求费用控制；首版不引入任务队列或并发 workers。预算包含 `max_requests`、每次 `max_output_tokens`、重试/超时与可用总 token 观测上限。所有网络尝试，包括格式失败和重试，都计入同一计划的持久化请求账本；恢复时不能重新清零预算。

每个请求以 provider profile/adapter、endpoint/model/revision、完整输入摘要、模板/参数/输出协议及候选序号计算 request key。命中已完成响应时校验 hash 后复用；改变影响输出的设置生成新 key。使用追加日志、原子产物发布与同一计划执行锁，避免两个进程重复消费预算。没有完整结束状态的响应不能当作成功缓存。

请求记录 pending/completed/failed/uncertain。超时不能证明供应方没有处理；支持幂等时按声明能力恢复，否则需要显式 `--retry-uncertain` 并保留重复费用风险。明确的 retry 仍受剩余请求预算限制。每次实际执行有独立 run_id，但可引用同一计划与已完成 request keys。

token 总量采用已报告 usage 与可保证的预留上限；若没有可靠的输入 token 计量，不能声称严格保证总 token/金额不超。始终可强制请求次数和发送的输出上限参数；供应方实际是否遵守及隐藏计量限制须记录。费用未返回时为 unknown，用户提供费率时只展示 estimate，不显示为账单或承诺硬金额上限。

### 7. 候选与训练记录分层，验证保持独立

先保留 generation candidates，再调用先行变更的策略服务构建 canonical 训练数据。候选保留输入 record/problem ID、candidate slot、request/response digest、生成完成原因和答案字段，内容地址身份可复查。

验证结果至少为 `correct`、`incorrect`、`unparseable`、`unsupported`、`unverified`，带 reference/source/verifier 版本和覆盖。复用现有抽取/数学等价能力时增加明确的状态包装；依赖缺失、解析异常或不支持不能落成 `incorrect`。verification claim 为“在此 reference 与验证器范围内最终答案等价”，不能扩展成推理无错误。独立人工抽查可另存证据，不作为首版强依赖。

- SFT：检查输出格式、非空、完整/截断事实、真实渲染长度、污染与最终答案。策略可输出经验证或明确未验证的探索数据。
- Preference：对同一训练问题生成多个候选，在 known correct/incorrect 中按已登记规则选取 chosen/rejected，并检查长度比、分组、非空且不同。全正确/全错误/缺验证覆盖时只保留候选与缺额，不补造 pair。可用正确候选保留在缓存中，转作其他阶段需要独立登记，不能自动跨固定阶段/评测分组。
- 预算耗尽时允许 partial 候选与输出报告，但不能用重复记录、未知当错误或截断推理来填满目标。

完全新问题只有在带独立可信 reference 或可验证构造过程时才进入 verified 路径。首个示例可用程序化整数运算构造题目与 reference，再让模型写解答；任意 LLM 自创题和自报 gold 的正确性需要后续 solver/人工证据，不能在此首版中假装已解决。

### 8. 质量对照采用可测量策略或受控扰动

提供三种清晰标记的实验状态：答案经验证、未经验证、`intentionally_noisy`。它们不是通用质量等级。经验证与未验证样本应报告实际检查覆盖与所选题目的分布变化，不能把缺验证直接等同于内容低质量。

首个噪声变换限制于支持的数值最终答案，例如确定性修改 boxed 最终答案并检查与 reference 不等价；记录变换版本、目标/实际比例、seed、被改记录和 parent IDs。保留格式、题目与来源，但公开说明这种扰动也可能造成解答与最终答案不一致，效果仅适用于该干预。无法可靠改写或验证的答案记录跳过，不能谎报达到目标比例。

匹配预算的质量对照使用相同训练问题及记录匹配、相同起点/recipe/评测；token 长度和分布仍如实报告。Preference 首版不做错误排序注入，避免将标签错误与候选本身质量同时混入一个未定义处理。

### 9. 生成血缘复用 v1 文件协议

每次调用批次产生标准 `data_build` run，`DatasetLineage.transform.name=llm_generate_v1` 标识操作，输出原始候选 dataset 和 `generation_report` file ref。quality_assess/data_build 后续生成阶段数据，train/merge/evaluate 沿既有身份链连接。这样不必为了第一版扩展所有 RunType Literal。

默认布局：

```text
artifacts/synthetic/<plan_id>/
  plan.json
  requests.jsonl
  private/responses/
  candidates/
  manifests/
  reports/generation_report.json
  reports/scenario_report.json
  reports/remediation_report.json
```

真实执行产物在已忽略的 artifacts 根，版本化提示词/示例配置保存在仓库可跟踪目录。UI 只读报告白名单字段，不通过 artifact viewer 暴露 requests 原文、private、认证信息或种子正文。生成模型 revision 不可得时保留 unavailable/reason；模型名称和 seed 不能被宣称为逐字复现保证，完整保存的响应是可校验重用依据。

### 10. 复测同时回答目标变化与其他能力变化

默认采用 `before_after_pipeline`：从诊断模型继续训练，复测相同 diagnostic 场景与通用保护指标，报告新增数据、训练预算、资源及绝对值/delta。这是整个追加流程的观察，不证明合成策略单独导致收益。

若用户选择 `matched_budget_ab`，复用先行数据效果比较：普通补数/定向补数两组从共同起点训练，固定声明预算、配方、seed 和评测。诊断起点结果单独作为参考，不伪装成具有相同训练预算的控制组。

`remediation_report` 同时保留诊断/复测身份、目标是否达到、保护指标是否违反、比较兼容性、样本/seed 限制及失败/OOM。训练 loss 或 reward 只是诊断。小分母、缺失 final 结果、不可比或单 seed 时不作稳定能力提升声明；零收益和退化是完整有效的实验结果。每轮新计划引用前一轮，记录诊断集已被反复使用，最终留出结果不自动成为下一轮生成输入。

### 11. 前端准备计划和检查证据，调用由显式命令执行

在现有 Data 与比较区域加入“薄弱场景”“定向补数计划”“生成与验证”“复测”四个可顺序到达的区块，复用本地 editor opt-in、来源跳转和草稿机制。用户选择已登记 provider profile、模板、输出阶段和预算后可保存/导出 CLI 配置；不通过浏览器输入或持久化 API 密钥。

显示实际场景分母、题型覆盖、候选/正确/错误/未知/超长/污染/缺侧数量、请求和成本证据。没有执行就显示待生成/待训练/待复测，fake 数据明确为本地模拟。真实调用与训练由 CLI 显式执行，本切片不增加收费网络 API 启动按钮。未来本地 launcher 复用相同计划，另行审阅任务生命周期与权限。

## Risks / Trade-offs

- [针对评测题补数造成泄漏] → 角色与祖先隔离、抽象场景输入、独立训练种子、污染检查及正式最终留出约定；不把改写视为新独立题。
- [提示词宣称高质量但答案/推理仍有错误] → 独立 reference 验证最终答案、记录支持范围和人工抽查扩展点，不能自报 reasoning verified。
- [目标场景改善但出现遗忘或偏科] → 预登记通用保护指标、补数混合比例及实际覆盖，报告所有结果，不自动选择赢家或继续迭代。
- [费用和重试不可控] → 串行执行、持久化预算和 uncertain 状态、明确估算边界；mock 验证所有成本控制契约，真实费用另行显式执行。
- [供应方变更导致不可逐字复现] → 缓存完整响应及摘要，记录 revision 不可用；复用与重新请求分别标注。
- [小样本和反复诊断产生过度结论] → 分母与 seed 可见，记录已使用诊断集和迭代历史，最终独立评测保持明确角色。

## Migration Plan

1. 先确认 data-policy-experiments 的依赖行为与测试已完成；新增场景/生成类型、fake provider 与本地 CPU fixtures。
2. 落地可选 provider、预算/恢复、验证导出和 lineage，再接入场景计划与效果报告；所有自动测试保持无真实网络调用。
3. 增加前端计划配置/导出和报告视图，同步中英文路线及中文 provider/数据操作说明；记录生成条款和公开产物边界。
4. 实施验收不自动调用真实服务或训练。用户之后配置所需资源并显式执行，真实 run/结果作为新增证据保存。
5. 回滚为停用生成 extra/计划入口；已有数据和报告继续可由内容摘要/既有 CLI 检查，原开源数据工作流无需迁移。

## Open Questions

没有阻止该切片实施的未决设计选择。具体供应方账号/模型、真实调用预算、业务场景阈值及 GPU 执行时间由用户在执行计划中提供，当前不选择或猜测，也不据此宣称补数一定有效。
