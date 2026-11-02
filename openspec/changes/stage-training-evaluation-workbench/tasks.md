# Tasks

## 0. 统一信息架构与现有页面迁移

- [x] 0.1 将主导航收敛为数据、训练与评测、实验 Runs、分析与改进，建立模块内二级视图及旧 URL 解析，隔离各模块筛选状态；用深链接/前进后退测试验证 data/build/stage/tab、workbench、policies、targeted、launcher、runs/run、comparison/matrix 仍能定位，并更新中英文 INFORMATION_ARCHITECTURE 产品文档。
- [x] 0.2 将已有数据集目录、来源/导入、样本预览、处理与质量、规则编辑整合到数据模块，以统一数据集详情承接质量和血缘；验证用户可由一个数据入口完成导入、查看样本和处理证据，并更新中英文 PAGE_SPEC 与数据工作台指南。
- [x] 0.3 将现有普通/seeded SFT 生成和定向生成的执行、审核、登记 UI 提取为数据模块共享流程；保持输入契约差异和真实阶段支持范围，验证普通生成无需手填方案 ID、定向上下文可回溯评测且 holdout 样本不进入生成输入，并更新生成指南。
- [x] 0.4 区分表单草稿/预检 token/持久任务引用，提供跨模块活动任务入口与恢复逻辑；验证生成中切换方式、切换模块、刷新、返回后原 execution/run 仍可查看，输入变更只使预检失效且不重复启动，并更新任务恢复说明。
- [x] 0.5 将阶段比较、已执行策略效果报告与薄弱场景诊断整合到分析模块，规则编辑仍返回数据处理，定向生成带上下文进入数据获取；用 fixture 验证报告/规则/目标交接保持同一身份且不重复登记，并更新分析操作指南。
- [x] 0.6 增加各模块独立 loading/empty/error/unsupported/partial 呈现，默认显示中文名称和指标解释、单位/分母，将 ID/hash/原始配置按需展开；验证数据目录失败不隐藏正常 Run、实际调用能力文案一致、关键操作在首屏可发现，并同步页面交互文档。

## 1. 阶段能力目录与输入预检

- [x] 1.1 定义服务端 stage/profile 注册表和 API DTO（阶段、算法、approved recipe、父模型规则、训练数据 stage/role、adapter、产物核验、可用评测 profile）；为 CPT/SFT/DPO/SimPO/GRPO 能力清单添加单测，并更新阶段能力文档。
- [x] 1.2 修正只读能力/预检 API 的证据解析：按登记 dataset ID 校验 manifest、内容 hash、role、quality/license evidence、父模型 lineage 和本机依赖；固定 Mini SFT 与通用工作台共享 swift 环境发现及父模型 lineage 校验；用嵌套 merge-run 与 Python Scripts PATH fixture 覆盖回归，并更新 Launcher API 指南。
- [x] 1.3 实现一次性预检确认摘要，绑定 stage/profile/config、父模型摘要、dataset ID/hash 和输出位置；测试篡改、过期、重复使用 token 均被拒绝，并记录安全边界文档。
- [x] 1.4 统一所有阶段构建/导入/生成登记器的 `canonical_path` 为相对 dataset artifact root 的路径；manifest 写入前核验路径 containment、文件存在性及 canonical/lineage SHA-256。增加旧 manifest 的安全修复流程：仅在唯一文件和原摘要完全匹配时更新；测试 OpenR1、FineMath、DeepMath、policy、calibration、upload/generation 入口及无法判定时 fail closed，并修复当前可验证的本地旧清单。
- [x] 1.5 为 stage profile 声明默认训练配置、可自定义参数白名单、类型/范围及参数间约束；用各阶段配置校验测试验证合法覆盖通过、未知字段/越界/不兼容字段被明确拒绝，并更新 profile 文档。当前各字段独立约束，跨字段约束为空；recipe 默认值也必须落在白名单约束内。
- [x] 1.6 将解析后的有效配置纳入预检确认摘要与一次性 token；测试参数任一变化会使旧预检失效，且预检与启动读到同一 config hash。SFT 阶段预检与实际启动预检交叉比对 hash；单次确认 token 绑定覆盖值/hash；Run 记录覆盖值与有效配置 hash，lineage hash 指向落盘 resolved_config.json。

## 2. 公共训练 Run 生命周期与 SFT 兼容

- [x] 2.1 抽取通用单活动 GPU supervisor 生命周期、持久状态、有界日志和进度解析接口；为 queued/starting/running/completed/failed/OOM/interrupted 状态添加状态机测试，并记录运行协议。
- [x] 2.2 将现有固定 Mini SFT Launcher 接入公共 supervisor，同时保留现有 API/URL/recipe 默认行为；运行 `tests/test_ui_launcher.py` 与相关前端测试，并更新 SFT Launcher 兼容说明。
- [x] 2.3 让所有新 stage Run 统一写出可校验的 run.json/lineage/model artifact 引用；测试成功、失败、合并失败只记录实际产物，并更新 run/lineage 使用文档。
- [x] 2.4 让已开放的受控训练 adapter 从 profile 默认值与已校验覆盖值构造最终 CLI 配置，不接受浏览器提供的任意 recipe 或命令参数；以 fake CLI 验证生效配置、run.json、lineage 与 config hash 一致，并保持 Mini SFT 默认配置兼容。当前网页仅 SFT adapter 开放，其他 profile 仍 fail closed。

## 3. CPT 与 SFT 阶段纵向切片

- [x] 3.1 实现 CPT 阶段 adapter：验证 CPT canonical token corpus、Base parent 和固定 recipe，适配数据并受控调用现有 CLI；用 fake CLI 覆盖成功/失败/OOM/父模型缺失，并更新 CPT Launcher 操作说明。
- [x] 3.2 实现 SFT 阶段 profile 与已登记数据选择，支持批准的 parent 选择规则但保留 Mini 固定 profile；添加 SFT dataset selector/preflight/run 端到端测试，并更新中文工作流文档。
- [x] 3.3 实现 CPT/SFT 评测 profile：复用统一 benchmark evaluator，CPT 接入有证据的 domain PPL；测试分母、协议与 not_measured 状态，并更新指标说明。

## 4. Preference 阶段 DPO / SimPO

- [x] 4.1 实现 DPO 与 SimPO 独立 profile，验证 Preference pair 数据、chosen/rejected 证据及同源 SFT parent，并保持两者独立 Run；添加阶段/契约/父模型不匹配测试并更新并行分支文档。
- [x] 4.2 实现 DPO/SimPO 训练监督、合并与 lineage adapter；用 fake CLI 验证实际输出登记、失败不伪造模型以及 SFT parent 关系，并运行新增训练 adapter 测试。
- [x] 4.3 实现 Preference held-out 评测 profile，与共享任务 benchmark 分开展示；测试禁止训练 pairs 作为评测输入及 partial/failed 指标状态，并更新 Preference 指标说明。

## 5. GRPO 阶段纵向切片

- [x] 5.1 实现 GRPO profile，校验 GRPO prompt schema、reward/reference contract 可执行性和 SFT parent；为缺契约、契约摘要不符及有效数据集添加测试，并更新 GRPO Launcher 文档。
- [x] 5.2 实现 GRPO standard/lowmem 中经批准的训练 adapter 与真实 rollout 诊断收集；用 fake CLI 验证 OOM/部分产物和未知 reward 指标保持真实状态，并更新 GRPO 指标解释。
- [x] 5.3 实现 GRPO 统一任务评测与 reward/rollout evidence 展示；测试未运行 rollout 时显示 not_measured、完整 lineage 可回溯，并更新阶段评测文档。
- [x] 5.4 将 targeted GRPO prompt builder 拆分为普通/定向可共享的确定性核心和上下文适配器；普通数据服务仅接受 OpenR1 固定 build protocol/revision 来源及逐条 `correctness_math_verify=true`、source record ID 证据，并复用批准 reward contract、canonical registrar 和质量报告。记录父 dataset/hash/record 与 contract/build protocol 摘要，明确上游标签非本系统独立验证。覆盖成功、来源不可信、答案缺失、伪造 contract/客户端字段、零可用记录、摘要/血缘闭合、定向上下文隔离和重复登记测试；失败不留半成品，现有定向 builder 行为保持兼容。

## 6. 阶段工作台 UI 与 Run/评测串联

- [x] 6.1 构建 CPT/SFT/DPO/SimPO/GRPO 阶段导航、可用性/blocker、阶段化数据集选择和预检摘要；增加 React 交互测试，验证阶段切换后选择/确认状态不会误用，并更新中文页面指南。
- [x] 6.2 将训练启动后的观察与评测统一到 Runs 详情，Launcher 与历史列表复用同一 Run 视图，呈现状态、实时日志、产物和 lineage；测试不同入口打开同一 run 不产生重复状态、失败/中断/缺模型时评测入口关闭，并更新 Runs 页面文档。
- [x] 6.3 实现各阶段评测确认、独立评测 Run 观察、指标分母/状态和回链；增加 API/UI 测试验证训练 Run 终态不被评测覆盖，运行 390px 与键盘可访问性测试并更新用户操作指南。
- [x] 6.4 接通数据登记/详情的“用此数据训练”、训练启动后的“查看运行”以及评测结果的“分析/补充数据”，自动传递阶段、真实对象 ID 和返回位置；以用户任务测试验证无需复制 ID、导航不启动任务、定向数据登记可返回原诊断，并更新中文端到端操作指南。
- [x] 6.5 将阶段、profile、父模型和数据集收敛为训练页唯一选择状态；把固定 Mini SFT 卡迁入 SFT 配方选择，使预检与启动消费相同身份/摘要并在输入变化时失效旧预检。其余阶段沿用相同操作路径、保留阶段专属输入；尚无网页启动 adapter 的 profile 明确阻断且不显示可启动状态。增加 SFT 启动绑定与跨阶段能力呈现测试，并更新中文操作指南。
- [x] 6.6 加固端到端 lineage 校验与 Runs/数据详情展示：验证 source/parent dataset -> registered dataset ID+canonical hash -> consuming training Run(dataset ID/hash/role、parent model ID/hash、profile/config) -> adapter/merged model -> evaluation Run(model ID/hash、training Run、evaluation data/protocol) 的引用闭合；覆盖 CPT->SFT、同一 SFT parent 并行分叉 DPO/SimPO/GRPO、多个训练数据身份、失败/部分产物及缺失/错 hash 断链。校验失败需定位对象并 fail closed；不得按名称推断或伪造关系。增加后端 fixture/API/UI 往返追溯测试并更新 lineage 指南。
- [x] 6.7 实现成功训练 Run 到下一阶段的模型交接：由 profile parent 规则计算兼容下游入口；Run/模型详情提供“用作下一阶段父模型”，深链传递目标阶段/profile 和精确 model ID；训练页加载后显示来源 Run 并预选该模型，但要求用户另选数据、重新预检并显式启动。父模型选择器显示阶段、artifact 类型、producing Run、短 ID 与 lineage 链接；对失败/缺 lineage/不兼容产物不给可启动 CTA 并说明原因。覆盖 CPT->SFT 与同一 SFT parent 到 DPO/SimPO/GRPO 分支及刷新/返回恢复测试。
- [x] 6.8 在阶段训练页提供默认配置快速路径和渐进展开的自定义参数区；仅渲染当前 profile 支持的参数及范围，展示有效配置/资源提示，参数修改后清除旧预检状态。用前端交互测试及中文操作文档验证默认不配置即可预检、自定义值正确回显、非法值可理解地报错、重新预检后配置绑定正确。当前网页仅 SFT profile 已有实际启动 adapter，其余 profile 继续明确阻断。
- [x] 6.9 在数据模块接通普通 GRPO seed 派生工作流：只列出来源协议、逐条参考答案证据与 train_seed 角色均合规的已登记输入，展示批准 reward contract、预览可用/拒绝条数、质量/许可限制和血缘后显式登记；无合规 seed 时说明证据缺项与可执行恢复入口。登记成功后以精确 dataset ID 返回 GRPO 训练页，不自动预检/启动；布局为后续定向上下文适配器保留扩展点。测试失败恢复、刷新状态、键盘/窄屏交互以及伪造来源/contract 字段拒绝。

## 6.10 用户自带客观题评测集

- [x] 6.10.1 增加 JSONL 单选/多选/判断题导入映射与 canonical 转换，保留源文件摘要、格式/转换版本和行级 lineage；用上传样例及 fixture 验证可变选项数（含 F）、缺失答案逐行拒绝、有效/拒绝计数准确，并更新中文导入格式指南。
- [x] 6.10.2 将转换结果登记为 evaluation-only 数据身份，禁止训练选择器消费并在预检检查目标训练 lineage 的 dataset ID/hash 冲突；增加导入、角色隔离和泄漏阻断 API 测试，并更新数据角色说明。

## 6.11 客观题确定性评测协议

- [x] 6.11.1 实现版本化 choice-accuracy evaluator：单选/判断精确匹配、多选答案集合完全匹配、无法解析单独统计；用混合题型、非 A–E 选项、错误预测和解析失败 fixture 验证各类型正确数/分母及 overall 统计，并更新协议文档。
- [x] 6.11.2 将自定义 suite 的 ID/hash、样本覆盖与 evaluator 版本写入评测结果/lineage；测试不同 suite/覆盖/协议不可直接比较且训练数据泄漏会阻断，并更新结果与可比性说明。

## 6.12 用户评测集前端闭环

- [x] 6.12.1 在数据模块提供评测集导入、预览拒绝原因、显式登记与详情入口；增加 React/API 交互测试验证坏行修正指引、零有效记录不可登记、登记后角色和身份清晰，并更新中文操作指南。
- [x] 6.12.2 在兼容阶段评测配置中选择已登记客观题 suite，预检展示版本、覆盖、协议和泄漏检查，结果按题型展示正确数/分母/解析失败并标识不可比项；增加 UI/API 集成测试验证不支持的 profile 不提供伪可用选项，并更新评测工作台指南。

## 6.13 训练时验证集与可复现随机切分

- [x] 6.13.1 为阶段 profile 增加三种显式验证来源：无训练时验证、已登记独立验证集、从训练集随机切分；服务端按 profile 校验验证 schema/角色、比例及最小样本数，切分使用固定算法版本和种子，Preference pair 保持原子并尽可能按稳定 problem/source group 分组。为随机切分生成不可变的 train/validation 成员清单及派生输入，保证成员摘要可进入 lineage；仅在能捕获并审计全部拆分成员时才允许直接使用 ms-swift 原生 `--split_dataset_ratio`。用测试验证相同数据摘要/算法版本/种子得到完全相同成员，不改变父数据文件，非法/空分区被阻断，final_holdout 不可作为训练或验证输入。
- [x] 6.13.2 在训练页增加互斥来源选择；选择切分时设置受限比例和随机种子、展示切分后的训练/验证数量及样本减少/潜在泄漏提示，默认仍不切分；验证来源变化清除旧预检/确认。用前端交互测试覆盖三种模式、边界比例和输入变化，并更新中文用户指南。
- [x] 6.13.3 将验证集身份或随机切分算法/比例/种子/成员摘要/数量绑定到 resolved config、确认 token、Run 与 lineage；阶段 adapter 将已核验的训练输入通过 `--dataset`、验证输入通过 ms-swift `--val_dataset` 传递，并按 profile 设置 `--eval_strategy`，不得同时启用 `--val_dataset` 和非零 `--split_dataset_ratio`。仅向明确支持训练期验证的阶段 adapter 传验证输入；结果明确区分训练指标与验证指标，不支持验证的 profile 显示 blocker/not_measured。用 fake CLI/API/lineage 集成测试验证预检与启动一致、参数互斥、刷新可追溯、final_holdout 仅在训练后的独立评测使用，并运行相关阶段回归。

## 6.14 已登记 Base 模型网页独立评测

- [ ] 6.14.1 为已登记且可核验的外部 base 模型增加受控网页模型目录/API 与评测预检/启动接口；支持数学基准和 `choice-accuracy-v1`，令牌绑定 model ID/hash、profile、evaluation-only dataset ID/hash（如适用）及输出，启动重新核验身份并使用单 GPU supervisor 写独立 Run/evaluation lineage。覆盖模型摘要不符、路径不可用、确认过期/重复使用、GPU 占用、客观题角色错误与接口失败，确认无 training Run 时 lineage 正确保留 external base root 身份。
- [ ] 6.14.2 在 Runs/评测页面增加 base 模型选择和 profile/客观题 suite 选择；展示人类可读的模型、评测协议、题数与摘要技术详情，完成预检、显式启动、运行观察、刷新恢复和结果追溯。增加前端/API 集成测试验证输入变更会使预检失效、模型和 suite 身份不串用、桌面/390px/键盘操作可用，并更新中文操作指南。

## 6.15 客观题逐题评测报告

- [x] 6.15.1 增加只读、受限的逐题报告 API：根据 evaluation lineage 核验 dataset ID、canonical SHA-256、protocol 与 prediction 文件，再按唯一题目 ID 联接 canonical 题干/选项/标准答案和预测/原始输出；缺失、重复或身份不符时显式返回不可用/不完整，不按行号猜测拼接。
- [x] 6.15.2 在 Runs 客观题评测详情展示分题型汇总和逐题报告，支持全部/答对/答错/解析失败与题型筛选；显示题干、选项、标准答案、模型答案和判定状态，原始输出默认折叠，提供命中数、空结果及窄屏/键盘友好交互，并更新中文操作指南。
- [x] 6.15.3 将逐题报告摘要放在指标/血缘网格之前；逐题明细和筛选控件默认折叠，同时保持报告标题与答题状态数量概览可见；使用键盘可操作的原生折叠控件，更新设计说明。

## 6.16 Final / best checkpoint 选择策略

- [x] 6.16.1 扩展阶段 profile 与有效配置，增加默认 `final`、可选 `best` 的 checkpoint policy；为支持 best 的 profile 声明验证指标、优化方向和 eval/save 频率约束。选择 best 而无可用验证集或指标时预检阻断，final holdout 不得作为候选选择输入。
- [x] 6.16.2 更新各训练 adapter 的 checkpoint 解析与模型合并：latest 继续用于续训，分别核验并保留 final 与 best；阶段模型依 policy 指向对应 checkpoint，同一 checkpoint 不重复存储。覆盖编号步数选择、指标缺失、best/final 分离及失败不登记伪产物。
- [x] 6.16.3 将 policy、验证数据身份、指标名/方向、final/best/latest 的 step/path/hash 和阶段模型采用者写入 Run、模型 artifact 与 lineage；UI 预检、运行详情和产物列表清楚展示选择结果及续训位置，并测试刷新后的可追溯性。
- [x] 6.16.4 确认固定 Mini SFT、CLI recipes、已有 stage profile 与旧 Run 的兼容行为：默认 final 保持现有语义；旧 Run 可读取且明确标记其历史选择规则；运行相关后端/前端回归并执行 OpenSpec strict validation。

## 6.17 Direct Base-to-SFT support

- [x] 6.17.1 扩展 SFT profile 的父模型规则，使工作台可从已登记模型目录选择 Base 或兼容 CPT 模型；保留固定 Mini SFT 入口及其默认行为，不接受浏览器提交任意模型路径。
- [x] 6.17.2 为 Base→SFT 预检实现登记记录、权重可用性、摘要、路径/符号链接边界、训练框架/架构与分词器/数据格式兼容性核验；对不兼容登记模型返回具体 blocker，覆盖 workspace 内外的安全路径情形。
- [x] 6.17.3 使 SFT adapter 从已核验的父模型身份生成最终训练配置，并将精确 Base model ID/hash、profile/config 摘要写入确认 token、Run、输出模型和 lineage；Base 作为无 producing Run 的外部根节点处理。
- [x] 6.17.4 增加 Base→SFT profile、预检/启动、失败阻断、血缘和 UI 选择覆盖；更新中文阶段能力与 SFT 操作文档，并运行 OpenSpec strict validation。

## 7. 跨阶段集成与兼容验收

- [x] 7.1 使用本地 fixture 完成五阶段端到端验证：登记数据选择、stage preflight、fake training、模型核验、post-training evaluation、run/model/dataset/eval lineage；运行全部相关后端/前端测试并保存验证摘要。
- [x] 7.2 验证单 GPU 锁、浏览器刷新恢复、服务停止中断、旧 Mini SFT API/CLI 兼容以及错误协议不作横向比较；执行 OpenSpec strict validation 和完整回归，记录剩余真实 GPU/网络验收 blocker。
- [ ] 7.3 使用浏览器完成三条完整任务走查：导入/普通生成并登记到训练、SFT 单入口预检/启动/独立评测和结果追溯、成功 Run 的模型交接到兼容下一阶段、薄弱场景到定向生成登记再训练；逐阶段确认默认配置与自定义参数、父模型和数据输入要求清楚，未接通的启动能力被如实阻断。覆盖参数修改导致预检失效、重新预检后配置一致、方式切换、页面返回和单接口故障，记录桌面/390px/键盘操作结果，并确认现有可用能力没有因导航迁移变成不可达。
  - GRPO 数据准备补充验收：从可追溯 train seed 构建并登记 GRPO 数据，核对逐条 reference/contract 证据、预检可通过及训练页精确回接；缺少可信 seed 时验证 fail-closed 与恢复引导。真实 GPU 训练/评测须另行显式确认，不作为浏览器任务测试的隐含副作用。
  - 2026-10-24 局部浏览器走查已覆盖工作台桌面/390px/键盘、Runs 列表与成功 SFT Run 详情、SFT→DPO/SimPO/GRPO 模型交接及精确 URL 身份恢复；修复并回归 merge `adapter_id` 识别和 Launcher 路由参数丢失。完整导入/生成登记、训练/评测启动及定向补数闭环仍待受控端到端验收。详见 `docs/reports/browser-walkthrough-2026-10-24.md`。
  - 追加复核：普通/seeded 生成方式切换无 POST；模拟 `/api/data` 503 后数据页局部报错且 Runs 仍可导航并读取 53 条记录；训练与薄弱场景页面在桌面/390px 渲染正常，现有 SFT Run/评测和下游交接入口可见。仍未新增真实数据、Run 或评测，因此 7.3 保持未完成。
  - 2026-10-24 真实探索性闭环追加：用户确认后以已登记 `ds-sft-generated-f88a69ec5766e5f42cede07c` 完成 SFT Run `run-mini-sft-1024-0fe815479943`（1/1 step）及独立评测 `run-launcher-mini-math-eval-41b6a382f28f`（GSM8K 0/5、MATH-500 1/5）；父模型、dataset ID/hash、adapter→merge model→evaluation 引用均经定向断言核对。工作区 lineage audit 现为 134 runs/151 artifacts/47 datasets/40 models/25 evaluations，74 errors/74 warnings，新增 Run 无 errors。此项是 API 实际运行与指标/血缘验证，不等于完整浏览器走查；普通导入/生成登记链路和薄弱场景→定向生成→登记→训练仍待受控验收，故 7.3 保持未完成。详见 `docs/reports/browser-walkthrough-2026-10-24.md`。
- 2026-10-26 普通 GRPO seed 派生与定向共享核心完成：普通流程只接受登记证据完整的 OpenR1 固定来源/revision、处理协议和逐条参考答案证据；服务端生成 reward contract 并校验父级 canonical SHA256/构建 Run，确定性选择后复用 canonical 注册与质量/血缘产物。普通/定向路径共用 `grpo_prompt_core`，计划上下文仅由定向 adapter 附加。数据工作台展示候选证据、许可限制、reward contract、可用/拒绝计数，并支持草稿刷新恢复、显式登记和精确 dataset ID 回接。验证：`tests/test_grpo_seed.py` 6 passed、`tests/test_targeted_grpo.py` 及 `tests/test_dataset_mixture.py` 回归通过；Vitest 工作台套件 12 passed；前端 build 和 OpenSpec strict validate 通过。真实候选为 OpenR1 375/375 与 8/8 条可用，但许可状态均为 `not_provided`，UI 明示仍需来源许可确认；没有启动训练。7.3 的完整浏览器/GPU 端到端走查仍未完成。
- [x] 7.4 在真实本地 artifacts 上运行只读 lineage integrity audit，覆盖数据 manifest/canonical hash、所有可发现 training run、parent/output models、merge artifacts、evaluation runs 与 metrics 引用；输出可复查的断链/摘要不一致报告和通过统计。对失败及历史对象保留真实状态，不批量猜测修复；对可安全修复项另行列出证据与建议，不在审计中改写用户 artifacts。报告：`docs/reports/lineage-audit-2026-10-24.md`（123 Runs、143 artifact 引用、47 datasets、34 models、23 evaluations；44 errors、100 warnings；只读未修复）。

## Workflow follow-up

- 评审后按 OpenSpec apply 实现；各阶段真实 GPU 冒烟通过后再归档变更。
- 2026-10-30 继续执行 7.3：普通 SFT 生成候选已审核登记，单条 smoke SFT Run、独立阶段评测、评测 lineage 与成功 Run→DPO 父模型预选/只读预检已验证；薄弱场景诊断报告缺少逐题标签，定向闭环及部分纯浏览器确认/参数失效走查仍未完成，因此 7.3 不勾选。完整边界与 Run IDs 见 `docs/reports/browser-walkthrough-2026-10-30.md`。
