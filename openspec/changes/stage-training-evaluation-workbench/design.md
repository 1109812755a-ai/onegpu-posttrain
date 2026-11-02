# Design

## Context

参见 `proposal.md`。当前 `src/onegpu_posttrain/ui/launcher.py` 的 `SFTLauncher` 是窄范围本地 supervisor：固定一个 Mini SFT recipe、单个已登记 SFT 数据集、loopback/token/单活动 GPU 约束；训练后可对核实的 M2 启动独立评测。其余阶段已有 CLI recipes、阶段化数据契约和评测命令，但没有同等的前端选择/预检/运行生命周期。现有数据/模型/run/evaluation lineage 和 `Run` 页面应作为事实源继续复用。

当前主导航同时存在“数据 Data”“数据集工作台”“策略对照”“定向补数”“训练 Launcher”“实验 Runs”“阶段比较”，并有尚不可操作的占位项。`main.tsx` 还把 data 的 build/stage/tab 筛选带到其他页面；普通 SFT 生成和定向页各有生成/登记交互。职责与上下文分散，会让用户难以判断入口和下一步。因此本设计同时确定模块边界，后续阶段能力和数据集组合沿同一结构接入。

## Goals / Non-Goals

**Goals:**

- 将阶段配置、训练输入验证、执行监督与评测 profile 声明为明确的受控能力，而非任意命令执行器。
- 同一 UI 工作流覆盖 CPT、SFT、DPO、SimPO、GRPO；不可运行的阶段/配置显示具体 blocker。
- 训练和评测均继续产生 CLI 权威 artifacts，各自有真实状态与 lineage。
- 每类对象与操作有明确的主要入口；用户从数据准备到训练、评测和改进时能带着实际对象继续操作。

**Non-Goals:**

- 不开放任意 recipe、白名单外/越界超参数、shell、远程 GPU、队列、并发或自动评测；自定义参数仅限 profile 明确声明的字段与范围。
- 不实现组合数据逻辑；工作台接收一个已登记 dataset ID，未来组合产物作为普通 ID 使用。
- 不以同一阶段一次真实 GPU 成功作为代码提案的前置条件；硬件/外部数据不可用需显示为真实阻断。
- 信息架构调整复用现有数据处理/生成接口；新阶段生成器、数据清洗引擎、全新 Overview 仪表盘不在本变更中实现。

## Decisions

### 1. 用服务端阶段 profile allowlist 驱动 Launcher

定义只读阶段能力目录，每个 profile 绑定阶段/算法、批准 recipe、默认训练参数、可由用户调整的参数白名单及范围、父模型解析器、dataset validator/adapter、固定训练 CLI 构造器、输出核验规则和允许的评测 profile。界面提供“使用默认配置”和可展开的“自定义参数”两种路径；默认路径无需填写参数。自定义仅可修改该 profile 声明的字段，服务端按类型、阶段、上下限及参数间约束重新解析和验证，拒绝未知字段、文件路径、命令、环境变量、recipe 指定和白名单外覆盖。前端提交 stage/profile ID、登记 dataset ID 与受限参数对象；服务端权威计算有效配置、资源提示和 blocker。

理由：用户可针对显存、数据规模和实验目的调整关键训练设置，同时无需理解或编辑 YAML；默认模式保留一键顺畅路径。服务端 allowlist 将易用的参数自定义限制在可验证范围内。替代方案是要求用户编辑/上传完整 recipe 或暴露任意命令参数；这会让本机 UI 成为通用 shell/训练服务，扩大安全与 OOM 风险，并削弱阶段数据、父模型和 lineage 校验。

### 2. 阶段依赖显式映射，不自动串跑上游

CPT profile 接受 Base 模型与 CPT corpus；SFT 接受已登记且通过对应 profile 兼容性预检的 Base/CPT parent 与 SFT dataset；DPO、SimPO、GRPO 各自从合规的 SFT parent 分叉，Preference dataset 只供 DPO/SimPO，GRPO 使用有可执行 reward/reference contract 的 GRPO dataset。每个 run 仅执行用户选择的阶段；缺少 parent 时提示先完成对应阶段，不自动触发其他训练。

SFT 的父模型选择由服务端模型登记目录驱动，允许选择任意已登记 Base 或兼容 CPT 产物；不把 Qwen3-0.6B 或单一父阶段写死在所有 SFT 配方中。每个 SFT profile 声明适用的模型/输入兼容条件和批准 recipe。预检针对所选登记模型核对 kind/stage、可解析本地权重、登记摘要、训练框架/架构支持、分词器与数据格式兼容性；结果及精确 model ID/hash 绑定到有效配置和一次性确认。Base 可作为没有 producing Run 的外部 lineage 根节点；其本地路径只从已验证登记读取，即使登记路径位于 workspace 外，也必须通过受信任的登记记录、严格路径/符号链接边界和内容摘要校验。UI 不接收任意模型路径；不兼容或不可用的登记模型保留 blocker，不触发下载、自动转换或训练。

训练数据都通过 manifest 解析并适配成 CLI 所需 native 输入，训练 run 记录 canonical dataset ID/content hash。第一版使用单个 dataset ID；新组合数据集已经是普通登记产物，不需要 launcher 理解其父数据列表。

所有数据集 artifact manifest 统一使用 artifact-root-relative `metadata.canonical_path`，不混用 manifest-directory-relative 路径或机器绝对路径。每个构建器在写 manifest 前验证目标是 artifact root 内的普通文件（拒绝符号链接/越界路径），并将文件摘要与 `hashes.canonical_records_sha256`、lineage content digest 对齐。`DatasetManifest` 的加载保持兼容历史文件；训练预检仍 fail closed。迁移旧产物时只允许从已知构建输出布局定位唯一 canonical 文件，并要求其现有摘要与 manifest/lineage 完全相同后才原子更新 manifest；不能证明时不猜路径、不改 hash。

### 3. 统一 supervisor 与不可变运行绑定

保留单机 loopback、显式启用、令牌保护、单活动 GPU 锁、一次性 confirmation token 和有界日志策略；将现有 SFT 实现逐步抽象为基于 profile 的公共生命周期。profile 默认值与本次经校验的用户覆盖值合成为不可变有效配置；预检 token 绑定 stage/profile 版本、有效配置完整内容及 hash、父模型 ID+digest、dataset ID+content hash 和输出目录。对 Base→SFT，解析后的有效配置必须指向所选登记模型的已核验权重，不能继续使用 profile 中固定的其他基础模型名称。任何参数变化都要求重新预检；启动端只能执行 token 对应的有效配置。Run 记录 profile 版本、默认值与用户覆盖值（或等价的完整 resolved config）、摘要和实际 CLI 配置。不得改变 Mini SFT 的既有默认行为。评测作为独立子运行有自己的 run ID/eval ID，训练记录关联评测引用而不覆写训练终态。

阶段、profile、父模型和数据集在训练页内维护为一份唯一的选择状态；只读预检和真实启动都消费这份状态。Mini SFT 是 SFT 下的一个固定 profile，原固定配方入口迁入 SFT 配方选择，不再保留一张要求重复选数据的并列启动卡片。各阶段采用同一交互顺序，但字段、验证和评测要求由阶段契约决定。网页启动能力以服务端 adapter 为准：没有 adapter 的 profile 显示不可启动原因；不得因为只读预检成功而开放假的启动按钮。

血缘的权威来源是登记的 manifest、Run/lineage 和 evaluation artifacts，而不是 UI state。数据构建/导入/生成记录 source 与父数据关系和 canonical 内容摘要；训练 Run 按实际消费记录每个 dataset ID/hash/role 及 parent model ID/hash；输出模型记录 producer Run、parent 与 adapter/merge 关系；评测 Run 记录精确模型摘要、训练 Run、评测输入和协议版本。统一校验器沿这些引用检查对象存在、摘要一致、关系方向正确和身份不重复；UI 只渲染已验证关系，并提供从 dataset/model/run/evaluation 任一节点双向追溯的入口。证据缺失时显示断链及对象 ID，禁止从名称或路径猜补。

下游模型交接基于服务端 stage profile 的 parent stage/kind 规则生成，而不是硬编码“训练完就接下一阶段”。成功 Run 详情对每个已登记且存在兼容 profile 的模型显示“用作下一阶段父模型”；对于同一 SFT parent 可分别进入 DPO、SimPO 或 GRPO。点击后通过 URL/context 传递目标 stage/profile 与精确 model ID，训练页加载后重新查询 registry 并预选父模型，继续要求用户选数据、预检和显式启动。父模型列表以“阶段 · artifact 类型 · producing Run · 短 ID”作为识别信息，提供 lineage 查看入口；展示“已登记/待预检验证”而非提前声称摘要已通过。失败 Run、缺失/冲突产物或不兼容模型不提供启动交接，并给出阻断原因。

替代方案是把训练和评测压成一个子进程/Run；这会混淆训练成功但评测失败的结果，也破坏既有独立 evaluation lineage。

### 4. 评测 profile 分阶段定义，通用与专属指标分区

评测 profile 由服务端登记且固定版本：CPT 适用统一任务评测和领域 PPL；SFT 使用统一任务评测；DPO/SimPO 使用统一任务评测及独立 held-out preference 指标；GRPO 使用统一任务评测及真实 rollout/reward 诊断。Profile 声明 benchmark/holdout ID、协议、样本上限、指标来源和可选/必选检查。评测数据必须是非训练用途；不能启动的 profile 在预检阶段给出具体缺失项。通用 benchmark 指标只有协议一致才比较；阶段专属指标独立呈现。未知或未执行的值保持未测，不从训练曲线推断。

保留已有 `run-model-eval`/专属 evaluator 的 CLI 产物职责。若某阶段的评测器不能证明相应指标，本版先标 unsupported/not_measured 或阻断该 profile，不能为了页面完整度伪造统一分数。

### 5. 主导航固定为四个用户任务，二级视图承接现有功能

| 主入口 | 负责的用户任务 | 二级视图与主要操作 | 交给下一模块的对象 |
| --- | --- | --- | --- |
| 数据 | 获得、理解并准备训练数据 | 数据集目录/详情；获取数据（已配置来源、本地导入、模型生成）；处理与质量（规则、预览、报告、血缘）；组合能力接入后在此创建版本 | 已登记 dataset ID、stage、可用性 |
| 训练与评测 | 配置并启动一次阶段实验 | 选择阶段、批准配方、兼容数据与父模型；预检；显式启动。启动后进入 Runs 的统一详情 | 实际 training run ID |
| 实验 Runs | 持续观察与追溯实际运行 | 列表/筛选；共享 Run 详情（概览、日志、评测、产物与血缘）；成功后“评测此模型”创建独立评测 run | model/run/eval ID 与协议 |
| 分析与改进 | 理解结果并提出下一轮数据改进 | 阶段指标比较、策略效果报告、薄弱场景诊断；从真实评测发起定向补数 | 场景目标和来源 eval/model/run 引用 |

“训练与评测”是进入完整流程的入口，运行中的训练和评测共享 Runs 详情，避免在 Launcher 和 Runs 维护两份生命周期视图。分析读取相同的 metrics/evidence，不维护另一份结果。来源类型和训练阶段仍是两个独立筛选维度。

旧“数据 Data”成为数据内的“处理与质量”视图；旧“数据集工作台”成为数据主入口的目录。旧“策略对照”拆出规则编辑/数据预览到数据模块、已执行实验的效果报告到分析模块，二者用同一 policy/experiment 身份互链。旧“阶段比较”成为分析的比较视图。旧“训练 Launcher”使用中文任务名称“训练与评测”；CPT/SFT/DPO/SimPO/GRPO 是其内部阶段选择，不再增加五个主导航。未提供实际内容的“概览”“对齐分析”占位从主导航移除；支持范围在相关页面就地解释。

备选是继续给每个新功能增加平级导航，或将所有步骤装入一个长表单；前者让入口膨胀，后者难以支持独立训练和历史回访。四个任务模块配合对象深链接能支持两种使用方式。

### 6. 定向补数复用数据生成流程，通过上下文体现差异

数据获取页是普通生成、seeded 生成和从分析发起的定向生成的共同入口。分析拥有诊断证据与目标选择，数据模块拥有 provider 选择、生成、质检、审核和登记。现有定向方案作为可选上下文传递，不作为普通生成必须由用户手工登记的步骤。两条路径在生成执行和数据登记处汇合：

```text
Data / ordinary generation ------------------+
                                             v
Analysis / eval / scenario --> target context --> Data / generation
                                                   |
                                                   v
                                         review / quality / register
                                                   |
                                                   v
                                  dataset ID --> Training --> Run --> Evaluation
                                                                  |
                                                                  v
                                                               Analysis
```

实现时复用现有生成/质检/登记/血缘服务，并提取共享的生成执行、审核与登记 UI；不要求定向和普通模式拥有相同输入 schema。定向页保留独有诊断与效果证据，移除第二套 provider、任务启动和登记表单。已配置 provider 可自动选取唯一可用项；用户主要输入目标、数量和方式，高级设置折叠。尚不支持的阶段明确说明可用范围。

定向上下文携带计划版本、诊断评测/模型/run 引用、目标场景与允许的 train seed；最终 holdout 仅作评测引用，禁止其样本作为生成输入。登记成功后显示新 dataset ID/质量摘要及“用此数据训练”，将 dataset ID 和 stage 交给训练页，不自动开始训练。

### 7. 对象详情、返回路径和任务恢复统一

数据集详情按“样本”“处理与质量”“数据血缘”“关联运行”组织；同一个 dataset ID 从任意入口打开均解析到同一登记版本。Run 详情按“概览”“日志”“评测”“产物与血缘”组织，活动日志自动刷新，成功且模型核验后出现评测操作。训练记录本身保持真实终态，启动评测创建新的独立 run，结果带模型、训练/合并和协议引用。

跨模块交接显式带对象 ID、目标 stage 及返回位置。例如数据集详情打开训练页后预选该 ID，评测结果打开定向生成后保留来源证据，登记后返回原改进上下文。用户不填写内部 ID、manifest 路径或 hash；默认展示名称、阶段、规模和状态，技术身份与原始证据放入可展开详情。

表单草稿、预检确认和活动任务是不同状态：草稿按模式/阶段保存；修改输入使旧预检失效；已开始的生成 execution ID 或训练/评测 run ID 不随表单重置而消失。切换主导航、生成方式、刷新页面或前进/后退后，从持久任务引用重新读取服务端状态。活动任务摘要在各模块均有“查看任务”入口，路由变化不会自动启动、取消或重试。任务仍由现有 supervisor/job store 管理，不增加调度系统。

### 8. URL 兼容与模块状态隔离

保留现有 page key 作为兼容入口，主导航标签按四模块统一；解析层规范化为 module/subview/context，页面仅序列化自己的状态。数据 `build/stage/tab/scope` 不再被全局 effect 写入其他模块；跨模块只传明确的 dataset/model/run/eval/target 身份及返回位置。

| 现有链接 | 新模块落点 |
| --- | --- |
| `page=workbench` | 数据 / 数据集，保留数据集定位和筛选 |
| 缺省 page 或 `page=data`，含 build/tab 上下文 | 数据 / 处理与质量，恢复旧 build/stage/tab |
| 缺省 page，无对象上下文 | 数据 / 数据集 |
| `page=policies` | 数据 / 处理与质量 / 策略编辑；效果报告通过独立链接打开分析 |
| `page=targeted` | 分析与改进 / 薄弱场景；生成子操作带目标进入数据获取 |
| `page=launcher` | 训练与评测 / 对应阶段；历史 dataset 预选仍有效 |
| `page=runs&run=...` | 实验 Runs / 同一运行详情 |
| `page=comparison&matrix=...` | 分析与改进 / 对应评测矩阵 |

每个模块独立显示 loading、empty、unsupported、failed 和 partial；数据目录请求失败只影响其视图及依赖它的操作，不能把全站标成断连或卸载运行详情。页面能力提示由实际接口返回，普通生成与定向生成均显示真实调用范围，避免“仅本地编辑”等固定文案和可执行能力冲突。

### 9. 布局与操作反馈保持一致

页面顶部使用模块名、对象/阶段和简短说明；重要操作位于首屏，精确配置、原始 JSON 和 hash 使用渐进展开。状态区说明当前在数据准备、训练、评测或分析的哪一步，失败就地给出恢复动作。指标同时给出中文含义、单位、分母和适用范围，质量状态、来源许可、数量目标与训练可用性分别展示。

桌面使用主导航、二级视图与内容区；390px 及以上窄屏保持单列与独立日志/表格滚动。主操作、焦点和错误提示满足键盘访问。当前没有有效操作的功能展示说明，不使用无法解释的禁用按钮。

### 10. GRPO 数据准备使用共享核心与上下文适配器

GRPO 数据准备放在数据模块，作为训练输入的确定性派生/登记流程，不把它伪装成普通模型生成。拆分现有 `targeted_grpo`：共享核心负责从 seed 记录抽取 prompt/reference、构造 canonical GRPO 行、执行 schema/reference/reward 检查并计算质量；普通适配器不依赖诊断或 TargetedStagePlan，定向适配器以后可将已注册方案的场景/评测/版本证据作为可选上下文传入同一核心。核心参数显式包含 source dataset/record 身份、允许的 reference evidence 和批准 contract，不接受任意 reward 代码/插件路径。

首版 seed 仅开放具备可信构建证据的已登记来源：OpenR1 固定 build protocol、来源仓库与 revision 必须存在；每条 SFT train_seed 必须带 `correctness_math_verify=true`、source record ID、非空问题和答案。该正确性是上游标签证据，不宣称本系统重新独立验证；质量报告须准确标注来源。用户上传即使自带同名字段也不满足来源协议。服务端验证 manifest/canonical hash、stage/role、每条记录与批准 reward contract，并将父 dataset ID/hash、父 record ID、reference 来源、contract 版本/实现/协议摘要和构建协议写进登记 lineage。登记后以精确 dataset ID 返回 GRPO 训练页，不自动预检或启动。

若当前没有符合条件的 seed，页面说明允许的来源协议和缺失证据并提供可执行的数据获取/导入入口；不得生成可启动但缺少 reference evidence 的 GRPO 数据。Selector 与训练预检调用同一契约核验逻辑，避免“可选但不能预检”的状态差异。

理由：GRPO 的 reward/references 是训练语义和质量结论的依据；普通模式与定向模式共享构建/登记规则，但只有定向模式需要诊断和 holdout 上下文。分离核心和适配器可扩展定向链路而不复制实现，并避免把客户端声明当成信任证明。

### 11. 用户自带客观题作为隔离的评测套件

客观题文件导入走数据模块的源文件保留与 canonical 转换链路，但注册为 evaluation-only 数据身份，不复用训练数据角色。首版接受 JSONL 的单选、多选和判断题；适配器负责将来源字段转换为版本化 canonical 行，保留题型、题目、可变长度选项、标准答案和来源行号。导入预览逐行报告无效答案、选项引用错误和题型无法识别等拒绝原因；只登记可评分记录，并保留源文件摘要、转换版本、有效/拒绝计数与映射 lineage。样例中的空 `correctAnswer` 记录必须成为显式拒绝项；六选项且答案为 F 的有效题目必须可正常导入。

评测时以已登记数据集 ID+canonical hash+evaluator protocol version 固定一次 suite。单选/判断采用规范化精确匹配，多选采用答案集合完全一致；模型输出无法解析时记为 parse failure 并纳入报告分母说明，不让 LLM 裁判替代确定性答案。数据集/版本/样本覆盖/协议任一不同均不能计算默认横向 delta。评测预检沿 lineage 检查该 ID/hash 是否曾被目标模型训练消费；冲突即阻断。候选集需与阶段 evaluator 能力匹配；不支持客观题 evaluator 的 profile 不显示该数据集为可选项。

理由：复用导入登记能保留用户文件来源和数据血缘；独立 evaluation-only 身份与训练 lineage 排除检查可降低泄漏风险。确定性客观判分适用于选择题，避免模型自评污染准确率。替代方案是把文件直接作为临时评测输入或作为普通训练集登记后任选用途；前者无法稳定复现，后者难以可靠防止训练/评测串用。

### 12. 训练时验证支持独立数据集或显式随机切分

训练页将验证来源建模为互斥模式：使用兼容的已登记验证集、从当前训练数据随机切分，或不做训练时验证。默认保持“不使用验证”，不因 UI 自动切分而悄悄减少训练样本；选择切分模式时由 profile 限定比例范围/候选值及最小训练和验证样本数，服务端使用固定算法和用户可见随机种子生成确定性成员划分。Preference pair 保持原子；数据中存在稳定 problem/source group 身份时，同组样本作为整体分配，避免近重复跨边界。

训练集加载和训练期评测复用 ms-swift 能力：独立验证集通过 `--val_dataset` 传入，评测频率通过 profile 白名单内的 `--eval_strategy`/对应参数控制；未启用验证时不传验证集并保持现有无验证行为。ms-swift 的 `--split_dataset_ratio` 与 `--data_seed` 可用于普通行级拆分，但不能仅依赖训练进程内部的隐式拆分来满足本项目的成员级 lineage 和分组隔离要求。首选由项目按固定、版本化算法先生成确定性的 train/validation 成员清单及本次运行的派生数据文件，再分别通过 `--dataset` 和 `--val_dataset` 交给 ms-swift；独立登记验证集同样解析为已校验 canonical 数据后通过 `--val_dataset` 传入。不得同时设置非零 `--split_dataset_ratio` 与 `--val_dataset`。只有当某 profile 能证明原生拆分产出的确切成员身份可被捕获并记录时，才可直接使用 ms-swift 原生随机切分。

预检身份需包含父训练 dataset ID 与 canonical hash、切分策略/算法版本、比例、随机种子，以及训练/验证成员集合摘要和数量；启动时重新计算并核对同一配置。切分子集作为本次 Run 的派生输入记录，不覆盖或另行改写已登记父数据。独立验证数据则记录精确 dataset ID/hash/role。两种模式都只把验证部分交给 profile 明确支持的评估逻辑，不能用于梯度更新；验证指标及其适用范围写入训练 Run。最终测试仍是训练后独立评测，不可被当作训练/验证输入或 checkpoint 选择反馈。

理由：独立验证集在数据准备充分时更容易审计；随机切分让单数据集场景也能得到训练期诊断，但可能降低训练样本量，并不能消除同源/近重复泄漏，因此须显式选择、可复现并尽量按分组切分。自动随机切分会静默改变训练数据规模和历史默认行为；只支持独立验证集又会让单数据集用户完全没有训练期观测。三种显式模式兼顾可复现性、数据利用率与能力诚实呈现。

### 13. 已登记 Base 模型的独立网页评测

在 Runs/评测工作区增加独立模型评测入口，列出本地已登记、`kind=base` 且模型路径与 SHA-256 可验证的模型。用户选择固定数学基准 profile，或选择 `choice-accuracy-v1` 与 evaluation-only 客观题套件；后者仍要求数据集 manifest、canonical 文件摘要和角色证据有效。页面只提交稳定 model/dataset ID 与 profile ID，不接收自由路径、recipe 或 CLI 参数。

预检与一次性令牌绑定 model ID/hash、profile/协议版本、dataset ID/hash（如适用）、运行 ID 和输出目录。启动时重新核验所有身份与摘要，使用现有单 GPU supervisor、状态与日志恢复能力创建独立 evaluation Run。Base 模型没有训练 parent，因此 evaluation lineage 应标记 external/base root model，而不是虚构 training run；评测套件通过 evaluation-only 角色约束与训练输入隔离。数学评测复用已批准的 Qwen3-0.6B Base smoke evaluation 配置/统一 evaluator，客观题评测复用确定性 choice evaluator。

理由：用户需要在网页中对登记的基础模型建立可追溯基线，也可将同一客观题 suite 用于后训练模型对照。复用现有登记身份、profile、supervisor 和 evaluator 可保持受控执行及结果可比较性，同时让 base 根节点没有训练 lineage 这一事实保持明确。

### 14. 客观题逐题报告

评测详情在后续操作卡片后、指标与血缘网格前显示逐题报告区域，确保用户能发现入口。报告标题与答对/答错/解析失败/未完成数量概览始终可见；题目明细容器使用原生可访问的折叠控件且默认关闭，展开后才呈现“全部 / 答错 / 答对 / 解析失败”及题型筛选、命中数量和空结果提示。每题卡片优先显示题号/题型、题干、选项、标准答案、模型答案及醒目的结果状态。模型原始输出另行默认折叠，避免长解释淹没答案对照。

现有 evaluator 已将题目 ID、标准答案、解析后预测、原始输出和判定保存到 `choice_predictions.jsonl`；canonical 题目保存于登记 dataset。后端逐题报告读取必须由 evaluation lineage 的 dataset ID、数据摘要及 protocol 锁定本次 suite，验证 manifest/canonical 摘要与 prediction IDs 一致后，按题目 ID 联接题干、选项和答案。优先新增专用、只读且有响应大小上限的评测报告 API；不得把任意文件名或路径交由浏览器读取。若证据不匹配，返回明确的 unavailable/partial 状态，不降级为按行号联接。

理由：单看准确率无法发现具体错题、解析失败或选项格式问题。逐题答案对照能支持复核和下一轮数据诊断；稳定 ID 与摘要联接则保证展示题目确属本次评测版本。

### 15. Checkpoint policy 默认 final，可选验证集 best

阶段训练增加显式 `checkpoint_policy`，允许 `final` 与 `best`，默认 `final`。`final` 取训练完成后步数最大的有效已保存 checkpoint；`best` 取训练期间按阶段 profile 声明的验证指标及优化方向排名最优的已保存 checkpoint。`best` 必须要求训练时验证已启用、验证数据通过现有隔离校验且指标确实产出；否则预检阻断，不做静默回退。最终测试集不进入选择过程。

保存/验证步频由 adapter 校验兼容，确保候选 checkpoint 对应可用的验证结果。latest checkpoint 继续供断点续训；final 与 best 的权重身份分别保留，若相同可复用同一目录。Run 与 lineage 记录 policy、指标及方向、验证集身份、final/best/latest 的 step/path/hash 和阶段模型实际采用者。阶段模型及下游 parent handoff 指向所选 policy 的模型；训练恢复仍明确使用 latest。

理由：主流训练框架把恢复训练的 latest 与按验证指标选出的 best 作为不同职责，并允许分别保留。保留 `final` 默认维持当前固定预算对比的稳定口径；`best` 为模型选型提供能力，但只能依赖隔离验证证据，并让选择依据可审计。替代方案是总是加载 best 或只保留 best；前者会悄然改变当前实验结果口径，后者会丢失精确续训位置和 final/best 差异证据。

## Risks / Trade-offs

- [不同 CLI/框架日志与退出语义不一致] → 各 stage adapter 仅负责受控命令、输出/进度解析；共用 supervisor 状态机，未知进度保持 unknown，并给每种 adapter 单测。
- [用户选择错误 parent checkpoint 或数据角色] → 服务端从 registry/lineage 解析 parent 与 manifest，浏览器只传 ID；预检核对阶段、role、hash 和契约。
- [质量或许可证据处于 exploratory/unverified] → 预检显示 evidence 与 blocker/acknowledgement；显式确认不升级质量或许可证结论。
- [单 GPU 显存不足或评测耗时较长] → 逐阶段给出 best-effort GPU/长度/资源提示和固定 profile；不并发、不自动重试，真实 OOM 留存为终态。
- [评测 profile 使用了训练数据] → 固定评测注册表只提供 benchmark/holdout role，并对 dataset ID+digest 做排除校验。
- [大量阶段首版导致测试范围扩大] → 阶段能力声明表驱动共同行为测试，另对 Preference 和 GRPO 的 pair/reward 契约保留专属测试。
- [导航迁移丢失历史链接或活动生成任务] → 模块解析与旧 URL 映射先行，先对旧链接、浏览器前进/后退、模式切换和服务端任务恢复做回归，再迁移页面归属。
- [GRPO 数据被错误标记为可信或使用过期 reward] → 只从已登记 train seed 派生，服务端重算批准 contract 身份并逐条校验来源/reference；缺证据时禁止登记或训练，不信任客户端声明。
- [用户评测文件字段不统一或答案质量不完整] → 通过有版本的格式适配器和逐行预览报告，拒绝歧义记录并保留有效/拒绝分母；缺少可评分记录时不登记。
- [自定义评测集泄漏进训练或造成不公平比较] → 使用独立评测数据角色并按 ID/hash 检查训练 lineage；报告绑定 suite 与协议版本，不同身份/覆盖/协议默认不可比较。
- [训练集随机切分导致样本减少或同源题目泄漏] → 默认关闭切分；预检展示双方样本量，使用固定种子和版本化算法，尽可能按 problem/source group 切分，并继续将 final_holdout 隔离于训练期流程之外。

## Migration Plan

1. 先落地四模块导航、旧链接解析和中文信息架构指南；迁移现有数据、生成、策略与比较视图，保持现有真实能力可操作。
2. 复用生成/质检/登记视图，接通分析到数据的定向上下文与活动任务恢复。
3. 定义阶段/profile 能力目录与公共 DTO；保留旧 `/api/launcher/*` SFT 端点和固定 SFT recipe 行为，确保旧客户端兼容。
4. 抽取公共 supervisor 生命周期，将 SFT 接入新阶段 API 并保持旧路径适配；新增 CPT、DPO、SimPO、GRPO adapter/profile，每类都复用登记数据解析与 CLI lineage。
5. 注册各阶段固定评测 profile，复用现有 CLI evaluator；补充 Preference/GRPO 阶段指标来源和 evaluation lineage 显示。
6. 将训练启动、Run 详情和结果分析连通；通过 API、前端和 fake-CLI 集成测试后才开放每个阶段。
7. 增加训练时验证来源选择，分别接通独立验证集和可复现的训练集切分模式；把切分身份加入预检/Run/lineage，并验证最终测试集仍只用于训练后的独立评测。

回滚时通过配置隐藏新增阶段 profile，历史 runs/artifacts 保持只读可见；旧 SFT 路径和 CLI 不删除。不得迁移或覆盖历史训练输出。

## Open Questions

无。第一版确定覆盖全部五阶段；无法由本机条件或真实评测证据支持的 profile 必须显示 blocker/未测状态，而不是缩减为仅 SFT 或伪造可用性。
