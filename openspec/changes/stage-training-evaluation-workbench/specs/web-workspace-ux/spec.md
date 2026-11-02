# Spec Delta

## MODIFIED Requirements

### Requirement: Data navigation returns to the data workspace
工作区 SHALL 从任意模块提供“数据”主导航，返回统一数据工作台；数据集目录为默认视图，处理、质量及登记操作在该模块内可发现。

#### Scenario: Return from targeted data
- **WHEN** 用户在薄弱场景或定向补数上下文中选择“数据”主导航
- **THEN** 打开统一数据工作台并正确标记当前主导航，原有诊断与活动任务不会因导航被删除

#### Scenario: Return from policy comparison
- **WHEN** 用户在策略效果分析中选择“数据”主导航
- **THEN** 打开统一数据工作台，并可从处理与质量视图定位相关规则或数据版本

## ADDED Requirements

### Requirement: Primary navigation follows user tasks
工作区 MUST 以“数据”“训练与评测”“实验 Runs”“分析与改进”四个主入口组织现有功能。数据获取/处理/质检归数据，阶段配置与启动归训练，运行详情归 Runs，比较与诊断归分析；尚未实现的占位功能不能混入可操作主导航。

#### Scenario: Find dataset management and processing
- **WHEN** 用户准备导入、生成、查看样本、检查处理质量或组合数据集
- **THEN** 用户从同一“数据”入口进入对应二级视图，不需要在两个同义数据主入口之间猜测

#### Scenario: Edit a data policy and inspect its effect
- **WHEN** 用户编辑规则或预览筛选效果，随后查看真实实验效果报告
- **THEN** 编辑/数据预览在数据模块，效果报告在分析模块，并通过同一策略/实验身份互相定位

### Requirement: Ordinary and targeted generation share the data workflow
普通生成与定向生成 MUST 在数据模块复用适用的生成、质检、审核和登记界面及协议；分析模块负责诊断和目标上下文，不提供另一套独立登记流程。普通生成不得要求用户手工登记内部方案身份。

#### Scenario: Generate data without a diagnosis
- **WHEN** 用户从数据获取选择模型生成
- **THEN** 可按当前支持的方式和阶段直接准备生成任务，系统管理方案与执行身份，并展示可用模型配置

#### Scenario: Generate for a diagnosed weakness
- **WHEN** 用户从薄弱场景选择补充数据
- **THEN** 同一数据生成流程预填目标并保留来源评测/模型引用，适用的 train seed 与最终 holdout 隔离，登记结果可追溯该上下文

### Requirement: Cross-module actions preserve object context
跨模块操作 MUST 传递明确的 dataset/model/run/eval 或目标方案身份、适用阶段及返回位置，并使用同一对象详情；用户无需手动复制内部 ID、hash 或路径。

#### Scenario: Train a registered dataset
- **WHEN** 用户从已登记数据集详情选择“用此数据训练”
- **THEN** 训练模块预选准确 dataset ID 与兼容阶段，并保留返回该数据集的路径；导航本身不启动训练

#### Scenario: Inspect one run from multiple modules
- **WHEN** 用户从训练页、关联数据或分析结果打开同一 run
- **THEN** 进入相同 Run 详情及真实状态，评测动作绑定该 Run 的核验模型，不维护第二份运行结果

### Requirement: Draft changes do not discard running tasks
页面 MUST 区分可编辑草稿、可失效的预检与已持久化执行；切换模块、阶段、生成方式或刷新后能够恢复活动 execution/run 的服务端状态，且不能自动取消或重启任务。

#### Scenario: Change generation mode while a task is active
- **WHEN** 用户在生成任务未完成时切换 seeded/prompt-only 方式或进入另一模块
- **THEN** 新草稿可独立编辑，原任务仍可从活动任务入口查看，执行身份与已生成内容保持不变

#### Scenario: Restore after reload
- **WHEN** 用户重新加载一个有活动训练、生成或评测任务的工作区
- **THEN** 页面根据已保存的任务引用读取真实状态，不要求重新提交启动操作

### Requirement: Navigation scopes URL state and supports existing links
工作区 MUST 保留旧页面深链接的含义、对象定位及浏览器前进/后退行为；各模块只管理自己的筛选/子视图参数，跨模块不得静默覆盖目标对象或携带无关数据页筛选。

#### Scenario: Open a legacy processing or comparison link
- **WHEN** 用户打开旧 data/build/stage/tab 或 comparison/matrix 链接
- **THEN** 分别定位到数据处理与质量或分析比较的对应内容，并正确恢复对象和主导航

#### Scenario: Navigate between data and runs
- **WHEN** 用户从数据打开 run 后使用浏览器后退/前进
- **THEN** 数据筛选与 run 身份分别恢复，数据页的 stage/build 参数不改写 Runs 的筛选或运行定位

### Requirement: Module failures and technical details remain local and understandable
页面 MUST 独立呈现所属模块的加载/空/错误/未支持状态与恢复动作；常规操作使用对象名称、含义、单位与分母，高级 ID/摘要/原始配置可按需展开。一个非依赖接口失败不能隐藏其他模块可用内容。

#### Scenario: Data catalog fails while a run is observable
- **WHEN** 数据目录请求失败而运行状态接口正常
- **THEN** 数据模块提供重试，活动 Run 的状态/日志仍可查看，页面不将整个工作区误报为离线

#### Scenario: Read a metric and trace its evidence
- **WHEN** 用户查看质量或评测数字
- **THEN** 可以看到指标含义、单位、实际分母和状态，并按需展开其 dataset/model/run/协议证据

### Requirement: Stage workbench presents a clear train-to-evaluate flow
阶段工作台 MUST 让用户识别当前阶段、数据选择、预检、训练、评测和结果状态，并按状态突出唯一的下一项主操作。

#### Scenario: Prepare a stage run
- **WHEN** 用户打开某阶段工作区且没有活动 run
- **THEN** 页面先呈现可选的兼容数据集、必要说明和预检入口，不要求用户填写内部路径或手工构造命令

#### Scenario: Continue from a completed training run
- **WHEN** 训练模型核验成功且尚无评测结果
- **THEN** 页面突出针对该模型的评测预检/确认操作，并保留训练摘要和 lineage 上下文

#### Scenario: Show a blocked or failed step
- **WHEN** 数据、父模型、资源或评测条件阻止工作流继续
- **THEN** 页面在相关阶段解释阻断原因和可执行的恢复动作，不将 blocker 隐藏在原始日志中

#### Scenario: Recover from missing GRPO reward/reference evidence
- **WHEN** 用户的 GRPO 训练数据缺少当前批准的 reward contract 或可追溯的可信参考答案
- **THEN** 数据选择器禁用该数据并说明缺失证据，提供前往 GRPO 数据准备的入口；准备流程只允许选择已登记可信 seed 和批准契约，登记成功后带精确 dataset ID 返回训练页，不自动预检或启动

### Requirement: Stage workbench remains usable on narrow screens and keyboard
阶段选择、数据集选择、训练/评测确认、状态、指标和 lineage 导航 MUST 在 390 CSS 像素及以上窄视口与键盘操作中保持可访问，不产生页面级横向溢出。

#### Scenario: Use the stage workflow on a narrow viewport
- **WHEN** 用户在 390 CSS 像素宽度查看阶段训练或评测
- **THEN** 页面采用可读的单列布局，关键操作可见可触达，日志可在自身区域滚动

#### Scenario: Navigate the workflow with a keyboard
- **WHEN** 用户仅使用键盘选择阶段、数据集和确认操作
- **THEN** 控件有程序化名称、明确焦点状态和符合流程顺序的键盘导航
