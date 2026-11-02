# Spec Delta

## ADDED Requirements

### Requirement: Evaluation profiles declare stage coverage
每个可从工作台启动的阶段评测 MUST 使用有版本的 profile，声明通用任务基准、阶段专属指标、评测数据角色、样本分母和不支持项。

#### Scenario: Compare stage-appropriate evaluation coverage
- **WHEN** 用户查看 CPT、SFT、DPO、SimPO 或 GRPO 的评测预检/结果
- **THEN** 页面显示该阶段 profile 的指标清单、数据身份/角色、协议版本和覆盖限制，不把不同协议的分数默认视为可比

#### Scenario: Stage-specific metric is unavailable
- **WHEN** 一项阶段专属指标缺少评测数据、评测器或真实执行证据
- **THEN** 该指标显示为 not_measured、unsupported 或 not_applicable 并说明原因，不填零或从训练指标推断

### Requirement: Stage comparisons use matching shared benchmarks
跨阶段比较 MUST 区分通用任务基准和阶段专属指标；仅当共享数据版本、样本 ID、提示/解码、解析器和评测覆盖一致时，才能并列比较通用指标。

#### Scenario: Compare SFT, DPO and GRPO on the shared protocol
- **WHEN** 多阶段模型均完成相同版本共享 benchmark
- **THEN** 比较页并列展示可比指标及各自训练/模型 lineage，同时单独展示不能跨阶段对比的专属指标

#### Scenario: Shared benchmark coverage differs
- **WHEN** 不同阶段的评测样本、数据版本或协议覆盖不一致
- **THEN** 系统标明不可直接比较及具体差异，不计算误导性的 delta

### Requirement: Import user-authored objective evaluation datasets
系统 MUST 支持将受支持的 JSONL 客观题源文件转换为已登记的评测数据集，并保留源文件、字段映射、转换版本及逐条转换结果的 lineage。

#### Scenario: Normalize supported question types
- **WHEN** 用户导入包含单选、多选或判断题的 JSONL 文件
- **THEN** 系统将有效记录转换为统一题目、选项、标准答案和题型字段；选项数量可变，不限制为 A–E

#### Scenario: Report malformed source rows
- **WHEN** 源记录缺少答案、答案不在选项中或无法识别题型
- **THEN** 预览报告按行显示拒绝原因与有效/拒绝计数，且无有效记录时禁止登记

### Requirement: Keep registered user evaluation data isolated from training
用户导入的评测数据 MUST 登记为评测专用身份，不能被训练数据选择器作为训练输入；评测前 MUST 检查数据身份及内容摘要未用于对应模型的训练 lineage。

#### Scenario: Select an evaluation-only dataset
- **WHEN** 用户在兼容的阶段评测中选择已登记的用户评测数据集
- **THEN** 系统展示数据集名称、版本、样本覆盖和协议，并仅将其作为 evaluation input

#### Scenario: Reject training/evaluation leakage
- **WHEN** 评测数据集 ID 或内容摘要与该模型训练 lineage 中的数据相同
- **THEN** 预检阻断评测并指出冲突的数据身份，不生成评测 Run

### Requirement: Score objective question types with an explicit protocol
系统 MUST 对已登记客观题使用有版本的确定性评测协议，分别计算单选、多选和判断题准确率，并记录有效分母、解析失败数及协议身份。

#### Scenario: Score variable-option single choice and true/false
- **WHEN** 题目为单选或判断题且模型输出可解析为一个选项或标准布尔答案
- **THEN** evaluator 按规范化答案精确判分，并记录该题型的正确数与有效分母

#### Scenario: Score multiple choice as an exact set
- **WHEN** 题目为多选题
- **THEN** 只有规范化后的预测选项集合与标准集合完全一致才记为正确，不以部分命中计为正确

#### Scenario: Preserve incomparable custom suite results
- **WHEN** 用户评测集的数据版本、样本覆盖或协议与另一结果不一致
- **THEN** 系统分开展示结果并标记不可直接比较，不计算跨套件分数差

### Requirement: Evaluate registered external base models from the web
网页 MUST 允许用户从已登记且摘要可验证的外部 base 模型中选择模型，使用受控且有版本的数学基准评测 profile 或已登记的 evaluation-only 客观题套件创建独立评测 Run。评测身份 MUST 绑定精确 model ID/hash、profile/协议、可选评测集 ID/hash 和输出位置；浏览器不得提交任意模型路径或命令参数。Base 模型没有训练 Run 时，报告应明确其外部根模型身份，不得伪造训练来源。

#### Scenario: Preflight and start a base-model benchmark evaluation
- **WHEN** 用户选择摘要可验证的已登记 base 模型和受支持的数学基准 profile，并显式确认
- **THEN** 网页预检展示模型、协议、指标覆盖和资源限制，启动独立评测 Run，并将评测 lineage 指向精确 base model ID/hash 与协议

#### Scenario: Evaluate a base model on an objective-question suite
- **WHEN** 用户选择已登记 base 模型及 `stage=eval`、`role=evaluation_only` 的客观题套件
- **THEN** 网页按 `choice-accuracy-v1` 启动独立评测，结果按题型给出正确数、分母和解析失败，并绑定 suite ID/hash；训练选择器仍不可消费该 suite

#### Scenario: Reject unavailable or changed base model identity
- **WHEN** base 模型未登记、路径不可用、内容摘要与登记不符，或预检后模型/套件身份发生变化
- **THEN** 预检或启动明确阻断且不创建评测 Run；启动使用一次性确认中绑定的模型、profile、套件及输出身份

### Requirement: Inspect per-question objective evaluation results
客观题评测详情 MUST 在总分和分题型汇总之外提供逐题报告。报告 MUST 按评测 Run 的 dataset ID、canonical SHA-256 与题目 ID 关联题干、选项和标准答案，并展示模型答案、判定结果及解析状态；不得按名称或仅按行号关联题目。

逐题明细 MUST 默认折叠；报告标题和答对、答错、解析失败、未完成的数量概览 MUST 保持可见。用户显式展开后才能看到筛选控件和逐题内容，原始模型输出仍默认折叠。
客观题逐题报告摘要 MUST 位于指标与血缘详情之前，避免入口落在首屏下方而不易发现。

#### Scenario: Filter correct and incorrect questions
- **WHEN** 用户查看客观题评测 Run 的逐题报告并选择全部、答对、答错或解析失败筛选
- **THEN** 报告仅展示符合筛选条件的问题，并对每题显示题型、题干/选项、标准答案、模型答案和明确的正确/错误状态

#### Scenario: Inspect an unparseable model response
- **WHEN** 模型输出无法解析为协议答案
- **THEN** 报告将其单独标记为解析失败/未作答，保留原始输出供按需展开查看，且不将其伪装为普通错误答案或隐藏于分数中

#### Scenario: Keep question report bound to evaluation evidence
- **WHEN** 逐题预测记录缺失、题目 ID 不唯一，或登记数据集摘要与 evaluation lineage 不一致
- **THEN** 页面说明逐题报告不可用或不完整，不通过名称、当前选择或行序推测性拼接题目

#### Scenario: Keep detailed question list collapsed by default
- **WHEN** 用户打开客观题评测 Run 详情
- **THEN** 报告标题及答题状态数量概览保持可见，筛选控件和逐题明细默认折叠，用户可通过键盘或鼠标展开查看
