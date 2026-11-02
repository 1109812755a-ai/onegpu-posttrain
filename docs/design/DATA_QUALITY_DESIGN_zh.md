# 数据质量设计

**版本：** Data Quality v1（设计提案）  
**状态：** `stage-data-quality-v1` 已实现阶段指标目录、证据覆盖与离线复评；质量阈值及语义审批仍未冻结  
**范围：** 各训练阶段的数据质量证据与训练前适用性检查

English: [Data Quality Design](DATA_QUALITY_DESIGN.md)。

数据处理步骤、artifact schema、受控模型效果实验和前端流程的详细设计见[数据处理、质量变化与模型效果设计](DATA_PROCESSING_EFFECT_DESIGN_zh.md)。

## 目的与产品定位

数据质量是与血缘、训练、评测、受控实验并列的一条主线。只有数量统计，不能证明实验输入有效、有代表性、无污染，或适合预定训练目标。

```text
数据
  ├── 血缘：数据从哪里来、如何变化
  ├── 质量：记录是否符合阶段特定要求
  ├── 分布：数据包含哪些样本、长度如何
  └── 泄漏 / 污染：与哪些数据可能重叠
       ↓ 校验后的输入
训练 → 评测 → 受控实验
```

本设计是对冻结的 v0.1.1 数据契约的增量补充，不会暗中改变既有准入规则，也不会重写历史结果。任何冻结门槛的变化都必须经过版本化设计修订并重建 artifact。

## 原则

1. **质量按阶段定义。** CPT、SFT、偏好、GRPO 和评测数据不能套用同一个通用分数。
2. **证据与判断分开。** 保存原始计数/比例和评估协议；只有存在版本化规则时才显示 gate 结果。
3. **不使用不透明综合分。** 不把正确性、覆盖度、泄漏和可用性揉成一个数字。
4. **显式表达不确定性。** 每项指标标记为 `measured`、`estimated`、`not_measured` 或 `not_applicable`，并保留分子、分母、方法/版本、来源和局限。
5. **启发式不冒充语义真值。** 例如模型判断的领域相关性是估计值；验证器判断正确性还取决于验证器覆盖率和可靠性。
6. **v0.1 阈值保持保守。** 初期先产出报告和警告；等 Mini 的真实分布出来后，再校准非普适阈值。数据切分隔离等安全不变量可独立执行。
7. **检查过程可复现、可检查。** 质量协议/配置、检查器和验证器版本属于 artifact 血缘的一部分。

### 当前实现边界（v1）

新的 `QualityReport`（schema `1.1`）记录 `quality_protocol`、checker 名称/版本、逐项指标状态及 `evidence_coverage`。历史 `1.0` 报告继续可读，但覆盖状态解释为 `legacy_unknown`。`complete` 只表示目录中的适用指标都已显式测量或标为不可测，不代表质量通过；没有独立验证器、解析结果或评分证据时必须保留 `not_measured`。Mini 复评通过 `assess-mini-data-quality` 输出独立、摘要绑定的 bundle，不改写数据、manifest 或历史报告。操作命令与限制见[Mini 就绪度 CLI 指南](../implementation/data/MINI_READINESS_CLI_zh.md)。

## 三层质量模型

| 层级 | 核心问题 | 常见证据 |
|---|---|---|
| 结构质量 | 记录是否可解析，是否符合声明的 schema/长度约束？ | schema 有效率、空值/空字段、解析率、精确重复、长度分布 |
| 语义质量 | 内容和标签是否符合任务语义？ | 领域相关性估计、答案/验证器正确性、推理完整性、偏好对顺序 |
| 实验适用性 | 数据是否适合目标实验，能否支持有效解释？ | 切分/基准重叠、来源/难度覆盖、可验证性，以及明确标记为依赖模型/recipe 的校准信号 |

三层分别报告。数据集可以结构检查通过，但语义证据较弱，或不适合当前实验覆盖要求。

## 分阶段评估

| 阶段 | 核心问题 | 建议指标（有证据时才报告） |
|---|---|---|
| CPT | 语料是否是可用文本、与领域相关、有足够多样性，并与评测材料隔离？ | 文档数和 token 数；schema/编码有效率；注明方法的精确/近似重复率；语言/来源/任务分布；按文档数和 token 加权的长度分布；带方法和审查覆盖说明的领域相关性估计；污染/重叠结果及检测局限 |
| SFT | prompt 和 target 是否完整、有效、在声明任务上足够正确，并能适配 recipe 预算？ | schema 有效率；空/格式错误比例；**验证器覆盖范围内**的验证器正确率；最终答案可解析率；格式有效率；依 tokenizer/recipe 的 prompt/response 长度和截断风险；问题重复率；来源/难度/任务分布；推理完整度仅作为明确标注的估计/抽查信号 |
| DPO / SimPO | 每个 pair 是否可追溯、无歧义且符合声明的偏好标签语义，切分是否隔离？ | 同 prompt 配对；格式错误/平局/歧义比例；标签一致性；**仅当偏好标签语义就是正确性排序时**报告 chosen/rejected 正确性及验证器覆盖率；长度和 chosen-vs-rejected 长度偏差；重复/同题一致性；train/eval 重叠。Pair yield 是**构建效率指标**，不是质量分；模型 margin/hard-negative 分数是依赖模型的诊断 |
| GRPO | prompt/reference 在结构上是否可用、验证器是否支持？另外，指定模型/recipe 的 rollout 信号是否可用？ | **静态数据：** reference 可解析性、验证器支持/覆盖、prompt 长度、任务/难度分布、重复率、预期答案类型。**校准：** 指定模型、reward/verifier、seed 和生成 recipe 下的基线通过率、非零 reward 率/组内方差、completion 长度和截断比例 |
| Evaluation | 数据集是否覆盖目标能力、评分稳定，且样本相关并足够独立？ | 来源/修订版和协议；题型/任务/难度覆盖及样本数；答案/解析器/验证器覆盖；与训练数据及其祖先的精确/近似重复和 problem-ID 重叠；污染检查方法及局限；分母/排除数；解析/评分失败率。模型准确率是评测结果，不是数据集质量指标 |

这张表是指标菜单，不要求虚构或计算每个指标。报告必须解释不可用指标，不能用 0 代替缺失。

### 代表性与指标类别

代表性始终相对于声明用途：数据集的任务、来源、难度、语言和长度覆盖越贴近目标总体，越能代表该用途。应声明目标总体并报告实际覆盖；来源比例平衡或单一相关性分数都不能单独证明代表性。Data-centric 研究同样强调底层问题的广度、难度和忠实度，而非只有数据清洁度（[DataPerf](https://proceedings.neurips.cc/paper_files/paper/2023/hash/112db88215e25b3ae2750e9eefcded94-Abstract-Datasets_and_Benchmarks.html)）。

每项指标必须标为以下一类：

- **数据集质量：** artifact 的有效性、正确性证据、重复、覆盖、泄漏和来源。
- **构建诊断：** 请求/接收数量、拒绝原因和 pair yield。它们解释 builder 效率与目标可达性，不能证明被接收样本好或坏。
- **模型/recipe 适用性：** 校准 rollout 观察到的通过率、reward 信号和生成截断；这些指标依赖模型、评估/reward 和生成参数，不是数据集固有属性。

MVP 优先覆盖结构有效性、ID/split、来源/版本、验证器支持覆盖、答案可解析性和重叠检查。长度/分布、重复、拒绝/yield 摘要作为诊断报告。语义估计、近似去重、人工抽查、模型 margin/hard-negative 分析，在方法和成本证明合理前均为可选项。未经验证的估算指标不能成为阻断 gate。

每个比例保留分子、分母、单位和排除项；抽样或估算指标尽可能记录抽样方法、样本量、不确定性/审查覆盖。将**检查结果**（`PASS`/`FAIL`/`NOT_EVALUATED`）和**后果**（`error`/阻断或 `warning`/继续）分开。建议性检查失败可以告警，但不阻断 run。

## Preference 产出率与 GRPO 校准

Preference yield 是产能规划/构建效率信号，不是质量分数。必须定义扫描单位和分母（例如每扫描一个唯一问题产生的有效 pairs），保留拒绝计数，并将目标达成情况与 pair 有效性、标签正确性分开展示。这样 Mini 可以估计需要扫描多少源数据，而不必为了凑数放宽正确性标准。“chosen 正确 / rejected 错误”只适用于声明了“按正确性排序”的偏好数据（如本项目可验证的数学 pairs），不能作为 DPO/SimPO 的通用定义。

仅靠静态 GRPO prompt 检查无法完全预测生成行为。可以另做**训练前校准报告**：使用目标 parent model 做有预算限制的基线 rollout，并记录 prompt 子集/seed、生成参数、验证器/reward 版本、completion 长度分位数、截断率、非零 reward 率和组内 reward 方差。这些是依赖模型与 recipe 的适用性测量，不是数据集永恒不变的属性。v0.1 先告警；硬门槛及其阈值需要校准协议和 Mini 证据支持后再评审。

## 数据处理轨迹与清洗前后质量

记录从原始来源到阶段数据集的真实处理顺序：按需包括规范化、去重、污染检查、按 problem 切分、答案验证、长度门槛、pair 构建和 token 预算选择。在现有版本化 data-build report 中增加有序的 `processing_steps`；最终数据集的 `quality_report.json` 继续由 manifest 引用。只有中间数据集被实际落盘并复用时，才需要为它分配独立 ID/manifest；内存中的每一步不必都生成完整数据副本。

[已实现的 Smoke 数据构建顺序](../implementation/data/SMOKE_DATA_BUILD.md)是起点：OpenR1 先规范化/精确去重、污染检查、数学可验证性检查，再按 problem 切分，最后应用阶段专属长度/转换规则；FineMath 在二次污染检查与分块之前，先按文档切分。设计示意图不能暗中改变这些 gate 顺序。若顺序或准入语义变化，必须对处理协议升级版本并重建受影响的 artifact。

每一步记录稳定的 step ID/顺序、输入输出 artifact 或逻辑数据池引用、转换/检查器名称与版本、配置摘要、**带单位**的输入输出计数、首个拒绝原因计数，以及可用的前后质量指标引用。跳过或未知的步骤及原因也应显式记录。若能追踪记录级祖先，接收/拒绝的样本 ID 可放在可选、遵守许可证要求的 sidecar 中；汇总报告不嵌入源文本。

```json
{
  "schema_version": "1.0",
  "build_id": "build-openr1-mini-v1",
  "source_dataset_id": "ds-openr1-source-v1",
  "processing_steps": [
    {
      "step_id": "verify-math-v1",
      "order": 4,
      "input_ref": "pool-after-decontamination",
      "output_ref": "pool-verified",
      "transform": {"name": "verify_math", "version": "1", "config_sha256": "<digest-or-null>"},
      "input_count": {"value": 1000, "unit": "examples"},
      "output_count": {"value": 820, "unit": "examples"},
      "first_rejection_reasons": {"unsupported_answer": 80, "incorrect_answer": 100},
      "quality_before_ref": "quality/decontaminated.json",
      "quality_after_ref": "quality/verified.json"
    }
  ]
}
```

示例数字仅用于说明。只有输入输出单位相同、且输出是输入子集时，才能计算保留率；配对、分块和 token 化可能改变计数单位。拒绝计数对应已声明规则顺序下的**首个失败原因**，不能解释成每条规则的独立贡献。清洗前后比较质量指标时，定义、检查器/版本、分母和抽样协议必须兼容。质量改善同时要展示覆盖损失，包括来源/任务/难度组成变化，以及可验证样本或少见样本的减少。原始来源可在有预算限制的样本上评估质量，但必须展示抽样方法和样本量；未测量项保持 `not_measured`。

### 将处理步骤连接到模型效果

清洗前后质量变化说明数据发生了什么，不能单独证明模型因此变好。验证训练效果时，应注册一个数据选择实验，记录源数据池、基线及清洗后数据集 ID/digest、唯一变化的处理规则或策略、parent 模型 hash、recipe、seed、预算、评测数据/协议，以及下游模型和评测 ID。基线仍需满足最基本的 schema、许可证和污染防护要求；“原始”不意味着训练格式错误或泄漏的数据。

明确区分两类实验问题：

| 比较类型 | 回答什么问题 | 结果如何解释 |
|---|---|---|
| **Pipeline Effect（流水线效果）** | 实际应用整套清洗策略后发生了什么？ | 报告真实的前后数量、token 体量、分布、质量证据、资源成本和模型评测。数据量与组成可能同时变化，模型 delta 只能解释为整条流水线的结果。 |
| **Controlled Quality Ablation（受控质量消融）** | 在训练预算匹配时，改变一个声明的筛选/处理策略会怎样？ | 固定可选源数据池和独立评测集，尽量匹配样本数或 token 数、optimizer steps、parent、recipe 和 seed；报告仍存在的分布差异及跨 seed 不确定性。这样才能更具体地解释该策略的效果。 |

两类比较都应报告数据质量/覆盖差异、模型指标 delta 和算力成本。仅凭训练 loss 或 reward 上升，不能证明任务能力提升。单次运行只能作描述性观察；更强的效果结论需要重复 seed，或明确标出不确定性/复现实验不足。验证器筛选后集合的条件正确率，也不能证明原集合所有被删标签都错误，或质量分数必然能预测 benchmark 提升。

候选实验包括 SFT 去重/验证器筛选、在相同 problem IDs/M2/算法/pair 预算下比较 preference 负例选择（随机错误、长度匹配、hard negative），以及基于有限 rollout 校准的 GRPO prompt 筛选。测试 GRPO prompt 筛选时应固定生成 recipe；改变 `max_completion_length` 是另一项 recipe 实验。按阶段同时比较任务指标与格式/解析表现、偏好 accuracy/margin 或 GRPO reward 方差/截断，并展示评测分母和协议。

Mini 基线和必需 OVAT 契约稳定后，可以做一个小规模、等量的可选实验；它不增加 v0.1 发布门槛，也不是第五个必需 OVAT。可复用的数据质量实验及比较 UI 放在后续数据中心/Experiment Studio 阶段。v0.1.1 只读 UI 可以展示已有处理步骤及其关联模型比较；没有受控证据时明确显示 `not evaluated`。

## Quality report 契约

每个构建的数据集都应在 manifest 旁关联 `quality_report.json`（或等价的版本化报告）。报告描述某个数据 artifact 在指定协议下的评估；数据或检查器协议改变时，需要生成新的报告身份/版本。

```json
{
  "schema_version": "1.0",
  "dataset_id": "ds-openr1-pref-mini-v1",
  "quality_report_id": "dq-openr1-pref-mini-v1",
  "quality_protocol": "preference-quality-v1",
  "assessment_scope": "pre_training_data",
  "created_at": "<RFC3339 timestamp>",
  "checker": {"name": "preference_validator", "version": "1"},
  "metrics": {
    "scanned_problems": {"value": 5000, "unit": "unique_problems", "status": "measured", "class": "build_diagnostic"},
    "valid_pairs": {"value": 2032, "unit": "preference_pairs", "status": "measured", "class": "build_diagnostic"},
    "pair_yield": {"value": 0.4064, "numerator": 2032, "denominator": 5000, "unit": "ratio", "status": "measured", "class": "build_diagnostic"},
    "chosen_correct_rate": {"value": 1.0, "numerator": 2032, "denominator": 2032, "status": "measured", "method": "math_verify@<version>", "class": "dataset_quality"},
    "train_eval_problem_overlap": {"value": 0, "unit": "unique_problems", "status": "measured", "class": "dataset_quality"}
  },
  "gate": {"summary_status": "WARNING", "policy_id": "preference-gates-v1", "checks": [{"check_id": "chosen_correctness", "result": "PASS", "severity": "error"}, {"check_id": "pair_yield_target", "result": "FAIL", "severity": "warning", "reason": "Thresholds are provisional"}]},
  "limitations": ["Verifier coverage does not establish correctness for unsupported answer forms"]
}
```

报告要求：

- 关联 dataset ID/content digest 和 manifest；manifest 在可用时反向引用该报告。
- 保留计数、单位、分子/分母、方法、检查器版本、范围和局限。
- 每项指标标注类别（`dataset_quality`、`build_diagnostic`、`model_recipe_suitability`），避免把构建产出率误当质量。
- 检查结果（`PASS`/`FAIL`/`NOT_EVALUATED`）与后果（`error`/阻断或 `warning`/继续）分开；汇总状态可为 `PASS`、`WARNING`、`FAIL`、`NOT_EVALUATED`。没有经过评审的策略时，不能暗示已通过。
- 区分构建时质量与依赖模型/recipe 的校准报告。
- 样本级诊断可选，并遵循隐私/许可证要求；不要把受限源文本复制到公开报告。

## 开源框架策略

项目自身掌握 canonical quality protocol、阶段语义、报告 schema 和可复现性保证。通用框架可以通过可选 adapter 接入，但核心项目必须不依赖任何外部质量框架也能运行。JSONL 校验优先复用项目的 canonical record schema；只有明确的 DataFrame 工作流带来实际收益时，才增加表格校验依赖。

| 候选工具 | 潜在用途 | 纳入策略 |
|---|---|---|
| [DataTrove](https://github.com/huggingface/datatrove) | CPT 语料过滤、去重和规模化文本处理 | 针对当前 CPT 输入做有限 PoC。保持可选；只有比较质量证据、内存/耗时、安装成本以及样本 ID/血缘保留情况后，才考虑 adapter。 |
| [Data-Juicer](https://github.com/datajuicer/data-juicer) | 面向 SFT 的过滤、诊断和评分算子 | 仅针对内置检查缺失的信号做定向 PoC。模型/LLM 评分标记为 `estimated` 诊断，不得当作正确性标签或自动拒绝 gate。 |
| [Pandera](https://pandera.readthedocs.io/en/stable/dataframe_schemas.html) | 表格/DataFrame 表示上的 schema 与约束检查 | 不作为 canonical JSONL 记录的核心依赖。若 DataFrame 分析成为长期维护的项目接口，再重新评估。 |
| [Argilla](https://docs.argilla.io/latest/) | 人工抽样、审阅和标注工作流 | 对歧义或主观样本可选使用；即使没有托管服务，核心报告仍必须可用。 |
| [Cleanlab](https://docs.cleanlab.ai/stable/tutorials/datalab/text.html) | 模型辅助发现标签问题、异常点和近重复 | 可选审计 adapter；结果依赖模型与数据，必须保留方法、模型及覆盖率信息。 |
| [NeMo Curator](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication) | 高级大规模语料过滤与去重 | 延后到明确的可选集成；当语料规模或所需算子超出轻量流水线能力时再评估。 |
| [Distilabel](https://github.com/lightonai/distilabel) | 合成数据和 AI feedback 流水线 | 它不是通用质量检查器，也不是核心依赖。仅当合成数据/AI feedback 进入项目范围后再考虑，并在届时核实维护状态与兼容性。 |

### 框架 PoC 验收标准

PoC 是非阻塞实验，不等于生产集成。使用符合许可证要求且能代表项目真实格式的样本，并与内置基线比较：(1) 是否补上了明确记录的能力缺口；(2) 信号是否有用、局限是什么；(3) 在支持的单卡/开发环境中的内存、耗时和安装负担；(4) 能否复现并记录版本；(5) 是否保留稳定 record ID、来源引用及 manifest/血缘；(6) 是否能写入版本化 `quality_report.json`，且不丢失指标类别和不确定性状态。只有带来实质价值才升级为可选 adapter；不得因此让核心安装依赖该工具。

## Gate 策略与版本管理

| 阶段 | v0.1 行为 | 潜在不变量 / 后续 gate 候选 |
|---|---|---|
| CPT | 报告结构检查、去重、长度和污染证据；未知项告警 | 编码/schema 有效；来源/许可证明确；无已知禁止性污染 |
| SFT | 报告正确性覆盖、解析/格式、长度和分布 | 无切分重叠；声明 schema 和最低验证器覆盖率；经评审的任务级正确性/长度阈值 |
| DPO / SimPO | 报告 pair 合格率、产出率、正确性证据、长度和重叠 | 在验证器支持范围内确认 chosen/rejected 顺序；无 train/eval 重叠；经评审的长度比例规则 |
| GRPO | 报告参考答案/验证器覆盖；对截断和 reward 信号做校准告警 | 参考答案可由支持的验证器验证；无切分重叠；绑定到模型/recipe profile 的校准边界 |
| Evaluation | 报告协议、覆盖、解析失败和重叠 | 固定/版本化评测协议；无已知训练重叠；评分覆盖达到经评审的协议要求 |

在没有验证器覆盖语义和真实分布之前，不要冻结“正确率必须 100%”或通用截断率/产出率门槛等示例值。Mini 证据用于提出阈值；采用的阈值必须版本化，并报告对接收/拒绝数量的影响。Gate 失败只应阻止受影响的正式实验路径；若以后设计显式 override，也必须保留失败警告，不得静默跳过。

## 前端与实验应用

只读 Data 体验应提供 Dataset Overview、Lineage、Quality、Distribution、Leakage 区块（可作为 Data 页内的标签，不需要变成五个顶级页面）。展示 gate 状态、指标定义、分子/分母、证据状态/可信度、协议、来源 artifact 和局限。默认显示汇总；样本查看应由用户主动进入并受访问策略约束。Overview 可突出低 preference yield 或 GRPO 校准截断偏高等行动性警告，并链接到证据。

数据质量消融应放在 Mini 和必需 OVAT 契约稳定之后。例如：等量高质量 vs 随机 SFT、hard vs random negatives、校准筛选 vs 未筛选 GRPO prompts。尽可能控制数据量和其他变量，版本化选择规则，使用相同 parent/评测协议，并直接报告数据质量差异，而不只是给数据贴“高质量”标签。

## 纳入路线

1. **现在完成设计：** 定义阶段指标、状态语义和报告关联；保持冻结的 v0.1.1 契约不变。
2. **dev7 / Mini 准备：** 暴露现有计数、拒绝原因、pair yield、切分/重叠、长度和截断证据；不支持的指标明确标为不可用。仅在发现明确缺口时，开展有限的 DataTrove（CPT）和 Data-Juicer（SFT）PoC；不得将它们加入核心依赖。
3. **Mini：** 收集真实分布与检查器覆盖率，评审候选阈值；不为达到样本目标而放宽正确性规则。按上述验收标准决定 PoC 是否值得形成可选 adapter。
4. **v0.1 发布：** 为正式数据集/run 发布质量报告并说明门槛与局限；不使用不透明综合分。
5. **v0.1.1 UI：** 用可访问表格、就地解释和血缘链接呈现质量证据。
6. **后续数据中心版本：** 开展受控的数据质量消融，之后再考虑自动 data CI / flywheel。

### 提前交付的定向合成数据切片

`targeted-synthetic-data` OpenSpec 变更提前实现一条范围受限、可选的研究路径：从逐题评测证据诊断带标签的薄弱场景；准备版本化计划及 provider/提示词/输出协议草稿；通过显式 CLI 生成数学 SFT 候选或有验证依据的同题 Preference pair；报告候选检查、策略选择、血缘、请求预算及未知/估算成本；训练与复测继续走分离的现有 CLI。离线 fixture 可用本地 fake adapter，OpenAI-compatible adapter 不会由 UI 触发。浏览器只读白名单摘要并导出未登记草稿，没有远程生成或训练控制入口。

生成数量和答案验证不是通用数据质量分。当前答案验证器不证明推理过程质量；新题需要独立 reference 或可验证构造。缺失标签/分母必须保持不可用；缺少证据时费用保持 unknown；最终 holdout 正文需隔离；`no_improvement` 或退化也是有效研究结果。此切片不实现自主数据飞轮、任意任务合成、CPT 语料生成或 GRPO prompt/reference 合成。

## 非目标

- 单一通用质量分或不透明的模型生成“质量等级”；
- 声称所有语义正确性都可被自动验证；
- 用尚未验证的阈值阻塞 smoke 迭代；
- 未经版本化修订就更改冻结的 v0.1.1 准入规则；
- 在报告中存储/重新托管外部数据集或敏感样本文本；
- 将 rollout 校准当作数据集内在且永恒不变的属性。
