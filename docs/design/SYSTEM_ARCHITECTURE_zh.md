# OneGPU-PostTrain 系统整体设计

**文档类型：** 持续维护的架构总览  
**状态：** v0.1 架构基线；未来版本组件在验收前均为规划  
**权威来源：** 产品方向见[产品愿景](PRODUCT_VISION_zh.md)；v0.1 实验契约仍以冻结的 [Design v0.1.1](v0.1.1/00-overall-design.md) 为准。

## 1. 目的与产品边界

OneGPU-PostTrain 是一个面向单张消费级 GPU 的开源、可复现后训练实验室，帮助用户理解 CPT、SFT、偏好优化和 RL 分别改变了什么。它不是新的 Trainer，而是组合数据准备、成熟训练/评测工具、受控实验和可检查产物，让工程师可以复现并解释阶段变化。

五条产品主线：

```text
Data Lineage · Data Quality · Training · Evaluation · Controlled Experiments
数据血缘 · 数据质量 · 训练 · 评测 · 受控实验
```

数据质量不是后期附加功能：每个正式数据构建都记录有序处理步骤、拒绝原因和可用的清洗前后质量证据。数据集血缘把数据版本连接到训练 run、模型和评测；`validate-lineage` 可只读检查本地引用、摘要、重复身份和环。模型效果结论需要受控比较；处理轨迹本身只说明数据发生了什么变化。详见[数据质量设计](DATA_QUALITY_DESIGN_zh.md)和[血缘设计](LINEAGE_DESIGN_zh.md)；在显式修订前，冻结的 v0.1.1 契约保持不变。

本文件将冻结的 v0.1 契约、当前本地 artifact 工作流、血缘设计和分阶段 UI/启动器规划连接起来，不取代各自的详细规范。

从来源处理 → 数据集质量 → 模型效果的详细契约见[数据处理、质量变化与模型效果设计](DATA_PROCESSING_EFFECT_DESIGN_zh.md)；本总览与版本路线共同指向该跨组件设计。

## 2. 系统上下文与边界

```text
项目用户 / 贡献者
      │
      ├── CLI + recipes ─────────────┐
      │                              ▼
      │                    OneGPU-PostTrain 核心
      │                  ┌───────────┼───────────┐
      │                  ▼           ▼           ▼
      │                Data       Training    Evaluation
      │                  │       (ms-swift)  (EvalScope +
      │                  │                   项目 evaluator)
      │                  └───────────┼───────────┘
      │                              ▼
      │                   本地文件系统 artifacts
      │                  manifests / runs / reports
      │                              │
      └── 未来本地只读 UI ───────────┘
```

### 范围内

- 数据源声明、准备、验证、切分/污染控制和训练框架适配；
- 阶段 recipe，以及围绕外部训练引擎的轻量执行边界；
- 统一评测和实验比较；
- 基于文件的数据、run、模型和评测 provenance；
- 本地 CLI，以及未来可选的本地 UI/启动器。

### 初始版本不做

- 重写成熟框架已有的 Transformer/Trainer 算法；
- 远程多租户服务、分布式调度、数据库 Registry 或强制托管跟踪；
- 在受控实验和证据可靠之前进行自动超参搜索；
- 将示例/占位数值包装成真实结果。

## 3. 核心领域与实验关系图

核心对象为 `ProcessingReport`、`DatasetArtifact`、`QualityReport`、`Recipe`、`CodeRevision`、`Run`、`ModelArtifact`、`EvaluationArtifact`、`DataEffectExperiment` 和 `DataEffectReport`。Dataset/run/model/evaluation 的身份规则见[数据与模型血缘设计](LINEAGE_DESIGN.md)；处理与数据效果契约见[数据处理与模型效果详细设计](DATA_PROCESSING_EFFECT_DESIGN_zh.md)。

```text
外部来源
   ↓ 有序、版本化的处理步骤（顺序由具体 builder 定义）
处理报告 + Canonical 问题池 + 数据集质量报告
   ↓ 先按 problem 切分，再生成阶段数据
训练数据池 ─────────── 独立 holdout / benchmark
   ├── CPT corpus                 │
   ├── SFT 样本                   │
   ├── Preference pairs           │
   └── GRPO prompts               │
          │                       │
Base → CPT run → M1 → SFT run → M2
                              ┌────┴─────┐
                              │共享 M2   │
                         DPO / SimPO / GRPO runs
                              │
                         adapter → merge → model artifacts
                              │
                              └→ evaluations
                                   ↓
                            受控数据效果报告
```

有向无环图（DAG）是关系事实源。除非明确声明为单独的串行 pipeline 实验，DPO、SimPO、GRPO 都应是共同 M2 parent 的兄弟实验。数据血缘和模型血缘通过 `run_id` 汇合；评测必须连回精确模型、数据/协议和 evaluator 版本。

## 4. 组件职责

| 组件 | 负责 | 不负责 |
|---|---|---|
| Data | 来源/版本/许可、有序处理轨迹、稳定 problem/record ID、过滤、切分、泄漏检查、manifest、阶段特定质量与分布报告、canonical→框架格式适配 | Trainer 语义或静默修改源数据 |
| Recipes / 实验定义 | parent、数据集、阶段、独立变量、控制变量、seed、预算、资源预期和评测计划；数据筛选研究连接策略、匹配控制和模型/评测产物 | 可变 run 状态或实际结果 |
| CLI / 编排 | 校验计划、解析 recipe、调用受支持命令、采集状态/退出码/日志/资源、连接输入输出 | 重写 ms-swift 算法；v0.1 不承担任意调度 |
| 训练适配器 | 将 canonical recipe/data 转换为固定版本的 ms-swift 调用并保存解析参数 | 宣称模型能力提升 |
| Evaluation | 统一运行任务/领域/偏好/RL evaluator；记录协议、分母、指标和原始引用 | 修改训练 artifact 或静默比较不兼容协议 |
| 实验分析 | 校验控制条件、计算兼容的 delta、汇总限制/资源并生成报告 | 补造缺失值或超出实验设计地作因果推断 |
| Artifact / provenance | 本地文件中的稳定 ID、manifest、处理/质量/效果报告、hash/状态、相对引用和 schema 版本 | 当前版本的数据库服务 |
| 未来 UI | 读取并可视化规范化 artifact、lineage 和兼容比较 | v0.1.1 中成为实验事实源或修改源 artifact |

## 5. 数据与模型契约

项目保持各阶段数据形态直观且可见：CPT 原始文本、SFT `messages`、偏好数据 `prompt/chosen/rejected`、GRPO prompt + verifier 标准答案。数学基线中，FineMath 用于 CPT，OpenR1 派生数据用于 SFT/Alignment，评测使用不相交 holdout 和固定 benchmark。

关键不变量：

- 按 problem 去重和切分，之后再生成阶段数据格式；
- 训练池与内部评测的 `problem_id` 不重叠；
- 作为 OVAT 比较时，DPO 和 SimPO 使用相同偏好数据 artifact；
- GRPO 标准答案只供 verifier/reward 使用，不作为 policy 的监督目标；
- 数据、recipe、reward 或评测语义发生变化时，必须版本化并记录 provenance。

冻结的过滤和 schema 见[数据设计](v0.1.1/01-data-design.md)，增量的数据质量报告提案见[数据质量设计](DATA_QUALITY_DESIGN_zh.md)，训练参数见[训练 Recipe 设计](v0.1.1/02-training-recipe-design.md)。

## 6. 执行与 artifact 生命周期

### v0.1 实现形态

```text
recipe + data manifests
        ↓
CLI 校验 / plan
        ↓
外部进程（ms-swift / evaluator）
        ↓
run.json + environment + logs + trainer outputs
        ↓
lineage.json + eval/comparison/report artifacts
```

- CLI 和本地文件系统是受支持的用户界面与持久化边界。
- 每次尝试（包含 OOM/失败）都有独立 run identity，并保留状态和证据。
- `run.json` 保留执行事实；`lineage.json` 记录输入/输出关系，不重复可变运行状态。
- 数据 manifest 和模型/评测引用连接关系图。路径用于定位；稳定 ID 与 digest 用于确认身份。
- `validate-lineage --root <artifact-root>` 只读检查 lineage；`--require-complete` 将普通 JSONL 等不完整来源作为门禁错误。校验完整性不代表数据质量合格或评测结果有效。
- 缺失或旧字段明确表示未知/不可用，不猜测回填、不静默重写历史 artifact。
- 默认不把第三方原始数据、密钥或模型权重放入 run artifacts。

## 7. 可复现性与结果解释

可解释的结果应尽可能关联：

- 模型与 tokenizer 来源/版本、parent/checkpoint identity；
- 数据来源/版本/许可、manifest、split 和样本/problem IDs；
- recipe、解析配置 digest、代码版本/dirty 状态、seed 和软件环境；
- 评测数据/协议/evaluator 版本、指标分母和原始报告；
- 硬件、耗时、吞吐、峰值显存及测量方法；
- 失败/OOM、缺失字段和有效性限制。

OVAT 比较要求 parent/data/evaluation 兼容，并声明独立变量。不同目标的训练指标不默认具有可比量纲。Smoke 证明流程/仪表可用，不证明算法质量优劣。

## 8. 分版本交互架构

| 版本 | 用户交互 | 架构边界 |
|---|---|---|
| v0.1 | CLI + 本地 artifacts | 无 Web 服务、数据库或调度器；recipe/CLI 是执行事实源。 |
| v0.1.1 | 可选本地只读 UI | FastAPI adapter 读取白名单路径；展示 artifacts 和 lineage；不提供写入/进程控制接口。 |
| v0.2 | 受限的本地启动器 | UI 通过现有 CLI 边界请求校验过的 recipe；仅允许 recipe/参数白名单；不执行任意 shell。 |
| v0.3 | Experiment Studio | 版本化 OVAT 定义、多 run/报告视图；只有批量需求证明必要时才引入本地串行队列。 |
| v0.4+ | Registry 视图 | ID/provenance/lifecycle 契约稳定后再考虑；持久化方式需要单独决策。 |

UI 细节见[前端产品设计](frontend/FRONTEND_PRODUCT_DESIGN_zh.md)和[前端路线图](frontend/FRONTEND_ROADMAP_zh.md)。UI 从属于实验系统，不是事实源。

## 9. 开源与安全约束

- 核心路径可从公开源码本地运行，不依赖专有托管服务；
- 固定并说明依赖、recipe、安装步骤和硬件预期；托管集成明确为可选；
- 分别追踪第三方代码、模型和数据许可；仓库许可不会覆盖上游资产；
- 避免提交大型数据集/权重、凭据、个人/受限数据或本机秘密；
- 未来 UI 文件访问必须限制在配置根目录内，使用路径 containment、artifact 白名单、限量读取和安全预览；
- 未来进程启动必须校验 recipe/参数，避免任意命令，记录日志/状态并要求显式本地操作。

按版本的发布要求见[开源发布计划](../project/OPEN_SOURCE_PLAN_zh.md)。

## 10. 文档事实源索引

- 产品方向和功能决策过滤器：[产品愿景](PRODUCT_VISION_zh.md)
- 原始动机与决策演进：[项目历史](../project/PROJECT_HISTORY.md)
- v0.1 冻结实验契约：[Design v0.1.1](v0.1.1/00-overall-design.md)
- 数据/模型/run/评测关系：[Lineage Design](LINEAGE_DESIGN.md)
- 阶段化数据质量：[数据质量设计](DATA_QUALITY_DESIGN_zh.md)
- 数据处理与模型效果闭环：[详细设计](DATA_PROCESSING_EFFECT_DESIGN_zh.md)
- UI 行为：[前端设计](frontend/FRONTEND_PRODUCT_DESIGN_zh.md)
- 当前阶段与证据：[开发计划](../project/DEVELOPMENT_PLAN.md)、[实施状态](../project/IMPLEMENTATION_STATUS.md)
- 版本顺序：[Roadmap](../../ROADMAP.md)、[开源计划](../project/OPEN_SOURCE_PLAN_zh.md)
