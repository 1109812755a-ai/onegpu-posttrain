# Design

## Context

参见 `proposal.md` 与三份 delta spec。当前数据登记以阶段化 canonical JSONL 和 manifest 为核心，训练/run lineage 已围绕一个明确的 dataset ID 组织。数据来源、训练阶段和质量状态是不同维度；组合不得把原始文件路径列表直接泄漏到训练入口。

## Goals / Non-Goals

**Goals:**

- 让组合产物继续遵循现有 DatasetArtifact / registry / quality report / lineage 协议。
- 组合可复现、可审查，并由现有 Launcher 以一个 dataset ID 消费。
- 保持四阶段各自的语义和安全边界，尤其是 Preference 成对关系、GRPO reward/reference 契约及 holdout 隔离。

**Non-Goals:**

- 本变更不实现通用数据清洗流水线、自动数据生成、在线混合采样或训练期间动态改权重。
- 不引入数据库、外部编排器，也不替换现有单数据集训练流程。
- 不把“达到某条数”或父数据集 PASS 自动解释为组合质量 PASS。

## Decisions

### 1. 先物化为标准派生 DatasetArtifact，再训练

组合构建生成一个确定顺序的 canonical JSONL 文件，随后走现有登记与阶段质检流程；Launcher 仍传递单一 dataset ID，由 registry 解析该 artifact。拼接按用户声明的父级顺序输出。加权采样由构建器离线完成，结果写入文件，不在训练时临时采样。

理由：既有训练、评测、缓存和 lineage 的输入契约是单 dataset artifact；物化可固定训练实际看到的样本，也使质量报告绑定实际内容。代价是额外磁盘空间和一次构建 I/O。

考虑过：直接把多个 JSONL 路径传给 ms-swift。该方式虽可利用底层列表能力，但难以让现有单 dataset manifest、组合后质量报告和递归血缘精确描述训练内容，因此不作为项目级默认路径。

### 2. 组合身份绑定有序父级、配置和实际输出

组合请求记录有序父 dataset ID 与父内容摘要、目标阶段、策略、预算/权重、seed、组合协议版本。派生内容摘要对最终 canonical JSONL 字节计算；dataset identity 由父身份与规范化配置决定，manifest 同时记录输出摘要。相同父内容及相同配置可得到相同内容身份；每次构建仍使用独立 build run ID。任何父内容或配置变化都创建新身份，不原位覆盖旧产物。

为了保留记录来源，构建器在产物级 provenance/sidecar 中记录每条输出对应的父 dataset ID 和源 record ID（若原始记录没有稳定 ID，则记录源文件摘要与行号作为定位信息）；不向训练样本正文注入可能改变 canonical schema 或模型输入的内部字段。

### 3. 四阶段复用 validator，但执行阶段专属契约校验

组合只接受同一 stage 的已登记训练数据角色。构建前读取每个父 manifest 并验证 schema、可训练状态和 holdout/diagnostic 隔离；Preference 保持 chosen/rejected 成对不可拆分，并沿用其可验证标签约束。GRPO 除 schema 外还要求 reward/reference 契约一致且可执行；契约不匹配时禁止组合进入可训练状态。CPT、SFT、Preference、GRPO 分别复用已有的格式与阶段质量检查，不用来源类型决定训练语义。

重复记录、跨来源重复和来源覆盖作为组合级报告项显式呈现；除现有明确阻断规则外不静默去重或丢弃样本。若后续增加去重，必须成为有版本、有配置且会改变身份的显式 transform。

### 4. 组合计划与已构建版本分离

UI 中的组合编辑内容是未登记的 plan；只有显式执行构建并完成校验后才产生不可变 dataset artifact。plan 可编辑，已登记版本不可编辑。计划保存父级 ID、策略、权重/预算、seed；再次执行时重新读取父级摘要并检查其与预览时一致，不一致则要求刷新预检，不能悄悄使用新内容。

第一版支持拼接与固定预算的确定性加权采样。权重必须为有限正数并归一化；采样预算、取整/余数分配规则、是否允许有放回及 seed 都写入 manifest。为避免暗含语义，默认无放回；若预算超过可用唯一记录总量则预检报错，除非后续规格显式加入有放回选项。

### 5. 质量与 lineage 在登记时绑定，不作为训练期附加计算

组合完成后先计算内容摘要、记录数和来源计数，再运行目标阶段质量 profile，并将报告绑定到该摘要。不可执行的检查明确标记为 `not_applicable`、`unsupported` 或 `not_measured` 并说明原因；不继承父级 PASS。现有 run lineage 只需引用组合 dataset ID，数据 manifest/lineage 保存全部父级和组合配置，沿现有递归解析即可追溯。

### 6. 组合创建属于数据模块，训练只交接已登记身份

遵循 `stage-training-evaluation-workbench/design.md` 的四模块信息架构：在“数据”的数据集目录/详情发起组合，共享样本、处理与质量、血缘、关联运行详情。训练与评测入口提供已登记组合版本选择；需要新组合时带阶段、已选父数据和返回位置进入数据模块，登记完成后返回训练表单并预选新 dataset ID。返回仅填充表单，不自动开始训练。两模块共用组合创建视图，不在 Launcher 另建一套编辑和登记表单。

两个变更不形成实现顺序上的硬依赖：本变更可先在现有数据集工作台暴露组合视图，后续导航迁移保持对象链接；若阶段工作台先落地则直接接其数据二级入口。跨模块明确传递 dataset ID/stage，不把 data/build/tab 筛选注入训练页。

## Risks / Trade-offs

- [大组合增加磁盘和构建时间] → 预检展示估算输入/输出大小；先流式写临时文件，校验和报告成功后再原子登记，失败清理临时文件。
- [规范化序列化差异影响摘要] → 固定 UTF-8、JSONL 换行与序列化规则版本，并将协议版本写入 manifest；摘要对最终文件字节计算。
- [重复样本改变训练分布] → 明确报告重复数及来源覆盖，不静默去重；Launcher 预检展示采样策略和有效样本规模。
- [父数据集登记后底层文件被改动] → 构建前后校验父内容摘要；父文件不可验证或摘要变化时拒绝生成并提示重新登记。
- [不同阶段质量规则并非全部可复用] → 将检查结果逐项标记适用性与覆盖分母，允许部分检查未测，但不得伪报整体通过。
- [GRPO reward/reference 配置来源不一] → 将契约作为父 manifest 的机器可比字段；缺失或冲突时阻止训练可用状态，而不是推断默认值。

## Migration Plan

1. 增量扩展 DatasetArtifact manifest/lineage 的多父级字段；旧单父级和无父级 manifest 继续按原样解析。
2. 新增组合构建服务，输出普通 canonical JSONL、组合配置和 provenance sidecar；复用既有登记与质量报告落盘入口。
3. 暴露预检/构建 API 与数据工作台组合 UI；训练 Launcher 仍选择单个登记 ID，仅增加对派生数据集的展示与校验。
4. 用各阶段 fixture 完成组合、摘要、质量报告与递归 lineage 集成验证后开放功能。

回滚时关闭组合 UI/API 入口即可；已生成的不可变数据集和历史 run 保留，单数据集训练不受影响。禁止回滚时删除或重写已引用的 artifact。

## Open Questions

无。本变更已固定第一版策略为拼接与无放回确定性加权采样；新增采样语义应走后续独立规格。
