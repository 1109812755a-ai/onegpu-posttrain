# Design

## Context

See `proposal.md` and `specs/dataset-ingestion/spec.md`. 项目已有阶段 canonical schemas、来源专用 builder、synthetic generation、manifest/quality report/lineage 和只读 Data catalog。各能力目前分散；现有 Mini SFT Launcher 仍是固定 recipe，不是多阶段调度器。

## Goals / Non-Goals

**Goals:**

- 将数据来源和训练阶段建模为正交维度：一个来源声明其可支持阶段，一个阶段构建引用兼容来源并产出数据集版本。
- 在统一工作台展示来源、可用阶段、处理/质量证据、dataset ID/hash 和消费它的训练 Run。
- 用可逐步扩展的来源适配器连接现有开源 builder、本地上传和显式调用的模型生成器，不重写其领域逻辑。
- 首版优先打通四阶段本地导入/登记/CLI 消费，并接入当前支持范围内的开源与模型生成来源。

**Non-Goals:**

- 不要求 source × stage 矩阵的每个格子都可用；不支持组合必须明确显示。
- 不引入抽象到可动态安装第三方插件的通用平台、数据库、对象存储、RBAC 或任务调度。
- 不在本变更内设计 CPT 合成语料生成或 GRPO reward/reference 系统。
- 不让模型生成绕过来源验证、阶段 schema、质量报告、成本预算或显式执行控制。

## Decisions

### 1. 采用三类领域对象，而非以页面或来源类型代表数据集

- **DataSource**：可重复读取/执行的来源身份，如开源仓库/修订、用户上传文件快照、合成生成计划及 provider/template/config 摘要；包含 source kind、可得 provenance/license 状态和适配器能力。
- **DatasetBuild**：一次来源获取/生成与一个目标训练阶段的明确处理执行，记录 source ID、stage、处理协议/配置、split、run ID、输入/输出 hash 和质量评估引用。
- **DatasetArtifact**：持久化 canonical 与训练适配数据版本，具有独立 dataset ID、阶段、数量、内容 hash、manifest 和质量报告；训练 Run 只消费此对象的显式 ID/hash。

一个来源可经过不同 DatasetBuild 产出多个阶段数据集；同一阶段也可由多个来源产生。来源不等于数据集，生成计划不等于已生成数据集，只有实际持久化产物进入血缘图。

**替代方案：**继续按每个来源建一套独立数据页/产物规则，会重复身份、质量和 lineage 语义；把 stage 放进来源类型中则无法表达同一来源的多个派生阶段。

### 2. 使用能力清单表达 source × stage 矩阵

每个适配器声明支持阶段、数据格式/版本、是否需要网络、是否有原始来源 revision、是否需要模型凭据及可用验证器。工作台基于能力清单显示可执行和未支持组合，后端每次执行前再次验证，不能只信任前端状态。

首版矩阵以实际能力为准：本地 canonical JSONL 导入支持 CPT/SFT/Preference/GRPO 四类；已存在的开源 builder 按其当前产物声明（例如 FineMath→CPT、DeepMath→SFT、OpenR1→SFT/Preference/GRPO）；模型生成先接 SFT，再接 Preference。GRPO 生成在可信可执行 reward/reference 建立前保持 unsupported；CPT 生成不复用数学题生成流程，另案设计。该矩阵是 capability data，不把所有组合写成一条全局假设。

**替代方案：**一开始强制所有来源都支持四阶段，或按来源建立独立数据模型。前者会制造虚假能力，后者会造成架构分叉。

### 3. 所有来源汇入同一阶段处理与 DatasetArtifact 协议

来源适配器提供带来源身份/原始摘要的输入记录；阶段构建将其校验/规范化为既有 CPT、SFT、Preference 或 GRPO canonical schema，再运行阶段适用的 processing 与 quality protocol，最后序列化为训练 adapter 所需格式。处理前后身份和 hash 分开记录；质量检查不适用或未运行时标 not_measured。现有开源 builder 与 synthetic exporter 先通过薄适配层产生共同 manifest/报告，不迁移或重写旧 raw cache。

**替代方案：**强迫所有上游数据从一种未经区分的 JSONL 入站。开源数据和生成响应拥有不同来源证据/验证语义，直接合并会丢失 provenance。

### 4. 以文件 artifacts 为事实源，并使 provenance 可缺省但不可编造

工作区沿用 `data/processed` 与现有 runs 结构，manifest 记录 source kind/reference、目标阶段、parent source/dataset IDs、原始与 canonical hash、transform/config/version、质量报告、license status 和 lineage references。source、revision、license、attribution 是可空/带状态字段；已知/用户声明/未核验/未提供需区分。记录级祖先能取得则保留，不能取得标 dataset_only。确定性 dataset ID 依据 stage、规范化内容和影响输出的 build 配置；同内容导入可复用，不覆盖产物。

文件命名和哈希只能校验内容/身份，不能证明版权、来源真实性或许可证。用户明确确认未知许可数据用于本地训练时，确认事件绑定 dataset ID 和训练 Run，但不改变 license evidence。

**替代方案：**使用 SQLite 作为第一事实源会增加迁移/备份和双写；把 source URI、文件名或上传者输入直接视为 verified 会伪造证据。

### 5. 分开 source acquisition 与 stage build/training 执行

打开工作台、浏览能力、保存草稿或执行本地格式预览不会触发下载或远程生成。开源获取和 LLM 生成是显式 source acquisition；受控转换/评估是 DatasetBuild；训练 Run 消费已登记 dataset ID。远程生成继续复用现有 explicit execute、预算、缓存、验证和密钥变量引用；首个 UI 生成切片支持 SFT，再按 Preference 的可信 pair 验证规则开放。GRPO 与 CPT 生成保持独立设计决策。

### 6. UI 用双入口表达两个维度，并保留单卡实验室边界

“数据来源”视图回答数据从哪来、能产哪些阶段、是否需联网/凭据；“数据集”视图回答阶段、版本/质量/血缘以及被哪些训练 Run 消费。两者可按 source kind、stage、数据状态筛选并互相链接。定向补数保留为以评测薄弱场景为入口的一种模型生成工作流，不再代表所有数据工作台。

首版 Data import 支持本地四阶段；Launcher 仅给当前 Mini SFT recipe 暴露匹配 SFT dataset 选择；其余阶段通过现有 CLI recipe 使用已登记的适配数据。模型生成 UI 先接 SFT，再接 Preference；所有生成仍需显式点击执行、预算和验证摘要。没有支持的 recipe/validator 时展示“尚未支持”，不提供看似可执行的控件。

### 7. 上传协议与安全边界

本地上传首版只接受单个 UTF-8 canonical JSONL 文件，不接受归档包或用户给出的磁盘目标路径。服务端流式写入工作区临时位置、在配置的字节和行数预算内校验 schema，再以原子目录提交；失败清理临时文件，既有数据不可覆盖。错误只返回有界行号/字段信息，不回显原始样本。建议上限为 256 MiB 与 250,000 行。

## Risks / Trade-offs

- [source × stage 组合增多使能力难以理解] → 由 capability matrix 声明支持，UI 同时显示未支持原因；不暗示全组合交叉积。
- [旧 builder 与新统一产物格式不一致] → 薄 adapter 渐进纳入，保留原缓存/报告并用来源摘要验证，不做一次性迁移。
- [LLM 生成被误认为已验证训练集] → 保留 candidate、验证、筛选和 dataset build 分层；Preference 必须有可信排序证据，GRPO/CPT 生成不提前开放。
- [大文件或异常 JSONL 消耗本机资源] → 流式读取、字节/行数预算、临时落盘、失败清理和最大错误列表。
- [用户 license 声明被误读为法务审查] → UI 与 manifest 记录声明来源/核验状态；本地确认只绑定训练 Run，不升级 license status。
- [泛化 Launcher 扩大本次范围] → 本次仅接入当前 SFT Launcher；其它阶段依照 CLI recipe 使用，通用多阶段 Launcher 另案。

## Migration Plan

1. 扩展 manifest/source schema 为兼容可选 provenance 字段；现有记录不迁移，旧 reader 继续读取。
2. 先实现本地导入四阶段 vertical slice 和共同 DatasetArtifact；接入既有开源 builder 时按来源逐个添加 adapter，不改变原始缓存。
3. Data 工作台先展示 sources、stage compatibility 和 datasets/lineage；模型生成按 SFT、Preference 的顺序逐步接入已有显式生成协议。
4. Launcher 先绑定兼容 SFT dataset，并将 dataset ID/hash 及未知许可确认写入真实训练 Run；其他阶段保留 CLI 接入。
5. 回滚只关闭新 UI/API 和新 adapters；已提交数据 artifact 不自动删除，旧构建和 CLI 保持原样。

## Open Questions

无。CPT/GRPO 模型生成、CSV/Parquet 映射和通用多阶段 Launcher 不属于本 MVP，需各自的验证/资源/UX 设计。
