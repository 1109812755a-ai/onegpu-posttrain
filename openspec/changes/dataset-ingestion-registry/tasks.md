# Tasks

## 1. 统一来源与阶段数据契约

- [x] 1.1 定义 DataSource、source-stage 能力清单、DatasetBuild 和 DatasetArtifact 的 manifest/schema；验证开源、本地导入、模型生成来源与四阶段的组合均可表达，且旧 manifest/lineage fixture 仍可读取。
- [x] 1.2 实现四阶段 canonical JSONL 流式导入与严格 schema 校验，覆盖坏行、空文件、未知字段、阶段不匹配和 byte/row 上限；运行单测验证失败不留下正式数据集。
- [x] 1.3 实现原文件摘要、确定性 dataset ID、临时目录原子提交、重复内容复用、来源/许可 unknown 状态；测试 hash 稳定性、磁盘失败清理和不得覆盖历史版本。
- [x] 1.4 为每个导入阶段生成 processing report、质量报告、manifest 及 data_build Run/lineage；用四阶段 fixture 运行 `validate-lineage --root <fixture> --require-complete` 并验证未测质量项保持 not_measured。
- [x] 1.5 编写中文 canonical schema/导入契约与四阶段示例，并按文档逐个成功校验 fixture；文档明确格式、质量和来源/许可是独立状态。

## 2. 来源适配与阶段构建

- [x] 2.1 定义并测试 source-stage capability matrix，列出现有开源来源、四阶段本地导入及生成来源的真实可用组合；未支持组合必须在 API/UI 返回原因。
- [x] 2.2 为现有开源构建器添加薄 source adapter，至少登记 FineMath CPT、DeepMath SFT、OpenR1 已有 SFT/Preference/GRPO 产物；测试来源 revision/摘要、阶段产物和旧缓存兼容，并补充中文来源说明。
- [x] 2.3 接入模型生成 SFT 来源，复用既有 provider 凭据、调用预算、缓存、候选验证和 explicit execute；以 fake provider 测试候选→质量检查→dataset ID/lineage，确认浏览/dry-run 不产生远程调用。
- [x] 2.4 在 SFT 来源链路稳定后接入模型生成 Preference，验证 chosen/rejected 的同题身份、可信排序证据和两侧祖先；测试无可信 wrong-side 时不产出伪造 pair，并更新中文指南。
- [x] 2.5 将模型生成 capability matrix 明确限制为 SFT、Preference；添加测试确保 CPT/GRPO 生成被明确拒绝/标 unsupported，且不调用 provider；在文档中说明需后续独立设计。

## 3. 数据来源与数据集工作台 UI/API

- [x] 3.1 增加本地安全导入 API 与来源/阶段能力 API，校验字节预算、路径边界、stage 值和结构化 provenance；用 API 测试覆盖越界、超限、错误响应不泄露原文及成功登记。
- [x] 3.2 扩展 catalog 将内置、上传、synthetic 数据集纳入共同视图，分别展示 source kind、可用阶段、dataset ID/hash、质量/许可状态和 lineage；用混合 fixtures 测试不支持/损坏 artifact 的降级展示。
- [x] 3.3 将 Data/定向补数体验提升为来源与数据集工作台：来源维度、阶段筛选、兼容矩阵、导入三步流程、quality/provenance 状态和 dataset→run 跳转；运行前端测试覆盖键盘/错误/取消/重试/上下文保留。
- [x] 3.4 编写中文工作台操作说明，分别说明开源、本地上传、模型生成来源与四个训练阶段的关系；在干净 workspace 按指南完成本地 JSONL 登记 walkthrough。

## 4. 训练消费与许可确认

- [x] 4.1 为已登记 CPT/SFT/Preference/GRPO dataset 复用现有阶段 adapter 生成 recipe 可消费的训练文件；用每阶段转换测试检查记录 ID、来源 metadata 和 dataset hash 能回链。
- [x] 4.2 将匹配的已登记 SFT dataset 接入当前 Mini SFT Launcher 选择和 preflight，并保持固定 parent/recipe 限制；测试阶段/配方不匹配时阻止，预检显示数据摘要和质量状态。
- [x] 4.3 对来源或许可未知的数据集，在本地训练启动时要求绑定 dataset ID 的显式确认，并写入 Run lineage；验证无确认被拒、有确认可执行、license 状态仍未核验。
- [x] 4.4 为其他阶段保留现有 CLI 训练消费路径，并编写从 workbench dataset ID 到 CPT/SFT/DPO/SimPO/GRPO recipe 的中文示例；测试解析配置的 dataset ID 与 manifest/hash 一致。

## 5. 端到端回归

- [x] 5.1 验证 source→stage build→quality→versioned dataset→training Run 全链路，对本地四阶段和当前开源来源逐个做 fixture 集成测试，并确认 lineage/hash 闭环。
- [x] 5.2 验证生成 SFT→Preference 两个阶段的启用顺序、预算/验证/失败状态展示，以及 GRPO/CPT 未支持状态不会被误报为可执行。
- [x] 5.3 运行后端全量测试、前端测试与生产构建、compileall、lineage 完整性和 OpenSpec strict validation；确认旧 smoke 数据可读，且验证过程未启动真实远程生成或 GPU 训练。验收：后端 `483 passed, 5 skipped`；前端 Node 测试与生产构建通过；四阶段登记/消费 fixture 的 `require-complete` 血缘校验通过。仓库级扫描排除顶层 pytest 临时夹具后检查 103 个 run/127 个 artifact，仍报告 19 个既有历史问题（8 incomplete、7 duplicate artifact ID、4 dangling ref），未修改或删除这些历史产物。
