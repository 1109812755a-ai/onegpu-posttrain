# Design

## Context

See `proposal.md` for motivation and scope. 当前 `/api/data/import` 只接受已是阶段 canonical 结构的 JSONL；`register_jsonl` 流式计算原始摘要、写入规范序列化 JSONL、生成 processing/quality 报告并以目录原子提交，但不会保留上传字节，也没有 source-format 转换步骤。manifest 通过 artifact-relative `canonical_path` 定位训练输入。旧 canonical import 的 dataset ID 及 manifest 必须继续可读、可复用。

## Goals / Non-Goals

**Goals:**

- 在数据导入边界增加纯本地、确定性、明确选择的格式适配器；转换输出统一进入现有 SFT schema、质检、catalog 和训练预检。
- 让预览复用实际转换逻辑但不登记；登记时保留原始文件与 canonical 版本，并以摘要和报告绑定二者。
- 限制上传与转换资源，保持原子提交、路径包含检查、错误脱敏及旧导入兼容。

**Non-Goals:**

- 不接入任意脚本、外部格式推断、网络获取、Provider、训练或评测。
- 不支持工具调用/多模态/非文本消息；不对答案正确性、许可或数据源作自动背书。
- 不改训练 schema、recipe 或现有注册数据；不提供数据集删除 UI/API。

## Decisions

### 1. 以显式 `source_format` 分派纯函数适配器

每个 adapter 输入一行 JSON object 和行号，输出 canonical SFT record、行定位 provenance 及不含原文的有界错误。支持 `canonical-v1`、`alpaca-v1`、`messages-role-content-v1`、`sharegpt-from-value-v1`；请求必须显式指定其中一种。既有 API 客户端省略该字段时默认现有 canonical 行为，不做结构探测。

Alpaca 将非空 `instruction` 与非空 `input` 以固定分隔组合为 user 消息和 `raw_problem`；两者至少一项必须非空，以兼容 instruction 留空、问题内容放在 input 的真实数据。非空 `system` 前置为 system，`output` 作为 assistant 目标。Messages 格式只接受 `system/user/assistant` 和字符串 `content`，按源顺序保留。ShareGPT 格式只接受 `conversations` 的 `from/value`，将 `human→user`、`gpt→assistant`、`system→system`；其他角色、非字符串载荷和工具调用均拒绝。对话格式 `raw_problem` 由 user turns 按顺序拼接；消息数组保持完整，不能只保留最后一轮。

### 2. canonical ID 使用确定性行定位，来源 ID 另行标注

本期不会把输入的任意 `id` 字段直接提升为已核验的上游稳定身份；若提供字符串 ID，则原样保存在 `metadata.source_record_id`。Canonical ID 由源行内容摘要和重复行序号确定，并在转换 provenance 中保存 1-based 行号。该生成 ID 仅是本地转换定位符，不能提升 `record_provenance` 为来源方 record-level 证据；manifest 继续使用 dataset-only 等级。映射规则、模板 ID 和格式协议版本纳入配置摘要。

### 3. 预览和登记共用 transformer，但隔离其副作用

增加只读转换预览操作，接受显式格式和有界上传，使用 workspace 内临时目录运行同一 transformer 与 canonical validator；响应只含摘要、最多限定数量的 canonical 样例、计数和行/字段错误，不创建 manifest、Run、质量产物或 Provider 调用。临时文件在成功/失败/取消后清理。确认登记时客户端重新提交文件；服务端重新校验并转换，避免依赖过期内存预览状态或把预览 token 当作有效登记证明。

### 4. 在现有原子 dataset artifact 内保存 raw 和 canonical

登记 staging 目录包含 `raw/source.jsonl`、`canonical/sft.jsonl`、manifest、processing/quality reports、build/quality lineage 和 registration metadata；源字节在解析前写入 staging 并同步计算 SHA-256。任何校验/磁盘失败都丢弃整个 staging。提交前验证 raw/canonical 文件都位于 workspace artifact root，摘要与报告相符，且二者合计不超过配置的 artifact storage budget；达到字节/行数/输出限制时拒绝整份登记，不截断或静默筛行。

Processing report 为 raw acquisition 后追加版本化 convert-source-format 与 canonical-validation 步骤，记录 source format、converter version、config hash、输入/输出计数、rejection count 和限制。错误日志与预览不回显原始 prompt/answer。现有 canonical API 仍默认旧处理协议与现有身份算法；新转换格式把 source format/converter config 摘要纳入 identity material，确保不同转换协议不会覆盖同一 canonical 内容的历史产物。

### 5. 保持 manifest 读取与训练消费边界

manifest 的 `canonical_path` 继续指向 `canonical/sft.jsonl`；raw 来源路径、摘要和转换报告放在受支持的来源证据字段及处理步骤中，不让训练 adapter 读取 raw 文件。老 manifest 未含转换字段时继续走 legacy/不完整血缘读取路径。原始文件与 canonical、报告共享数据集 artifact 的保留生命周期，不新增单独删除 API。

### 6. 前端分阶段提交

现有阶段选择和来源/许可表单中，SFT 额外展示格式下拉框及“预览转换”步骤。预览列出转换示例、映射协议、预计输出/错误数量；错误后允许返回改格式或替换文件；只有预览无结构错误且来源/许可声明已确认后可显式登记。格式切换、文件替换或来源内容变化使预览失效。Canonical 选项保留当前流程与文案兼容；其它训练阶段不展示这些 SFT 专属格式。

## Risks / Trade-offs

- [用户把通用 role/content JSONL 当成任意 ShareGPT 变体] → UI 为每种格式展示字段示例；只接受契约字段并指出不支持角色，不做模糊兼容。
- [原始副本与 canonical 同存导致占用增加] → 登记前计算双份实际字节并应用单数据集总量限制；报告实际占用，超限失败原子回滚。
- [预览与登记之间文件被更换] → 预览只作提示，登记时全量重新解析并重新生成摘要；界面在文件/格式改变时立即清除旧预览。
- [来源不带 ID 时生成 ID 被误认为上游身份] → manifest/provenance 明确标记 synthetic row locator 和 dataset-only；不将行号声明为上游稳定 ID。
- [对话轮次中 assistant/user 交替不规则] → 原样保留所有受支持消息；必要 user/assistant 目标检查单独报告，不自动重排或修补。

## Migration Plan

1. 先添加可独立测试的适配器、转换 provenance schema 与 processing protocol；保留旧 canonical importer 默认路径。
2. 扩展后端 preview/register，使 staged raw 与 canonical artifact 原子落盘；旧 API 请求兼容，新增格式必须显式传格式 ID。
3. 增加 SFT UI 格式选择/转换预览，更新指南；已有数据不迁移、不补写 raw 文件。
4. 通过旧 canonical 导入回归、各适配器 fixture、资源/路径/失败原子性测试、浏览器窄屏/键盘走查和 OpenSpec strict validation 后发布。

回滚时可隐藏/禁用新格式选项并保留只读访问已登记 artifact；旧 canonical 导入代码路径和既有登记数据不依赖新 adapter。新 raw 文件随整个 dataset artifact 管理，不单独清理。
