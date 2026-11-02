# Spec Delta

## Purpose

让用户选择常见 SFT JSONL 来源格式，并在登记前检查其如何转换为项目 canonical 记录。转换保留来源、角色和版本证据，同时让既有训练入口继续只消费 canonical 数据。

## ADDED Requirements

### Requirement: SFT 来源格式必须显式选择
本地 SFT 导入 MUST 要求用户明确选择来源格式；系统 MUST 支持项目 canonical JSONL、Alpaca JSONL、role/content messages JSONL 和 ShareGPT conversations/from/value JSONL，且不得根据样例静默猜测格式。

#### Scenario: 导入已支持的格式
- **WHEN** 用户选择一种已支持格式并提交匹配的 JSONL
- **THEN** 系统按该格式的版本化映射生成 canonical SFT 记录，并继续执行项目 schema 校验

#### Scenario: 格式与文件结构不匹配
- **WHEN** 文件字段不符合用户选择的格式或记录混用多种格式
- **THEN** 系统指出对应行和字段并拒绝登记，不自动切换格式或跳过该行

### Requirement: 对话来源转换必须保留受支持的轮次与角色
转换器 MUST 保留输入记录中的消息顺序和全部受支持轮次；每个 canonical 记录 MUST 至少包含非空 user 输入和 assistant 目标。无法表达的角色、工具调用或载荷 MUST 明确拒绝，不得静默删除或降格为普通文本。

#### Scenario: 转换 Alpaca 指令样本
- **WHEN** Alpaca 行包含 `instruction`、可选 `input`/`system` 和非空 `output`，且 `instruction` 或 `input` 至少一项非空
- **THEN** 非空的 `instruction` 与 `input` 按固定分隔组成 user 消息，非空 `system` 作为 system 消息，`output` 作为 assistant 目标，原始字段保持在来源文件中

#### Scenario: 转换 role/content messages
- **WHEN** 消息数组使用 `system`、`user`、`assistant` 角色及字符串 `content`
- **THEN** 系统按原顺序保留消息及完整多轮内容，并为 canonical 记录补充所需身份和模板字段

#### Scenario: 转换 ShareGPT conversations
- **WHEN** `conversations` 数组使用受支持的 `from`/`value` 字段与 `human`、`gpt` 或 `system` 角色
- **THEN** 系统依次映射为项目支持角色、保留完整对话顺序，并生成至少含 user 和 assistant 的 canonical 记录

#### Scenario: 输入包含工具角色或非文本载荷
- **WHEN** 输入包含未声明映射的角色、tool/function 调用或非字符串消息载荷
- **THEN** 系统报告行级不支持原因并拒绝该文件，不丢弃消息后继续训练

### Requirement: 转换预览与登记错误必须可审查
系统 MUST 在登记前提供有界转换预览，说明所选格式、映射版本、实际扫描数量、输出数量和行级拒绝原因；预览 MUST NOT 创建已登记数据集、训练 Run 或外部模型请求。

#### Scenario: 用户预览 Alpaca 转换
- **WHEN** 用户对本地文件请求预览
- **THEN** 系统展示源字段到 canonical messages 的规则及有限样例，且不创建数据集或调用 Provider

#### Scenario: 文件包含无效行
- **WHEN** 任一已扫描行 JSON 无效、字段缺失或内容无法映射
- **THEN** 报告实际行号、字段和原因，保持全文件登记失败且不静默过滤

### Requirement: 转换记录必须稳定且不伪造来源身份
每条转换记录 MUST 有可复现的 canonical ID 和到原始文件行位置的本地映射；系统 MUST 将生成的 ID/行位置标记为转换定位信息，不得宣称其为来源方提供的稳定记录 ID。

#### Scenario: 来源行没有稳定 ID
- **WHEN** Alpaca 或 ShareGPT 记录未提供可信来源记录 ID
- **THEN** 系统生成转换所需 canonical ID、记录来源行定位，并将记录来源等级保持为 dataset-only 或等价的非来源方 ID 状态

### Requirement: 转换产物沿用训练数据质量与安全边界
来源格式转换 MUST 受现有大小、行数、路径包含性、原子登记和 canonical 校验约束；转换成功 MUST NOT 自动表示答案正确、来源可信、许可已核验或正式数据集 readiness 通过。

#### Scenario: 转换成功但语义正确性未测
- **WHEN** 文件通过格式转换和 canonical schema 校验，但没有独立答案验证证据
- **THEN** 质量与训练预检保留未测量状态，并要求使用者按既有许可规则确认，不提升为已验证

#### Scenario: 转换或磁盘写入中断
- **WHEN** 转换失败、超出限制或持久化中断
- **THEN** 系统不留下 catalog 可见的半成品，不覆盖已存在的数据集，并返回可恢复的错误摘要
