# 提案：SFT 多来源格式导入

## Why

常见 SFT 数据以 Alpaca、聊天 `messages` 或 ShareGPT 对话形式提供，而当前工作台只接受项目内部 canonical JSONL，要求用户自行转换且容易丢失来源与转换信息。提供显式、可预览、可追溯的来源格式适配，能降低导入门槛，同时继续使用现有 canonical 校验、质量检查、许可确认和训练血缘。

## What Changes

- 在 SFT 本地导入中增加显式来源格式选择；首版支持 Alpaca JSONL、`messages: [{role, content}]` 与 ShareGPT `conversations: [{from, value}]`，并保留现有 canonical JSONL 导入兼容行为。
- 对各格式执行版本化、确定性转换：Alpaca 的 `instruction/input/system/output` 映射至 system/user/assistant；messages 保留顺序和角色；ShareGPT 将受支持的 `human/gpt/system` 等角色映射至项目角色并保留完整多轮。未知角色、工具调用、无法表达的载荷或坏行显式报错，不静默丢弃。
- 登记前展示转换规则、代表性样例、输入/输出条数和有界行级错误；转换结果仍须通过项目 SFT canonical schema 和现有质量检查。结构通过不得表述为回答正确、来源可信或许可已核验。
- 将原始上传文件作为数据集产物中的只读来源证据保留，并记录原始摘要、格式/转换器协议版本、映射配置摘要、输入/输出行数和 canonical 摘要；登记失败不得留下半成品，产物不得覆盖既有数据。
- 明确原始文件本地存储和删除边界；上传字节、行数、转换输出与总磁盘占用继续受限。
- 本期不支持任意用户定义脚本或未声明的格式变体，不改变训练配方/算法，不自动修复语义问题或开始训练。

## Capabilities

### New Capabilities

- `dataset-source-format-import`：将明确选择的常见 SFT 来源格式安全、可预览地转换为 canonical 数据，并保留原件和可审计转换证据。

### Modified Capabilities

- `dataset-lineage`：导入转换产物须绑定原始文件摘要、格式/转换版本与配置摘要、canonical 内容摘要，并保留原始来源证据。

## Impact

涉及 `web/src/DatasetWorkbench.tsx` 及导入 API、`src/onegpu_posttrain/data/importer.py` 与 `dataset_import.py` 的格式适配和原子登记、manifest/processing report/lineage 的来源证据、原始文件落盘与容量/路径安全、导入指南及后端/前端测试。现有 canonical JSONL API、阶段 schema、训练入口和已登记数据保持兼容。
