# Proposal

## Why

数据集工作台目前只展示来源、阶段、数量、质量和血缘元数据，用户无法在训练前快速确认实际样本结构与内容。样本预览应覆盖所有来源中有可验证 canonical 数据的已登记数据集，而不是只对本地上传开放。

## What Changes

- 在数据集卡片提供只读样本预览，覆盖本地上传、开源适配和模型生成等来源。
- 预览仅基于已登记 manifest 指向且摘要匹配的 canonical JSONL；限制返回行数、字段和内容长度。
- 对旧格式、证据不完整、文件缺失或摘要不匹配的数据集，显示明确的不可预览原因，不猜测路径或读取任意文件。
- 预览不修改、导出或重新登记数据，也不宣称样本通过语义质量评估。

## Capabilities

### New Capabilities
- `dataset-sample-preview`: 按已登记数据集身份安全读取少量 canonical 样本供用户检查。

### Modified Capabilities
- 无

## Impact

影响本地 UI 的数据 API、dataset manifest/canonical 文件解析、`DatasetWorkbench` 卡片交互与样本展示、相关后端/前端测试和中文工作台说明。不引入新依赖，不改变现有导入、训练或质量评估语义。
