# Spec Delta

## Purpose

让用户在训练前直接检查已登记阶段数据集的少量真实样本，确认 canonical 结构与内容是否符合预期。预览覆盖不同来源，但仅对身份和内容摘要可验证的数据提供记录正文。

## ADDED Requirements

### Requirement: 所有来源的已登记数据集均可按阶段预览
工作台 SHALL 对本地上传、开源适配和模型生成等来源使用统一 dataset ID 预览其已登记的 canonical 样本，不得仅因来源类型不同而限制预览。

#### Scenario: 预览不同来源的数据集
- **WHEN** 用户打开任一具有可验证 canonical 数据的已登记 dataset ID
- **THEN** 工作台按其训练阶段展示样本，并继续显示来源、许可和质量状态，不推断未记录的来源事实

#### Scenario: 数据集缺少 canonical 证据
- **WHEN** 已登记条目属于旧格式、缺少 canonical 路径/摘要，或关联文件不存在
- **THEN** 工作台显示“暂不可预览”及可理解原因，不按文件名猜测路径

### Requirement: 预览只读取登记数据且核验内容完整性
样本预览 SHALL 只通过 dataset ID 定位 manifest 声明的 canonical 文件，确认路径位于对应数据集 artifact 根目录内，并在返回样本前核对 manifest 中的 canonical SHA256。

#### Scenario: canonical 内容与登记摘要一致
- **WHEN** dataset ID 唯一解析到有效 manifest 且 canonical 文件摘要匹配
- **THEN** API 返回该阶段的有界样本预览和 dataset ID、stage、内容摘要

#### Scenario: 内容被修改或路径越界
- **WHEN** canonical 摘要不匹配，路径越界或解析出多个不一致 manifest
- **THEN** API 不返回记录正文，前端显示证据校验失败且不把该数据标为可预览

### Requirement: 预览内容与规模有界
样本预览 SHALL 为只读并限制单次记录数、字段集合及每字段文本长度，不得返回整个数据集或任意附带 metadata。

#### Scenario: 查看阶段样本
- **WHEN** 用户请求一个合法的预览数量
- **THEN** 响应最多返回系统上限条记录，仅包含该阶段 canonical schema 的可读训练字段和记录 ID；超长字段被截断并明确标记

#### Scenario: 无效或超限的预览数量
- **WHEN** 请求数量为非正数、非整数或超过上限
- **THEN** 请求被拒绝或安全地限制在声明上限，且不会增加读取范围

### Requirement: 预览状态不代表数据质量或使用权
工作台 SHALL 将样本可预览状态与 schema/质量、许可状态独立展示；预览本身不得改变 dataset、质量报告、血缘或许可状态。

#### Scenario: 样本可以预览但质量未评估
- **WHEN** canonical 摘要验证通过而质量指标仍为 NOT_EVALUATED/not_measured
- **THEN** 工作台显示样本内容与“质量未评估”状态，不将预览解释为质量通过

#### Scenario: 预览后检查数据集产物
- **WHEN** 用户完成、关闭或重复打开预览
- **THEN** canonical 文件、manifest、质量报告和 lineage 内容保持不变
