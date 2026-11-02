# Dataset Lineage Specification

## Purpose

本能力让每个新构建的训练数据集都能追溯到来源、转换过程和实际内容，从而把数据质量报告与后续训练结果准确关联到进入相应阶段的数据版本。

## Requirements

### Requirement: 带版本的数据集身份
每个新构建的数据产物 SHALL 具有稳定、唯一的数据集 ID 和受支持的血缘 schema 版本。数据内容或转换配置改变时 SHALL 产生不同身份；展示名称或文件路径本身 SHALL NOT 作为身份依据。

#### Scenario: 相同输入重复构建
- **WHEN** 以相同来源修订、转换配置和输出内容再次构建
- **THEN** 数据集身份和内容摘要保持相同

#### Scenario: 过滤结果改变
- **WHEN** 过滤使输出内容改变
- **THEN** 新产物具有不同的数据集身份和内容摘要

### Requirement: 数据集祖先与来源证据
新数据集 manifest SHALL 记录来源数据集或仓库、已知修订与许可／使用元信息、父数据集 ID、转换名称／版本／配置摘要、阶段与划分、数量及单位、内容摘要。未知来源事实 SHALL 明确标为不可得，不得编造。

#### Scenario: 派生偏好数据集
- **WHEN** 从规范化数据构建偏好数据集
- **THEN** manifest 标明规范化父数据集、来源修订或明确的未知状态、配对转换及输出摘要

#### Scenario: 外部来源无不可变修订
- **WHEN** 无法确定上游修订
- **THEN** manifest 将其标为不可得，且不宣称来源可按不可变版本复现

### Requirement: 处理与质量证据关联
新数据集 manifest SHALL 指向实际产生它的数据构建 `run_id`，并保留处理报告引用与可用摘要。质量报告 SHALL 有独立身份，指向被检查的数据集和实际质量评估 `run_id`、检查协议／版本；数据集可反向索引该报告。缺少报告 SHALL 与报告全部检查通过区分开。仅持久化的真实产物进入图；不得为内存中临时状态伪造 ID。

#### Scenario: 构建与质量评估均完成
- **WHEN** 数据构建 Run 从源数据生成清洗后数据集和处理报告，质量评估 Run 随后检查该数据集
- **THEN** 可经两个不同 `run_id` 追到输入、转换、输出、处理报告及质量报告，并验证可得摘要

#### Scenario: 报告未生成
- **WHEN** 数据构建或质量评估未生成其应有的报告
- **THEN** 消费方展示不可得／未知，而非通过

#### Scenario: 清洗前后均有真实数据版本
- **WHEN** 清洗前后两个持久化数据集各有质量评估
- **THEN** 消费方可比较质量报告并追溯父子关系，但不得仅凭质量变化宣称训练指标因清洗而提升

### Requirement: 可用时保留记录级来源
新规范化训练记录 SHALL 保留可从上游取得的稳定记录 ID 和来源引用；派生偏好对须包含题目与两个父 generation 的 ID。若无法建立可靠的记录级来源，产物 SHALL 标为 `dataset_only`，且 SHALL NOT 虚构来源记录引用。

#### Scenario: 两个 generation 构成偏好对
- **WHEN** chosen／rejected 来自两个源 generation
- **THEN** 结果记录能指向题目及两个父 generation 记录

#### Scenario: 上游缺少行标识
- **WHEN** 来源没有可靠的行或 generation 标识
- **THEN** manifest 将记录来源等级标为 `dataset_only`

### Requirement: 旧版数据兼容
读取器 SHALL 继续接受缺少血缘 v1 字段的现有合法数据集 manifest，并将其血缘完整度标为旧版或不完整。遇到不支持的未来主版本 SHALL 明确报错，不得默认为 v1。

#### Scenario: 现有 Smoke manifest
- **WHEN** 读取不含血缘 v1 祖先信息的现有 manifest
- **THEN** 原有数据集和报告元信息仍可访问，同时显示旧版血缘状态

#### Scenario: 不支持的主版本
- **WHEN** manifest 声明了读取器不支持的血缘主版本
- **THEN** 读取器明确报告版本不受支持
