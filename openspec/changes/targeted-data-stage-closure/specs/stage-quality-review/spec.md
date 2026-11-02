# Spec Delta

## ADDED Requirements

### Requirement: Stage-appropriate quality review before registration
每个定向生成或导入的数据集在登记前 MUST 提供目标阶段适用的质量审查摘要，绑定准确的数据集内容摘要、profile 版本、检查覆盖与拒绝原因；无适用检查须标为 not_applicable 或 not_measured。

#### Scenario: 数据数量低于计划目标但质量证据可用
- **WHEN** 数据集非空、格式有效但数量低于期望目标
- **THEN** 审查按实际分母报告质量与覆盖，允许用户确认登记 partial 数据，不因数量不足自动标记质量失败

#### Scenario: 数据数量达到目标但质量检查未通过
- **WHEN** 数量达到期望目标但有适用质量检查失败或未测
- **THEN** 数量达标不得覆盖失败/未测状态；登记与后续训练必须展示相应质量风险和审核决定
