# Spec Delta

## ADDED Requirements

### Requirement: Quality review measures the composed output
组合数据集 MUST 绑定对组合后实际内容执行的阶段质量报告，报告按组合输出分母呈现适用检查、父来源覆盖和来源间重复情况；父数据集质量结论只能作为上下文，不能代替该检查。

#### Scenario: 父数据集均有各自质量报告
- **WHEN** 系统组合多个已有质量报告的数据集
- **THEN** 新组合数据集仍获得内容摘要匹配的独立质量报告，并保留各父报告引用供对照

#### Scenario: 某检查无法应用于组合内容
- **WHEN** 组合数据集的阶段质量 profile 无法执行某项检查
- **THEN** 报告将其标为 not_applicable、unsupported 或 not_measured 并说明原因，不能继承父报告中的 PASS
