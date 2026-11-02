# Spec Delta

## ADDED Requirements

### Requirement: Stage-aware targeted plan
定向计划 MUST 记录目标训练阶段及该阶段适用的数据目标、父模型、数据角色、配方、预算和复测/保护指标；所有字段均须可追溯到计划版本。

#### Scenario: 用户切换目标阶段
- **WHEN** 用户在同一能力缺口下选择 CPT、SFT、Preference 或 GRPO
- **THEN** 系统显示该阶段适用的数据输入与验证信息，且不得静默沿用其他阶段的 schema 或质量结论

### Requirement: Targeted plan connects to actual execution evidence
计划状态 MUST 关联实际生成/导入、质量审查、训练和评测 run；未执行步骤须显示未执行，执行失败须保留失败状态和已知输入，不得由草稿推断执行成功。

#### Scenario: 定向计划执行失败
- **WHEN** 生成、登记、训练或评测任一步失败
- **THEN** 计划时间线保留实际失败 run、阶段和错误类别，并只显示已验证存在的数据集/模型产物
