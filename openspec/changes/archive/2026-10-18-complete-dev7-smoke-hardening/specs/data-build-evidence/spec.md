# Spec Delta

## Purpose

为数据集构建提供可复核的目标达成和短缺证据，使用户能区分样本筛选产率不足、扫描预算耗尽与数据源耗尽，而无需放宽既定质量规则。

## ADDED Requirements

### Requirement: Target-driven build termination
数据构建 MUST 持续扫描输入，直至所有配置的输出目标达成，或输入源耗尽，或达到显式配置的原始行扫描上限；未达目标时 MUST 以失败状态结束。

#### Scenario: Targets are reached after scanning additional rows
- **WHEN** 初始候选行不足以满足输出目标，但后续行含有合格样本
- **THEN** 构建继续扫描并在目标达成后停止，且报告 `targets_met`

#### Scenario: Scan cap or source exhaustion leaves targets unmet
- **WHEN** 扫描上限已达或输入源已耗尽，且至少一个输出目标未满足
- **THEN** 构建报告对应停止原因、未满足目标，并以非成功状态结束

### Requirement: Build shortfall diagnostics
每次构建报告 MUST 包含原始行扫描数、各输出的目标数与选中数、样本拒绝原因计数及停止原因；诊断信息 MUST 保留质量筛选规则的原有语义。

#### Scenario: User reviews a shortfall report
- **WHEN** 任一输出目标未达成
- **THEN** 报告可用于识别主要筛选/拒绝阶段及扫描停止原因，且不得将未知语义质量指标伪报为已验证
