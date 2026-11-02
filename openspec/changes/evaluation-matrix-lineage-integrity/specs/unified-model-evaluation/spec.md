# Spec Delta

## ADDED Requirements

### Requirement: Evaluation matrix verifies evidence identity
统一评测矩阵 MUST 校验指标文件、评测运行记录与评测 lineage 指向同一评测执行、模型身份及终态；缺失或不一致时 MUST 明确报告来源完整性问题，且不得将该项当作完整成功评测或可比较证据。

#### Scenario: Evaluation sidecars reference different executions
- **WHEN** 矩阵读取到的 `evaluation_run_id` 与 lineage 的 `produced_by_run_id` 不一致，或评测/模型身份与配置不匹配
- **THEN** 对应模型的来源状态标记为不匹配或不可验证，不纳入完整可比结论，并保留已读取指标及具体问题

#### Scenario: Failed or missing lineage accompanies metric output
- **WHEN** 指标文件存在，但运行状态失败/部分、lineage 缺失或 lineage 状态未完成
- **THEN** 矩阵保留可用指标和分母，同时将评测状态标为失败、部分或来源不完整，不宣称该模型的评测完整

### Requirement: Matrix provenance remains inspectable
评测矩阵 MUST 保留足以定位其源评测产物及身份校验结果的信息；历史结果缺少必要身份字段时 MUST 标为旧版/不完整，而不得猜测或修补来源。

#### Scenario: Legacy evaluation lacks stable identity
- **WHEN** 旧评测有指标但不含可核实的评测身份或 lineage
- **THEN** 矩阵继续展示可用指标，并明确显示来源不完整及缺失字段

#### Scenario: Source references are repaired by a new matrix build
- **WHEN** 用户依据已核实的评测产物重新生成矩阵
- **THEN** 新矩阵引用真实源路径和稳定身份，且不修改历史评测指标或血缘文件
