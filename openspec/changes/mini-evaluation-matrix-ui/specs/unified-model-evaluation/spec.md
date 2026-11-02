# Spec Delta

## ADDED Requirements

### Requirement: Evaluation matrix is presented with coverage and denominators
阶段比较消费方 MUST 展示矩阵中的基线、预期模型、指标值、请求与实际样本分母及相对基线变化；不得将缺失、失败或部分结果显示为零分或完成。

#### Scenario: All models completed under a shared protocol
- **WHEN** 矩阵包含已完成的 M2 基线及多个候选模型
- **THEN** 页面逐模型显示各指标、样本数和 delta，并保留矩阵身份

#### Scenario: A model or metric is missing or failed
- **WHEN** 矩阵标记模型或指标为缺失、失败或部分完成
- **THEN** 页面显示该状态和可用覆盖，不为缺失值补零或隐藏失败

### Requirement: Matrix comparability and limits are visible
阶段比较消费方 MUST 显示协议版本与可比性结论，并突出样本量、截断或评测覆盖等限制；不可比矩阵 MUST NOT 被呈现为受控的直接比较。

#### Scenario: Matrix protocols are consistent
- **WHEN** 矩阵协议、配置摘要和评测数据集身份一致
- **THEN** 页面显示可比状态及协议版本，并仍展示评测限制

#### Scenario: Matrix protocols or dataset identities differ
- **WHEN** 矩阵报告不一致、不可比或关键证据不可用
- **THEN** 页面清晰标识原因，不将模型差值描述为有效对照结论

### Requirement: Matrix results retain resolvable lineage
阶段比较消费方 MUST 为可解析的模型、训练和评测身份提供来源信息；无法解析的血缘 MUST 标记不可用，不得从展示名称推断关系。

#### Scenario: Model and run provenance resolve
- **WHEN** 矩阵结果可关联模型产物和训练/评测运行记录
- **THEN** 页面展示稳定 ID 和可访问的来源详情

#### Scenario: Model or training lineage cannot be resolved
- **WHEN** 某个矩阵条目缺失本地模型或训练来源
- **THEN** 页面保留指标状态并明确显示来源不可解析
