# Spec Delta

## ADDED Requirements

### Requirement: 转换导入保留不可变原始来源
以来源格式转换登记的数据集 MUST 在其数据集产物内保留原始上传字节的只读副本，并在 manifest/处理报告中绑定原始文件摘要、声明的来源格式、转换器协议/版本与配置摘要、canonical 输出摘要及输入/输出记录计数。

#### Scenario: 完成格式转换登记
- **WHEN** 用户确认登记一个有效的 Alpaca、messages 或 ShareGPT 文件
- **THEN** 数据集可同时追溯原始文件和 canonical 输出，且逐项核对其摘要与转换报告一致

#### Scenario: 以不同转换配置重新导入
- **WHEN** 相同原始文件经不同格式声明、转换版本或映射配置处理
- **THEN** 新转换身份和数据集 lineage 能区别于既有产物，即使 canonical 内容相同也不得覆盖旧产物

#### Scenario: 整个数据集产物被清理
- **WHEN** 本地保留策略或维护操作清理整个转换数据集产物
- **THEN** 原始副本、canonical 文件、manifest 和处理/质量证据遵循同一产物生命周期，不提供独立于数据集的原件删除或残留路径
