# Spec Delta

## ADDED Requirements

### Requirement: SFT 支持无种子数据集的提示词生成
系统 MUST 允许用户以主题、能力点、难度和可选约束直接准备 SFT 生成任务，不要求选择现有数据集。请求与输出仍须使用项目限定的 SFT 格式和有界预算。

#### Scenario: 按主题生成新 SFT 样本
- **WHEN** 用户选择提示词生成模式并提交有效主题、能力点和目标数量
- **THEN** 系统使用项目默认提示结构生成包含新问题及解答的 SFT 候选，不读取或伪造种子数据集

#### Scenario: 浏览或准备 prompt-only 方案
- **WHEN** 用户打开页面、保存/准备方案或执行 provider-free 预检
- **THEN** 系统不发起远程模型调用，并记录提示词配置摘要和目标预算

#### Scenario: 无法独立验证新题答案
- **WHEN** prompt-only 候选没有可信的外部 reference 或可验证构造过程
- **THEN** 系统将语义正确性标记为未验证，不依据模型自评或 JSON 格式通过而提升为 verified

### Requirement: Prompt-only SFT 必须人工审核后登记
Prompt-only 生成结果 MUST 经过 canonical schema、记录身份、重复及适用长度检查；生成数量不得作为质量门槛，候选必须经用户预览和显式确认后才能登记为训练数据集。

#### Scenario: 部分候选通过校验
- **WHEN** provider 返回格式错误、重复或超出声明长度的候选
- **THEN** 系统分别报告可用数量与拒绝原因，保留符合最低格式要求的候选供审核，不自动降低规则补足目标

#### Scenario: 登记无种子生成结果
- **WHEN** 用户确认审核已生成的 SFT 候选
- **THEN** 系统创建独立 dataset ID 和内容摘要，并在登记前重新验证 canonical 格式
