# Proposal

## Why

项目已有可选 LLM provider、定向 SFT 候选验证、数据登记、Mini SFT Launcher 和评测 CLI，但用户目前要在多个 CLI/页面间手工衔接，尚不能从生成目标一路追到训练模型与评测结果。先打通一条 SFT 纵向闭环，可验证“生成数据是否可被真实训练消费、训练后指标如何变化”，不要求先凑齐 Mini 目标数量。

## What Changes

- 在数据工作台提供 SFT 模型生成的显式配置与执行入口，复用已登记 provider/profile、训练种子、提示词与现有生成/验证协议；浏览页面、打开配置和 dry-run 不调用远程服务。
- 分开展示期望可用样本数、最大请求/重试预算、实际候选数、筛后可用数、失败/未验证数及短缺原因；目标数量是期望而非质量 gate，不足时允许审阅 partial 产物。
- 将通过 canonical schema、身份/污染/长度验证的 SFT 输出登记为 `model_generated` 数据集，保存 generator config/model/prompt、输入祖先、执行 Run、内容 hash、质量状态和 lineage；不覆盖用户或历史数据。
- 允许用户在现有 Mini SFT Launcher 选择已登记生成数据，保留许可/质量/长度提示与显式启动确认；训练产物继续沿现有模型 lineage 关联 dataset ID。
- 训练完成后提供对应评测启动/状态/结果入口，显示数据集、训练模型、评测协议和指标之间的可追溯关系；无评测证据时明确显示未执行，不推断效果。
- 本期只实现 SFT。Preference 后续单独解决有效 pair 目标与可信 chosen/rejected 产率；CPT 和 GRPO 继续标注为不支持模型生成，分别等待语料构建与 reward/reference 契约设计。

## Capabilities

### New Capabilities

- `sft-generation-workflow`: 连接显式 SFT 合成生成、partial 审阅、数据登记、Mini SFT 训练与评测，提供可追溯的单阶段端到端用户流程。

### Modified Capabilities

- `synthetic-data-generation`: 扩展显式生成入口及候选目标/预算语义；远程请求仍仅由用户明确执行触发，凭据和成本边界不变。

## Impact

- 影响数据工作台、合成生成 API/CLI、dataset registry/import、Mini SFT Launcher、训练后评测入口与相关 lineage/report API。
- 复用现有 OpenAI-compatible provider、候选验证、canonical SFT schema、数据集 manifest、Run/Model/Evaluation lineage 和固定 Mini SFT recipe；不增加必需的外部服务或数据库。
- 自动验收使用 fake provider、CPU fixture 和模拟训练/评测，不触发付费 API、下载模型或 GPU。真实 provider、RTX 4060 训练和评测效果需用户显式运行并作为独立证据记录。
