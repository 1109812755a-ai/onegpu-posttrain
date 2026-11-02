# Proposal

## Why

固定版 DeepMath-103K 全量离线校准显示，在 Qwen3-0.6B tokenizer/template 下，2048-token 候选可提供 15,046 条 exact-answer-match SFT 记录，足以覆盖 Mini 的 5K 数量目标；但该校准产物仍是探索数据，且既有 2048-token 短程探测的显存采样已达 7,931 MiB，不能据数量直接转为正式 Mini。现在需要一条独立、可审阅、可追溯的 Mini SFT 构建与单卡验证路径，同时保持项目基模和 Smoke 契约不变。

## What Changes

- 新增 DeepMath 固定快照到 Mini SFT 候选的离线构建路径：使用 Qwen3-0.6B-Base 固定 tokenizer/template、完整渲染长度上限 2048、不静默截断，并创建新的数据/处理/质量/血缘产物。
- 使用现有 FineMath 5,000,368-token CPT 训练集创建独立 Mini CPT recipe，从固定 Qwen3-0.6B-Base 训练并产出该 Mini SFT 的 parent M1；记录实际 token exposure，不把训练集预算等同于实际消费量。
- 对题目身份先做确定性 train/eval 分组，再从合格训练题中确定性选择最多 5,000 条；与项目评测/holdout 做去重和污染检查，分母、拒绝原因及 stop reason 均可审计。
- CPT/SFT 所用数据与运行先保持 `exploratory`：当前 FineMath CPT 质量 review 是 `pending` 且 usage 为 `legacy_unknown`。只有匹配的数据质量 review、许可/来源审阅、处理证据和完整 lineage 全部满足 Mini readiness 契约后，才允许维护者显式批准为正式 Mini 输入。
- 增加独立的 Mini CPT 与 Mini SFT recipes/run identities，固定基模为 Qwen3-0.6B-Base；Smoke recipe 保持 1024，2048 SFT 使用 micro-batch 1、gradient checkpointing，并在开始完整训练前重新确认 GPU 空闲显存和短程可行性。
- 仅在数据门槛及资源预检通过后运行 Mini SFT 与统一评测；结果绑定数据、配置、parent checkpoint、环境和评测血缘。结果不声称长度上限本身带来因果提升，也不自动完成其他 Preference/GRPO Mini 缺口。
- 将 Qwen3-1.7B-Base 留作后续独立 pilot，不与本次数据源/长度变更混在同一实验中。

## Capabilities

### New Capabilities

- `mini-sft-source-build`: 从固定、可审阅的本地数学数据快照构建具有确定性拆分、完整 provenance 和质量门槛的 Mini SFT 数据。

### Modified Capabilities

- `mini-dataset-readiness`: 让新 SFT 来源继续经过既有预算、质量 review、处理证据、使用状态和 lineage 验收，且不能因达到 5K 自动成为 official。

### Explicitly Unchanged

- `sft-length-calibration`：校准输出仍为探索性证据，不能直接晋升或覆盖正式数据；Mini 构建必须从 pinned source 重新生成独立 artifact。
- Smoke recipe `recipes/smoke/20-sft.yaml` 的 1024-token 默认值。
- 项目主基模 Qwen3-0.6B-Base，以及 CPT、Preference、GRPO 的数据来源与验收门槛。

## Impact

- 影响数据构建器/CLI、Mini 数据预算配置、CPT/SFT recipes、质量 review 与 readiness、血缘校验、统一评测及中文 Mini 操作文档。
- 数据来源为本地已缓存的 `zwhe99/DeepMath-103K` 固定 revision `5cf055d1fe3d7a2eb19719ac020211469736ae44`；需记录该 revision 的来源与许可证信息，运行中不隐式下载。
- 复用现有 Qwen3-0.6B-Base pinned tokenizer、canonical data、quality review、lineage 和 evaluation 协议，不引入新训练框架或必需服务。
- 2048-token RTX 4060 8GB 训练存在高显存风险；若 GPU 空闲显存不足、短程 probe OOM 或数据 review 未批准，则保留探索性产物并停止完整训练，不降级质量门槛、不自动改用 1.7B。
