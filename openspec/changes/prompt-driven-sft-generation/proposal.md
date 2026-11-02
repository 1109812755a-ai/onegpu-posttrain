# Proposal

## Why

当前普通 SFT 生成只能从已登记数据集中取现有问题，用户仍需先有一份数据才能生成数据。为让新主题快速形成可审查的训练集，需要支持只凭生成目标和项目默认格式从零生成候选，同时不把模型自报答案误标为正确。

## What Changes

- 在普通 SFT 生成中增加“按主题/提示词生成”模式，与“参考已有数据集生成”并列；前者不要求选择或伪造父数据集。
- 用户配置主题、能力点、难度及可选约束，项目固定提示词结构与 canonical SFT 输出协议；目标数量和请求预算在执行前确认。
- prompt-only 结果执行格式、重复、长度等可验证检查；没有独立 reference 时答案正确性标记为未验证，必须先人工审核再登记。
- 将 prompt/template/provider/profile 摘要、生成 execution、候选摘要和登记 dataset ID 纳入血缘；根来源明确标记为 prompt-only。

## Capabilities

### New Capabilities

<!-- None: this extends existing generation and lineage contracts. -->

### Modified Capabilities

- `synthetic-data-generation`: 增加不依赖种子数据集的 SFT prompt-only 生成模式及质量/审核要求。
- `dataset-lineage`: 明确无父数据集的模型生成数据如何记录可追溯根来源。

## Impact

- 后端：普通 SFT plan/bundle、prompt-only workflow、candidate validation、generation API 和 manifest registration。
- 前端：SFT 生成模式选择与按主题输入表单；不展示路径或密钥。
- 数据与血缘：prompt-only SFT manifest 可没有 dataset parents，但必须保存 generation execution 和 prompt/provider 摘要。
- 文档和测试：更新中文生成指南；使用 fake provider 覆盖从输入、生成、审核到登记的离线流程。
