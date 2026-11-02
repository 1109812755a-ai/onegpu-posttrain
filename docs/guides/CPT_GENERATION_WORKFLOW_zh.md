# CPT 模型生成、审核与登记

数据工作台 → “模型生成的数据阶段” → “CPT 语料”提供受控探索性生成入口。CPT 生成的是连续预训练文本语料，不是 SFT 问答，也不应把模型生成的事实视为已验证。

## 操作顺序

1. 选择“普通生成”或“定向补数”。普通生成按领域和子主题创作，不需要种子。定向补数必须填写已登记 CPT 计划 ID 与版本，且领域、子主题和预算须与不可变计划一致；其训练种子只按 dataset ID/hash 精确读取 `train_seed` 记录，拒绝 `diagnostic_dev`、`final_holdout` 和身份不匹配的数据。
2. 设置 Provider、目标 token 数、请求/候选上限及单次输出上限，先准备方案，再执行只读预检。准备和预检不会调用模型 Provider。
3. 核对预检给出的 tokenizer、配置摘要和污染筛查范围，再显式确认开始。界面会保存 execution ID 并在刷新后恢复状态；服务重启后显示 interrupted，不会自动重放可能重复计费的请求。
4. 生成结束后预览候选并人工审核，再点“确认审核并登记”。没有有效候选的失败任务不能登记。登记会运行 canonical schema 校验、写入质量报告、处理报告、manifest 和数据 lineage。
5. 返回训练与评测工作台，在 CPT 阶段按新登记 dataset ID 选择数据并单独执行训练预检。生成/登记不等于启动训练或阶段评测。

## 当前边界

- Provider 输出正确性始终标记为 `not_verified`；静态 schema/重复检查不是语义事实核验。
- 登记血缘会保留 Provider profile、模型声明和可用 revision；身份状态为 `profile_declared_not_independently_verified`。当前不会仅凭相同模型名关联项目登记模型 ID/hash。
- 当前工作区污染筛查只有 1 条本地示例基准记录，预检会标为 `example_fixture_only`。这只能拦截与该示例的命中，不构成正式污染排除、基准覆盖或质量验收结论。
- 许可证状态保留为 `not_provided`，请操作者核实 Provider 输出和种子来源的使用/再分发条件。
- 定向生成会记录计划 ID/version 和精确父 dataset ID/hash；生成只使用 train_seed，绝不读取诊断集或最终 holdout 正文。
- 请求结果未知时须先查看 `/api/synthetic/cpt/request-review/{bundle_id}` 并核实 Provider 侧记录。确认可能重复计费后可以登记“已核查未知请求”状态，但系统仍不会声称原请求失败。
- 本指南描述本地实现能力；实际 Provider、GPU 训练和评测须单独受控验收，不以 fake Provider 或 CPU 测试代替。
