# Tasks

## 1. 数据组合协议与产物身份

- [x] 1.1 扩展 DatasetArtifact manifest/lineage schema，记录多父级 ID/摘要、规范化组合配置、输出摘要和 build run；为旧 manifest 增加兼容性测试并更新 schema 文档，验证旧单数据集样例仍可读取。
- [x] 1.2 定义组合配置 canonicalization、身份计算和 JSONL 内容摘要实现；为父级顺序、配置变化、父内容变化、换行/序列化版本编写单元测试，验证相同输入配置可复现相同身份及字节摘要。
- [x] 1.3 增加逐记录来源 provenance sidecar（父 dataset ID、原 record ID 或源摘要+行号）；测试记录数/顺序与输出一一对应，并说明 sidecar 不进入训练样本正文。

## 2. 组合构建与阶段兼容校验

- [x] 2.1 实现同阶段父数据集预检，校验 canonical schema、登记可用性、训练角色和 holdout/diagnostic 祖先隔离；针对 CPT/SFT/Preference/GRPO 增加正反例测试并记录用户可读冲突信息。
- [x] 2.2 实现声明顺序拼接与确定性无放回加权采样，记录预算、权重、seed、归一化及余数分配；为零/负/非有限权重、超预算、空输入、seed 重放和样本重复编写测试，并在数据工作台文档中说明两种策略。
- [x] 2.3 实现原子化组合构建：临时输出、父摘要前后校验、内容摘要、来源计数、构建 run 及失败清理；测试失败时不产生半登记 dataset，且重复构建不覆盖既有版本。
- [x] 2.4 为 Preference 与 GRPO 增加专属组合约束：Preference pair 不可拆分且保留排序证据；GRPO reward/reference 契约必须一致且可执行；添加不匹配阻断测试并更新阶段数据契约说明。

## 3. 组合产物质检与递归血缘

- [x] 3.1 将组合产物接入现有阶段质量 profile，生成绑定输出摘要的独立质量报告，包含来源覆盖、组合后分母、跨来源重复和检查适用状态；测试父级 PASS 不会被继承为组合 PASS，更新质量报告文档。
- [x] 3.2 将组合登记接入现有 lineage 查询/校验，递归解析多父级及构建 run；覆盖缺父级、摘要不匹配、循环引用和父级新版本场景，验证旧 lineage 校验仍通过并更新 lineage 文档。

## 4. 数据工作台组合体验

- [x] 4.1 增加组合预检与构建 API，明确区分可编辑 plan 和不可变已登记版本；为阶段不匹配、隔离角色、父摘要变化、策略参数错误及成功结果增加 API 测试，并更新 API 使用说明。
- [x] 4.2 在统一数据模块的工作台增加多选父数据集、拼接/加权采样配置、预检摘要与显式构建确认，复用数据集详情的质量/血缘入口；组件测试验证冲突可解释、未执行不产生 artifact、构建后可查看报告，并更新中文操作指南。

## 5. Launcher 与端到端验证

- [x] 5.1 让各阶段 Launcher 选择已登记组合 dataset ID，预检显示规模、策略、父数据集和质量证据；创建新组合时携带阶段/返回位置导航到同一数据组合视图，登记后预选新 ID 返回且不自动启动；测试交接、单 dataset ID 消费与旧单数据集流程，并更新 Launcher 文档。
- [x] 5.2 增加四阶段端到端集成测试：组合数据集登记、质量报告绑定、训练 run 引用单一组合 ID、lineage 递归回溯全部父级；运行相关回归测试及 OpenSpec strict validation，记录验证命令和结果。

## 验证记录

- `pytest tests/test_dataset_mixture.py tests/test_preference_stage_adapter.py tests/test_lineage_validation.py -q`：45 passed（四阶段端到端用例已在 `test_dataset_mixture.py` 覆盖）。
- `pytest tests/test_training_preflight.py -q`：18 passed。
- `pnpm run test:dataset-mixture`：2 passed；覆盖预检冲突、显式构建前不产生 artifact、构建后质量/血缘详情及训练工作台交接。
- `pnpm run test:dataset-preview` 与 `pnpm run test:policy-editor`：10 passed；`pnpm run build` 通过。
- Ruff 检查通过；`openspec validate dataset-mixture-composition --strict --json` 通过。
- 全仓 `pytest -q` 本次结果为 678 passed、7 skipped、4 failed；失败均在 `tests/test_calibration_registration.py`，集中于 calibration 候选 exploratory preflight 缺少 DataPolicy 兼容分支；与本 change 的组合数据集/训练 Run 路径无关。该独立问题未在本 change 中扩大处理范围。

## Workflow follow-up

- 通过评审后使用 OpenSpec apply 实现；全部任务完成并验证后再归档变更。
