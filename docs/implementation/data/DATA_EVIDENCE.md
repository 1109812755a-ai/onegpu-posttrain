# 数据处理轨迹与静态质量报告：首个开发切片

实现日期：2026-10-16。沿用现有 `build-openr1` / `build-finemath` CLI；核心安装使用现有 Pydantic，不增加第三方数据质量框架依赖。

阶段化审核、用途分类、可选独立答案核验与 Mini readiness 接入现已实现；详见[阶段数据质量审查](QUALITY_REVIEW_zh.md)及[build-02 审查记录](MINI_BUILD02_REVIEW_zh.md)。审查结果仍不得与模型效果或清洗因果收益混为一谈。

## 已实现

OpenR1 输出 11 个有序聚合步骤：规范化、精确去重、去污染、generation 完整性/解析检查、数学核心筛选、problem split，以及 SFT / Preference / GRPO / Holdout / Preference Eval 五条分支。分支顺序描述构建 DAG，不表示串行训练。

FineMath 输出 5 个步骤：非空文本检查、document split、二次去污染、tokenizer chunking、token budget 选择。输入输出单位不同的步骤不计算保留率；达到配额而未选中的数据记为 `unselected`，不算质量拒绝。

```text
<output>/
├── canonical/                         # 既有 canonical 文件
├── manifests/<stage>-smoke-v1.yaml     # 既有文件名兼容，内部 dataset_id 按内容/配置生成
└── reports/
    ├── processing_report.json
    ├── processing_report.md           # 步骤汇总表及按步骤展开的证据
    ├── quality/<stage>.json
    ├── quality/<stage>.md
    └── <source>_data_report.json/.md   # 保留原报告，并增加关联信息
```

Pydantic 契约位于 [`data/evidence.py`](../../../src/onegpu_posttrain/data/evidence.py)。`ProcessingReport.processing_steps` 是步骤列表；各步骤使用 `operation`、`operation_version`、`kind`、带单位 input/output counts、首次拒绝原因、非质量排除原因、`metrics_before/metrics_after` 和局限说明。可按单位/子集语义对账的步骤由 schema 校验计数守恒。

manifest 新增可选的 `processing_report` 和 `quality_report` 引用；`path` 相对于 manifest 所在目录，另附报告 ID、SHA256 和 `relative_to: manifest`。旧 manifest 可以继续读取，新字段默认空。文件名是定位信息，内部 `dataset_id` 才表示数据集身份：它关联 canonical 内容摘要、阶段、来源/版本、tokenizer、构建配置和污染索引指纹。相同输入/配置重建得到相同 dataset ID，每次调用获得新的 build/report ID。建议为每次正式构建使用新的 output 目录，保留历史报告与其 hash。

污染索引缺失或为空时步骤显示 `skipped`，不宣称“污染率为 0”；可用索引记录方法、阈值、shingle 大小、条目数和规范化内容指纹。报告只覆盖实际处理的源数据前缀，`stop_reason` 区分 `targets_met`、`max_raw_rows`、`source_exhausted`。

## 当前质量检查范围

每个最终数据集检查 canonical schema、重复 record IDs 和本次构建的训练/评测 ID 全局交集，并统计记录数、可用 token 长度和来源分布。SFT/Preference 单独报告输出 metadata 与上游数学正确性标签的准入一致性。

OpenR1 当前使用源数据的 `correctness_math_verify` 标签和 complete/finish-reason 信息，**没有重新运行 Math-Verify**。因此独立语义正确性、近重复质量、GRPO rollout 校准等没有测量的项目保持 `not_measured`。空集合比例的 `value` 为 `null`，分母为 0，不报告成 0% 或 100%。过滤后 100% 的完整/可解析率是筛选条件下的结果，不是独立验证结论。

结构检查失败时质量报告为 `FAIL`，CLI 返回未达验收的退出码 3。结构检查通过时总体 gate 为 `NOT_EVALUATED`，因为完整语义门槛尚未校准；单项检查保留 PASS/FAIL。现有目标数和切分检查仍参与 CLI 验收。

## 验证与后续

2026-10-16：完整 CPU suite **65 passed**。新增 11 个测试覆盖两条流水线报告/schema/hash 引用、确定性数据身份、前后比例、空集合、扫描上限、配额排除、重复 IDs、污染索引指纹与旧 manifest 兼容。此次数据/报告模块和相关测试 Ruff 通过；全仓仍有 36 项待清理 findings（包括 CLI 原有项），不能视为全仓 lint 已通过。

当时的下一步是扩充带 method/coverage 的阶段质量证据、在正式 Mini 上检查真实分布，并完善 run/model/evaluation 血缘。后续 Lineage v1 与策略实验切片已完成部分接口和 CPU 验证；仍需补足官方 Mini 数据及真实受控训练结果。当前切片没有执行新的 GPU 训练，也未改写已有真实 Smoke 产物。

## 2026-10-18 策略与模型效果实验切片

OpenSpec [`data-policy-experiments`](../../../openspec/changes/data-policy-experiments/proposal.md) 增加 SFT/Preference 策略身份、固定问题级 pool/split、offline preview/build、预登记和 `pipeline_effect` / `controlled_quality_ablation` 报告。数据质量、数量和覆盖先独立呈现；模型分数只通过已登记的 dataset/run/model/eval、配置、预算、seed 和评测协议精确关联。真实分数保留在原始 run 证据中，不匹配时不生成可比 delta。

新增 Data 页面“策略对照”：默认只读，使用 `serve-ui --enable-policy-editor` 才能用同源本地端点做最多 1,000 行/30 秒的离线预览和保存到专属新草稿目录；页面可只读查看比较报告。中文指南为 [`POLICY_EXPERIMENT_UI_zh.md`](../../guides/POLICY_EXPERIMENT_UI_zh.md)，范围与 UX 约束见 [ADR-002](../../design/frontend/ADR-002-policy-editor.md)。

CPU 完整示例命令：`.venv/Scripts/python.exe scripts/data_policy_offline_example.py`。2026-10-18 本机运行成功：A/B 各 2 个项目自有 fixture 样本、同一来源池和 split，实验预登记成功；报告为 `NOT_EVALUATED`，无 GPU 训练、外部 API 或下载。这验证协议链路，不验证数据语义质量或模型收益。真实 Mini assessment 仍为 `NOT READY`，参见 [`DEVELOPMENT_PLAN.md`](../../project/DEVELOPMENT_PLAN.md)。

## dev7 真实 OpenR1 扫描验证（2026-10-17）

通过本地代理 `127.0.0.1:7897` 对 pinned OpenR1 revision 做了只读 streaming 验证。先读取 1,000 行确认数据流正常；随后使用新的输出目录、5,000 行显式上限运行完整构建流程。报告位于 [`openr1_data_report.json`](../../../data/processed/dev7/openr1-real-check/reports/openr1_data_report.json)，处理步骤证据位于 [`processing_report.json`](../../../data/processed/dev7/openr1-real-check/reports/processing_report.json)。

该执行实际处理 5,000 raw rows，`stop_reason=max_raw_rows`，CLI 以非零状态退出（未满足配额，符合预期）。目标与产出为：preference train **0/200**、preference eval **1/50**、SFT **8/200**、GRPO **200/200**、holdout **100/100**、alignment core **1,716/300**。主要首拒绝计数为 `preference_missing_valid_pair_side=1,716`、`preference_eval_missing_valid_pair_side=585`、`no_verifier_correct_trace=1,541`；train/eval overlap 为 0。报告因此定位为偏好候选有效 pair 产率短缺，而非代理连接失败。

此结果只覆盖 pinned source 的前 5,000 行，**不能**推断整个 OpenR1 数据源已耗尽，也不能视为 200/50 preference 目标已完成。更大上限的无进度扫描因耗时过长被安全中止，未写出报告；本次有界报告保留了真实源前缀的质量/拒绝证据。测试 fixtures 另验证目标驱动持续扫描、上限终止、源耗尽短缺和 CLI 非零结果。此次复核未发现 builder/CLI 合同缺陷，因此没有放宽过滤门槛或改变数据语义。
