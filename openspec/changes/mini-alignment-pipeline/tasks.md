# Tasks

## 1. Mini exploratory recipes

- [x] 1.1 为 DPO、原始 SimPO、GRPO-standard 与 GRPO-lowmem 增加独立 Mini recipes，引用 M2 与本地 Preference/GRPO manifest，显式声明 exploratory、实际预算和有限训练步数；增加配置测试确保 SimPO `cpo_alpha=0`、四分支 parent 一致，并确认 Smoke recipes 未改。
- [x] 1.2 补充中文操作说明，列出真实数据量、质量报告限制、可复现命令和失败语义；逐条验证文档所列文件/命令存在，不将数据量写成启动门槛。

## 2. 数据适配与血缘预检

- [x] 2.1 将 canonical Preference/GRPO 数据适配为训练器输入并保留 manifest 身份、内容摘要和行数关联；增加测试验证真实 Mini 输入被选中、Smoke fixture 被排除，缺失/摘要不符时失败关闭。
- [x] 2.2 为四分支增加 lineage/preflight 测试，核对 M2 model ID/hash、数据集 ID/hash/count、resolved config、seed 与 exploratory/review 状态均被记录，且短缺数量不阻止探索运行。

## 3. 实际训练与模型产物

- [x] 3.1 从共同 M2 依序执行 DPO 与原始 SimPO；保存每次训练和合并 run、真实 adapter/merged model IDs、耗时/显存/实际样本暴露量，并确认 4-pair 运行未被误报为官方 Mini。
- [x] 3.2 从同一 M2 执行 GRPO-standard 与 GRPO-lowmem 有界训练；记录 profile、实际 prompt/completion 暴露量、reward/资源测量及成功或失败状态，只登记实际存在的模型输出。
- [x] 3.3 对成功和失败的四分支运行 lineage validator；确认父模型一致、数据/recipe/config 引用可追溯、失败分支不含虚构产物，保存机器可读验证结果与中文阶段记录。

## 4. 统一评测与交付

- [x] 4.1 使用同一 evaluation identity/protocol 对 M2 与可用分支运行受控评测；验证模型、数据版本、样本 ID、解析器和预算一致，未完成模型/指标明确标记而非填零。
- [x] 4.2 生成中文与机器可读 Mini alignment 总结，汇总四分支状态、真实数据规模/质量限制、训练与评测指标、资源、父子血缘，并明确不构成官方 Mini 完成或算法有效性结论。
- [x] 4.3 运行相关单元/集成测试、lineage 校验和 `openspec validate mini-alignment-pipeline --strict`；验证 Smoke 配方与历史输入/产物未被覆盖。

## Workflow follow-up

- 评审后归档本变更；`mini-sft-2048-pilot` 中因 OOM 未完成的任务单独保留，不由本变更自动关闭。
