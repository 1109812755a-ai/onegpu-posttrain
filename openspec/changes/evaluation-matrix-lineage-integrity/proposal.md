# Proposal

## Why

阶段比较矩阵目前可能把 `eval.json`、评测运行记录和评测血缘侧车文件来自不同执行的证据拼在一起；这会让指标看似完整，却无法证明它们对应同一次评测。Mini 已有真实矩阵暴露出 GRPO 评测引用与本地侧车身份不一致，需先修正证据关联，才能可信地验收完整流程。

## What Changes

- 矩阵生成时校验指标文件、评测运行记录和评测血缘的身份及终态一致性。
- 对缺失、失败、身份不匹配或路径不一致的评测保留可用指标，但标为来源不完整/不可验证，不作为成功的完整评测纳入可比结果。
- 增加真实 Mini 评测产物对账与回归覆盖；只从可核实的源产物重新生成矩阵，不改写历史指标或猜测 GRPO 来源。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `unified-model-evaluation`：矩阵必须验证所引用评测文件与运行/血缘身份一致，且区分有效完成结果与不一致证据。

## Impact

影响 `src/onegpu_posttrain/reporting/evaluation_matrix.py`、`src/onegpu_posttrain/lineage_validation.py`（如交叉校验适合放在通用校验器）、对应测试，以及 Mini 的矩阵/阶段总结产物。保持文件式存储和旧评测可读性；不增加数据库、依赖、训练或评测运行，不改动指标值。
