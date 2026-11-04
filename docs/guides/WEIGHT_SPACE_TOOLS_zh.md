# 权重空间实验：张量工具边界

本实验的目标权重是登记的 `Qwen/Qwen3.5-0.8B` 检查点；Bailian `qwen3.8-max` 只负责提出检查和编辑请求。后端只公开 `inspect_tensor` 与 `propose_edit` 两个固定工具，不执行模型提供的 Python、Shell 或文件路径。

## `inspect_tensor`

- 参数：`tensor_name`、从 0 开始的扁平 `offset`、`length`。
- 只接受 `model.language_model.layers.` 下已在 Safetensors 索引登记的张量。
- 每次最多返回 32 个元素；越界、未知张量、非有限值或符号链接权重文件均拒绝。
- 返回张量名称、形状、数据类型、起点和数值切片。

## `propose_edit`

- 参数：`tensor_name`、`offset`、新 `values`、`parent_sha256`。
- 每次最多 32 个浮点元素；必须匹配当前父检查点摘要、现有张量形状及浮点数据类型。
- 单元素新值相对原值的绝对变化不得超过 `0.1`；非有限值一律拒绝。
- 提案通过验证后才可创建独立候选。父检查点只读，候选目录记录父摘要、候选摘要、元素数量和前后数值。

一次编辑器迭代最多调用 Bailian 4 次、使用工具 8 次、运行 120 秒；每次生成最多 512 个输出 token。外发题目和反馈由已登记的 300 道练习题构造，81 道保留题的 ID、题干和逐题结果不得进入请求。最终结果只通过固定的 `choice-accuracy-v1` 评测流程产生。
