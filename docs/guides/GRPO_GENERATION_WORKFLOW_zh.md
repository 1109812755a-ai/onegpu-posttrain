# GRPO 模型生成与可信数据派生

数据工作台 → 数据集 → “模型生成的数据阶段” → “GRPO 题目候选”。本入口的目标是生成题目候选，不会将模型自报答案或 reward 当成可信证据。

## 普通生成

1. 选择“普通生成”，填写数学主题/提示（每行一条），选择已配置且凭据可用的 Provider。
2. 设置请求数、候选数和单次最大输出 token；Web 上限为 100 条请求/候选和 2,048 tokens，候选数不能超过请求数。
3. “准备方案并预检”只校验配置，不调用 Provider。只有点击“确认并生成候选”并再次确认后才会发出请求。
4. 页面保存 execution ID，可刷新后恢复状态；运行中可请求取消，当前进行中的单次远端请求可能仍会完成。任务重启后不会自动重放不确定请求。
5. 完成后可预览候选和错误/拒绝摘要。普通生成没有可信 reference，因此候选显示为不可登记、不可训练。

## 定向补数

选择“定向补数”并挑选已登记 GRPO 计划。服务端会核对计划版本、训练种子 dataset ID/hash 与当前 reward contract，仅把精确 train_seed 记录作为输入；诊断和最终 holdout 不会交给 Provider。请求、候选和输出预算不能超过不可变计划。

生成题目通常不等同于原种子题，系统不会把种子的参考答案错误地复制给新题。因此定向生成的候选同样保持 reference 缺失、不可登记/训练，直到有单独可信的 reference 绑定和验证证据。

## 需要可训练的 GRPO 数据时

如果目标是把已有可信参考答案转换为 GRPO prompts，请回到工作台下方“从可信训练种子派生 GRPO 数据”。该路径仅接受满足来源 revision、逐条上游正确性证据、lineage 和 processing report 校验的 OpenR1 SFT train_seed，并将实际父 dataset ID/hash 与 reward-contract 身份写入登记血缘。来源许可证仍需操作者核实；派生数据质量标记为 report-only，不代表模型 rollout 或训练已完成。

模型生成任务的状态 API 为 `GET /api/synthetic/status/grpo/{execution_id}`，需要本机生成 token。它只返回持久状态，不启动生成、登记、训练或评测。
