# 客观题准确率评测协议

`choice-accuracy-v1` 面向显式登记为 `stage=eval`、`dataset_role=evaluation_only` 的客观题 suite。支持单选、多选和判断，不从测试集自动生成训练数据。

## 计分规则

- 单选：只解析第一行中的一个有效选项标签；预测标签与标准答案完全相同才正确。
- 多选：解析一个或多个标签；预测与标准答案的集合完全相同才正确，不给部分匹配分。
- 判断：将 `true/false`、对/错、正确/错误及 ✅/❌ 归一化后精确比较。
- 无法解析、标签超出选项范围、空输出均记为错误，并单独增加 `parse_failures`；仍计入该题型和 overall 的分母。
- 每个题型输出 `correct`、`denominator`、`parse_failures`、`accuracy` 和 `status`。未出现的题型为 `not_measured`，不会伪造 0 分。

运行固定使用贪心生成（temperature=0、max_new_tokens=16），prompt 要求仅输出答案。每次 Evaluation Run 保存逐题预测、`choice_metrics.json`、汇总 `eval.json`、`evaluation_lineage.json` 与 `evaluation_run.json`。报告同时记录模型 ID、来源训练 Run、suite dataset ID、manifest/canonical SHA-256、样本数、配置摘要和 evaluator 版本。

## 可比性与隔离

preflight 和持久化结果使用同一 `comparability_key`，由协议版本、canonical suite SHA-256、样本数和评分版本共同确定。模型结果仅在 key 相同时可直接比较；题目内容、覆盖数或协议变化都会改变 key。用户可将不同 suite 的结果分别报告，但本协议不自动声称其横向可比。

数据登记时固定 `evaluation_only` 角色。训练数据选择/预检只接受阶段允许的训练角色；阶段评测预检及 CLI 启动时重新读取 manifest 与 processing evidence、重算 canonical SHA，并以目标训练 Run 的 dataset ID 和可用 digest 做重叠检查。发现身份或摘要重合时 fail closed。该检查只能识别可观察到的完全相同内容，不代替近重复/语义污染审计；未实现的污染检测保持 `not_measured`。

## 阿里百炼大模型评判

客观题评测可在 Runs 中选择 `阿里百炼大模型评判`，也可保持默认的精确匹配。LLM 评判使用独立协议 `choice-llm-judge-v1`，不会覆盖或与 `choice-accuracy-v1` 自动合并比较。

当前固定使用 `configs/synthetic/local/provider.dashscope-qwen3.6-plus.local.yaml` 中声明的 DashScope 兼容 profile（`dashscope-qwen3.6-plus-local` / `qwen3.6-plus`）。评测服务从 `.env.local` 或进程环境读取该 profile 声明的 `ONEGPU_SYNTHETIC_API_KEY`；密钥不进入浏览器请求或 Run 产物。缺少凭据时预检会阻止启动。

每题单独向百炼发送题目、选项、标准答案和被测模型原始回答；本地题目 ID、文件路径和凭据不会发送。预检显示模型和请求数，用户显式点击启动即确认外发这些评测内容。API 响应按 `correct`、`incorrect`、`undetermined` 结构解析；超时、无效响应和 Provider 错误单独记录。准确率分母只包括已判定的 correct/incorrect，报告还显示判定覆盖率、无法判断和 Provider 错误数量。超时后可能已在 Provider 端处理的请求不会静默重试。

LLM 评判结果记录 provider/model、revision（若可得）、提示词版本/摘要、解码配置和请求 ID。模型别名及 LLM 判断可能变化，因此这些分数应与精确匹配分数分开解读；它们不构成标准答案或模型回答正确性的独立证明。
