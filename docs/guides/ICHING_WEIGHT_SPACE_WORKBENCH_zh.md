# 《易经》权重空间实验工作台

本工作台把论文中的“语言模型读取并写入权重”方法用于本项目的《易经》客观题。它固定使用 `Qwen/Qwen3.5-0.8B` 作为被编辑模型，百炼 `qwen3.8-max` 作为编辑提议模型。任务和数据与论文原实验不同，81 题结果只能说明该题集上的表现。

## 环境和固定身份

- 在 Windows 的 WSL `Ubuntu-22.04` 中运行 Docker，并确保 `docker run --gpus all` 能看到 NVIDIA GPU。
- 推理镜像固定为 `hiyouga/llamafactory@sha256:2167de797ca005b1c9b013820e0264db17b799945588cdfe11e9d78cc42576a4`，包含 PyTorch `2.6.0+cu124`、Transformers `5.8.0`。主项目的 Transformers `<5` 环境保持原有版本。
- 被编辑模型固定为 `Qwen/Qwen3.5-0.8B` revision `05f056712f8c582c8708395156518d963b46fb29`。注册模型 ID 为 `model-89d12ba9eb8fb14d6eb6`；处理器模板 SHA-256 为 `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`。
- 练习集为已登记的 `ds-eval-upload-14aee30e5c7ec6ec0959`，300 题；最终集为 `ds-eval-upload-54ca42ca021d295d8390`，81 题。工作台在调用前核对登记信息与内容摘要。
- 模型加载使用 `AutoProcessor` 和 `AutoModelForMultimodalLM`。推理使用处理器聊天模板、贪心生成、`max_new_tokens=16`，评分使用 `choice-accuracy-v1`。

先在本机完成 Docker 镜像和模型快照准备，再以启用 Launcher 的 loopback 模式启动项目 UI。打开“训练与评测 → 权重空间实验”，或直接访问 `?page=weight-space`。远程模式下该 API 不可用。

## 百炼凭据和费用确认

编辑器配置在 `configs/synthetic/provider.dashscope-qwen3.8-max.yaml`，使用本地环境变量 `ONEGPU_SYNTHETIC_API_KEY`；可沿用项目的 `.env.local` 凭据加载方式。浏览器只看到凭据是否可用，不接收密钥值。

点击“检查本轮调用”只生成一次性确认凭据，不调用百炼。工作台披露发送内容和上限后，勾选确认框并点击“确认并开始本轮”才会发起请求；确认凭据 5 分钟后失效。每轮最多 4 次百炼调用、8 次工具调用、每次 512 输出 token，最长 120 秒。当前配置没有可靠价格，页面显示“未知”；实际费用以百炼账单为准。

每轮从练习基线中选取最多 20 道错误题，把题目、正确答案及模型原始输出发给百炼。模型可以请求读取最多 32 个张量元素；读出的数值也会返回百炼。81 道最终题的题干、答案、逐题预测不用于编辑请求。工具只接受已登记张量的有界读取和数值编辑，不执行模型提供的代码。详细限制见 [张量工具边界](WEIGHT_SPACE_TOOLS_zh.md)。

## 候选、评测与回退

一个有效编辑创建独立的候选检查点，并登记父模型摘要、编辑元素、修改前后数值和候选摘要。候选先在 300 题练习集评测。练习报告完成后，保留状态的候选可手动运行 81 题最终评测；已丢弃的候选需要先执行“恢复”。基线是同一 Qwen3.5 模型在 81 题上的 `33/81`，而不是旧 Qwen3-0.6B-Base 的结果。

实验文件位于 `runs/weight-space/exp-weight-space-baseline01/`。每个评测目录包含 `eval.json`、`evaluation_lineage.json`、`evaluation_run.json`、`choice_metrics.json` 与逐题预测；Runs 页面可查看已完成报告。工作台只把完成的报告显示为分数；运行中和失败状态不计分。比较时必须匹配数据集摘要、样本覆盖、模型版本、处理器、镜像、推理配置和评分协议。

要回退，在工作台丢弃候选并继续使用登记的原始 Qwen3.5 模型。丢弃不会删除候选或历史报告，以便核对实验来源。一次最终集评测后反复挑选候选可能使结果受该题集影响，应把最终分数当作这 81 题的观察值。
