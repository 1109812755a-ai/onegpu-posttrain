# 多阶段真实 Provider 与 GPU 验收清单

本清单用于把软件契约测试与真实运行验收分开记录。fake Provider、CPU fixture、预检通过和已有 Run 均不能代替本清单中的新一次真实验收。每次只验证一个明确的数据/模型/配置组合；付费 Provider 请求、GPU 训练或 GPU 评测须逐项获得用户确认后才执行。

## 每次验收记录

| 字段 | 必填内容 |
| --- | --- |
| 验收编号与时间 | 独立 acceptance ID、开始/结束时间、操作者 |
| 阶段与路径 | CPT / SFT / DPO / SimPO / GRPO；普通生成或定向补数 |
| 数据身份 | Provider 生成 execution ID（若有）、build/quality report ID、dataset ID、manifest SHA-256、canonical SHA-256；训练组合则记录每个父 dataset ID/hash 与 provenance |
| 模型身份 | 父 model ID/hash；训练后 checkpoint/merged model ID/hash；Provider 声明模型、可得 revision 和身份核验状态 |
| 计划与配置 | targeted plan ID/version（如适用）、recipe/profile ID/version、resolved config SHA-256、评测协议/version/config SHA-256 |
| 许可与证据 | 每个来源的 URI/name、revision 状态、license 状态、使用确认、质量状态、验证器/version、已测分子/分母及 limitation |
| Provider 预算与费用 | 请求上限/实际请求数、输入/输出 token（可得时）、重试、报价/计费来源、实际费用或明确 `unknown`；不得把输出 token 估值写成总费用 |
| 设备与训练 | GPU 型号、驱动/CUDA/PyTorch/ms-swift 版本、开始/峰值/结束显存（可得时）、step 与耗时、退出码、Run ID、错误/中断说明 |
| 阶段评测 | 独立 evaluation Run ID、被评模型 ID/hash、eval dataset ID/hash、协议/config、指标 numerator/denominator、状态、日志和 lineage 路径 |
| 结论 | `passed` / `failed` / `partial` / `not_run`；仅描述观察结果，不推断数据导致模型提升 |

## 阶段路径台账

以下每一行均须分别覆盖普通与定向路径。定向路径还须核对精确 plan/version 与 train_seed 来源，确认 diagnostic/final holdout 的正文、答案及已知派生祖先未进入生成输入或训练数据。Preference 阶段分别记录 DPO 与 SimPO 训练分支；GRPO 的 prompt-only 模型生成候选没有 reference 时只能记录候选生成，不能记作可训练数据或训练验收通过。

| 阶段/路径 | Provider 生成 | 候选审核/登记 | 训练 Run | merge/model | 阶段评测 Run | 当前真实验收 |
| --- | --- | --- | --- | --- | --- | --- |
| CPT 普通 | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | 待用户逐项授权 |
| CPT 定向 | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | 待用户逐项授权 |
| SFT 普通 | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | 待用户逐项授权 |
| SFT 定向 | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | 待用户逐项授权 |
| DPO 普通 | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | 待用户逐项授权 |
| DPO 定向 | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | 待用户逐项授权 |
| SimPO 普通 | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | 待用户逐项授权 |
| SimPO 定向 | `not_run` | `not_run` | `not_run` | `not_run` | `not_run` | 待用户逐项授权 |
| GRPO 普通 | `not_run` | `blocked_without_reference` | `not_run` | `not_run` | `not_run` | 待可信 reference/reward 证据及用户授权 |
| GRPO 定向 | `not_run` | `blocked_without_reference` | `not_run` | `not_run` | `not_run` | 待可信 reference/reward 证据及用户授权 |

`not_run` 表示本次没有执行，不表示失败或通过。许可为 `not_provided` / `reported` 时先呈现并要求使用者核实；本清单不会把许可状态自行升级为 `verified`。若资源、依赖、权重、Provider 凭据或 evidence 不足，应保留 `not_run`/`blocked` 并写明具体缺项，不以旧结果、mock 或人工猜测填充。
