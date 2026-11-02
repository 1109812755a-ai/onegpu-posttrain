# 定向补数阶段工作流与血缘

## 身份如何串联

一次定向补数方案使用不可变的 `plan_id + plan_version`。修改目标、数据或预算时保存新版本，不回写旧方案。每次实际生成/导入、质量审查、登记、训练、合并和评测都保留各自的 `run_id`（生成任务还保留 `execution_id`）；方案身份不是运行身份。

```text
plan/version
  -> generation/import run
  -> quality run
  -> registered dataset_id + content hash
  -> training run + input dataset_id + parent model_id
  -> produced model_id
  -> evaluation run + model_id + protocol
```

`dataset_id` 绑定阶段 canonical 数据及内容摘要；训练必须引用实际登记的数据集和父模型；评测必须引用实际训练/合并产出的模型。状态文件只保存已经发生或明确阻塞的证据。失败运行可以有 `run_id`，但没有真实 checkpoint 时不能出现模型产物；partial 只列出磁盘上已核实的产物。

## 四阶段数据不是同一种格式

- CPT：领域连续语料，目标量通常以 token 计；不得伪装成 prompt/answer。
- SFT：指令与回答，目标量以可用 records 计。
- Preference：同一问题的 chosen/rejected 候选及排序依据，目标量以有效 pairs 计，可供 DPO 和 SimPO。
- GRPO：训练 prompts 及版本化、可执行的 reward/reference 契约，目标量以 prompts 计。

期望数量与最大候选数、请求数是不同概念。预算耗尽时可以审查、登记有效的非空 partial 数据；数量不足本身不等于质量失败，数量达标也不等于质量通过。无法独立验证的事实/答案保持 `not_measured` 或 `unverified`。

## 当前能力状态

本地 UI 的只读能力接口为 `GET /api/targeted/stages`，可查看各阶段的数据 schema、父模型阶段、配方、验证/reward 契约、GPU 观测、评测支持和阻塞原因。定向工作台展示这四类能力状态；能力清单不是运行完成证据。

当前可复用的离线能力包括：CPT 有专门语料 schema、定向语料生成与注册、注册数据训练 preflight/CLI；SFT 的生成审核注册可选择已登记 shared plan，并由 Mini SFT Launcher 将 plan ID/version、实际 dataset ID、训练 run 与评测 run 串联；Preference 可从同题候选中仅对“独立验证正确/错误”的一对构造记录并登记 partial 数据；GRPO 有版本化、代码摘要绑定的本地 reward contract，可从带可信 reference 的 train seed 建立 GRPO prompt 数据并检查 rollout reward 统计。GRPO 阶段评测现在可为已绑定计划的已完成训练生成效果摘要；缺少匹配起点评测、计划目标指标或协议证据时会显示不可比较/未测量，不推断为提升。

## 训练入口：多数据集与受控自定义参数

在“训练与评测”中先选阶段、已批准的训练方案和父模型，再勾选一个或多个同阶段、已登记的数据集。多个数据集不能直接被 Mini SFT 固定训练命令作为多个路径传入；应使用 Launcher 的“组合所选数据集”控件先执行组合预检，再明确选择拼接或加权抽样并登记组合数据集。训练随后只消费新生成的一个组合 `dataset_id`，其 lineage 保留所有父数据集 ID、内容 hash、选择顺序、随机种子及组合策略。数据集组合会形成新的质量报告，父级质量/许可证结论不会自动升级。

“训练参数”展开项只开放当前方案声明的字段与取值范围。改动后必须重新预检，预检会固定解析后的配置摘要；不接受上传任意配方文件、shell 命令或本机路径。当前 Mini SFT 的真实 Web 训练入口仍使用批准的固定配方与固定父模型，其开放参数仅限方案白名单。CPT、DPO、SimPO、GRPO 的 Web Launcher 尚未全部接通执行 adapter；选择这些阶段时如仅显示兼容性预检或 blocker，不要据此认为已经启动训练，可使用对应受控 CLI 流程。

## 四阶段执行边界

| 阶段 | 训练数据 | 必要证据 | 当前注意事项 |
| --- | --- | --- | --- |
| CPT | 连续领域语料 / token | 来源与内容 hash、去重/污染和语料质量报告 | 使用 CPT 专用 schema 与受控 recipe；不能把问答格式当语料覆盖证据。 |
| SFT | prompt / response 记录 | canonical 校验、质量/许可提醒、父模型与 dataset lineage | Mini SFT 可从一个已登记数据集训练；多选时先建立组合数据集。未核验许可仍须用户显式确认。 |
| Preference（DPO / SimPO） | 同题 chosen / rejected pair | 独立排序/验证依据、pair 完整性、相同 SFT 父模型身份 | DPO 和 SimPO 是不同训练算法；可共享合格 Preference dataset，但各自 Run 独立。Web adapter 未接通时不要从预检页面声称已训练。 |
| GRPO | prompt + 可执行 reward/reference | reward 实现和协议摘要、reference 覆盖、rollout 真实分母 | 缺可信 reward/reference 时阻断正式训练；未生成 rollout 指标显示 `not_measured`，不补成 0。 |

每阶段均遵循：选择阶段与方案 → 选择父模型和训练数据 → 检查/预检 → 显式确认训练 → 查看真实 Run 和产物 → 使用匹配的阶段评测协议复测。阶段切换不会把尚未执行的计划变成成功记录。CPU/fake 测试只证明软件契约与血缘，不代替真实 Provider/GPU 验收。

## 按阶段继续操作

- [CPT 阶段工作流](CPT_STAGE_WORKFLOW_zh.md)：语料登记、质量审查、Base 模型要求、CLI 训练与评测边界。
- [定向合成数据与 Provider 配置](TARGETED_SYNTHETIC_DATA_zh.md)：fake 离线流程、OpenAI-compatible/DashScope profile、密钥环境变量约定、预算/确认、生成与恢复。
- [阶段能力与评测契约](STAGE_CAPABILITIES_zh.md)：Preference/DPO/SimPO 与 GRPO 的 schema、parent/reward/reference 要求、CLI 路径和评测数据隔离。
- [本地 Launcher](LOCAL_EXPERIMENT_LAUNCHER_zh.md)：Mini SFT 网页多数据集组合、受控参数、训练/评测 Run 与失败边界。
- [定向补数 UI 交互走查](TARGETED_DATA_UI_WALKTHROUGH_zh.md)：窄屏、键盘、数据选择、失败提示和刷新恢复的界面验收步骤。

参考机器为 RTX 4060 8GB；这不是每种阶段/序列长度都能运行的保证。每次在训练前使用当前阶段预检核对已登记父模型、数据、配方、依赖与 GPU 观测；历史显存值仅是参考，GPU 快照也不等于 OOM 保证。模型/配方/显存不满足时先降低序列长度或训练预算，再明确确认启动，不要用预检通过推断正式 Mini 验收。

最终效果需通过相同评测协议、数据集版本、样本覆盖和解析器比较。单组补数前后变化是流水线观察；只有共同父模型、匹配训练预算和受控单变量差异时，才可称为匹配对照。目标指标、保护指标和资源成本均应同时保留。
