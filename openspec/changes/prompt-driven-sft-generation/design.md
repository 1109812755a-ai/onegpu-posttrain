# Design

## Context

现有 `OrdinarySFTPlan` 以 `source_dataset_id` 和数据摘要为必填字段；workflow 从数据行抽取问题后才构造 provider prompt。生成 job、候选审核和登记接口已支持异步执行，但注册器默认新 manifest 至少有一个父 dataset。prompt-only 需要新增独立输入与根 lineage，同时尽量复用既有 provider/job/review 机制。

## Goals / Non-Goals

**Goals:**
- 在 SFT 普通生成入口并列提供“参考数据集”和“按主题生成”。
- prompt-only 输入仅包含有界主题、能力点、难度和可选约束；固定系统指令与输出协议由项目控制。
- 将无 reference 的解答诚实标记为语义未验证，支持人工审核后登记和下游 dataset ID 消费。
- 公开 lineage 只含摘要和身份；执行前页面不得发起 Provider 调用。

**Non-Goals:**
- 不提供任意 system prompt、任意 JSON schema、任意路径或浏览器密钥输入。
- 不宣称 LLM 新生成题目/解答已被证明正确；本期不引入数学证明器或外部 verifier。
- 不把 prompt-only 复用为 Preference、CPT 或 GRPO 数据生成。

## Decisions

1. **与 seeded generation 使用不同 plan 类型，共用受控执行设施。** 新增 prompt-only plan/bundle，固定 `mode=prompt_only`，绑定 prompt 字段、默认模板/协议、provider profile 摘要、quality policy 摘要、目标和预算。避免将缺失来源伪装成空字符串塞入 `OrdinarySFTPlan`，同时保留共同的 Provider adapter、异步 job 状态、审核预览和登记流程。

2. **输入为结构化生成意图，而非自由完整提示词。** UI 收集主题、能力点、难度和可选附加约束；server 负责长度限制、模板渲染与固定格式要求。这样用户能指定训练场景，又不易覆盖输出/安全契约。原始生成意图留在 workspace 内受限 bundle，manifest 与 API 只暴露其摘要。

3. **provider 响应与 canonical SFT 记录分层。** 专用候选协议要求生成题目文本、解题内容、最终答案；本地以 execution ID 和候选序号分配稳定 record/problem ID，再映射至 SFT canonical。候选解析/格式失败被统计为拒绝，不接受模型给出的任意 ID 来混淆记录身份。

4. **无父数据集的 lineage 是合法根，而非缺证据。** 登记 dataset manifest 使用空 `parents`，`source_evidence` 明确记录 `source_kind=prompt_only`、execution/bundle ID、prompt/template/protocol 摘要、provider profile/model 摘要和输出摘要。记录不得伪造源 dataset/record IDs；处理与质量 run 仍产出真实 lineage。

5. **正确性与格式分开报告。** Prompt-only 输出接受 schema、空值、重复、长度等可执行检查；没有独立 reference 时 correctness 为 `not_independently_verified`，不得沿用 seeded 模式从 source comparison 得到的结论。UI 在人工确认前不能登记，并展示未验证限制。

6. **小步复用与迁移。** generation job 通过受信 plan ID 区分 seeded/prompt-only；共享预览/显式登记 API 根据 execution result 的模式解析已验证产物。已有 plan、job 和 manifest 保持可读；新模式不修改既有 run 的 lineage。

## Risks / Trade-offs

- [无 reference 时错误解答可能通过结构检查] → 状态保持未验证，强调人工审核，不能标为 quality pass。
- [用户附加约束可能很长或含敏感内容] → 设置长度上限、不写进公开日志/manifest；用户确认后才发给 Provider。
- [无数据集父项易被误认为血缘缺失] → manifest 与 UI 明确显示 prompt-only 根来源和 prompt hash。
- [与 seeded 共用 job 增加分支复杂度] → 单独 plan/workflow 模块并用 fake Provider 端到端测试两种模式互不串线。
