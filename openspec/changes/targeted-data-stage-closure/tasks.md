# 任务清单

## 1. 工作流契约与共享执行证据

- [x] 1.1 为 CPT、SFT、Preference、GRPO 建立带版本的定向计划 schema，定义阶段专属字段、不可变计划版本、目标与预算语义、训练/诊断/holdout 数据角色；验证 schema 测试会拒绝跨阶段字段并保留历史版本。
- [x] 1.2 建立共享的工作流状态/读取模型，将计划版本关联到真实生成/导入、质量、登记、训练、合并和评测 Run ID；验证持久化与重载能保留 partial、failed、blocked、completed 状态且不伪造产物。
- [x] 1.3 增加阶段能力/预检响应，检查所需数据 schema、父模型、配方、验证器/reward 契约、硬件输入和评测支持；验证不支持阶段会返回可执行的阻塞原因，且不会启动 Provider 或训练进程。
- [x] 1.4 增加从计划 → 数据构建/质量 → dataset ID → 训练 Run → model ID → 评测 Run 的血缘完整性检查，并编写共享身份说明；验证既有 SFT 生成产物仍可读取且严格血缘测试通过。
- [x] 1.5 建立不可变训练配置契约，支持受控内置配方、边界受限的自定义参数和一个或多个已登记训练数据集，并与定向生成种子分开；将多数据集选择固化为已登记 mixture。验证顺序/hash、采样或合并策略、配方边界、预检/Run/模型血缘，以及拒绝任意命令/路径。

## 2. CPT 定向语料切片

- [x] 2.1 定义 CPT 语料 canonical 输入/输出和领域覆盖计划字段，不要求 prompt/answer pair；验证有效/无效 JSONL fixture 与角色/holdout 隔离测试。
- [x] 2.2 实现有界 CPT 语料生成/导入、去重、污染检查、阶段质量报告和显式 partial 登记；验证 fake Provider/offline 测试会将报告和 manifest 绑定到真实内容 hash，且不会把未经验证的事实性标为已验证。
- [x] 2.3 将已登记 CPT 数据集接入现有 CPT 配方/训练 CLI，提供明确的基础模型预检、实时状态/日志和失败恢复；验证 fake/CPU 预检及 Run 血缘测试，并说明真实 GPU 证据需单独验收。
- [x] 2.4 将 CPT Run 产物接入可用的领域 PPL 和下游评测协议，生成前后资源/效果报告；验证缺失指标保持 `not_measured`，不推断不受支持的场景分数。

## 3. SFT 定向切片

- [x] 3.1 在定向计划 ID 下复用仅 prompt 和基于种子的 SFT 生成，将诊断目标场景映射到安全训练种子，并保留独立正确性状态；验证候选保留 seed/prompt/execution 祖先且不会读取诊断集/最终 holdout 记录。
- [x] 3.2 登记经审核的 SFT partial/full 数据集并接入 SFT Launcher 预检、训练、评测和效果报告；验证离线端到端测试会将同一 dataset ID 串联到 Run/model/evaluation lineage，并报告真实计数/资源。

## 4. Preference 与 DPO/SimPO 切片

- [x] 4.1 实现同题候选生成和 pair 构造，保存独立答案/排序证据、拒绝原因及可用 pair 数；验证双正确、双错误、未知、重复和预算耗尽 fixture 不会被伪造成有效 pair。
- [x] 4.2 增加 Preference 质量审核/登记和 DPO/SimPO 预检，让两者消费同一已登记 Preference 数据集及相同 SFT 父模型身份；验证成对 Run 测试可发现 dataset/parent/config hash 不一致，并保留独立 DPO/SimPO Run ID。
- [x] 4.3 将两个训练分支接入共享评测/比较，包含 Preference 指标、任务指标、保护指标和成本；验证离线匹配配置测试能区分缺失/失败结果与数值零，并将不匹配的前后评测标为观察结果。

## 5. GRPO 定向切片

- [x] 5.1 定义带版本的可执行 reward/reference 契约和定向 prompt 格式；验证契约 fixture 执行具确定性、暴露不支持的输出，并在生成/登记前拒绝 holdout 祖先。
- [x] 5.2 实现 GRPO 候选生成/导入、reward 证据、质量报告、partial 登记和训练预检；验证缺少/不可执行 reward 时不能启动正式训练，并以真实分母报告 reward 方差、验证器覆盖、长度和截断。
- [x] 5.3 将 GRPO 配方执行接入任务评测和效果报告，使用显式配置的目标指标/保护指标（不得猜测阈值），并保留 Run/model/data 血缘；验证 fake/CPU 集成覆盖失败、OOM、无 checkpoint 和成功路径，且不伪造模型产物。

## 6. 前端工作流与集成验收

- [x] 6.1 将 SFT 专属定向区改为阶段化工作流，展示当前步骤、阻塞原因、可恢复任务、候选/审核数量、dataset ID、配方选择/边界受限自定义参数、Run 日志、血缘和评测；支持选择多个兼容训练数据集及明确组合策略，并通过 UI 测试覆盖阶段切换、刷新恢复、窄屏和显式确认。阶段能力卡增加直达对应训练与评测 Launcher 的入口。
- [ ] 6.2 增加跨阶段计划历史与效果页，对比目标指标、保护指标、协议/覆盖、训练预算、耗时及显存；验证渲染能区分 unknown/unsupported/not-run 与数值零，并区分观察结果和匹配 A/B。
- [x] 6.3 编写中文指南，覆盖 CPT、SFT、Preference/DPO/SimPO、GRPO 工作流、Provider 配置、reward/reference 前置条件、partial 数据、holdout 安全和 4060 资源预检；验证离线 walkthrough 可按文档执行。2026-10-25 使用项目自有 fixture 执行 `scripts/targeted_data_offline_example.py` 成功；结果明确为未生成/未训练/未复测、Provider 调用 0、未联网，未将离线走查描述为真实训练证据。
- [ ] 6.4 运行各阶段 fake Provider/CPU 端到端测试、完整后端/前端测试、compile/type/build 检查、血缘校验和 `openspec validate targeted-data-stage-closure --strict`；除非真实运行，否则将 Provider/GPU 验收记录为待完成。
