# Design

## Context

See `proposal.md` for motivation and `specs/` for observable contracts. 当前定向页面以报告和配置草稿为主，实际运行的生成、登记与训练闭环以 SFT 为主；训练 Launcher 也是固定 Mini SFT 方案。项目已有阶段 canonical dataset registry、阶段质量报告、provider job/恢复、训练 CLI、Evaluation Matrix 和 dataset/run/model/evaluation lineage，适合在其上连接，不适合再造存储或调度系统。

## Goals / Non-Goals

**Goals:**
- 以单一计划身份连接目标、阶段适配器、数据集版本、训练/评测 run 和效果报告。
- 四阶段共享状态、预算、日志、恢复、取消和审阅体验，但保留不同数据语义与训练/评测条件。
- 用户可以逐阶段运行；真实 provider、训练或评测未发生时明确显示，不以 fixture 代替真实证据。
- 所有训练只消费已登记 dataset ID 和固定父模型/配方，不从浏览器接收任意命令或路径。

**Non-Goals:**
- 不增加后台队列、数据库、远程 worker 或跨机器调度。
- 不承诺合成数据一定提升指标，也不把样本量当 quality gate。
- 不允许模型自评充当 Preference 正误标签或 GRPO reward 的独立验证。
- 不修改既有不可变 dataset/run/model/evaluation 身份规则；不删除或回写旧记录。

## Decisions

8. **训练配置与训练输入显式、可复现。** 定向计划的 generation seed 与训练输入集合是不同概念。训练时允许选择多个已登记、阶段兼容的数据集，并要求显式记录其 ID、hash、顺序及组合策略；支持内置配方和受限 schema 下的自定义参数，两者都解析成固定配置摘要后再预检/执行。多数据集输入先登记成有父级 lineage 的 dataset mixture，再由训练消费其精确 ID。自定义不等于任意命令、脚本或任意路径。
9. **GRPO 目标判定不猜阈值。** 目标效果配置来自用户/评测定义，记录指标、方向、阈值来源、阈值、最小样本数及保护指标；没有目标时可报告实测差值，不可宣称达标。

1. **统一 workflow identity，阶段子任务独立 run。** 使用不可变 `plan_id`/`plan_version` 关联目标和配置；生成、导入/清洗、质量审查、训练、合并、评测仍分别分配真实 `run_id`。每次 execution 可恢复，但不得把 plan ID 当 execution ID，也不得把失败步骤写成成功产物。相较新增通用 job scheduler，这复用现有 workspace 文件和本地 supervisor，保持单机架构。

2. **阶段 adapter 负责语义，不建立一个万能生成 prompt。** 每个 adapter 实现 canonical 读写、验证/拒绝分类、质量 profile、适用训练配方和可用评测证据。CPT 是领域连续语料及知识覆盖；SFT 是 prompt/response；Preference 是同题候选对和可审计排序证据；GRPO 是 prompt 与版本化可执行 reward/reference。共用 provider transport、预算和 job lifecycle，但不共用候选 schema。

3. **按共享的数据契约登记和消费。** 生成/导入输出先作为私有候选，审查后由显式动作登记为不可变 dataset ID；manifest 记录输入祖先、阶段、来源/许可已知状态、转换/提示摘要、内容 hash、build run 和 quality report。训练预检只接收 registry 中的 ID，并冻结父模型摘要、recipe/config 摘要。既有 CLI 对普通 JSONL 的兼容不改变，但该路径不可标为本闭环完整。

4. **训练与评测复用现有执行路径，按阶段扩展能力声明。** 后端 capability/preflight 返回阶段可用性、父模型、配方、硬件估计与缺项；显式确认后在本机子进程启动现有 train/eval 命令，实时读取日志/状态。DPO 与 SimPO 从同一 SFT 父模型平行分支并可复用同一 Preference dataset；GRPO 只有 reward 契约可导入并执行检查后才允许启动。若现有阶段配方或评测工具不支持，能力显示 blocked，不另造假结果。

5. **效果比较按证据等级呈现。** 单次补数前后比较固定评测集、协议与解析设置并列出训练数据、步数/token/时间和峰值显存差异，称为 pipeline observation；只有共同父模型、受控唯一差异和匹配预算时才标记为 matched A/B。CPT 展示可用的 domain PPL 与下游能力评测；Preference 额外展示可测的 pairwise 指标；GRPO 展示 reward/KL 等训练诊断及任务能力评测。指标缺失显示 not_measured，而不是零。

6. **前端以任务步骤和阻塞解释为中心。** 一个计划详情展示目标证据、阶段数据配置、生成/导入结果、质量/审核、dataset ID、训练日志、模型 lineage、评测及效果。阶段切换明确保留未完成 execution 并禁止无提示丢弃；不要求用户重复输入已登记 profile、seed 和 parent。写入、付费调用、训练、取消和登记均由单独明确动作触发。

7. **按垂直切片交付并用离线契约测试。** 先固定共用 plan/run/data contract 和阶段能力预检，再分别接 CPT、SFT、Preference、GRPO adapter；每阶段 fake provider/fixture 覆盖 schema→quality→registration→train preflight→模拟执行记录→evaluation artifact→effects report。离线测试只证明软件链路和 lineage，不证明真实模型提升；GPU/Provider acceptance 记录为显式用户运行证据。

## Risks / Trade-offs

- [可配置项可能让用户以为可以提交任意训练命令] → 只暴露服务端声明的安全参数及范围，最终仍使用受控 recipe adapter。

- [四阶段的数据与训练机制差异大，单一大改容易扩散] → 共用薄 workflow shell，分阶段 adapter 和任务清单；每个阶段独立能力开关、测试和真实验收状态。
- [合成 CPT 文本、Preference 排序或 GRPO reward 可能不可信] → 把独立 verifier/reward 契约作为阶段证据；未验证状态不能因格式正确升级。
- [本地进程/浏览器刷新可能导致用户误以为任务丢失] → 执行状态落盘、轮询恢复，不在刷新/页面切换时自动重启，历史 plan 和 run 不覆盖。
- [单卡资源不足或阶段模型产物缺失] → preflight 检查实际父模型、配方、磁盘与显存边界；允许失败和 partial 证据，但阻断缺少必要 parent/reward 的训练。
- [定向反复使用诊断集造成污染或过拟合] → 固定 data role 与祖先检查，诊断开发集和最终 holdout 分离，效果报告注明暴露与使用次数。
- [真实评测协议不覆盖目标场景] → 不从总分推断场景；显示 not_measured 并提供评测扩展缺项。

## Migration Plan

1. 只新增 plan 和子 run 记录；保留现有 SFT plan/job/dataset/run 可读，不自动迁移或重写。
2. 在本地 UI 增加按阶段声明的 capability；先以只读预检提供缺项，再逐阶段启用已通过回归的操作。
3. 新 workflow 写入独立目录并通过 registry 扫描登记产物；旧记录可从 Runs/Data 页面访问并标记 lineage 完整度。
4. 回滚时关闭新 UI capability 和执行入口；保留已登记 dataset、质量报告、run 与 lineage，不做破坏性删除。

供应方是否支持幂等键及准确 usage 统计依服务而异；无法确认的请求/费用继续按现有 `uncertain`/unknown 规则呈现，不影响阶段功能边界。
