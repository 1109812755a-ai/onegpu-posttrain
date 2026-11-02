# 多阶段数据生成职责与依赖矩阵

本矩阵服务于 `multistage-generation-closed-loop` 的实施，记录 2026-10-27 起点；入口或函数存在不等于其已完成真实 Provider/GPU 验收。

| 能力 | 当前证据与状态 | 本变更职责 | 依赖或边界 |
| --- | --- | --- | --- |
| 阶段模型生成能力 | `ui/targeted_capabilities.py` 当前只将 SFT 标为 Web 生成可用 | 增加按阶段和普通/定向模式的真实能力与阻断；复用 SFT 状态 | 不因文件或 mock 存在宣布阶段就绪 |
| 普通及定向 SFT 生成 | `/api/synthetic/sft/*` 与 `ui/sft_generation_jobs.py` 提供预检、执行、预览、取消和登记 | 构建兼容的阶段分发并保留旧 API、历史 ID 和记录 | 不覆盖既有输出；生成与登记仍显式确认 |
| CPT 数据生成 | `data/targeted_cpt.py` 有阶段生成函数，非当前 Web 生成入口 | 复用语料协议，接入普通/定向任务、质检、审核及登记 | 不将语料当 SFT 问答；事实状态独立报告 |
| Preference 数据生成 | `data/targeted_preference.py` 有同题候选、验证配对和登记函数 | 接入共享执行与普通/定向入口；DPO 和 SimPO 共用数据 | 排序证据和正确性验证不能从模型自评推导 |
| GRPO 数据 | `/api/data/grpo-seeds/*` 与 `data/grpo_seed.py` 提供可信训练种子派生，不是模型生成 | 增加受控模型候选路径，保留并区分现有派生入口 | 训练继续检查可信 reference 和可执行 reward 契约；不放宽 gate |
| 数据登记和组合 | 既有 manifest、质量报告、`/api/data/mixtures/*` 已提供登记证据和组合流程 | 复用其数据 ID/hash、质量及来源，多个训练来源先登记组合 | 不新建第二套登记、组合或血缘系统；缺证据时保持未验证/阻断 |
| 阶段训练与评测 | `/api/training/stage/*`、`/api/training/evaluation/*` 和阶段 profile 管理已有启动与评测 | 传递精确身份，独立预检、确认并记录实际结果 | 不复制 Launcher 或评测；父模型要求保持现有 profile 契约 |
| 定向计划与效果 | `targeted-data-stage-closure` 已有阶段计划/部分效果能力，其 6.2 历史与效果页、6.4 集成验收仍未完成 | 定向入口连接共用数据准备及真实 plan ID/version | 原变更继续负责其 6.2/6.4，不因本变更自动勾选或重复实现 |
| 浏览器整体走查 | `stage-training-evaluation-workbench` 的 7.3 仍未完成 | 验证本变更的数据准备、交接和证据，并可供原走查复用 | 本变更的局部测试不自动完成原 7.3；真实运行需要单独确认 |
| 未来多轮分析 | 当前不提供自动下一轮；Provider 类型为 fake/OpenAI-compatible | 保留可得模型身份、只读结果、数据来源及评测可比性证据 | 不实现自动晋级、回退、轮次编排、本地模型服务或放宽父模型规则 |

## 核验与实施顺序

1. 先建立阶段/模式能力与共享任务兼容层，保护旧 SFT API 和历史数据；未接通的阶段继续显示阻断。
2. 分别接入 CPT、SFT、Preference 与 GRPO 的生成/审查/登记，再复用现有训练与评测路径；普通和定向路径共用阶段适配器。
3. 通过 fake Provider、CPU fixture、API/UI、lineage 和严格规范校验验证功能；真实 Provider/GPU 和许可证据另行确认并记录。
4. 检查现有路由和负责模块：`rg -n 'api/(synthetic/sft|training/stage|training/evaluation|data/mixtures|data/grpo-seeds)' src/onegpu_posttrain/ui/server.py`。对照两个现有变更的任务清单，确保原任务状态不因本变更被自动改动。
