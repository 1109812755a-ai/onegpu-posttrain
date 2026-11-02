# 前端路线图

**状态：** 分阶段路线图。Data 浏览与策略对照编辑已在 `data-policy-experiments` 切片实现；Runs 列表/详情与只读本地 API 在 `launcher-path-safe-evaluation-run-workspace` 切片实现。完整 v0.1.1 UI 验收仍未完成，UX 是每个阶段的发布门槛。

本文的版本号指产品发布版。产品发布版 v0.1.1 与冻结的设计修订版 v0.1.1 不同；后者针对产品发布版 v0.1。

## 原则

- 先消费 artifacts，不让训练依赖 Web 服务；
- 即使 UI 增强，CLI 仍是稳定执行接口；
- artifact schema 和 provenance 成熟后再扩功能；
- 没有查询/并发需求之前不引入数据库或远程服务；
- 保留协议兼容、分母、seed、缺失指标等科学上下文。
- 数据质量证据是一等视图：区分阶段化实测证据、估算值和依赖模型/recipe 的校准结果，不生成通用综合分。
- 交互以用户任务为中心：让用户能找到目标数据/run、理解状态与缺失证据、沿血缘查看来源，并保留筛选和返回上下文。
- 键盘可操作、清楚的焦点/选中状态、可读对比度、响应式布局，以及加载/空/错误/过期状态均为验收项；颜色不能是唯一状态提示。
- 图表和流程视图须提供可访问的表格/列表替代；高级信息按需展开，避免主界面拥挤或隐藏分母、单位、限制。

## 版本里程碑

| 版本 | 范围 | 退出门槛 |
|---|---|---|
| v0.1 | CLI-only smoke 与 Mini 研究基线 | 现有 v0.1 门槛通过；Web UI 仍是明确非目标。 |
| v0.1.1 | 本地 Experiment UI：Overview、Runs、Run Detail、Stage Comparison、Data（血缘/质量/分布/泄漏）、策略对照编辑器、定向补数（场景诊断/计划草稿/生成验证/复测摘要）、GRPO/Preference 分析、硬件/资源和 artifact viewer。artifact 浏览默认只读；策略预览与草稿保存为显式启用的受限例外；定向生成仍只由 CLI 显式执行。 | 正确呈现真实 artifacts；质量分母、不确定性、缺失数据和协议警告有效；文件访问受限；没有 provider 调用或训练进程控制入口；启停/安全说明齐全；完成可用性任务 walkthrough 和键盘/窄屏检查。 |
| v0.2（首个纵向切片） | 显式启用的本地 Mini SFT Launcher，只允许固定 `recipes/mini/23-sft-1024-fallback.yaml`（307 条探索数据、1,024-token 上限）；不允许参数/路径覆盖或启动评测。 | 仅 loopback；只读预检 + 确认精确配置；单进程监督；真实状态/日志/血缘；关闭浏览器不影响训练、关闭服务记录中断；假 CLI 端到端测试通过。更多 recipe/参数覆盖需另行设计。 |
| v0.3 | Experiment Studio：定义 OVAT、多 run 比较、曲线/表格/delta、seed 对比和报告导出 | 实验定义可版本化复现；比较会检查兼容性；导出包含 provenance 与限制。 |
| v0.4+ | Dataset、Model、Experiment Registry UI | 稳定 ID/provenance/lifecycle 已建立；Registry 有真实使用理由；持久化方案单独评审。 |
| v1.0 | 平台级工作台：任务队列/调度、资源与状态视图、Dataset/Model/Run/Evaluation 工作流、部署配置与运维入口 | 经真实用例验证的任务生命周期、失败/中断/恢复和资源反馈；权限及安全边界、部署升级和数据导出经过验证；交互任务 walkthrough、可访问性和响应式验收通过。 |

前端路线与模型/数据研究路线并行：v0.2 首个切片先基于现有 0.6B Mini 基线验证“浏览器→SFT”；后续 recipe/参数扩展再与选定的 1B/3B 实验协同。v0.3 将 Studio 与数据中心实验结合，并在 OVAT 批量使用需要时引入本地串行队列；v0.4+ 建设 Registry 视图；到 v1.0 再把已验证的任务调度、资源管理和部署运维整合成平台工作流。

### Runs 工作区切片（本地只读）

Runs 读取 `runs/` 下 CLI 运行记录、评测记录及 Launcher 状态，不建数据库、不启动任务、不提供任意路径或命令输入。用户可以按阶段/状态/关键词筛选，打开稳定 `run_id` 详情查看状态、配置、真实指标与样本分母、受限日志、输出和已记录 lineage；状态缺失、失败/部分完成及未解析的 parent 都显式呈现。Launcher 完成视图可跳转到同一训练 run 的 Runs 详情。此切片属于本地 UI 的渐进实现，不代表 v0.1.1 所有页面、图表及完整 UX walkthrough 已通过发布验收。

## v0.1.1 只读架构提案

```text
React + Vite + TypeScript
          │ 本地 artifact API + opt-in 策略编辑 API
          ▼
FastAPI artifact adapter
          │ 白名单读取
          ├── runs/
          └── docs/results/
```

建议组件库为 Ant Design + ECharts。存储继续使用文件系统中的 JSON、JSONL、Markdown 和日志，不用数据库。这些是规划默认值，依赖冻结前需要 ADR。

## 顺序与门槛

1. dev7 期间只提前实现经明确优先排序的 Data 页面；其余 UI 范围仍按 v0.1.1 门槛推进，不能把首屏预览视为完整 UI 发布；
2. 根据真实 artifact 稳定最小索引/规范化读取 schema，支持 schema 版本和可选字段，不改写历史文件；
3. artifact API 只读扫描配置根；策略编辑 API 单独 opt-in、限制来源 ID/预览预算/新草稿目录，覆盖路径穿越、坏/大文件、secret/权重暴露测试；
4. 用完整和不完整 smoke artifact 验证，包括当前高 clipping 和偏好数据偏少的情况；执行任务式 UX 检查：新用户能否找到数据来源/处理步骤/质量报告、看懂缺失或告警、筛选后返回并保持上下文；
5. 只读 UI 有价值后，再由 v0.2 单独设计进程启动授权。

## v0.1.1 明确不做

不做任意命令执行、训练进程控制、recipe 修改、模型权重服务、默认远程绑定、用户/角色管理、数据库、Dataset/Model Registry 写入或自动调度。受限策略编辑仅写入专属新草稿目录，并复用 CLI 的数据策略协议。
