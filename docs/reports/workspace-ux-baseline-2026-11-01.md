# 工作区体验基线（2026-11-01）

范围：数据、训练与评测、Runs、分析与改进四个主模块。浏览器快照由本机 Edge headless 在 `http://127.0.0.1:8769/` 捕获；桌面为 1440×900，窄屏为 390×844。此轮快照是只读/初始加载状态，**不代表**训练、评测或真实 GPU 链路通过。任务状态问题同时参照用户此前提供的运行中、评测中和失败截图，以及当前组件代码。

| 模块 | 桌面 | 390px | 真实任务路径及当前主要摩擦 |
| --- | --- | --- | --- |
| 数据 | [桌面](workspace-ux-data-desktop.png) | [390px](workspace-ux-data-mobile.png) | 导入/生成→预览与质量许可核对→登记→交接训练。来源格式、文件、预览、许可连续出现，但原生文件选择器与其他控件风格不一致；长数据集 ID 和证据可能抢占首屏。 |
| 训练与评测 | [桌面](workspace-ux-launcher-desktop.png) | [390px](workspace-ux-launcher-mobile.png) | 阶段配置→预检→确认启动→Run→评测→结果。现有“完整流程”已展示后续步骤，但运行态与评测态分散；提交中、已创建 Run、GPU 运行容易被同一“启动中/处理中”表达混淆。390px 快照中说明文字存在横向溢出。 |
| Runs | [桌面](workspace-ux-runs-desktop.png) | [390px](workspace-ux-runs-mobile.png) | 搜索 Run→详情→评测/交接。Base 评测入口和 Runs 搜索并列，读屏顺序不够清楚；大段 ID/hash/路径与业务信息混排，预检和评测执行的按钮反馈不够明确。 |
| 分析与改进 | [桌面](workspace-ux-comparison-desktop.png) | [390px](workspace-ux-comparison-mobile.png) | 评测差异→薄弱场景→定向补数→返回。快照主要处于读取态；诊断上下文与下一步动作需保持可见，错误与空态不应伪装成完成。 |

可复查问题索引：

1. **状态真实性**：`Runs.tsx` 的评测预检和启动共用 `evaluationBusy`；`Launcher.tsx` 另有 `busy`、训练 Run 与评测 Run 状态。网络请求、Run 已创建及后台执行应分层显示，不可用“处理中”概括整个长任务。用户此前截图曾出现按钮长期“处理中”、评测日志在下方而当前步骤不清楚。
2. **首屏信息层级**：Launcher/Run 详情反复展示模型 ID、dataset ID、输出路径和原始配置；失败原因、当前步骤、可信进度、下一步应优先于技术证据。技术证据仍须可展开、可复制。
3. **控件一致性**：原生 `select`/文件控件与定制按钮、徽标和提示差异明显，焦点/禁用/错误态需要统一；尤其 390px 下任何两列控件都不应超宽。
4. **重复提示**：相同 GPU 锁/未启用或评测入口阻断可在页面头部、动作栏及卡片内重复出现；保留一处主解释与就近修复入口即可。
5. **后续可见性**：训练已有流程路线图，数据和 Runs 的跨模块下一步仍需显式呈现；刷新后应从精确 Run 恢复，而不是依赖一次性按钮状态。

截图索引仅用于 UI 基线；快照中部分目录仍在异步读取、Launcher 为只读模式，不能据此判断服务端数据缺失。后续任务 5.1 须使用可控 fixture 再做完整路径验收。

## 共享控件/主题复查（首轮）

使用 `node scripts/capture_workspace_ui.mjs docs/reports` 在真实 Edge 中分别设置浅色、深色，并捕获桌面和 390px 视口。例图：[训练浅色 390px](workspace-ux-launcher-mobile-light.png)、[训练深色 390px](workspace-ux-launcher-mobile-dark.png)、[Runs 深色桌面](workspace-ux-runs-desktop-dark.png)；其余文件同名规则 `workspace-ux-{data|launcher|runs|comparison}-{desktop|mobile}-{light|dark}.png`。所见选择器、边框和背景随主题变化；训练首屏原有横向溢出已收敛。组件测试覆盖状态色、禁用、键盘切换主题；纯 CSS 焦点/错误态由统一规则约束。此截图仍是页面初始态，不代替任务 5.1 的完整交互验收。
