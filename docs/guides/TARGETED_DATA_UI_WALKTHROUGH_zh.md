# 定向补数 UI 交互验收走查

本走查说明“定向补数”页面的诊断/草稿区交互边界。配置草稿不会发起 provider 请求；真实 SFT 生成由同页单独 opt-in 的本地执行区控制，训练和复测仍需在 Mini SFT Launcher 中分别预检、确认。页面不是通用任务调度器。

## 走查路径

1. 打开左侧“定向补数”，先查看场景准确率、正确数/可评分分母、总样本数、解析失败、不支持和缺预测数。没有逐题报告或没有有效分母时显示“未评估”，不以 `0%` 代替缺失值。
2. 选择 SFT 或 Preference、场景和 provider profile，编辑 endpoint/model/凭据环境变量名（仅名称）、提示词、输出字段和请求预算。切换 fake profile 时显示“本地 fixture”，不把 fake 结果描述成真实模型能力。
3. 导出 `targeted-synthetic-workbench-draft.json`。导出字段包含 `draft_not_registered` / 未执行状态；不会将密钥值写入文件，不会访问 provider 或读取进程环境变量。费用没有 usage/费率证据时为“未知”。此草稿区始终保持未执行状态。
4. 点击“刷新报告”：如果当前配置/诊断证据与上次导出内容不一致，显示“草稿已过期”；刷新失败时保留已有输入和报告快照，并显示错误提示。
5. 查看生成验证区与复测区。partial、失败、本地模拟、请求数、候选目标、成本状态、起点模型、目标场景、追加数据和比较限制分别展示。缺报告时分别显示待生成、待训练、待复测，不推断收益。
6. 若已有登记的 SFT workflow bundle 且 UI 以 `--enable-synthetic-generation` 启动，可在下方生成区预检预算、显式确认后执行、查看进度，并在登记前预览样本。刷新页面会恢复 execution 状态；人工审核后才创建新 dataset ID。详见 [SFT 生成闭环指南](SFT_GENERATION_WORKFLOW_zh.md)。

## 键盘与窄屏检查

- 页面控件均为原生 `button`、`select`、`input`、`textarea`；可用 Tab/Shift+Tab 顺序访问，Enter/Space 触发按钮，方向键操作原生下拉框。
- `:focus-visible` 为可见 3px 轮廓；场景按钮使用 `aria-pressed`，加载失败提示使用 `role=alert`，导出结果提示使用 `role=status`。
- 视口宽度不大于 1000px 时两列工作区改为单列；不大于 680px 时流程步骤为两列、配置/证据/引导区为单列，长来源 ID 可换行，按钮与表单控件保持全宽可触达。
- 报告刷新只更新证据，不重置用户已输入的配置；本页没有 API key 输入控件。Provider 执行按钮仅存在于单独 opt-in 的 SFT 工作流区，且必须二次确认。

## 自动化证据

实现验收记录：

- `pytest tests/test_ui_targeted.py tests/test_ui_policy.py`：API 白名单、敏感哨兵隔离、unknown denominator/cost、fake profile 与只读路由检查通过。
- `pnpm build`：TypeScript 检查及 Vite 生产构建通过。
- `pnpm test:policy-editor`：既有策略 editor 草稿预览新旧响应测试通过。

窄屏布局与键盘可达性依据本文件的 DOM/ARIA 与响应式 CSS 逐项源码审查；当前自动化环境未配置浏览器驱动，因此不把这项记录冒充真实浏览器/屏幕阅读器测试。发布前应在实际浏览器按上述路径复核。
