# 浏览器走查记录（2026-10-24）

## 已验证

- 用本机 Chrome 对当前工作台进行桌面与 390px 窄屏浏览；工作台在两种视口都能渲染，窄屏 `body.scrollWidth` 为 390px，没有横向溢出。键盘 Tab 可进入品牌链接、主导航、模块导航和筛选控件。
- Runs 列表和已有成功 SFT Run 详情可加载；Run 输入、产物、hash、配置和日志可见。API 与页面无 JavaScript 错误；仅观察到未提供 favicon 的 `/favicon.ico` 404。
- 检查真实 SFT Run `run-mini-sft-1024-7775274a9fad`：其 registered merged model `model-5d5771daad7e9c646d52` 与成功 merge Run 的输入/输出摘要相符。
- 修复 merge 模型登记通过 `adapter_id` 指向训练 Run 输出时的交接核验；Run 详情现显示 DPO、SimPO、GRPO 的兼容交接入口。浏览器点击 DPO 入口后 URL 保留 `stage=dpo`、`profile_id=dpo-smoke-v1`、精确 `parent_model_id`、`source_run_id` 和 `return_to=runs`；不传训练数据、不自动预检或启动。
- 当前代码的临时 UI 服务只绑定 `127.0.0.1:8768`，验证后已停止；用户现有 8766 服务未停止。Runs 总数在验证前后均为 53；没有创建训练或评测 Run，也没有登记或生成数据。
- 之后确认用户的 8766 服务仍运行旧后端（详情 API 缺少 `handoff_models`），且 Launcher 最近一条 SFT Run 和关联评测均已完成；在没有活动训练/排队任务后，按原开关重启到项目 `.venv` 当前代码。重启后 `/api/health`、Launcher capability、Run 详情均正常；最新 Run `run-mini-sft-1024-726393dc21ba` 为 `completed`、47/47 steps，关联的 `run-launcher-mini-math-eval-000015d6f010` 评测 `completed`，lineage `complete`，交接模型已识别。

## 本次修复与回归

- `run_workspace.py` 在验证 merge 关系时同时读取模型登记中的 `parent_model_ids` 与 `adapter_id`，仍要求 merge Run 完成、输入/输出 ID 与 SHA-256 精确匹配、模型路径安全且登记唯一。
- `workspaceNavigation.mjs` 在切换到 Launcher 时保留 `profile_id`、`parent_model_id`、`source_run_id`、`return_to` 等已确认的对象交接上下文；此前路由归一化会把这些参数丢掉。
- 回归：`tests/test_ui_run_workspace.py` 10 passed、1 skipped；前端完整 `node --test tests/*.test.mjs` 65 passed；`npm run build` 通过。

## 后续只读浏览器复核

- 在用户当前 8766 服务上重新检查数据、训练、Runs 与薄弱场景页面；桌面/390px 下数据工作台、训练页和薄弱场景页均无页面级横向溢出，页面无 JavaScript 异常。
- 选择 SFT 后，工作台显示批准 recipe、父模型、阶段数据集、预检入口和已有训练 Run；从主导航进入 Runs 可读取 53 条既有运行记录。已有 SFT Run 页面可见四个兼容模型交接入口，以及已完成的关联评测 Run；未点击评测预检/启动。
- 在普通 SFT 生成表单中切换“参考已有数据集”与“按主题生成”，切换前后没有 POST 请求，确认仅修改草稿。没有点击生成、登记、训练或评测启动。
- 对现有 `ds-sft-generated-f88a69ec5766e5f42cede07c` 执行服务端只读 SFT 预检：`ready_for_confirmation`，数据 `role=train`、1 条记录、canonical hash 与 manifest hash 均通过，父模型 `model-93611b3dbc4e10161349` 为有效 merged CPT 模型；只返回质量 `report_only` 与许可 `not_provided` 两条提醒，`execution_started=false`。没有签发确认 token 或启动训练。
- 浏览器仅对 `/api/data` 注入 503：数据目录显示局部错误与重试动作；通过主导航仍进入 Runs，并正常读取 53 条运行记录。证明目录接口故障不会阻断运行记录模块。
- 这轮浏览器访问没有写入 artifact，也没有改变 Run 数量；此前完整 pytest/build/OpenSpec 验证结果保持不变。

## 尚未验收

7.3 仍不能标完成：本次没有实际导入/登记数据、调用付费模型生成、启动新的 GPU 训练或创建独立评测 Run；这些会写入本地 artifacts、消耗外部调用或占用 GPU。阶段评测预检在此前浏览器尝试中也未在等待窗口内返回，因此不声称新预检/启动已通过浏览器验收。弱场景到定向生成再登记并训练仍需受控端到端验收；生成方式切换与单接口故障隔离已通过只读浏览器检查。

原 8766 服务当时没有返回 `handoff_models`；本次已重启并核验当前代码。刷新用户打开的 `http://127.0.0.1:8766/?page=workbench` 即可加载新版本；训练和合成数据开关保持启用。

## 追加：真实探索性 SFT 与阶段评测（2026-10-24）

- 用户确认后，通过当前 Launcher API 对已登记数据 `ds-sft-generated-f88a69ec5766e5f42cede07c` 启动一次固定 Mini SFT；单条样本，仅作探索性验证。训练 Run `run-mini-sft-1024-0fe815479943` 完成（1/1 step，92 秒），LoRA merge 成功，产出 merged model `model-2643167210361f8b56a7`。训练输入 lineage 记录父模型 `model-93611b3dbc4e10161349` 与 dataset ID/canonical SHA-256。
- 独立评测 Run `run-launcher-mini-math-eval-41b6a382f28f` 完成；GSM8K 0/5、MATH-500 1/5（20%），仅为每项 5 条的探索性结果，不能视为正式 benchmark。训练 Run 的关联评测状态与 `/api/runs` 详情均显示 completed；8766 `/api/health` 正常，Runs 页面 HTTP 200。
- 数据集许可仍为 `not_provided`、质量为 `report_only`，且生成样本语义正确性未独立验证；训练记录保留了用户确认和质量提醒。没有把该数据标记为正式 Mini 验收通过。
- 9 项定向关系断言通过：manifest canonical hash 与文件相符、训练消费同一 dataset ID/hash、父模型身份一致、adapter 由训练 Run 产出、merge 消费该 adapter、评测消费该 merge model，评测协议与分母为实际 5 条。
- 优化 `validate-lineage --root .` 的扫描：一次遍历并在下探前剪掉 `.git`/`.venv`/`node_modules` 后，全工作区审计在约 90 秒内完成。当前总计 134 runs、151 artifact references、47 datasets、40 models、25 evaluations；74 errors、74 warnings，主要为历史 artifacts。上述新训练/评测记录无 lineage errors；评测 benchmark 数据摘要不可用保留为 warning。独立 run 子目录校验会因引用对象在 workspace 其他目录而报告 unresolved，因此以全 workspace 审计和定向身份/hash 断言为准。
- 本次尚未完成 7.3：没有执行导入/普通生成登记到训练，也没有执行薄弱场景→定向生成→登记→训练；本次训练/评测由 API 启动，不能冒充完整浏览器任务走查。生成/登记及额外训练仍待后续受控验收。
