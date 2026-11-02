# 阶段工作台浏览器走查（2026-10-30）

本记录是 OpenSpec `stage-training-evaluation-workbench` 任务 7.3 的阶段性证据，不代表该任务已完成。

## 已验证

- 隔离 Chrome/CDP 桌面视口与 390px 视口均无横向溢出；键盘 Tab 导航可见焦点。
- SFT 普通生成页面：主题 `基础算术`、能力点 `10以内加法`、目标 1 条；页面方案准备与只读预检通过（最多 2 次 Provider 请求）。随后通过本机受控 API 发起一次生成，实际 1 次请求生成 1 条候选。审核样本为“3 个苹果再给 4 个，共 7 个”，人工核对算术正确后显式登记。
- 登记数据集：`ds-sft-generated-752b39a81fa2ca6136a5c6c4`。训练预检核对 canonical schema、数据/hash/lineage、SFT 父模型和批准配方均通过；质量为 `report_only`、许可为 `not_provided`，只以 smoke 数据使用并保留警告。
- SFT 训练 Run：`run-mini-sft-1024-ee626a965d65`，exit 0，训练及 LoRA merge 成功，登记模型 `model-0573d3916cbd290dbb46`、`model-378db444be75f5f47cc4`。
- 独立阶段评测 Run：`run-sft-sft-math-eval-v1-e967db17d4c3`，状态 completed，`math-eval-v1`；GSM8K 0/5、MATH-500 1/5。结果明确为小样本 smoke，不代表正式基准表现；评测 lineage 可追溯到模型和训练 Run。评测启动曾超过客户端等待时间，但服务端任务继续成功；重复提交被 GPU 锁拒绝，没有生成第二条评测。
- 用成功 SFT Run 的模型打开 DPO 工作台，浏览器已预选正确的 SFT merged 模型及来源 Run；使用兼容的 Preference 数据 `ds-preference-generated-b25cd7cb2e5e3630230e` 做只读阶段预检，状态 `ready_for_confirmation`。未启动 DPO 训练。
- 重启前检查到模型体验会话持有活租约；经用户明确同意，重启了本地 8769 服务、恢复其退出后形成的陈旧租约，随后服务健康检查通过，GPU 租约恢复 available。

## 尚未完成

- 薄弱场景页面当前没有可展示的逐题场景诊断（`/api/targeted/overview` 的 `scenarios` 为空）；页面明确提示不能从总分推算场景准确率，需先用逐题 predictions 和场景标签生成 scenario report。当前 smoke 评测没有本项目可验证的场景 taxonomy/标签，不能把 0/5 的 benchmark 汇总分数伪造成弱场景。因此尚不能诚实完成“薄弱场景 → 定向计划 → 定向生成/审核/登记 → retrain”。任务 7.3 保持未完成。
- UI 的 Provider 确认按钮通过 CDP 未稳定提交；为避免重复计费，本次只通过已经预检的同一 bundle 和相同预算在本机 API 执行一次 Provider 调用。需要后续补做真正浏览器内完整生成确认/登记、单接口故障与返回导航走查。
- 训练参数修改/预检失效的细节验证仍需在浏览器中补齐；相关组件和 API 自动化测试已覆盖，但不能替代这项手工验收。

## 验证状态

本次此前运行：前端 build 成功；Vitest 32/32；web Node 测试 83/83；选定后端测试 51 passed、1 skipped；OpenSpec strict validation 通过。以上自动化通过不改变 7.3 的未完成状态。
