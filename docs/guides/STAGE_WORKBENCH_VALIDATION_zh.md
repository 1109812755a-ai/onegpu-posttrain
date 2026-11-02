# 阶段工作台本地验证摘要

日期：2026-10-23  
范围：本地 fixture 与 fake subprocess；不下载模型、不调用远程 provider、不使用真实 GPU。

## 阶段纵向覆盖

| 阶段 | 已登记数据 / 预检 | fake 训练与模型登记 | 评测 / lineage 验证 | 主要测试 |
| --- | --- | --- | --- | --- |
| CPT | CPT canonical registration、固定 profile、parent/数据预检 | fake CLI 终态与模型 artifact 登记 | CPT holdout PPL 的真实分母/lineage、holdout 隔离 | `test_targeted_cpt_generation.py`, `test_cpt_stage_adapter.py`, `test_evaluation_lineage.py` |
| SFT | SFT 数据构建、模型/数据 lineage | fake train/merge 及真实产物摘要 | 模型评测、dataset/model/run/evaluation lineage 校验 | `test_lineage_e2e.py`, `test_training_preflight.py` |
| DPO | Preference train/eval pair 登记、同源 SFT parent 预检 | fake DPO train/merge | held-out pair 评测、分母与独立 eval Run | `test_preference_stage_adapter.py`, `test_preference_evaluation_adapter.py` |
| SimPO | Preference 数据与 profile 预检 | fake SimPO train/merge；独立 stage/run/model ID | held-out pair 评测、分母与独立 eval Run；与 DPO 共享协议但保留各自身份 | `test_preference_stage_adapter.py`, `test_preference_evaluation_adapter.py` |
| GRPO | prompt/reward contract/parent 预检 | fake GRPO train/merge，核实 reward contract 与模型 lineage | 注册模型评测、真实分母及未测 rollout 指标保真 | `test_training_preflight.py` |

这是按阶段拆分的纵向 fixture 测试集，不声称在单一 fixture 中真实训练五阶段；各阶段执行边界和失败状态保持独立。Stage Evaluation API/Run 观察、训练 Run 终态保护、指标分母及 UI Run 回链由 `test_preference_evaluation_adapter.py` 与 `test_ui_launcher.py` 覆盖。

## 本次回归结果

阶段 / lineage / Launcher 相关后端回归：

```text
97 passed, 2 skipped
```

随后全仓 Python 回归：

```text
626 passed, 7 skipped
```

全量结果覆盖单活动任务锁、重复启动拒绝、服务停止/重载后的 interrupted 状态恢复、Mini SFT 兼容、评测协议不一致时不横向比较等现有测试。期间修正了一条过时的 GRPO OOM CLI 退出码断言：实际行为按 OOM 返回 `4`，并保留失败状态、不登记模型。

前端所有 Node 测试：

```text
61 passed
```

前端 `tsc --noEmit && vite build` 成功；本次涉及的 Python 文件 Ruff 检查通过。真实 RTX 4060 训练、模型下载和真实 benchmark 未在此 fixture 回归中执行。浏览器三条真实用户任务走查仍需在 UI 服务启动后执行，不把源码测试当作浏览器实测。

可复现命令（仓库根目录）：

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_targeted_cpt_generation.py `
  tests/test_cpt_stage_adapter.py `
  tests/test_preference_stage_adapter.py `
  tests/test_preference_evaluation_adapter.py `
  tests/test_training_preflight.py `
  tests/test_lineage_e2e.py `
  tests/test_stage_profiles.py `
  tests/test_ui_launcher.py `
  tests/test_ui_run_workspace.py -q

Push-Location .\web
node --test tests\*.test.mjs
Pop-Location
pnpm --dir web build
```
