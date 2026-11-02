# 提案：Launcher 实时日志与训练后评测

## Why

当前 Launcher 虽会自动轮询状态，但把 CLI 汇总 JSON 当作训练日志，训练期间看不到真实 step、loss 和进度；训练完成后只能回到终端运行评测。用户无法在一个连续流程中观察训练并验证模型效果，削弱了项目“可观察、可复现”的核心体验。

## What Changes

- 训练运行期间持续展示实际训练日志和可解析的 step/指标，页面刷新后仍恢复同一 run；保留有界日志与真实未知状态。
- 训练和模型合并成功后，显示评测预检与明确的“运行评测”操作；评测使用固定、版本化且有界的 Mini 探索性协议，不开放任意命令或参数。
- 评测作为独立受监督执行：展示状态、日志、真实指标和失败/部分状态，并通过模型产物身份与摘要关联回训练及其血缘。
- 重组 Launcher 页面，突出当前阶段、主操作、运行状态和结果；配方细节可折叠，日志、评测结果、产物/血缘分区展示，并覆盖窄屏与键盘交互。
- 训练和评测共享单任务/GPU 生命周期约束；不引入队列、并发或通用调度器，评测也不会随训练自动启动。

## Capabilities

### New Capabilities

- `local-experiment-lifecycle`: 覆盖本地训练实时观察、显式评测启动及从评测结果回溯训练模型的连续流程。

### Modified Capabilities

- `web-workspace-ux`: 将 Launcher 的训练/评测工作区纳入窄屏可用性要求。

## Impact

- 影响 `src/onegpu_posttrain/ui/launcher.py`、`src/onegpu_posttrain/ui/server.py`、`web/src/Launcher.tsx`、工作区样式及相应 API/前端测试。
- 训练继续复用固定 SFT recipe 与现有 CLI；评测复用 `run-model-eval` 及 `math-eval-v1` 协议，不新增训练或评测框架。
- 新评测通过独立评测 `run_id` / `eval_id` 记录，并遵循既有 `evaluation-lineage` 和 `run-model-lineage` 约束；评测执行需要再次占用本地 GPU，须显式确认。
