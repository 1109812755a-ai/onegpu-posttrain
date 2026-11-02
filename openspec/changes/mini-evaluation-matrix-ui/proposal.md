# Proposal

## Why

Mini 的 M2 与四个对齐分支已完成同协议评测，且已有机器可读矩阵，但用户目前只能在数据页查看逐条关联指标、或打开报告文件查看 delta。把矩阵接入现有“阶段比较”入口，可以让实验结果、样本边界和模型/数据来源在一个可读页面里核查。

## What Changes

- 启用只读“阶段比较”页面，读取已有 unified evaluation matrix artifacts，不触发训练或评测。
- 展示共同基线、各模型状态、指标值/样本分母、相对基线变化及协议可比性；缺失或失败指标保持明确状态，不显示成 0。
- 展示模型与评测 run 身份、可解析的训练/父模型/数据血缘，以及小样本、截断和质量审查限制。
- 增加安全受限的本地读取 API，并覆盖空数据、损坏报告、不可比协议及窄屏/键盘交互。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `unified-model-evaluation`：消费方能够在阶段比较页展示矩阵覆盖、分母、delta 和可比性状态。
- `web-workspace-ux`：阶段比较导航可用，加载/空态/错误态及结果表格具备清楚、可访问的交互。

## Impact

影响 `src/onegpu_posttrain/ui/server.py`、`web/src/main.tsx`、`web/src/types.ts`、`web/src/style.css`，以及对应的 API、前端导航和交互测试。复用现有 `evaluation_matrix.json` 与 lineage artifacts；不增加数据库、训练调度、训练/评测命令或新依赖，不修改历史指标和数据产物。
