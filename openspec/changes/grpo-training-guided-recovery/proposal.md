# GRPO 训练引导与恢复

## Why

当前 GRPO 预检会正确拒绝缺少可信 reference 或 reward contract 身份的旧版、上传数据，但训练页仅将其禁用，用户难以知道下一步应去哪里准备数据。项目已有可信 OpenR1 SFT train seed 派生流程和受控 GRPO Web adapter；需要把两者连成可发现、可恢复、可验证的训练路径，并同步过时的能力说明。

## What Changes

- 在 GRPO 数据选择器中逐项解释阻断原因，并在没有可用数据时提供显眼的“准备 GRPO 数据”入口，直达数据模块现有 seed 派生流程。
- 在数据派生与返回训练页之间保留 GRPO profile、父模型及来源位置；登记成功后以精确 dataset ID 回到 Launcher，重新预检后由用户显式确认启动。
- 保持缺少 reference/reward 证据的数据 fail-closed；不通过补写声明字段、绕过契约检查或自动启训来消除阻断。
- 修正文档中“GRPO Web 启动尚未接通”等与当前代码不符的说明，并记录真实 GRPO smoke 验收边界。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `web-workspace-ux`：阶段数据阻断需要提供可执行的恢复入口，并在跨数据/训练模块往返时保留对象与配置上下文。

## Impact

影响 `web/src/Launcher.tsx`、数据工作台 GRPO seed 派生与模块导航、工作台交接状态/API、相关前后端交互测试，以及 `docs/guides/STAGE_CAPABILITIES_zh.md` 等阶段操作文档。复用现有 `/api/data/grpo-seeds`、GRPO 派生登记器、阶段预检/启动 adapter 和 reward contract；不新增数据生成器、奖励插件或训练协议。
