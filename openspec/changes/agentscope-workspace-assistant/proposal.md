# 提案：基于 AgentScope 的工作区实验助手

## Why

当前助手依靠固定关键词分类和一次性证据读取，真实工作区问题出现检索失败和上下文超限。现有框架选型只完成资料比较，用户决定使用 AgentScope 实现一版可交互的实验问答助手。

## What Changes

- 用 AgentScope Agent 与受限工具调用流程处理实验问题，模型沿用已配置的阿里百炼。
- 提供工作区运行、评测、数据和模型血缘的只读查询工具；回答引用实际读取的证据，比较前检查可比性。
- 延续现有聊天入口、默认助手、会话、专用助手配置、长期偏好、上下文压缩、外发授权和可选 Langfuse 记录。
- 在界面显示正在查找的内容、结论、证据和需要用户补充的信息；失败时明确说明原因。
- 用真实工作区问题与可控样例验证完整链路，记录百炼真实调用的结果与限制。

## Capabilities

### New Capabilities

- `workspace-agent-evidence-dialogue`：使用受限工作区工具完成有证据的实验问答与评测比较。

### Modified Capabilities

- 无。现有 `workspace-experiment-agent` 变更提供本次实施的既有设计和契约。

## Impact

涉及 `src/onegpu_posttrain/assistant/`、工作台 API 和聊天界面、依赖配置、助手文档及验证样例。需验证 AgentScope 与项目 Python 环境及百炼模型的实际兼容性。训练执行、数据生成、联网搜索、MCP 和多 Agent 协作留待后续变更。
