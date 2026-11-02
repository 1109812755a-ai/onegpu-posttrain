# Proposal

## Why

模型体验页目前只允许调整最大生成长度和 Temperature，其他影响采样与重复输出的参数无法由用户控制；页面默认的 512 token 也可能让重复内容持续很久。需要让用户能使用一组清晰、可恢复且可复现的解码设置体验各阶段模型。

## What Changes

- 在模型体验页为单模型对话和同题对照提供一致的生成参数配置，包括最大新 Token、Temperature、Top-p、Top-k、重复惩罚和 Qwen3 思考模式。
- 提供推荐默认值及恢复默认值操作；初始推荐值为 256、0.7、0.8、20、1.05、关闭思考模式。重复惩罚值是温和的产品默认建议，不宣称为 Qwen 官方推荐值。
- 后端校验并应用有效参数；对不支持思考模板选项的模型明确反馈，不得静默声称该选项已生效。
- 在单聊及对照结果中展示实际生效的参数，以支持用户理解和复现；不改变体验会话不持久化、不生成 Run/评测/训练数据的边界。

## Capabilities

### New Capabilities
- `model-playground-generation`: 配置、校验、应用并展示模型体验的解码参数。

### Modified Capabilities
- None.

## Impact

- 前端：`web/src/ModelPlayground.tsx` 的参数表单、单聊/对照请求及结果参数摘要。
- 后端：`src/onegpu_posttrain/ui/model_playground_session.py` 的默认值、参数范围校验、生成参数传递和模板选项处理。
- 测试：模型体验会话/对照 API 与前端组件测试；不新增运行时依赖，不修改训练或正式评测的生成参数。
