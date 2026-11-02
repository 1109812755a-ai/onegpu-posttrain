# Proposal

## Why

远程合成数据生成目前只接受进程环境变量，用户需要在每个终端会话中重复设置凭据。支持本机 gitignored 文件可简化显式生成操作，同时保持密钥不进入受版本控制的 provider profile 或产物。

## What Changes

- 显式执行生成时，可从本地 `.env.local` 读取 provider 凭据；已有进程环境变量优先。
- dry-run、配置校验、UI 浏览及其他命令不得读取该文件或发起请求。
- 提供安全示例和中文使用说明；不保存或使用已在对话中暴露的密钥。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `synthetic-data-generation`: 增加显式生成时从本地凭据文件解析 provider secret 的行为和安全边界。

## Impact

影响 synthetic generation CLI/provider 凭据加载、相关测试和定向合成数据中文指南；不新增运行时依赖，不改变 provider profile schema。
