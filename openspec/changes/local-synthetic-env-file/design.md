# Design

## Context

Provider profile 只保存 `api_key_env` 名称；远程 provider 在进程环境中解析密钥。仓库 `.gitignore` 已忽略 `.env.*`，当前 CLI 以 `--execute` 作为唯一远程请求入口。

## Goals / Non-Goals

**Goals:** 在显式生成时安全读取仓库根目录 `.env.local`，提供环境变量优先级和可测试解析规则。

**Non-Goals:** 不把密钥写入 YAML、UI、计划、日志或产物；不增加密钥管理服务、自动请求或新的运行时依赖。

## Decisions

- 仅 `generate-targeted-data --execute` 在输入准备完成后加载 `.env.local`；dry-run 和其他命令完全不触碰此文件。
- 只解析 `KEY=value`（可选 `export `、单/双引号）；不执行 shell、不做变量插值。允许的凭据键限定为当前 profile 的 `api_key_env`。
- 进程环境中的非空值优先；文件值只填补缺失变量。重复键、无效行和不允许键均 fail closed，错误不包含值。
- 文件从当前项目工作目录的 `.env.local` 读取，文档要求在仓库根目录运行；文件不存在时保持现有缺凭据错误。
- 不写真实 `.env.local` 或暴露密钥；提供仅含占位符的 `.env.local.example`，并说明复制后填写经轮换的新密钥。

## Risks / Trade-offs

- 本地明文凭据可能被同机其他用户读取 → 文件权限由用户/操作系统管理；明确提示不要提交或分享，并建议轮换曾暴露的旧密钥。
- 轻量格式不支持完整 dotenv 插值 → 保持解析行为确定且避免代码执行；文档列出支持语法。

## Migration Plan

复制 `.env.local.example` 为 `.env.local` 并填入新凭据；无需修改 provider profile。移除文件即可回退到现有环境变量配置。
