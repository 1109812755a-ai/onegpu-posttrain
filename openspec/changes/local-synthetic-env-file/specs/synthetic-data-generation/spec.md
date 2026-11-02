# Spec Delta

## MODIFIED Requirements

### Requirement: 生成配置明确且凭据不落入公开产物
系统 MUST 接受明确的 provider/endpoint/model、凭据变量引用、提示词模板、输出协议、生成参数及预算；密钥只能在实际调用时解析，不能保存到计划、日志、UI、异常或导出报告。

#### Scenario: 缺少凭据执行生成
- **WHEN** 远程生成所需凭据变量不存在
- **THEN** 执行前报明确缺失且不发请求，配置和错误不包含任何秘密值

#### Scenario: 配置含带密钥的 URL
- **WHEN** endpoint 包含内嵌认证或被识别的敏感查询字段
- **THEN** 配置校验拒绝并要求独立凭据引用，不将 URL 原文回显到公开错误

#### Scenario: 显式执行从本地文件加载凭据
- **WHEN** 用户从项目目录显式运行远程生成，进程环境中缺少 profile 指定的凭据变量，且本地 `.env.local` 含该变量
- **THEN** 系统在发请求前读取凭据并仅用于 provider 认证，不将值写入任何生成产物或输出

#### Scenario: 进程环境变量优先
- **WHEN** 进程环境与 `.env.local` 都提供 profile 指定的凭据变量
- **THEN** 使用进程环境变量，不以文件内容覆盖它

#### Scenario: 本地凭据文件格式无效
- **WHEN** `.env.local` 含重复键、格式错误或不允许的键
- **THEN** 执行失败且不发请求，错误只指出行号或键名，不回显任何值

### Requirement: 生成是显式可选操作
系统 MUST 仅在明确执行生成命令时发出模型请求；配置校验、计划保存、dry-run 和页面浏览均无远程调用，核心开源数据/本地导入流程不依赖生成服务。

#### Scenario: 没有 API 账户运行项目
- **WHEN** 用户仅使用开源来源、本地数据或 fake provider
- **THEN** 可以完成相应数据与实验流程，生成能力显示为未配置或本地模拟，不能被误认为远程真实结果

#### Scenario: 未执行操作不读取本地凭据
- **WHEN** 用户执行 dry-run、配置检查或打开 UI，而本地 `.env.local` 存在
- **THEN** 系统不读取凭据文件且不发出 provider 请求
