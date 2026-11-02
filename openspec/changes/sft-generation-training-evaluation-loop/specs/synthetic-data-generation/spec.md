# Spec Delta

## MODIFIED Requirements

### Requirement: 生成是显式可选操作
系统 MUST 仅在用户明确执行生成操作时发出模型请求；配置校验、计划保存、dry-run、页面浏览和打开生成配置均不得调用 provider，核心开源数据/本地导入流程不依赖生成服务。

#### Scenario: 没有 API 账户运行项目
- **WHEN** 用户仅使用开源来源、本地数据或 fake provider
- **THEN** 可以完成相应数据与实验流程，生成能力显示为未配置或本地模拟，不能被误认为远程真实结果

#### Scenario: 通过命令明确执行
- **WHEN** 用户运行显式生成 CLI 并授权执行
- **THEN** 系统才可按计划预算发出 provider 请求

#### Scenario: 通过本地界面明确执行
- **WHEN** 用户打开生成计划并主动确认“开始生成”
- **THEN** 本地服务按已审阅计划执行生成；仅打开页面、切换阶段或保存草稿不发请求

## ADDED Requirements

### Requirement: 生成目标与调用预算分别记录
系统 MUST 分别记录期望候选或可用样本目标与实际 provider 请求预算，且不得承诺目标数量必然达成。

#### Scenario: 达到目标数量
- **WHEN** 成功生成数量达到计划目标
- **THEN** 报告目标数、实际数量、请求数与停止原因

#### Scenario: 预算耗尽或候选不足
- **WHEN** 请求预算耗尽、来源种子耗尽或请求失败导致产出低于目标
- **THEN** 保存 partial/failed 状态、已生成数量、缺额和停止原因，不以重复或未验证记录填满目标
