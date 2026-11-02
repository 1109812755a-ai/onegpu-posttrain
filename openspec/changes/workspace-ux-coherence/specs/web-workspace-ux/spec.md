# Spec Delta

## ADDED Requirements

### Requirement: Task pages present one current step and next primary action
数据准备、训练、评测和分析中的可执行任务 MUST 展示当前对象、当前步骤及其唯一的下一项主操作；其他适用操作 MAY 作为次要操作保留。配置、预检、确认、提交、后台运行、完成/失败 MUST 具有可区分的文案和状态，不能让旧预检与当前草稿并列为可启动状态。

#### Scenario: Move from evaluation preflight to a run
- **WHEN** 用户对已核验模型完成评测预检并确认启动，服务端返回评测 Run 身份
- **THEN** 页面明确表示任务已启动，提供该精确 Run 的查看入口，不继续将启动按钮显示为“处理中”或再次提供可重复提交的主操作

#### Scenario: Change inputs after a preflight
- **WHEN** 用户修改模型、数据集、阶段、评测协议或受控参数
- **THEN** 旧预检和确认操作失效，页面返回配置步骤并提示需要重新预检；已启动的 Run 不受草稿变化影响

### Requirement: Long-running work exposes truthful, recoverable progress
后台任务运行期间，页面 MUST 从服务端身份恢复状态，并展示已用时间、最近状态更新时间、进度是否可测及日志入口；刷新或跨模块导航后 MUST 能继续查看同一任务。未有可信进度时 MUST 显示“进度未知”而非伪造百分比；结束时 MUST 明确呈现结果或失败及可执行的下一步。

#### Scenario: Evaluation lasts several minutes
- **WHEN** 评测已启动但尚未完成，且暂时没有可量化进度
- **THEN** 用户看到评测 Run、运行中、已用时间、最近更新时间、进度未知及日志/返回 Run 入口，而非无限期“启动中”

#### Scenario: Reload while a task runs
- **WHEN** 用户在训练或评测运行期间刷新或从另一模块返回
- **THEN** 页面读取持久任务身份与服务端状态，恢复当前步骤，不自动重启或重复提交任务

#### Scenario: Running task completes or fails
- **WHEN** 服务端报告任务完成、失败、部分完成或中断
- **THEN** 页面停止运行中提示，展示实际结果或错误原因及相应的查看结果、查看日志、重试预检等恢复操作

### Requirement: Diagnostic evidence is available without dominating primary tasks
工作区 MUST 优先展示可读名称、对象角色、状态和用户下一步；完整 ID、hash、文件路径、原始配置及日志 MUST 保持可访问但默认处于次级信息层。安全阻断、许可/质量限制及 lineage 不一致 MUST 保留明确提示，不得因视觉收纳被隐藏或误示为通过。

#### Scenario: Inspect a training or evaluation object
- **WHEN** 用户打开训练 Run、评测 Run 或模型/数据详情
- **THEN** 首层能识别对象名称、阶段、状态、主要输入和下一步；可展开查看完整身份、协议、摘要和原始日志

#### Scenario: Preflight is blocked
- **WHEN** 预检发现数据、模型、GPU、许可或评测条件不满足
- **THEN** 当前步骤显示具体阻断和可执行的恢复入口；原始诊断仍可展开，页面不会用笼统失败文案代替原因

### Requirement: Shared controls and feedback remain consistent across workspace modules
相同语义的选择器、输入框、按钮、步骤、状态提示和结果卡片 MUST 在数据、训练与评测、Runs、分析模块使用一致的可辨认样式与交互；桌面及 390 CSS 像素窄屏下 MUST 保持主要操作可见、无页面级横向溢出，并保持键盘焦点、可访问名称及错误提示关联。

#### Scenario: Continue from data to training and evaluation
- **WHEN** 用户从已登记数据集前往训练，再从成功 Run 发起评测
- **THEN** 三处的主次按钮、字段提示、运行状态和返回入口遵循同一视觉与交互规则，不因页面不同而退化为难辨认的原生控件

#### Scenario: Use a narrow or keyboard-only viewport
- **WHEN** 用户在 390 CSS 像素宽度或仅用键盘完成配置、预检、确认和查看结果
- **THEN** 主要操作可见可触达，焦点顺序跟随任务步骤，状态变化可被识别，日志/技术详情在自身区域滚动且不造成页面级横向溢出
