# Spec Delta

## ADDED Requirements

### Requirement: Stage comparison navigation opens the evaluation workspace
工作区 MUST 提供可访问的“阶段比较”导航入口，并在矩阵页面标记当前导航位置；从比较页可返回数据工作区。

#### Scenario: Open and leave stage comparison
- **WHEN** 用户选择阶段比较或从比较页选择 Data
- **THEN** 对应工作区显示，当前导航项正确标记

### Requirement: Matrix page explains loading and availability states
阶段比较页面 MUST 区分加载中、无矩阵、读取失败和有结果状态；读取失败时 MUST 提供重试，不得把空值表示为零。

#### Scenario: Matrix has not been produced
- **WHEN** 本地工作区没有可读取的评测矩阵
- **THEN** 页面说明尚无矩阵，并保留返回或重新加载操作

#### Scenario: Matrix request fails
- **WHEN** 本地矩阵 API 返回读取错误
- **THEN** 页面保留可见错误说明和重试入口

### Requirement: Comparison results remain usable on narrow screens and by keyboard
阶段比较页面 MUST 在 390 CSS 像素及以上视口避免页面级横向溢出，并使导航、矩阵选择和血缘详情可由键盘操作且具有可见焦点。

#### Scenario: Compare models on a narrow viewport
- **WHEN** 阶段比较在 390 CSS 像素视口显示
- **THEN** 指标和差值仍可读，页面不产生横向溢出

#### Scenario: Navigate comparison with a keyboard
- **WHEN** 用户仅使用键盘浏览矩阵页面
- **THEN** 交互项顺序可预期、焦点可见且状态可感知
