# Proposal

## Why

OneGPU-PostTrain 已能登记多个阶段数据集，但训练入口目前一次只接收一个 dataset ID；底层 ms-swift 支持多数据集加载，项目层却尚未把多来源组合做成可复现、可审查的实验输入。直接把多个路径临时传给训练会弱化数据质量复核与多父级血缘，因此需要将组合本身登记为新的不可变数据集版本。

## What Changes

- 在数据集工作台增加同阶段多数据集组合：支持拼接和显式权重采样，记录采样预算与 seed；来源类型保持独立于训练阶段。
- 将组合结果作为派生数据集登记，产生新的 dataset ID、内容摘要、阶段质量报告和指向所有父 dataset ID 的数据血缘；训练仍只消费一个明确的 dataset ID。
- 组合前校验阶段 canonical schema、数据角色/holdout 隔离和父级证据；GRPO 组合要求 reward/reference 契约一致，Preference 保留每对样本的排序证据。
- 组合数据集在统一数据模块创建，训练入口可选择已登记组合版本，或带阶段上下文进入数据模块创建后返回；预检和 run lineage 固定组合 dataset ID、父数据集及组合配置摘要。
- 不改变单数据集路径；不自动放宽质量规则、不将样本数量作为质量门槛，也不自动下载或生成数据。

## Capabilities

### New Capabilities

- `dataset-mixtures`: 显式、版本化地把同一训练阶段的多个已登记数据集组合为可复现的单一派生 dataset ID。

### Modified Capabilities

- `dataset-lineage`: 组合数据集必须保留全部父 dataset ID、组合配置摘要、实际内容摘要和构建 run。
- `stage-quality-review`: 组合后的实际数据集必须按目标阶段质检，报告按父来源覆盖/分母，不以父级报告代替组合产物质检。

## Impact

- 后端：dataset registry/import 与派生数据构建、stage-aware validation、Launcher/CLI 数据输入解析和 lineage。
- 前端：数据集工作台的多选、组合规则/权重/预算配置、预检摘要和 Launcher dataset 选择。
- 验证：四阶段组合 schema 与阻断条件测试、组合内容/父级摘要回归、训练 run 单 dataset ID 的 lineage 集成测试。
