# Proposal

## Why

当前 SFT 数据生成准备过度依赖定向诊断证据，导致普通的“生成一批符合 SFT 格式的数据并走完训练链路”也被场景报告、训练/holdout 角色证据阻塞。用户需要先跑通数据生成→审核登记→训练→评测，再逐步启用有诊断证据的定向补数。数据数量是可配置的实验目标，不应作为质量 gate。

同时，前端不应要求用户理解 provider endpoint、模型 ID、token 限额和密钥变量等配置细节。优先复用工作区已配置的 provider profile；密钥只从本地环境读取，不在浏览器填写或展示。

## What Changes

- 将 SFT 生成分成两种明确模式：普通 SFT 生成以已登记 SFT 数据集提供训练主题，并使用项目默认生成模板，不依赖诊断报告；没有登记 SFT 数据时先引导用户导入/构建数据，避免虚构样本来源锚点。定向补数仍需有效的逐题评测、场景映射及隔离证据。
- 普通模式允许用户指定目标生成数量；数量仅为停止/完成目标，不作为数据质量或流程 readiness gate。达到/未达到目标均如实记录。
- 自动发现并默认选择已配置且凭据可用的 provider profile。普通用户只需确认数据集、目标数量和生成启动；高级预算/参数折叠，页面不要求输入 API key、endpoint 或模型名。
- 保留 provider 调用前显式确认、canonical SFT 格式/身份/答案一致性校验、质量报告、人工审核、显式数据集登记以及训练/评测的独立确认。
- 所有生成样本继续保留来源、生成 execution、输入 dataset、provider profile 摘要及后续训练模型 lineage；不得把未验证的语义正确性标记为已验证。

## Capabilities

### New Capabilities

- `guided-sft-generation-setup`: 从工作区已有数据和 provider 配置准备常规或定向 SFT 生成；以少量必要选择完成生成准备，并保留质量、审核和 lineage。

### Modified Capabilities

- 无。既有生成执行、数据登记和训练/评测链路保持兼容；本变更定义更轻量的生成准备入口。

## Impact

- 影响 SFT 生成 UI/API、workflow bundle/plan 构建、数据集/provider 发现、质量报告、数据登记和本地用户文档。
- 复用 canonical SFT schema、现有 provider profile、生成 execution、数据集 manifest 与 lineage；不增加数据库或远程服务。
- 自动测试使用隔离 fixture 与 fake provider；真实 provider、训练和评测仍需用户显式操作，不会自动下载模型或启动 GPU 作业。
