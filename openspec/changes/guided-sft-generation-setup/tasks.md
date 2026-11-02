# Tasks

## 1. Workspace 输入发现与就绪状态

- [x] 1.1 实现只读 SFT 准备状态服务，按登记 ID/manifest 发现阶段兼容的数据集、provider profile、项目默认模板/协议/policy、匹配的诊断报告和隔离数据；输出来源、摘要、资格/缺项原因且不返回密钥值；用 workspace fixture 测试身份匹配、过期文件、缺项及零 provider 调用。
- [x] 1.2 增加受 token 与 loopback 保护的 discovery/readiness API，并补充中文接口/状态说明；API 测试验证不接受路径、命令或任意配置覆盖，错误不会暴露凭据。
- [x] 1.3 readiness 安全识别 `.env.local` 中指定凭据是否存在（不加载或返回密钥），并提供被 Git 忽略的本地 DashScope profile；测试确认密钥值不会进入响应。
- [x] 1.4 扩展发现/readiness 为普通 SFT 与定向补数分别判定：普通模式不要求诊断/holdout 证据，定向模式继续要求有效诊断与隔离证据；测试确认样本数量不会成为阻断 gate。

## 2. 自动组装并登记不可变方案

- [x] 2.1 增加 ID 驱动的 prepare 操作与独立 `OrdinarySFTPlan`/bundle：普通模式绑定已登记 source dataset 摘要、profile、template、protocol、policy 和目标数量，不依赖 report/scenario/role/holdout；定向模式继续使用现有 plan 与严格证据校验。API 不接受路径、命令或任意配置覆盖；覆盖缺证据、身份/holdout 冲突、路径边界、摘要变化及不覆盖历史方案。
- [x] 2.2 验证 prepare/重开页面不调用 provider，且 bundle/execution lineage 记录模式、输入摘要、provider profile 摘要、目标数与生成身份；更新中文指南，说明普通生成无需诊断报告、如何审核登记及如何单独训练/评测，保留 CLI 作为高级入口。

## 3. 前端生成准备向导

- [x] 3.1 提供普通 SFT 生成与定向补数的明确模式入口；普通流程只要求确认数据集/默认模板、目标数量和默认 provider，定向流程展示诊断证据与隔离 readiness，不展示任意路径表单。
- [x] 3.2 自动选中唯一凭据可用的 provider profile；页面不接收密钥，多个 profile 才要求选择，无可用 profile 时给出短本地配置说明；预算与 endpoint/model/token 等技术设置折叠。连接 prepare → 预检 → 明确调用/预算确认 → 生成。
- [x] 3.3 显示请求目标、实际候选数、拒绝原因、质量/未验证项；候选须人工审核并显式登记，生成数量不作为质量 gate；训练和评测保持独立动作。补充刷新恢复、错误状态和零未确认 provider 调用测试，完成前端构建。

## 4. 端到端回归

- [x] 4.1 添加离线集成测试，覆盖普通模式数据/默认模板 → 自动方案准备 → fake generation → 质量审核 → 显式登记，并断言生成 execution、dataset manifest 和后续 run lineage 身份一致；另测定向路径在诊断证据缺失时被阻断而普通路径可继续。
- [x] 4.2 运行相关后端/前端测试、完整测试套件、编译与 OpenSpec strict validation；报告真实结果，不触发真实 API、GPU 训练或模型下载。
