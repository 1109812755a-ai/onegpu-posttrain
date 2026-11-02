# Tasks

## 1. 版本化 SFT 格式适配器

- [x] 1.1 定义 canonical、Alpaca、role/content messages、ShareGPT from/value 的格式 ID、字段规则和 converter config 摘要；验证未知/混用格式不会自动探测，旧 canonical schema fixture 不变。
- [x] 1.2 实现三种外部格式到 SFTRecord 的确定性转换，支持 system/user/assistant、human/gpt 映射和完整多轮顺序；用每种格式的 fixture 验证 canonical 字段、空值处理、可选 system/input 和输出顺序。
- [x] 1.3 实现可复现记录 ID 与源行定位 provenance，并区分生成 locator 和上游稳定 ID；测试重复源行、内容变化、缺失 ID 时 dataset-only 语义及不伪造来源身份。
- [x] 1.4 为缺字段、坏 JSON、空输入/回答、未知角色、tool/function、多模态/非字符串载荷定义有界安全错误；测试行号/字段准确、错误不回显输入正文且全文件失败时无静默跳行。

## 2. 转换预览与原子登记后端

- [x] 2.1 增加纯本地、有界的格式转换预览接口，共用正式转换器和 canonical 校验；API 测试确认返回限定样例/计数/错误、不创建 manifest/Run/质量产物且临时文件在成功、错误和取消后清理。
- [x] 2.2 扩展 SFT 注册过程，在同一 staging artifact 保留 raw/source.jsonl 与 canonical/sft.jsonl，并绑定 raw/canonical SHA、source_format、converter 版本/配置、行数及报告；测试路径包含性、总存储预算、原子提交和失败回滚。
- [x] 2.3 让转换配置摘要参与新格式身份且不改现有 canonical API 的 dataset ID 规则；测试同文件/同配置复用、不同 converter 版本或映射不覆盖旧 artifact、旧 manifest/catalog/训练读取兼容。
- [x] 2.4 将来源格式转换过程写入 processing report、manifest 和 build lineage，并保持 quality checks、许可状态和 canonical_path 语义；验证 lineage audit 可复核两份摘要且语义正确性仍为未测量。
- [x] 2.5 更新导入 API/中文数据工作台指南，说明各格式示例、角色映射、原件本地保存、资源边界、许可与不支持载荷；按文档执行各格式导入 API walkthrough。

## 3. 数据工作台用户流程

- [x] 3.1 在 SFT 导入流程加入显式来源格式选择和字段示例，保持其他阶段不出现 SFT 专属选项；组件测试覆盖 canonical 旧路径及 Alpaca/messages/ShareGPT 选项。
- [x] 3.2 增加转换预览、映射说明、有限 canonical 样例、行级错误与输入/输出计数；前端测试验证文件/格式变化会使预览失效、失败时可返回修改，且预览不直接登记。
- [x] 3.3 将预览通过、来源/许可声明和显式登记串成清晰流程；UI 测试验证重复提交防护、结构错误不可登记、质量/许可状态不被显示为已验证，并覆盖窄屏及键盘操作。

## 4. 集成与验收

- [x] 4.1 使用用户提供的 Alpaca 与 messages-role/content 样例，加上 ShareGPT from/value 和多轮 fixture，端到端验证预览→转换→原件留存→dataset ID/hash→训练预检消费 canonical；不得启动 Provider 或训练。
- [x] 4.2 运行相关完整后端和前端测试、类型检查、生产构建、lineage validation 与 `openspec validate sft-source-format-import --strict`；记录结果、失败与未执行项，确认 canonical 导入/API 向后兼容。
- [ ] 4.3 执行真实浏览器导入走查，核对格式切换、预览状态失效、拒绝错误行、来源许可提示、重复确认和登记后查看 lineage；覆盖桌面/390px/键盘并保存验收报告。

## Workflow follow-up

- 实现任务由 `$openspec-apply-change` 执行；本提案不代表任何来源格式已在前端可用。
- 归档前确认原始副本保留策略、工作区磁盘配额及数据生命周期说明已由项目负责人审阅。
