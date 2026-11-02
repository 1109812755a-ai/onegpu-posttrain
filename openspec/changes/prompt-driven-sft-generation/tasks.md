# Tasks

## 1. Prompt-only 方案与不可变输入

- [x] 1.1 新增独立 prompt-only SFT plan/bundle schema，约束主题、能力点、难度、附加约束长度、目标和预算；绑定模板、协议、policy、provider profile 摘要且不要求 source dataset；单测覆盖越界输入、摘要变化、稳定身份和历史不覆盖。
- [x] 1.2 加入只读 prompt-only readiness/preflight，验证 profile 凭据状态及默认依赖但不解析或回显密钥；API 测试证明浏览/准备/预检 provider 调用为零，且不接受路径和任意 prompt 模板覆盖。

## 2. 无来源锚点的 SFT 生成与质量检查

- [x] 2.1 实现固定 prompt 渲染与专用候选协议，将生成题目/解答映射至 canonical SFT 并由本地分配稳定记录 ID；fake-provider 单测覆盖有效输出、格式错误、重复、截断、请求预算和正确性未验证状态。
- [x] 2.2 注册 prompt-only SFT dataset：manifest `parents=[]`，记录 source_kind、execution/bundle、prompt/template/protocol/provider 摘要、目标与内容摘要；端到端测试沿 build/quality run lineage 验证无伪造父数据集且摘要一致。
- [x] 2.3 更新中文 SFT 生成指南，说明 seeded 与 prompt-only 的输入差异、正确性限制、审核登记流程及训练/评测分离；按文档完成 fake-provider prompt-only walkthrough。

## 3. API 与异步任务接入

- [x] 3.1 将 prompt-only bundle 接入现有 job 状态/进度/取消/恢复机制，并隔离 seeded 执行分支；API 测试覆盖显式确认才调用 provider、错误脱敏和 execution 身份持久化。
- [x] 3.2 复用候选预览与显式登记 API，禁止未审核、空数据或无效 canonical 数据登记；集成测试验证登记 dataset ID 可在数据目录发现并供 SFT Launcher 预检选择。

## 4. 前端双模式生成向导

- [x] 4.1 在数据工作台增加“参考已有数据集 / 按主题生成”模式选择；prompt-only 表单仅显示主题、能力点、难度、可选约束、目标数和已配置 Provider，不要求 dataset 且不显示路径/密钥；前端回归覆盖窄屏、空状态和模式切换。
- [x] 4.2 串接 prepare → preflight → 明确调用确认 → 实时进度 → 候选预览 → 人工确认登记；刷新后恢复 execution，显示拒绝原因及未验证限制；UX 回归确认浏览/准备/预检不触发生成。

## 5. 集成验收

- [x] 5.1 运行 prompt-only 与 seeded 定向回归、前端测试/生产构建、完整 pytest、compileall 和 `openspec validate prompt-driven-sft-generation --strict`；不调用真实 API、不启动 GPU 训练、不下载模型。
