# SFT 来源格式导入验收记录（2026-10-31）

## 验收范围

- 通过本机 FastAPI TestClient 对两份用户提供的 JSONL 执行 `preview → import`；输出写入临时 workspace，验收后清理。
- 验证原始文件字节留存、source SHA256、canonical SHA256、行数及 converter 输出。
- 不调用 Provider，不启动训练或评测。
- 真实浏览器桌面/390px/键盘走查另见 OpenSpec task 4.3；本记录不代替该走查。

## 用户文件结果

| 文件 | 实际结构/显式格式 | 预览 | 输入/输出 | dataset ID | 原始 SHA256 | canonical SHA256 |
| --- | --- | --- | ---: | --- | --- | --- |
| `datasets-lLdQc-f3blWz-alpaca-2026-10-31.jsonl` | Alpaca `instruction/input/system/output`，含空 instruction、非空 input；`alpaca-instruction-v1` | 通过 | 182 / 182 | `ds-sft-upload-224887c3781f902b78d1` | `6f59cd30ae9b2d98aef513bd98a926299421219c6b1437cd4095e0e933be20ed` | `a59f1a7ef6d0b664a84c293d7fe597d2072981db9a0833f609397dc58b144336` |
| `datasets-lLdQc-f3blWz-sharegpt-2026-10-31.jsonl` | 文件名虽含 ShareGPT，但内容实际为 `messages[{role,content}]`；显式选择 `messages-role-content-v1` | 通过 | 182 / 182 | `ds-sft-upload-09e4238810aea98e35fd` | `b5f15f75b21c4a0d2d2ea964416aa17f29f128d90a3c43e62e4947fd417ac484` | `c7861a171cf215c54d3eb794e2ba3d20e87cc4228269d1bfc0f65810b7831339` |

两份文件各自登记后，artifact 中 `raw/source.jsonl` 与上传字节逐字节相同，原始 SHA 与登记响应一致；canonical 文件的摘要与 manifest 一致。空 instruction 兼容规则已落实：Alpaca 的 instruction/input 至少一项非空，output 必须非空。另以 ShareGPT `conversations[{from,value}]` 多轮 fixture 验证 `human→user`、`gpt→assistant` 与对话顺序。

转换后的 messages canonical artifact 已由只读 SFT training preflight 消费测试验证：规范 schema、manifest/lineage 与 canonical SHA 可读；预检不启动执行。

## 自动化验证

- Python：完整后端 `pytest -q`：**834 passed, 8 skipped**（3:22）；覆盖转换器、错误安全、预览无副作用、登记原子性、身份摘要、lineage 和训练预检消费。
- 前端：完整 Vitest：**10 files, 34 passed**；新增 UI 覆盖无效预览阻断、重复确认只提交一次、许可不升级为已核验、键盘操作；Node UX 检查 **2 passed**，核对 390px 以下导入按钮/步骤的响应式规则。
- `npm run build` 通过；生产包提示既有单 chunk > 500 kB warning。
- `openspec validate sft-source-format-import --strict` 通过；lineage API 审计测试通过。

## 尚未执行

- OpenSpec task 4.3 需要真实浏览器桌面/390px/键盘走查。本会话可用环境没有浏览器交互控制能力；尝试启动临时浏览器检查时被运行策略拒绝，因此不把 Vitest 或 CSS 规则检查冒充成真实浏览器验收。导入页的服务端 API 与前端自动化测试均已通过。

## 8769 服务运行态复核

- 用户反馈预览路由返回 404 后，确认 8769 上的 UI 服务为旧进程（启动于 2026-10-30），进程未加载当前源码路由；没有活动 GPU 租约或运行中训练。
- 重启为当前工作区 `.venv` 中的服务后，health 返回 `ok`；用户的 Messages JSONL 文件以 `messages-role-content-v1` 预览返回 HTTP 200、182/182 条有效、只读且 Provider 调用数为 0。预览没有执行登记。
- 重启会轮换本地临时操作令牌；已打开的旧页面需强制刷新以重新获取令牌。

