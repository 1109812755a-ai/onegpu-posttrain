# ADR-001：提前交付只读 Data 页面

日期：2026-10-16。状态：采用。用户在本地明确同意提前实现 Data 页面；其余五页和 v0.1/Mini 发布门槛仍按路线推进。

## 选择

首个页面使用 React 19.3.1、React DOM 19.3.1、TypeScript 7.0.2、Vite 8.3.4。接口使用可选 `ui` extra：FastAPI 0.143.3、Uvicorn 0.54.0；API 测试使用 HTTPX 0.28.1。前端精确版本与锁文件固定；核心 CLI 的基础安装不需要 Web 依赖。

本页只需要步骤列表、证据表和分布条形图，采用原生 CSS 与语义 HTML；暂不引入 Ant Design/ECharts。完整六页发布前，依据实际交互复杂度评估组件库。前端由 Vite 构建，FastAPI 可同时提供同源生产静态文件，开发时 Vite 代理 `/api`。服务仅绑定 loopback。

## 数据与用户体验

- 优先显示真实 Smoke 的历史聚合报告，并明确标记旧版/推导/缺失；不会修改或回填历史文件。
- 演示数据由独立脚本显式生成，保存到 `artifacts/ui-demo/`，以“合成示例”标注并与真实数据分组。
- 展示处理轨迹、拒绝原因、逐步指标、阶段质量与分布、可验证的训练/评测关联。没有受控结果时显示“模型效果未评估”。
- 只按实际 dataset ID 血缘关联 run，不用目录名或训练 stage 猜测因果关系。
- 路径发现限制在声明的数据根与 run 根，按文件名/位置白名单读取，限制扫描深度/文件大小，阻止越界符号链接。浏览器按 opaque artifact ID 获取经过脱敏的汇总 JSON，不提供任意路径、原始样本、权重、命令或环境转储。
- 页面只读，提供刷新、URL 过滤状态、键盘可访问标签页和来源弹层；不提供数据修改或训练启动。

## English summary

Deliver a single read-only Data page early, explicitly reprioritized by the user. Keep React/Vite/TypeScript and an optional FastAPI extra, with exact dependencies and a frontend lockfile. Use semantic HTML/CSS for this slice. Read real legacy reports without rewriting them, distinguish synthetic demos, preserve missing values, and join runs only through explicit dataset lineage. Enforce bounded allowlisted filesystem reads, loopback binding, same-origin static delivery, and no process-control or artifact writes from the server.

References: [Vite requirements](https://vite.dev/guide/), [FastAPI server deployment](https://fastapi.tiangolo.com/deployment/manually/), [page specification](PAGE_SPEC_zh.md).
