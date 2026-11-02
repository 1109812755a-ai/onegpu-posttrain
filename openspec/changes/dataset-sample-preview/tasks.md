# Tasks

## 1. 安全的多来源 canonical 预览 API

- [x] 1.1 实现按 dataset ID 查找 manifest 与来源适配 canonical locator，支持 local upload、open-source 和 model-generated fixtures；测试路径 containment、符号链接、重复冲突 manifest、缺失文件和旧格式明确不可预览。
- [x] 1.2 增加有界只读预览 API 与 stage 字段 allowlist；测试 1–10 条限制、默认数量、字段/文本截断、metadata 不返回、SHA256 不匹配时不返回正文，以及无写入副作用。

## 2. 工作台预览交互

- [x] 2.1 为所有来源 dataset 卡片增加按需“预览样本”操作与预览面板；前端测试覆盖加载、空结果、不可预览、失败重试、关闭面板和键盘可达性。
- [x] 2.2 明确区分样本完整性校验、数据质量、来源/许可状态，并保留当前筛选与卡片状态；前端测试确保预览不会显示质量 PASS 或改写 catalog 元数据。

## 3. 文档与集成验收

- [x] 3.1 更新中文数据集工作台指南，说明三类来源预览范围、字段与行数上限、旧数据限制和隐私边界；按文档在本地验证四个阶段示例可预览。
- [x] 3.2 运行后端相关回归、前端测试及生产构建、compileall 和 OpenSpec strict validation；确认本地、开源、模型生成三类来源可预览，坏摘要/旧产物安全降级，且未修改 dataset/manifest/quality/lineage 文件。
