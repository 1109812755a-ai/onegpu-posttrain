# Design

## Context

See `proposal.md` - Why. 当前 `DatasetWorkbench` 将 catalog 中的 manifest 汇总为卡片，卡片只有 hash/count/source 等摘要。canonical 路径因来源适配器而异：本地上传通过 manifest metadata 声明，开源构建器通过 processing report 的 stage output 声明；旧报告可能没有完整 lineage、processing report 或 canonical 文件。现有通用 artifact 预览会主动隐藏样本字段，不适合直接复用来查看训练记录。

## Goals / Non-Goals

**Goals:**
- 以统一 dataset ID 为入口，为所有来源类型提供同一预览交互。
- 只返回 stage-aware 的训练字段和少量记录，并在读取前验证路径边界与内容摘要。
- 让无法验证的旧产物可理解地显示“暂不可预览”，不影响目录其他数据。

**Non-Goals:**
- 全量数据浏览、全文搜索、编辑、下载或导出。
- 以样本预览替代完整质量审核、许可审核或训练 readiness。
- 自动修复旧 manifest 或回写/规范化 canonical 文件。

## Decisions

1. **按 dataset ID 查找并验证，不接受客户端路径。** 后端从工作区 catalog 的已登记 manifest 建立候选映射；必须唯一解析到同一个 stage/hash。避免 API 变成任意本地文件读取器。若多份 manifest 指向同一 dataset ID 且摘要与 stage 一致，可选一个完整有效副本；证据互相冲突则拒绝预览并报状态。

2. **保留来源适配差异，统一成内部 canonical locator。** 本地上传从 manifest metadata 中的 `canonical_path` 解析（相对 dataset artifact root）；开源和模型生成构建从 manifest 引用的 processing report 中取得当前 stage 的 canonical output path（相对该 report 的 artifact root）。两类路径只按各自已声明契约解析，不扫描目录猜文件；统一拒绝符号链接逃逸、目录外路径、缺失文件及摘要不匹配。遇到旧报告只提示缺失证据。

3. **新增专用只读预览 API，而不放宽通用 artifact sanitizer。** 使用 `GET /api/data/datasets/{dataset_id}/preview?limit=3`；limit 默认 3、范围 1–10，API 逐行流式读取到上限并返回 total_count/truncated。每个 stage 使用字段 allowlist：CPT text；SFT raw_problem/messages/prompt_template_id；Preference raw_problem/prompt/chosen/rejected/prompt_template_id；GRPO raw_problem/prompt/answer/prompt_template_id。显示 record ID；忽略任意 metadata；字符串字段统一截断到 2,000 字符并给出截断标志。

4. **前端按需加载并局部展开。** 每张 dataset 卡片增加“预览样本”操作，打开卡片内折叠区或对话面板；只在用户点击后请求，不随 catalog 全量加载。显示阶段、来源、dataset ID、返回条数和摘要校验状态；错误和不可预览原因在该卡片内呈现，可重试，不清除其他页面筛选。

5. **状态不混合。** 样本预览成功只说明 canonical 字节与 manifest 摘要一致。现有 quality、license、revision、lineage 状态继续独立显示；UI 不将预览视为质量 PASS。

## Risks / Trade-offs

- [预览文本可能含用户或来源数据中的敏感内容] → 仅本地 API 按需访问，默认最多 3 条、硬上限 10 条，字段 allowlist、不返回 metadata、不提供批量导出；沿用 UI loopback 默认与远程访问显式 opt-in。
- [大文件或坏 JSON 行导致请求耗时/内存增加] → 流式读到限定样本数，单条序列化输出有界；digest 可通过流式 SHA256 校验，不在内存加载整份文件。
- [旧开源构建产物缺少 canonical 证据] → 卡片保留既有摘要显示，预览单独显示暂不可用原因，不让一个条目失败阻断 catalog。
- [同 dataset ID 的重复 manifest 证据不一致] → 拒绝显示正文并记录非敏感诊断，不静默采用任意一份。

## Migration Plan

不迁移、不修改已有数据或 manifest。先部署 API 与 UI，再用本地上传、开源、模型生成三类 fixture 验证预览；旧数据自动按可用证据呈现“可预览/暂不可预览”。回滚时移除新预览路由和 UI 操作即可，既有数据产物及 catalog 格式不变。
