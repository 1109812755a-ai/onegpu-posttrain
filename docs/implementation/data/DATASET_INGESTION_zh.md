# 数据集导入与登记契约

本文定义本地数据导入的首版格式。数据来源、文件格式、阶段结构、质量状态和许可状态是不同维度：格式通过只说明记录符合该阶段 schema，不代表语义质量合格、来源真实或许可证已核验。

## 支持范围

本地导入需要用户显式选择 `cpt`、`sft`、`preference`、`grpo` 或仅评测的 `eval`，上传 UTF-8 JSONL。训练阶段非 SFT 只接受项目 canonical 格式；SFT 可显式选择 `canonical-v1`、`alpaca-instruction-v1`、`messages-role-content-v1` 或 `sharegpt-conversations-v1`；`eval` 可显式选择 `objective-choice-jsonl-v1`。不自动探测格式，也不自动映射 CSV/Parquet、不接受压缩包或用户指定落盘路径。源文件最多 256 MiB、最多 250,000 行；转换结果与原件合计最多 512 MiB。SFT 格式转换遇到无效行会拒绝整份文件；客观题评测集则逐行拒绝无效记录，只有至少一条有效记录时才允许登记，预览及登记结果必须清楚显示输入/有效/拒绝数量和行号原因，不会静默丢弃。

模型生成来源当前只支持 SFT 与 Preference（后者必须通过可信 chosen/rejected 质量策略）；CPT 生成尚无领域语料覆盖设计，GRPO 生成尚无可信 reward/reference 契约，因此二者明确为 unsupported。生成计划验证阶段能力时先于 provider 初始化和调用执行。

每一阶段的示例均位于 [`docs/examples/datasets/`](../../examples/datasets/)；这些文件同时作为 schema 回归 fixture。

## 阶段记录格式

| 阶段 | 必需内容 | 说明 |
| --- | --- | --- |
| CPT | `id`, `text`, `source` | 一条文档/文本记录；可选 `metadata` |
| SFT | `id`, `raw_problem`, `messages`, `prompt_template_id` | `messages` 必须含 user 与 assistant；message 仅允许 system/user/assistant |
| Preference | `id`, `raw_problem`, `prompt`, `chosen`, `rejected`, `prompt_template_id` | chosen/rejected 必须非空且不同；格式校验不证明排序正确 |
| GRPO | `id`, `raw_problem`, `prompt`, `answer`, `prompt_template_id` | schema 校验不证明 reward/verifier 可执行或可信 |
| Eval | `id`, `problem_id`, `raw_problem`, `prompt`, `answer`, `eval_prompt_version`, `question_type`, `options` | 仅供客观题评测；`single_choice`、`multiple_choice`、`true_false`；登记 role 固定为 `evaluation_only`，不属于训练数据 |

所有 canonical schema 均拒绝未声明字段。可选 `metadata` 是 JSON object，但导入器不会据此推断来源真实性、正确答案或记录级祖先。

## SFT 来源格式适配

格式必须与文件真实结构匹配，并在工作台显式选择：

| 格式 ID | 每行结构 | 规则 |
| --- | --- | --- |
| `canonical-v1` | 项目 SFT canonical 记录 | 原有路径和 dataset ID 规则不变 |
| `alpaca-instruction-v1` | `{"instruction":"…","input":"…","output":"…","system":"…"}` | `instruction` 或 `input` 至少一个非空；`output` 必填非空；`input`、`system` 可选；转换为 system? → user → assistant；若二者都有内容按固定分隔拼接 |
| `messages-role-content-v1` | `{"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}]}` | 保持完整对话顺序，角色仅允许 system/user/assistant，最后一轮必须是 assistant |
| `sharegpt-conversations-v1` | `{"conversations":[{"from":"human","value":"…"},{"from":"gpt","value":"…"}]}` | human→user、gpt→assistant；也接受 system/user/assistant；保持多轮顺序，最后一轮必须是 assistant |

转换预览使用 `POST /api/data/import/preview?stage=sft&source_format=<format-id>`，正文为 JSONL 字节。预览共用转换和 canonical schema 校验，最多返回 3 条样例及有界行错误；不会创建 Run、manifest、质量报告或登记产物。正式登记使用 `POST /api/data/import?stage=sft&source_format=<format-id>`，服务端会重新转换和校验，避免预览后文件变化造成绕过。成功产物在同一 workspace artifact 内保留原始字节 `raw/source.jsonl` 和训练读取的 `canonical/sft.jsonl`，manifest/processing report/lineage 均绑定来源格式、转换器版本/配置摘要以及原始与 canonical SHA256。失败时不登记半成品。

不支持 tool/function 角色、非字符串内容、多模态载荷、未知角色、缺少回答或不完整对话；遇到一行无效会拒绝整份文件，错误不回显正文。自动生成的 ID 是由源行内容与重复序号派生的本地 locator，不是上游稳定样本身份；若源文件提供字符串 `id`，仅作为 `metadata.source_record_id` 留存。导入仅做结构转换，不验证答案正确性、许可、质量或污染；用户仍须声明来源并确认使用权限。原件占用本地工作区磁盘，当前没有数据删除 API，请据此管理空间与来源数据生命周期。

## 来源与许可

用户可提供来源名称、URI、revision、许可标识及 attribution。每个信息分别记录状态：`known` / `unknown` / `not_provided`；许可可区分 `verified`、`user_declared_unverified`、`unknown`、`not_provided`。本地上传用户不能自行标记为 `verified`；用户声明不等于独立核验。来源信息缺失不阻止本地登记，但后续训练前会要求用户针对具体 dataset ID 确认有权本地使用；该确认不修改许可证状态，也不代表允许公开分发。

## 身份、内容摘要与血缘

- `original_file_sha256` 针对上传原始字节，作为来源快照摘要。
- `canonical_records_sha256` 针对按阶段 schema 解析、稳定序列化后的 JSONL 内容。
- 既有 canonical 导入的 dataset ID 继续由阶段和 canonical 内容摘要确定；外部格式导入还将来源格式转换配置摘要及原始文件摘要纳入身份，避免不同原格式/配置覆盖或误复用。匹配身份可复用，不覆盖既有产物。
- manifest 记录来源声明、两类摘要、样本数、处理协议、processing report、quality report 及 `data_build` Run；质量评估另有 `quality_assess` Run。
- 无法取得记录级祖先时标为 `dataset_only`，不使用文件名或行号伪造来源。

## 质量状态

导入会执行 canonical schema、重复 ID 等静态检查，并保存阶段质量报告。语义正确性、近重复、训练/评测交叉污染、答案可解析性等未执行的指标保持 `not_measured`；不将“schema 有效”提升成“质量通过”。建议样本量属于实验 readiness 信息，不是导入 gate。

## 示例校验

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dataset_ingestion_docs.py -q
```

正式导入入口会复用同一 canonical JSONL 契约。登记后可用 `onegpu-posttrain validate-lineage --root <dataset-artifact> --require-complete` 校验 manifest、报告摘要和 build/quality Run 闭环。

SFT 外部格式的 API walkthrough（先预览，再正式登记；示例均为一行 JSONL）：

```powershell
$format = "sharegpt-conversations-v1"
$body = [System.Text.Encoding]::UTF8.GetBytes('{"conversations":[{"from":"human","value":"1+1?"},{"from":"gpt","value":"2"}]}' + "`n")
$preview = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8765/api/data/import/preview?stage=sft&source_format=$format" -ContentType "application/x-ndjson" -Body $body
$preview | ConvertTo-Json -Depth 8
$result = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8765/api/data/import?stage=sft&source_format=$format&license_status=not_provided" -ContentType "application/x-ndjson" -Body $body
$result | Format-List dataset_id, record_count, original_sha256, canonical_sha256, manifest_path
```

将 `$format` 切换为 `alpaca-instruction-v1` 或 `messages-role-content-v1`，并同步替换 JSON 字段，可执行相同流程。自动化 walkthrough 覆盖三种转换格式、原件 SHA、canonical 消费文件和 lineage 完整性：`tests/test_dataset_import_api.py::test_external_sft_formats_api_walkthrough_preview_then_registration`。

## 客观题评测集导入与评分

外部 JSONL 每行格式为 `questionType`、`question`、`correctAnswer`，可选 `id`、`options`、`tags`。题型值为 `single_choice`、`multiple_choice`、`true_false`；选项允许 JSON 数组或 JSON 编码的数组字符串，因此兼容常见导出器。选项数为 2–26，按原顺序映射为 A–Z，答案使用标签；多选答案是标签数组（如 `["A","F"]`）。判断题答案可用 `true/false`、`✅/❌`、`对/错`、`正确/错误`。空题干、缺答案、越界标签、非法选项和重复显式 ID 按行拒绝，并给出行号/字段，不回显试题正文。有效行变为 `EvalRecord`，源件保存在 `raw/source.jsonl`，规范件为 `canonical/eval.jsonl`，报告包含格式/转换器版本、源和规范摘要及逐行拒绝明细。

登记只表示结构可评分，不证明题目质量、答案正确性、授权或代表性。`evaluation_only` 数据集只会出现在兼容模型的客观题评测选择器中；训练预检禁止使用它。启动评测时再次核验登记证据和 canonical 摘要，并与目标训练 Run 的 dataset ID/hash 对比，发现重合即阻断。`choice-accuracy-v1` 使用确定性 exact match：单选/判断精确匹配，多选答案集合完全相同才算正确；解析失败仍计入分母并单列。结果记录 suite ID、canonical SHA-256、样本覆盖和协议版本；仅当这些身份相同时可直接比较不同模型结果。导入和评测不是自动发生：必须由用户显式登记并在训练 Run 详情中选择评测协议与数据集。

