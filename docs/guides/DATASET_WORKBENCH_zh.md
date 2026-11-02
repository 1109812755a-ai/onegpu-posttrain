# 数据来源与数据集工作台使用说明

数据来源回答“数据从哪里来”，训练阶段回答“数据要按什么结构处理”。来源不是阶段：同一来源可以产生不同阶段数据，不同来源也可以分别提供同一阶段数据。只有完成规范校验并写入 manifest 的数据才获得明确 dataset ID。

## 当前来源能力

| 来源 | CPT | SFT | Preference | GRPO | 说明 |
| --- | --- | --- | --- | --- | --- |
| FineMath（开源） | 可用 | — | — | — | 当前 builder 产出 CPT 文档块；固定来源 revision 写入 manifest |
| DeepMath（开源） | — | 可用 | — | — | 当前 builder 产出 SFT；具体质量/许可证据以 manifest 和报告为准 |
| OpenR1（开源） | — | 可用 | 可用 | 可用 | 按现有 builder 产出阶段数据；Preference 可供 DPO/SimPO 使用 |
| 本地 JSONL | 可用 | 可用 | 可用 | 可用 | 非 SFT 必须 canonical；SFT 支持 canonical、Alpaca、messages-role/content、ShareGPT 显式转换 |
| 模型生成 | 可用（探索性） | 可用（探索性） | 普通审核排序及定向独立核验均可登记（探索性） | 可生成 prompt 候选；没有可信 reference 不可登记/训练 | CPT 生成必须经人工审核；当前污染筛查仅有本地示例覆盖，不构成正式排除证明；Preference pair 必须有可审计排序证据；GRPO 模型候选仍需可信 reference/reward 证据 |

模型生成 CPT 目前提供受控探索路径：按领域/主题设置有限预算，普通生成不需要种子；定向生成必须绑定已登记 CPT 计划 ID/版本及其精确 train_seed 数据集身份。方案准备和预检不调用 Provider，只有显式确认后才启动；完成后先预览审核，再确认登记。文本事实仍是 `not_verified`。当前本地去污染索引只有 1 条示例，因此只能显示“示例范围筛查”，不可据此宣称正式污染排除或评测覆盖。Preference 普通生成须人工对同题候选排序，定向生成须通过独立答案核验后才登记；登记产物可交接 DPO/SimPO，但仍标记探索性。GRPO 工作台现提供普通 prompt 提示和定向计划两种候选生成方式；候选任务可恢复/取消、可预览，但由于生成结果不含可信 reference，不提供登记按钮，也不能直接训练。需要可训练 GRPO 数据时，可使用下方“可信训练种子派生”，仅接受绑定精确 OpenR1/SFT train_seed 证据的输入。模型自报答案或 reward 不构成证据。

模型生成执行可通过受 token 保护的只读 `GET /api/synthetic/status/{stage}/{execution_id}` 查询 CPT、SFT、Preference、GRPO 的进度、输入身份、请求错误、登记数据集及缺失证据；刷新查询不会重放 Provider 请求。按登记 dataset ID 查询结构化质量和来源血缘可使用 `GET /api/data/datasets/{dataset_id}/status`；它会返回已核验的 canonical hash、质量报告覆盖/缺失状态、来源许可提醒和父 lineage，不返回样本正文。训练 Run 与评测结果/lineage 使用 Launcher 的只读 Run/evaluation 接口。

## 本地文件登记

1. 打开 UI 左侧“数据”主模块下的“数据集”视图，找到“已登记数据集”。
2. 选择 CPT、SFT、Preference 或 GRPO，并选取 UTF-8 JSONL（最多 256 MiB、250,000 行）。SFT 还要显式选择 Canonical、Alpaca、role/content messages 或 ShareGPT 格式；不会猜格式。点击“预览并校验”查看转换后的 canonical 样例、输入/输出数量及有界行级错误。任何一行错误都会拒绝整份文件，不会静默删行。
3. 预览通过后点击“继续核对来源与许可”，填写已知来源、revision、许可证和 attribution；确实未知时选择“未知”，没有信息时保留“未提供”。最终确认时服务端会重新转换并校验；SFT 外部格式的原始文件保留在 `raw/source.jsonl`，训练仍只读取 `canonical/sft.jsonl`。源文件与转换结果总大小不超过 512 MiB。
4. 登记成功后页面显示 dataset ID、样本数和实际质量状态。在对应数据卡片选择“查看质量与血缘详情”，即可查看质量指标、处理步骤、登记清单和明确关联的 Run；“打开完整处理报告”可回到处理与质量视图，“查看 Run”打开同一条运行记录。选择“在训练工作台选择此数据”会按数据阶段进入训练与评测模块，并把精确 `dataset_id` 预选到相符的阶段训练数据中；Preference 默认进入 DPO（也可在 Launcher 改为 SimPO）。该导航不会预检或启动训练。CPT、Mini SFT、DPO、SimPO、GRPO 均有受控 Web 启动路径，但必须重新预检并由用户显式确认；本地依赖、数据契约、父模型或 GPU 不满足时会显示具体阻断原因。

### GRPO 数据准备与阻断恢复

GRPO 训练只接受已登记且逐条绑定可执行 reward contract 与可信 reference 的 `train_prompts`。旧数据或普通上传数据即使格式正确，也不会因此自动满足 reward/reference 信任条件；它们会继续显示后端给出的不可训练原因。若 GRPO 数据都被阻断，训练工作台显示“准备 GRPO 数据”，直达本页的 GRPO 数据准备卡片。

在“从可信训练种子派生 GRPO 数据”卡片选择已登记 OpenR1 SFT `train_seed`、设置目标条数与 seed，先点“预检 GRPO 数据”，检查可用数量、contract 与上游正确性证据，再点“确认派生并登记”。派生预检的完整要点与 SHA-256 会绑定到新数据集 manifest 和每条 canonical 记录；流程同时生成质量报告和数据血缘。它不是模型生成，也不声称独立复核上游正确性标签。`not_provided`/未知许可会继续显示为提醒，需要使用者自行确认来源使用权限。

从 Launcher 进入时会保留 `grpo` 阶段、当前 profile、父模型和返回目标。登记后点击“去 GRPO 训练工作台使用”，只会将这个精确 dataset ID 带回并重新核对 profile/父模型兼容性；不会沿用旧预检或启动令牌，也不会自动启动。返回 Launcher 后再次预检，查看所有提醒，必要时确认许可并显式启动。

### Preference 普通生成与定向补数

在模型生成卡片切换到“Preference 偏好对”。普通模式下每行输入一道题，先准备方案，再执行只读预检（Provider 调用数为 0），最后明确启动候选生成。候选不会带有模型自报的正确性/偏好标签；逐题选择不同的 chosen 与 rejected 候选并填写审阅者标识和排序理由后，才可显式登记。审阅者身份是本地用户声明，系统不会声称已认证其真实身份。

定向模式选择已登记的 Preference 定向计划；服务端按计划中的精确 `train_seed` dataset ID 与 canonical hash 读取训练种子，不读取诊断集或 final holdout。只有独立最终答案/reference 校验能区分正误时才形成偏好对；该校验不评价推理过程质量。生成任务会持久化 execution ID，刷新后可重新读取候选和排序证据；服务重启不会自动重放未知 Provider 请求。登记只创建 Preference 数据集，不启动训练；随后可把同一精确 dataset ID 分别用于独立的 DPO 或 SimPO Run，二者父模型及 Run 仍各自预检和确认。质量与许可状态不会因登记而提升。

来源或许可未知仍可登记。本地上传只接受用户声明未核验、未知或未提供，不能自行标记“已独立核验”。声明状态与质量状态互不覆盖：格式通过不代表语义正确、没有重复、没有污染或许可已核验。没有运行的质量指标继续显示 `not_measured`。建议数量不是导入 gate。

SFT 外部来源格式转换仅保留结构和对话顺序：Alpaca 的 `instruction` 或 `input` 至少一项非空，`output` 必填，`input`/`system` 可选；messages 格式只接受 system/user/assistant；ShareGPT 另外把 `human` 映射为 user、`gpt` 映射为 assistant。tool/function、多模态和非字符串内容不支持。自动记录 ID 是本地源行 locator，不代表上游样本身份；原始文件和转换配置摘要进入登记血缘。预览不会登记数据集或运行训练/Provider。

## 组合多个训练数据集

在“组合已登记数据集”卡片中，先选阶段，再勾选至少两个同阶段的数据集。可选“全部拼接”（按勾选顺序合并全部记录）或“按权重确定性采样”（指定目标条数、每个来源权重和随机种子；无放回采样，重复运行可复现）。点击“预检组合”只读校验阶段 schema、内容 SHA-256、训练角色和可追溯父级；确认摘要后再点击“确认创建组合数据集”。组合会创建一个新的 dataset ID，不修改输入数据集；输出会重新跑质量报告，质量状态不会从父集继承。完成后可直接把新 ID 带到训练工作台，但不会自动启动训练。

Preference 的一行是完整 chosen/rejected pair，组合和采样不会拆分 pair。GRPO 只有在所有来源具有相同且已验证的 reward/reference 契约时才允许组合。诊断集、holdout、benchmark 等隔离数据不能进入训练组合，且会递归检查已知组合祖先。组合版本保留父 dataset ID 与摘要、配置摘要和逐条 provenance sidecar（父 ID、父摘要、原 record ID、原行号）；sidecar 位于 artifact 的 `provenance/records.jsonl`，不会进入训练样本正文。组合数据仍需逐一确认各父来源许可。

本地 API 对应 `POST /api/data/mixtures/preflight` 和 `POST /api/data/mixtures/build`，请求体形如：

```json
{"stage":"sft","dataset_ids":["ds-sft-upload-a…","ds-sft-upload-b…"],"strategy":"concat"}
```

加权采样还需提供 `budget`、`seed` 和以 dataset ID 为 key 的正数 `weights`。build API 只监听本机 loopback 模式，不支持远程服务；创建成功后返回组合 dataset ID、质量状态、manifest 路径和来源贡献计数。

## 样本预览

在任一已登记数据集卡片上选择“预览样本”，工作台会按需从本机 API 读取最多 3 条；不随目录刷新批量加载。预览适用于本地导入、开源构建和模型生成三类来源，前提是 manifest 与 processing report（如适用）能明确定位 canonical 文件，且文件 SHA256、阶段和记录数均与登记证据一致。旧产物缺少路径或摘要、文件缺失、证据冲突或摘要不匹配时，会在该卡片显示原因并允许重试；不会猜路径，也不影响其他卡片和当前筛选。

“查看质量与血缘详情”会按需读取该数据集所在构建的处理报告，展示质量指标、处理步骤及明确引用该构建的 Run。报告缺失时可重试；若没有显式 lineage 关系，页面会显示“尚无明确引用”，不会仅根据相同阶段或相似名称推断。

预览字段遵循阶段 schema：

| 阶段 | 预览字段 |
| --- | --- |
| CPT | `text` |
| SFT | `raw_problem`、`messages`、`prompt_template_id` |
| Preference（DPO / SimPO） | `raw_problem`、`prompt`、`chosen`、`rejected`、`prompt_template_id` |
| GRPO | `raw_problem`、`prompt`、`answer`、`prompt_template_id` |

每条记录的 ID 会显示；任意 metadata 不返回。单字段文本最多显示 2,000 个字符，超长字段会被截断并标出。后端接口可用 `limit=1..10` 调试，UI 默认固定显示前 3 条；不会提供全量浏览、编辑或导出。

预览提示“canonical SHA256 已核验”只表示当前字节与登记摘要一致，不表示质量 PASS、许可证已核验或数据可用于任意训练。卡片上的质量、来源、许可、revision 和血缘保持独立，不会因预览而改写。样本正文可能包含敏感或受许可约束的内容；仅在本地 UI 中按用户操作读取，使用前仍应检查来源授权，分享截图或日志前也请确认样本不含敏感信息。

## API walkthrough（干净 workspace）

以下命令使用仓库中的一条 SFT 示例，只写入当前 workspace 的 `data/processed/user-upload/`；示例 URI/许可未提供是有意的。请先启动本地 UI 服务，再在项目根目录运行 PowerShell：

```powershell
$body = [System.IO.File]::ReadAllBytes((Resolve-Path ".\docs\examples\datasets\sft.jsonl"))
$uri = "http://127.0.0.1:8765/api/data/import?stage=sft&source_name=docs-example&revision_status=not_provided&license_status=not_provided"
$result = Invoke-RestMethod -Method Post -Uri $uri -ContentType "application/x-ndjson" -Body $body
$result | Format-List dataset_id, stage, record_count, original_sha256, canonical_sha256, quality_status, manifest_path
```

预期结果：`stage=sft`、`record_count=1`、返回稳定 `dataset_id` 和两个不同用途的 SHA256；quality status 不会被 schema 校验自动提升为 PASS。可在工作台刷新后找到该数据集，或运行：

```powershell
$manifestPath = Join-Path (Get-Location) $result.manifest_path
$artifactRoot = Split-Path (Split-Path $manifestPath -Parent) -Parent
.\.venv\Scripts\python.exe -m onegpu_posttrain validate-lineage --root $artifactRoot --require-complete
```

直接校验 API 返回的 artifact 路径也可用仓库内测试验证四阶段 fixture：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_dataset_import_api.py tests/test_dataset_ingestion_docs.py -q
```

## 训练消费

登记只代表获得稳定 `dataset_id`，不代表适用于任意训练配方。训练前按 manifest 验证 stage 和 canonical hash，再转换成 ms-swift native JSONL；转换产物包含 `.adapter.json` 与 record lineage sidecar，训练 CLI 可据此回链到 canonical dataset manifest。适配产物是不可覆盖的新目录，不会改写登记数据。

训练与评测模块的阶段 Launcher 可按已登记 dataset ID 选择 CPT、SFT、Preference（供 DPO/SimPO）或 GRPO 数据；从数据集卡片进入时会传递并预选对象 ID，但仍需用户选择父模型、检查阶段 profile 并手动预检/启动。Mini SFT Launcher 仍固定 Mini M1-CPT、1024-token、seed 42 和 LoRA 参数。预检展示 dataset ID、canonical/native hash、质量和许可状态。非 `verified` 许可需要显式勾选确认；确认只绑定本次 dataset ID/Run，不会把许可证状态升级。

```powershell
$manifest = ".\data\processed\user-upload\<dataset_id>\manifests\sft-user-upload-v1.yaml"
$nativeDir = ".\data\adapted\<dataset_id>"
.\.venv\Scripts\python.exe -m onegpu_posttrain adapt-registered-dataset --manifest $manifest --output $nativeDir
```

CLI 会拒绝 stage 与 recipe 不匹配的数据。将 recipe 的 `dataset_id` 与 manifest 路径设置为本次登记产物，再把转换输出作为 `--dataset` 输入：

```powershell
$cli = ".\.venv\Scripts\python.exe -m onegpu_posttrain"
& $cli run-peft-cpt --config <cpt-recipe.yaml>     --model <parent-model> --dataset "$nativeDir\native\cpt.jsonl"        --output <run-output> --merge
& $cli run-sft   --config <sft-recipe.yaml>        --model <parent-model> --dataset "$nativeDir\native\sft.jsonl"        --output <run-output>
& $cli run-dpo   --config <dpo-recipe.yaml>        --model <sft-model>    --dataset "$nativeDir\native\preference.jsonl" --output <run-output> --merge
& $cli run-simpo --config <simpo-recipe.yaml>      --model <sft-model>    --dataset "$nativeDir\native\preference.jsonl" --output <run-output> --merge
& $cli run-grpo  --config <grpo-recipe.yaml>       --model <sft-model>    --dataset "$nativeDir\native\grpo.jsonl"        --output <run-output> --merge
```

`<run-output>` 每次必须是新目录；DPO 与 SimPO 可消费同一已登记 Preference dataset，但要分别使用各自 recipe/Run。真实样本量由实际登记 artifact 决定，不设 5K/2K 数量门槛；语义质量与许可状态继续按报告原样展示。适配验证可运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_registered_dataset_adapter.py tests/test_launcher_registered_datasets.py -q
```
