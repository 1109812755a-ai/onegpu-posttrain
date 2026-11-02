# Design

## Context

动机见 [proposal.md](proposal.md)。本变更跨数据构建、实验登记、CLI、报告及可选 Web API，需要明确共享语义与默认只读边界。

已核对的实现与证据：

- `data/openr1.py` 的 `_make_sft` 将无正确候选和正确候选全部超长合并计为 `sft_no_eligible_correct_trace`；`_make_preference` 在长度过滤后才记录缺侧。当前报告不能单凭这些计数定位语义问题。
- Mini build-02 的 `processing_report.json` 记录 SFT/Preference 长度上限均为 1024。其正确候选渲染长度中位数分别为 4747/4684 tokens，错误候选中位数为 7308 tokens。该分布支持优先检查资源适配，但不能推出放宽长度后精确可得到多少样本。
- `_pool` 当前按目标配额加权分组，因此 A/B 不能通过修改目标隐式重分配题目。正式对照需要单独固定来源池和分组。
- `data/evidence.py`、`manifest.py`、`lineage.py` 已有处理/质量/来源摘要和 run/model/evaluation 身份；`data_effect_experiment.json` / `data_effect_report.json` 仍是设计，不能假设已实现。
- `ui/server.py` 只有 GET 路由，`tests/test_ui_api.py` 校验 POST 被拒绝；`web/src/main.tsx` 已有处理流程与质量页面。ADR-001 和前端路线原先排除写入与配置编辑，需要在实施本变更时显式修订。
- `cache-raw-dataset-sources` 已提供本地原始来源能力，但当前没有要求建立实际源缓存；用户已要求暂缓大规模下载。已有筛选后 JSONL 不能还原被拒绝的候选。

## Goals / Non-Goals

**Goals:**

- 用同一执行服务支持 CLI、有限预览和前端配置，策略身份可追溯，真实诊断能解释筛选与资源预算。
- 允许已有小规模数据完成探索工作流，为正式 Mini 保留独立验收；探索不会伪造 verified、完整 reasoning 或 Mini 已达标。
- 落地预登记和可复查的数据效果报告，为下一变更的评测驱动定向合成提供基础。

**Non-Goals:**

- 当前切片不提供任意规则脚本、完整图形流程编排、训练启动按钮、队列/调度器、数据库或自动迭代。
- 不调用外部模型，不下载源数据/tokenizer，不执行 GPU 训练，不将新功能验收等同于模型提升。
- 首个编辑/构建切片覆盖 SFT 和数学正确性排序的 Preference；CPT/GRPO 保留现有构建/训练路径，后续按实际需要扩展策略编辑。

## Decisions

### 1. 先建立共享协议，再实现规则表单

新增数据策略类型和共享执行服务，沿用 Pydantic、JSON/YAML、本地文件。协议包含 `schema_version`、`policy_id/version`、`stage`、`usage`、`rules`、`resource_limits`、tokenizer/模板引用和解析配置摘要。不可变身份按规范化配置摘要派生；显示名称不能用作身份。

三类设置分别展示：

| 类别 | 首个切片行为 |
|---|---|
| 质量策略 | 完整性、最终答案解析、SFT 正确性采用 `filter` 或探索用 `report_only`；Preference 正确性排序保留真实 true/false 依据，长度比为独立选择规则 |
| 资源预算 | 渲染后 token 上限、样本/pair 目标、扫描预算，显示依赖 tokenizer 与 recipe，不显示为综合质量分 |
| 固定完整性 | 格式、来源/许可声明、引用摘要、固定评测隔离、污染防护；未知项保留状态，不能在实验中关闭或标成通过 |

模板提供兼容现有筛选语义的基线和显式探索策略；独立数学验证尚未实现时显示不可用，不能靠开关假装执行。策略不包含 Python 表达式。相较于仅将现有参数搬到浏览器，这一协议使 CLI、合成数据和 UI 使用同一语义。

### 2. 新策略路径版本化，保留旧 builder 默认行为

新路径采用 `openr1-policy-build-v1`，抽取并复用现有规范化、候选解析、token 计数和记录导出逻辑。原 `build-openr1` 不指定新路径时仍使用历史协议与资格语义；增加诊断可作为附加字段，旧字段继续按历史含义读取。

每个候选先保留观察事实，再由策略选择，避免 `_trace_candidates` 的提前删除使 `report_only` 无法工作。未知 correctness 不转换成错误；SFT 记录允许明确的非 verified 探索状态；Preference 不因放宽长度或完整性就伪造 chosen/rejected 正确性。

首拒绝计数与重叠诊断分开。SFT 至少区分 `no_known_correct_candidate`、`correct_candidates_all_over_length`；Preference 分别记录长度过滤前缺少 correct/wrong/known labels 和过滤后的缺侧。每个数值带问题/trace/pair 单位和分母。历史合并计数不能被自动拆解或回填。

### 3. 固定来源池与问题分组，再应用 A/B 规则

`source_pool.json` 固定来源 revision/content digest、规范化与污染索引版本、tokenizer/模板和输入范围；`split_snapshot.json` 固定问题身份算法、seed、阶段分配权重与子 split。分配配置与输出目标解耦，必要的 problem-ID 映射以不含正文的 sidecar 保留摘要。

新策略先在相同的非空规范化、去重及污染防护来源池上确定问题分组，再应用各组资格和选择规则。该顺序与旧路径不同，因此必须使用新协议和新的 manifests。A/B 的数量目标不得成为 split 函数的隐式输入；相同题目的所有候选与派生数据继承问题分组。不能用不稳定的处理行号作为祖先身份。

若用户只选择已筛选数据，将其声明为 `existing_dataset` 来源池，允许重新评估或进一步筛选；界面明确无法恢复先前丢弃数据，也不能将它称为原始全源对照。

### 4. 有限预览与完整构建调用同一服务

建议 CLI 边界为 `validate-data-policy`、`preview-data-policy`、`build-policy-data`，明确提供策略、source-cache manifest 或本地 JSONL、split snapshot、本地 tokenizer 与输出 root。未知参数与冲突来源输入拒绝。完整构建采用现有 report/manifest/lineage writer，写入新的构建目录。

UI 只引用服务器预先配置并发现的本地来源 ID。预览默认处理 200 个 raw rows、最大 1000 个，执行预算默认 30 秒并使用协作式检查；这些预算作为版本化服务设置与报告范围保存，不是质量阈值。报告注明确定性前缀抽样偏差、不完整扫描及 stop reason。完整 hash 校验和输入准备耗时独立记录，不能声称 30 秒预算可以中断阻塞的文件系统调用。

缺少本地来源、污染索引或 tokenizer 时返回结构化缺失状态。预览不调用源 materializer，也不以字符数冒充真实 tokenizer。只在用户显式执行完整 CLI 构建时处理全量本地来源，且不会从缺失本地输入隐式切回远程流式下载。

### 5. 探索用途与正式 Mini 分别记录

`usage=exploratory` 记录实际 counts、所用策略、检查状态及可训练资格；正数数据可导出训练配置，不把 5K/2K 等正式配额作为探索阻断。空数据、格式/引用错误、评测污染仍阻断。用户可基于现有完整 manifest 做小规模执行；小样本不意味着模型结果具有统计代表性。

新策略 manifest/experiment 引用 `usage` 和 policy ID，训练 resolved config/lineage 保留它们。基线、探索和后续 `intentionally_noisy` 是用途/来源事实，不是三档质量分。新策略数据不能只凭相同数量自动通过官方 Mini；需要对应新协议经审核的 Mini profile。旧 `mini-data-readiness@1` 仍仅回答自己的正式验收问题。

### 6. 落地两类实验与兼容性汇总

沿用详细设计的 `data_effect_experiment.json`，新增登记/校验与 `compare-data-experiment` CLI。计划保存 source/split、两组 policy/dataset refs、parent hash、recipe、预算类型及实际值、seed、eval datasets/protocol、主指标和保护指标；训练之前登记并固定配置。

- `controlled_quality_ablation` 需要一个声明变量，其余控制条件匹配；按 examples/tokens/steps/completions 中明确的预算进行校验，不能把其中一种当成其他预算相同。难度/长度等残余分布差异仍须报告。
- `pipeline_effect` 允许策略带来的真实数量与组成变化，但不把模型 delta 归因于某条过滤器。
- SFT 对照从相同 M1 开始；DPO/SimPO 对照从同一 M2 分叉。一个实验组的模型不能默认成为另一组的 parent。
- 若提高数据上限需要同时改变训练 max_length 或其他 recipe，显式作为流水线/资源策略实验登记。不能把增大的显存成本或训练截断隐藏在“更高数据质量”结论中。

汇总通过 dataset/run/model/eval IDs 与摘要连接，兼容既有 reporting 的状态语义。数据-only 报告可先生成，模型结果为空；实际 lineage 不匹配、评测协议/集合不一致或预算违反声明时标 `NOT_COMPARABLE` 并保留原值。每个比例保留分母，单 seed 仅作描述，失败/OOM 不从比较中消失。

### 7. 前端按一个用户任务逐步展开

复用 Data 页面，增加“策略对照”区域：选择来源与阶段 → 复制模板并编辑 A/B → 查看规则差异和校验 → 明确范围后预览 → 保存/导出计划与命令 → 关联已有结果。数据集血缘和当前 URL 筛选继续可用。

默认展现目标问题、修改项、数量及关键警告；完整配置、方法、分母和来源按需展开。每项规则有含义、单位、默认值和恢复操作；质量与资源不同分组。预览结果绑定配置摘要，编辑后标过期；请求序号/摘要防止旧请求覆盖新草稿。无来源、加载、超预算、失败、无有效样本与未训练均有具体下一步。键盘、焦点、窄屏和表格替代作为验收条件。

以 `serve-ui --enable-policy-editor` 显式启用受限写入；默认 GET 行为和只读 health 声明保持。编辑模式使用新增版本的能力声明，写接口只接收强类型配置、已发现来源 ID 和实验 ID。服务生成文件名并仅向专用 experiments 根写入新文件，检查解析后路径及符号链接，拒绝覆盖历史产物。非 GET 操作执行同源/Origin 和应用请求令牌校验，不接受任意命令、路径或脚本。CLI 导出用固定模板，不将浏览器字符串拼成 shell 命令。预览在进程内有界执行，每个服务实例最多一个预览，不建立后台任务调度系统。

### 8. 与定向生成闭环及发布计划衔接

下一变更 `targeted-synthetic-data` 消费本变更的策略、数据效果报告及实验身份，实现：诊断评测的场景缺口 → 定向生成计划 → 本地训练种子/生成候选 → 验证与策略分组 → 训练 → 目标场景及通用保护指标复测。生成供应方作为额外数据来源接入同一协议，不成为核心安装或实验复现前提。

实施时同时更新中英文 frontend roadmap、ADR-001 的后继 ADR、Development/Open Source Plan 和 Data Processing Effect Design：明确提前交付数据策略交互与有限比较，完整 Studio/launcher/调度器按原路线后续评审；合成数据的小型实验切片提前，但自动飞轮与复杂平台仍属长期方向。官方 v0.1 的四项必需 OVAT 不因本变更自动增加一项或改变门槛。

## Risks / Trade-offs

- [仅凭聚合报告无法确定各规则的独立损失] → 保留历史不确定性，新增诊断只对新执行输出；缺原始输入就不声称已完成预览。
- [更长样本突破 8GB 预算] → recipe 兼容检查、资源变量单列；探索生成短而完整答案由下一变更评估，不直接截断推理充数。
- [筛选引起数据量和难度混杂] → 事先登记比较类型与预算，同时呈现覆盖；不作普遍的质量分数/能力因果宣称。
- [本地写接口扩大原只读服务边界] → 显式 opt-in、受限文件根、强类型字段、同源校验和无进程控制的 API 回归测试。
- [新旧切分/数据协议混用] → 新协议与 split snapshot 摘要参与身份，旧数据不原位迁移；跨协议比较必须满足登记的固定控制条件。

## Migration Plan

1. 先实现策略/实验类型、诊断及离线 fixture，确保 legacy 默认输出和质量含义不变。
2. 实现固定 pool/split、共享预览/构建和数据效果报告，使用项目自有小样本完成 CPU 证据链。
3. 增加默认只读下可拒绝、opt-in 可保存/预览的 API 和前端任务流，保留独立 CLI。
4. 完成中文操作指南与中英文路线同步，记录新协议和探索用途；不在本阶段执行下载或 GPU 工作。
5. 回滚为关闭 editor 并停用新策略路径；历史策略、数据和报告仍可只读检查，旧 CLI 工作流继续可用。

## Open Questions

没有阻止当前切片实施的未决问题。实际可用本地来源规模、预览性能和 GPU 资源结果在后续明确执行时测量，不作为已知事实写入此设计。
