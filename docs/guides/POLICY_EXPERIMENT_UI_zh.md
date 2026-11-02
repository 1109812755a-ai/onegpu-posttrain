# 数据策略实验与前端操作指南

本文介绍本地策略对照编辑器和离线示例。它用于观察筛选规则如何改变数据数量、质量诊断和覆盖；没有真实训练/评测证据时，模型效果始终显示为“未评估”。

## 启动

基础启动仍是只读模式：

```powershell
onegpu-posttrain serve-ui --workspace .
```

显式启用本机策略编辑、预览和草稿保存：

```powershell
onegpu-posttrain serve-ui --workspace . --enable-policy-editor
```

服务默认只绑定 `127.0.0.1`。如确实要从同一局域网的其他设备访问，可显式运行：

```powershell
onegpu-posttrain serve-ui --workspace . --host 0.0.0.0 --port 8766 --allow-remote
```

然后在其他设备打开运行服务电脑的局域网 IP，例如 `http://192.168.1.20:8766`。远程模式会接受非本机 Host，并让所有能连接该端口的客户端读取本地 UI 数据；同时启用 `--enable-policy-editor` 时，还会开放策略预览/草稿保存接口。请只在可信网络短时使用，完成后停止服务。该命令不会创建或修改防火墙规则；防火墙、路由和沙箱网络仍可能阻止访问。远程模式不是带认证的多用户服务。

编辑器只列出 `configs/data/policy-sources/*.yaml` 中的操作员来源配置；浏览器不能填写本机输入路径、tokenizer 路径、shell 命令或任意输出目录。预览仅读取已登记的本地来源，最多扫描 1,000 行/30 秒；缺少缓存、污染索引或 tokenizer 时显示缺失，不会自动联网。每次预览单实例运行。

## 操作流程

1. 在“策略对照”选择 SFT 或 Preference，以及同一个本地来源。已筛选来源会明确提示不能恢复上游已删除的候选。
2. 为 A/B 载入阶段模板，按需调整正确性、完整性、答案解析、长度比和资源上限。规则说明中区分 `filter`（筛除）与 `report_only`（保留但报告）。未知正确性标签不会变成错误答案；Preference 的排序依据仍必须是已知的正确/错误标签。
3. 查看“当前差异”。受控质量消融应只有一个声明变量；数据数量/覆盖也要一起解释。恢复默认会重置当前组的规则。
4. 显式点击“预览 A/B 数据”。两组使用相同 source pool 和 problem split。结果展示扫描范围、诊断分母、选中数及停止原因。改动任何输入后，旧结果立即标记过期；没有原始来源时不会用示例替代。
5. 保存会在 `artifacts/data-experiments/draft-<id>/` 创建新策略草稿目录，不覆盖已有文件，也不会启动构建、训练或评测。
6. 补齐模型起点及摘要、recipe 摘要、训练参数、共同预算、seed、评测集合/协议和指标。页面用共享 schema 校验并保存 `experiment.yaml`；A/B manifest 路径由服务按草稿 ID 生成。计划状态仍为未登记。
7. 使用页面生成的命令在该目录构建 A/B 数据，再在训练开始前用 `register-data-experiment` 登记。之后训练/评测仍由 CLI 执行；用 `compare-data-experiment` 关联实际 run/eval。页面读取比较 JSON 并同时显示逐项数据质量/覆盖、模型指标、分母及兼容性状态。

训练分支要从共同 parent 平行启动：比较两种 SFT 数据时，A/B 都从同一个 M1-CPT（若未做 CPT，则都从同一个 Base）；比较 Preference 数据时，DPO/SimPO 对照都从同一个 M2-SFT 起步。不要让 B 接在 A 的 checkpoint 后训练。两边还需固定 recipe 参数、实际预算和 seed 列表，使用完全相同的评测数据、协议和配置。若预算或覆盖不匹配，应报告流水线整体差异，不解释成某一条规则的因果收益。

## 离线完整示例

项目自有小样本只用于验证协议与交互，不代表上游数据质量，也不包含真实模型效果：

```powershell
.venv/Scripts/python.exe scripts/data_policy_offline_example.py
```

脚本建立两份 SFT 数据、校验同一来源池/切分、登记单变量实验并生成 `NOT_EVALUATED` 报告。它使用仓库自有 JSONL 和明确标注的空格测试 tokenizer，不下载数据、不调用外部模型、不训练。报告中 fixture parent/eval 身份为协议占位符，不能用于真实训练结果归因。

## CLI 示例

```powershell
onegpu-posttrain validate-data-policy --policy configs/data/policies/sft-example.yaml
onegpu-posttrain preview-data-policy --policy configs/data/policies/sft-example.yaml --source configs/data/policy-sources/example.yaml --rows 100
onegpu-posttrain build-policy-data --policy configs/data/policies/sft-example.yaml --source configs/data/policy-sources/example.yaml --output artifacts/policy-example/new-build
```

构建必须使用新的空输出目录。真实来源需自行配置已声明许可、固定来源版本、隔离污染索引和真实 tokenizer。本例配置为 fixture，不能当作正式 Mini 数据。

## 状态解释

| 状态 | 含义 |
|---|---|
| 数据预览 | 来源前缀上的离线观察；不是全源统计，也不是训练数据已达标 |
| `NOT_EVALUATED` | 没有符合身份和评测协议的模型结果；模型效果缺失，不是 0 |
| `NOT_COMPARABLE` | 存在结果但 parent、数据、recipe、预算、seed、评测集或协议不兼容；保留原始分数，不生成可解释为收益的 delta |
| `COMPARABLE` | 仅说明声明的兼容性检查通过；单 seed/小分母仍是描述性证据，不自动构成统计显著或普遍因果结论 |

现有报告可用 `GET /api/policies/reports` 只读索引查看，服务仅扫描 `artifacts/data-experiments/` 中受限大小的比较 JSON。默认服务没有策略写入/预览权限；本机同源令牌只是 loopback 编辑保护，不是多用户认证或远程服务权限系统。

## 界面 walkthrough 与 UX 验收

在桌面和约 390px 窄屏分别复核以下路径：

1. 用键盘进入“策略对照”，选择 SFT/Preference 和已登记来源；缺失文件或阶段不匹配时确认预览按钮禁用且原因可读。
2. 载入 A/B 模板，改一条规则；确认当前差异列出规则路径，`controlled_quality_ablation` 多变量时显示阻止提示，并可改为 `pipeline_effect`。
3. 预览后改变规则、来源或扫描范围；旧数字应保留并带“已过期”状态。触发缺 tokenizer/benchmark 的请求后，错误提示不清除表单。
4. 通过键盘完成 parent/recipe/evaluation SHA、training JSON、预算和 seed 输入；保存政策草稿、校验并保存 experiment.yaml、导出配置，再返回原 Data 页面。检查焦点可见、控件有标签、表格可横向查看、状态不只靠颜色。
5. 打开 fixture 对照报告，检查数据质量表含单位/分母/方法；无 run/eval 时显示 `NOT_EVALUATED`。真实不兼容报告应保留原分数并展示问题，不能出现可解释为收益的 delta。

### 走查记录（2026-10-18）

使用本机 headless Chrome 对隔离临时工作区执行了端到端 walkthrough；策略编辑器只读取项目自有 fixture，临时服务绑定 `127.0.0.1:8767`。原有 `127.0.0.1:8766` 保持只读且未重启。走查创建的策略草稿、实验计划、离线报告和下载文件均位于系统临时目录，结束后清理；没有访问外网、调用模型、下载数据、启动训练或生成模型评测结果。

| 路径 / 验收点 | 结果 |
|---|---|
| Data 选择来源证据 → 打开来源摘要 → 按 Escape 关闭 | 通过；原触发按钮重新获得焦点 |
| 从 Data 进入策略对照 → 返回 Data | 通过；URL、搜索词 `OpenR1`、SFT 阶段筛选及 build 上下文保留 |
| 选择项目自有 SFT fixture → 只修改 B 组完整性规则 → 预览 A/B | 通过；两次预览都成功，source pool 与 split 相同；显示扫描/选中数和“部分前缀预览”停止原因 |
| 保存策略草稿 → 填写 fixture 实验合同 → 校验保存 → 导出 YAML | 通过；实验计划经共享 schema 校验，下载到临时目录；未触发构建/训练 |
| 比较报告、状态和质量表 | 通过；离线 fixture 报告以文字显示“未评估”，质量表包含“指标、结果、分子/分母、方法与局限”表头；模型效果明确显示未评估，没有显示模型 delta |
| 键盘焦点与窄屏 | 通过；键盘可聚焦控件且焦点轮廓可见；390px 下页面无横向溢出，质量表保持表头与单元格可读 |

本次走查发现并修复了两项问题：

1. A/B 预览原先并行请求，与后端“单实例预览”锁冲突，造成一个 409 后整组结果丢失。现在按 A 后 B 串行请求，并有回归测试。
2. 预览过期指纹原先包含实验合同字段，填写 parent / evaluation 信息就会让刚完成的数据预览过期、禁用计划保存。现在预览指纹只跟踪来源、阶段、两组策略和扫描行数；合同字段继续单独保存，并有回归测试。

验证命令：`pnpm --dir web test:policy-editor`（4 项通过）；`pnpm --dir web build`；`.venv/Scripts/python.exe -m pytest tests/test_ui_policy.py`。以上浏览器验收使用 fixture，不代表真实数据质量或模型效果；未覆盖真实比较报告、屏幕阅读器及真实模型训练结果。
