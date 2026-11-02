# 数据策略与小规模探索

数据策略首版支持 SFT 和按数学正确性排序的 Preference。CPT/GRPO 继续使用既有命令。策略页面与 CLI 使用同一组规则；不需要数据库、外部模型调用或任务调度。

质量规则包括正确性、完整性、答案解析。`filter` 会筛除失败/缺证据样本；`report_only` 保留事实并可供探索使用，不将未知标成错误或 verified。正确性排序 Preference 始终需要已知正确/错误两侧。长度比是独立选择规则，不是通用质量分。

`resources.max_length` 以指定 tokenizer、训练模板渲染后的 prompt+response 计量；不是推理质量分数。提高长度上限后仍要检查 recipe 和显存预算。训练前若完整数据超过 recipe max_length 将报错，不能悄悄截断。`target_count` 是数据预算；探索数据可以少于 Mini 的 5K/2K 正式配额，空集合则不能训练。

格式、来源/使用范围声明、唯一身份、评测隔离与污染检查是固定要求。不要关闭它们来制造对照。独立语义验证和近重复质量检查未执行时保持未测量；通过结构检查不等于质量已全面通过。

## 完全离线的项目自有样例

以下样例使用显式标记的空格测试 tokenizer，只验证数据/配置/血缘流程，不能作为真实模型 token 长度或训练收益证据。可用仓库虚拟环境运行；每次构建请使用新的输出目录。

```powershell
.venv/Scripts/python.exe -m onegpu_posttrain validate-data-policy --policy configs/data/policies/sft-example.yaml
.venv/Scripts/python.exe -m onegpu_posttrain preview-data-policy --policy configs/data/policies/sft-example.yaml --source configs/data/policy-sources/example.yaml --rows 8
.venv/Scripts/python.exe -m onegpu_posttrain build-policy-data --policy configs/data/policies/sft-example.yaml --source configs/data/policy-sources/example.yaml --output artifacts/policy-example/sft-01
.venv/Scripts/python.exe -m onegpu_posttrain validate-lineage --root artifacts/policy-example/sft-01 --require-complete
```

Preference 使用 `configs/data/policies/preference-example.yaml`，输出另选 `artifacts/policy-example/preference-01`。样例数据为项目自有内容，不触发下载。schema/来源/污染索引/tokenizer 缺失会明确报错，不改用远程数据或用示例结果冒充真实结果。

## 使用本地真实数据

复制 source 配置，由操作者填写本地 JSONL 或已验证的 source_cache manifest（二选一）、真实来源/revision/许可说明、本地非空评测索引及 pinned tokenizer。source 配置中的文件路径相对于该配置文件；`tokenizer.local_path` 可指定已准备好的 tokenizer 目录。真实数据将 `test_only` 设为 false，加载使用 local_files_only，不会下载。

`input_scope: existing_dataset` 需要声明 existing_stage，仅允许进一步评估/筛选当前 canonical 样本。它无法恢复过去丢弃的候选，不能据此统计原始来源的规则产出。原始来源与已筛选来源使用不同 pool 身份。

同一个来源、split snapshot、策略和 tokenizer 产生确定性数据身份；每次构建仍有新的 run_id。A/B 固定 split，不能通过改配额重新分配题目。新策略路径使用 `openr1-policy-build-v1`；旧 `build-openr1` 的默认协议保持，历史报告不回填新诊断。

## 看懂预览与拒绝原因

预览默认是前 200 行、最多 1000 行，具有前缀采样偏差。显示实际处理行数、停止原因和 partial；不要将保留率外推为全源产出。完整数据准备/hash 校验与实际预览耗时分开记录，时间预算是协作式停止。

`no_known_correct_candidate` 表示候选不具备已知正确标签；`correct_candidates_all_over_length` 表示存在正确候选但全部超长。Preference 另区分原本没有错误侧与错误侧全部超长。首拒绝原因可按问题数对账，额外 trace/problem 诊断可重叠，不能直接相加。

## 进入现有训练路径

将 canonical JSONL 通过 `adapt-ms-swift` 导出后使用现有 plan/run SFT、DPO 或 SimPO 命令。训练预检保留 policy/usage、实际数量和警告，并校验原始/adapter 摘要及 recipe 长度；执行仍需用户准备模型和训练环境。少量 pairs 能验证执行路径，不能支持稳定算法排名。正式 Mini 继续单独运行 `validate-mini-data`，不会因探索成功自动就绪。

数据策略对照与后续定向生成计划见 [data-policy-experiments](../../openspec/changes/data-policy-experiments/proposal.md) 和 [targeted-synthetic-data](../../openspec/changes/targeted-synthetic-data/proposal.md)。生成接口与模型效果尚须各自实际实现/执行，不能由本页数据计数推断。
