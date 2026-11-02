# Design

## Context

现有 `calibrate-sft-length` 已对固定 DeepMath 快照全量校准，但其输出被明确隔离为探索数据；现有 Mini readiness profile 的 SFT key 仍是 `openr1.sft_train`。Mini SFT 目标为 5,000 条。全量校准在 2,048-token 门槛下得出 15,046 条 exact-answer-match 候选，在 1,536 下为 4,820 条；校准报告同时指出 exact matching 不等于 reasoning correctness，也没有人工质量批准。既有 2,048-token、3-step RTX 4060 probe 通过，但系统级显存采样为 7,931 MiB，余量很小。

用户选择先训练 Mini CPT，再以其 checkpoint 为 SFT parent。本地现有 5,000,368-token FineMath CPT artifact 可用，但对应 quality review 仍为 `pending`、usage=`legacy_unknown`；本地真实训练 checkpoint 只有 smoke M1。故本变更产出的 CPT/SFT run 在数据审核获批前均须标记 exploratory，不得宣称正式 Mini baseline 或 readiness PASS。详见 proposal 与相关 delta specs。

## Goals / Non-Goals

**Goals:**

- 将校准发现转成一套从 pinned source 重建、可复现、可审核的独立 Mini SFT 数据/训练路径。
- 先以现有 5M-token FineMath 数据训练独立 Mini CPT，产出可追溯的 M1；再以此 parent 测试 Qwen3-0.6B-Base、2,048 token、5,000-example Mini SFT。
- 在数据 review 未批准时允许开展标记为 exploratory 的工程/研究 run，但明确隔离于 official readiness 与发布结论。
- 保证数据选择、质量 review、资源可行性和模型评测分别有证据，未满足门槛时安全停止。

**Non-Goals:**

- 不将校准 JSONL 直接复制/晋升为 official 数据，不放宽完整答案、验证、去污染或长度规则。
- 不把 DeepMath 与 OpenR1 的 SFT 数量默认为可合并；不改变 CPT、Preference、GRPO 来源或 Mini 总目标。
- 不修改 `recipes/smoke/20-sft.yaml` 的 1,024 默认值，不执行 1.7B pilot。
- 不声称该单臂训练能证明 2,048-token 长度本身或 DeepMath 来源造成因果提升。

## Decisions

1. **Mini CPT 先行，并以独立 run 产出 parent。**
   用现有 build-02 FineMath `cpt_train` artifact（5,000,368 tokens）训练新 Mini CPT LoRA run；recipe 使用训练集预算对应的完整训练过程而不是 Smoke 的 20 steps。记录 optimizer steps、实际 token exposure、parent Base revision/hash、resolved config、资源和 checkpoint hash。若该 CPT 数据使用状态仍是 `legacy_unknown`/pending，则模型 run 标记 exploratory，但仍可作为相同实验的后续 SFT parent。

2. **新增正式 builder，而非晋升 calibration artifact。**
   从指定本地 DeepMath Parquet 快照重跑规范化、答案提取、完整渲染长度、拆分、去污染和选择，写入新的 `data/processed/mini/<build-id>/deepmath-sft/` artifact root。校准报告作为预检参考写入 lineage，不作为训练输入或审核证明。固定 snapshot revision `5cf055d1fe3d7a2eb19719ac020211469736ae44`、source file hashes、seed、prompt/template 与 tokenizer revision；缺失资源只报告，不在线下载。

3. **先按题目分组，再做 5K 预算选择。**
   使用 `problem_id` 做确定性 train/eval 分配，先排除已知 benchmark/eval/holdout 重叠题，再在 train pool 内基于稳定 ID/hash 选择最多 5,000 条。每题最多一条 trace，沿用当前校准的候选排序规则；完整题目和 trace 身份均进入 manifest。若污染索引或 split 证据缺失，产物只能为探索性；数量不足则报告 shortfall，不填充或放宽门槛。

4. **建立与校准隔离的 Mini source cohort。**
   以 `deepmath.sft_train` 作为单独 key 加入 versioned Mini profile； readiness 按 source cohort 独立报告数量、quality approval、processing evidence 和 lineage。OpenR1 SFT 保留原 key 与历史计数。只有明确版本化的 cohort-combination policy 才可合并不同来源预算，本次不创建该 policy。

5. **质量状态默认从 exploratory 开始。**
   生成 structural/duplicate/length/distribution/contamination 报告；将 upstream `final_answer`、exact string agreement、semantic verifier 结果、reasoning completeness 的人工抽审分别标注其证据来源、分母和覆盖。复用现有 `mini-review-v1` 及 digest-bound review bundle。自动筛选或答案一致不能代替人工 reasoning review；维护者必须基于报告显式批准匹配数据身份的 review，readiness 才可评估为官方 SFT 输入。未批准时仍可保留数据报告，但不得运行 official Mini baseline。

6. **为 Mini 建独立 CPT/SFT recipes，不改 Smoke。**
   新建版本化 Mini CPT recipe，固定 Base Qwen3-0.6B-Base 与 pinned revision，LoRA rank 8、micro-batch 1、gradient checkpointing，完整消费现有 5M-token 数据预算；训练前确认本地数据 review 状态及显存。新建 Mini SFT recipe，parent 固定为本次 CPT run 产生的 M1，max_length 2048，LoRA rank 8、micro-batch 1、gradient checkpointing，记录同一套确定性训练预算。训练前做配置解析和资源 probe；OOM、CUDA 资源不足、parent 不可用或显存保护失败则停止相应完整 run，记录证据，不自动量化、不自动改短长度、不自动换基模。数据 review pending 时只可执行显式标记的 exploratory run。

7. **CPT 与单臂 Mini SFT 结果按 stage outcome 解释。**
   SFT 数据、CPT 数据和本地评测资源就绪且显存 probes 通过后，按顺序执行 Mini CPT → Mini SFT → 统一评测；若质量 review 仍 pending，训练和评测产物只标记 exploratory。记录 resolved configs、dataset hashes、source revisions、parent/model hashes、code/environment、实际训练预算、资源与评测协议。报告 CPT/SFT 相对各自 parent 的阶段变化和限制，但不得与不同数据量/长度的旧 Smoke SFT 作受控因果比较，也不能由此宣称 2,048 长度优于 1,024。

8. **许可证信息来自锁定版本并保留上游归属。**
   数据集卡当前标注 MIT，但实现时仍需将 pinned revision 对应的 card/许可文本及来源链接快照纳入 source manifest/文档审查；无法在本地复核时不将该数据标为可公开再分发。训练产物遵循项目数据策略，不把大体量源语料复制进 git。

## Risks / Trade-offs

- [校准中的答案筛选只覆盖最终答案，不保证推理正确、完整或可教学] → 独立 quality report、versioned human reasoning review；未批时禁止 official readiness。
- [DeepMath 的 benchmark decontamination 不等价于本项目当前评测/holdout 全量不相交] → 在本地已有索引上做显式交叉检查；索引缺失时保留 exploratory 状态并停止 official promotion。
- [2,048-token 在 8GB RTX 4060 上余量小，环境差异可能造成 OOM] → fresh preflight + same-config probe；失败就停止，不自动降低长度或改变量化设置。
- [DeepMath 是 SFT 新来源，和既有 OpenR1 preference/GRPO cohort 不完全同源] → 在 profile、UI、报告和 lineage 中独立显示来源 cohort；本次不将跨来源结果描述成受控对照。
- [FineMath 质量 review 当前 pending 且 usage 为 legacy_unknown] → Mini CPT 和后续 SFT 均先作为 exploratory run，保留 exact data/model lineage；不伪造 approval，维护者可后续对完全匹配的 digest 提交 review。

## Migration Plan

1. 仅新增实现、Mini profile 版本、recipe 和 artifact root，不覆盖 `build-02`、Smoke recipe、calibration artifacts 或任何旧 run。
2. 先完成离线 fixture/本地 snapshot build 与审计验证，再输出新的 review bundle；source/license、quality 或 lineage 不满足时停止在 exploratory/pending 状态。
3. 本地 parent、数据处理证据、短程 GPU probe 和评测资源均就绪后，可运行明确标记 exploratory 的 Mini CPT/SFT 与统一评测；只有对应数据 review 和 source/license 审阅获批准后，相关数据与模型才可参加 official readiness。每次运行使用新 `experiment_id`/`run_id` 与独立 lineage。
4. 回滚通过不将新 cohort 配置加入生效 Mini profile、或停止使用新 build 实现；历史输入与 artifacts 仍保持不变。

