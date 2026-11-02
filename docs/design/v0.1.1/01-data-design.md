# OneGPU-PostTrain — Data Design

> **Design Version:** v0.1.1  
> **Target Release:** v0.1  
> **Freeze Date:** 2026-10-12  
> **Status:** FROZEN FOR IMPLEMENTATION


---

## 1. 设计目标

数据层必须做到：

1. 每阶段“吃什么数据”清晰；
2. Train/Eval 严格隔离；
3. SFT/DPO/SimPO/GRPO 尽量来自同一数学问题分布；
4. 不把 Canonical Schema 绑定到 ms-swift；
5. 长序列过滤透明，不静默截断；
6. Preference negative 可信而不是垃圾样本；
7. 数据构建 deterministic；
8. 所有 artifact 可追溯到 source revision 与 row IDs；
9. 数据许可证与 attribution 清晰。

---

## 2. 数据源与 Pinning

### 2.1 CPT

```text
repo: HuggingFaceTB/finemath
config: finemath-4plus
revision: PINNED_IN_RECIPE
```

用途：数学领域 PEFT-CPT。FineMath-only baseline 被定义为 **domain-only educational stress test**。

### 2.2 SFT / Preference / GRPO

```text
repo: open-r1/OpenR1-Math-220k
config: PINNED_IN_RECIPE
revision: PINNED_IN_RECIPE
```

需要使用的逻辑信息：

- problem / answer；
- multiple generations / reasoning traces；
- `correctness_math_verify`；
- `correctness_llama`（仅审计辅助）；
- `is_reasoning_complete`；
- finish reason；
- source / type / difficulty proxy（如有）。

### 2.3 Benchmark

- GSM8K；
- MATH-500。

Benchmark 只用于 eval/decontamination，同样 pin revision。

---

## 3. Canonical Schema

### 3.1 CPT

```json
{
  "id": "cpt_000001",
  "text": "...",
  "source": "finemath-4plus",
  "metadata": {
    "source_row_id": "...",
    "document_id": "...",
    "language": "en"
  }
}
```

### 3.2 SFT

```json
{
  "id": "sft_000001",
  "raw_problem": "...",
  "messages": [
    {"role":"user","content":"..."},
    {"role":"assistant","content":"..."}
  ],
  "prompt_template_id": "qwen3-math-v1",
  "rendered_template_hash": "sha256:...",
  "metadata": {
    "problem_id":"...",
    "source_row_id":"...",
    "answer":"...",
    "trace_tokens":0,
    "correctness_math_verify":true,
    "is_reasoning_complete":true
  }
}
```

### 3.3 Preference

```json
{
  "id":"pref_000001",
  "raw_problem":"...",
  "prompt":[{"role":"user","content":"..."}],
  "chosen":"...",
  "rejected":"...",
  "prompt_template_id":"qwen3-math-v1",
  "metadata":{
    "problem_id":"...",
    "chosen_tokens":0,
    "rejected_tokens":0,
    "length_ratio":0.0,
    "chosen_correct":true,
    "rejected_correct":false
  }
}
```

### 3.4 GRPO

```json
{
  "id":"grpo_000001",
  "raw_problem":"...",
  "prompt":[{"role":"user","content":"..."}],
  "answer":"...",
  "prompt_template_id":"qwen3-math-v1",
  "metadata":{
    "problem_id":"...",
    "verifier":"math_verify",
    "problem_type":"verifiable_math",
    "prompt_tokens":0
  }
}
```

`answer` 只供 reward/verifier，绝不进入模型输入。

### 3.5 Internal Eval

```json
{
  "id":"eval_000001",
  "problem_id":"...",
  "raw_problem":"...",
  "prompt":"...",
  "answer":"...",
  "eval_prompt_version":"qwen3-math-eval-v1"
}
```

---

## 4. OpenR1 构建顺序

```text
Raw Dataset (pinned)
 ↓
Normalize
 ↓
problem_id
 ↓
Exact Dedup
 ↓
Near-Duplicate Analysis
 ↓
External Benchmark Decontamination
 ↓
Programmatically Verifiable Math Core
 ↓
Problem-level Split
 ↓
SFT Pool / Alignment Core / Internal Holdout
 ↓
Tokenizer + Final Template Length Eligibility Gate
 ↓
Stage Transform
 ↓
ms-swift Adapter
```

禁止 Stage Transform 后再各自 random split。

---

## 5. Problem Normalization

最小 normalization：

1. Unicode normalize；
2. trim；
3. collapse whitespace；
4. normalize 常见数学空格；
5. 保留公式语义；
6. 不做会改题意的 rewrite。

```text
problem_id = SHA256(normalized_problem)
```

exact hash 不能捕获所有语义重复，因此 near-duplicate 单独报告。

---

## 6. Benchmark Decontamination

必须**先于 split**：

```text
Normalized OpenR1
 ↓
GSM8K/MATH-500 exact + high-confidence fuzzy check
 ↓
remove contaminated candidates
 ↓
Verifiable Core
```

记录：exact hit、fuzzy candidate、removed problem IDs、benchmark revision、protocol version。

FineMath 即使上游做过 decontamination，最终抽取 corpus 仍做二次 verification scan。

---

## 7. Verifiable Math Core

v0.1 Core 的正负样本优先依赖 programmatic math verifier。

### SFT 正样本基础条件

```text
correctness_math_verify == true
is_reasoning_complete == true
final answer parseable == true
not truncated == true
```

### Preference rejected 基础条件

不能把 parser failure、空白、截断、乱码当作负样本。Rejected 必须：

- completion 完整；
- final answer 可提取；
- verifier 明确判错；
- 无明显垃圾/重复。

### GRPO

只使用稳定 ground truth、可程序验证题型。排除：proof、notfound、需要主观 judge 的题目。

---

## 8. Split Strategy

```text
Verifiable Core
  ├── train_pool
  │    ├── sft_pool
  │    └── alignment_core
  └── internal_holdout
```

同一 `problem_id` 只能属于一个顶层 split。

Alignment Core 平行派生：

```text
alignment_core
  ├── preference_pairs → DPO / SimPO-original
  └── prompt+answer    → GRPO
```

Preference 与 GRPO 可以共享部分 problem_id，因为它们是从 M2 出发的平行分支；必须报告 overlap。

### Mini Target

```text
SFT Pool          5,000 eligible
Alignment Core    3,000 eligible
Internal Holdout    500 eligible
Preference        2,000 valid pairs
GRPO              2,000 valid prompts
```

这里是最终 eligible 数，不是 raw candidates。

---

## 9. Length Eligibility Gate

这是 v0.1 的核心数据规则之一。

长度必须用：

```text
pinned Qwen3 tokenizer
+
final prompt/template rendering
```

### 9.1 SFT

```text
rendered_tokens <= sft_max_length
```

不截断长 reasoning。存在多条正确 trace 时，优先选择**最短的完整、正确、可解析 trace**。

### 9.2 Preference

Prompt + chosen 和 Prompt + rejected 都必须完整落入预算。

### 9.3 GRPO

```text
prompt_tokens <= max_prompt_length
```

### 9.4 Selection Bias Report

Data Report 必须比较 gate 前后：

- candidate count / retention rate；
- P50/P90/P95/P99 length；
- source distribution；
- problem type；
- difficulty proxy。

8GB 导致的“偏短样本选择”必须显式暴露。

---

## 10. SFT 派生规则

同题多条正确 trace 的 deterministic priority：

1. math verifier 正确；
2. reasoning complete；
3. final answer parseable；
4. length eligible；
5. 最短完整正确 trace；
6. stable key tie-break。

禁止不记录 seed/规则的随机选择。

---

## 11. Preference Pair 派生规则

合法 pair：

```text
same problem
chosen = complete + parseable + verifier-correct
rejected = complete + parseable + verifier-wrong
both length eligible
```

多个 rejected 时：选择与 chosen token length 最接近的 valid incorrect trace。

初始 length gate：

```text
0.5 <= rejected_tokens / chosen_tokens <= 2.0
```

若后续根据真实分布调整，必须升级 protocol version。

每条 pair 记录：chosen/rejected tokens、length ratio、source generation id、verifier outcome。

---

## 12. GRPO 派生规则

```text
problem + ground_truth
```

不提供标准 reasoning。

必须支持：

```text
completion
 ↓
answer extractor
 ↓
normalize
 ↓
math equivalence verifier
 ↓
reward
```

解析失败 deterministic reward=0，不终止训练。

---

## 13. FineMath / CPT 数据

### 13.1 Document-level Split First

```text
documents
 ├── train_docs
 └── holdout_docs
```

先 split，再分别 chunk/pack。

### 13.2 Token Budget

```text
Smoke     500K
Mini        5M
Standard   20M
```

Dataset-size 实验严格：

```text
D5 ⊂ D10 ⊂ D20
```

避免独立随机采样带来的额外变量。

---

## 14. Holdout

Internal Holdout：

- 与 OpenR1 训练源同分布；
- problem-level 未见；
- 固定 revision/hash；
- Mini 500，Standard 1000。

FineMath Domain PPL Holdout 必须来自独立 documents。

---

## 15. Duplicate / Leakage

硬门槛：

```text
Train ↔ Internal Holdout exact overlap = 0
```

另输出：

```text
near_duplicate_candidates.jsonl
contamination_report.json
```

v0.1 near-duplicate 以报告为主，不假装简单 fuzzy 能彻底解决语义污染。

---

## 16. Dataset Manifest

```yaml
dataset_id: sft-mini-v1
stage: sft
source:
  repo: open-r1/OpenR1-Math-220k
  config: ...
  revision: ...
tokenizer:
  repo: Qwen/Qwen3-0.6B-Base
  revision: ...
prompt_template_id: qwen3-math-v1
protocol_versions:
  normalization: v1
  decontamination: v1
  verifiability_gate: v1
  length_gate: v1
split_seed: 42
sample_count: 5000
source_row_ids_sha256: ...
canonical_records_sha256: ...
processed_artifact_sha256: ...
```

---

## 17. Data Report

每次 build 生成 `data_report.json` + `data_report.md`：

```text
source repo/config/revision
raw candidates
normalized/dedup candidates
benchmark contamination removed
verifiable-core count
eligible final count
retention rate
sample/token count
length percentiles
preference length-ratio distribution
overlap results
near-duplicate candidates
source/type/difficulty before vs after gates
CPT document count / packing utilization
artifact SHA256
```

---

## 18. Adapter Boundary

```text
Canonical Dataset
  ↓
ms-swift Adapter
  ↓
Native Training Format
```

v0.1 只实现 ms-swift adapter，但 Canonical Schema 不锁死训练框架。

---

## 19. License / Compliance

- Qwen3-0.6B-Base：按 pinned model license attribution；
- OpenR1：Apache-2.0 attribution；
- FineMath：ODC-BY attribution；
- Git repo 不提交 GB 级原始数据；
- example/derived sample 明确 source。

---

## 20. Data Acceptance Criteria

- [ ] all revisions pinned
- [ ] benchmark decontamination before split
- [ ] Verifiable Math Core deterministic
- [ ] problem-level split deterministic
- [ ] SFT complete/correct/parseable gate
- [ ] tokenizer-rendered Length Gate
- [ ] preference verifier-negative + length control
- [ ] GRPO verifiable answer gate
- [ ] FineMath document split before chunking
- [ ] exact overlap = 0
- [ ] near-duplicate report
- [ ] retention/distribution-shift report
- [ ] source IDs + canonical/artifact hashes
- [ ] license attribution documented
