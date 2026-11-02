# OneGPU-PostTrain — Evaluation Design

> **Design Version:** v0.1.1  
> **Target Release:** v0.1  
> **Freeze Date:** 2026-10-12  
> **Status:** FROZEN FOR IMPLEMENTATION


---

## 1. Evaluation 的角色

项目的核心问题不是“loss 降了吗”，而是：

> 模型经过一个训练阶段后，哪里变好了、哪里变差了、代价是什么？

没有 stage-end evaluation 的 checkpoint 不进入官方结果。

---

## 2. 四层体系

```text
Layer 1 — Training Dynamics
Layer 2 — Domain Capability
Layer 3 — Alignment / RL Behavior
Layer 4 — General Capability Guardrail
```

---

## 3. Layer 1 — Training Dynamics

所有 stage 记录：

- algorithm-specific loss；
- learning rate；
- grad norm；
- optimizer steps；
- samples/tokens；
- wall time；
- throughput；
- peak VRAM。

GRPO 额外：prompts、completions、generated tokens。

### 核心规则

```text
CPT loss
SFT loss
DPO loss
SimPO loss
GRPO objective
```

**绝对值不能跨算法排名。** Loss 只用于相同 objective 内观察收敛与超参变化。

---

## 4. Layer 2 — Domain Capability

固定：

```text
GSM8K accuracy@1
MATH-500 accuracy@1
Internal Math Holdout accuracy@1
FineMath Domain PPL
```

### 4.1 accuracy@1

官方主结果使用冻结的 deterministic/low-variance generation protocol，每题一个主输出，因此命名 `accuracy@1`。

Sampling-based pass@k 仅用于 RL 分析，不进入主 stage leaderboard。

### 4.2 Internal Holdout

Mini=500，Standard=1000。

记录：accuracy@1、parse rate、format accuracy（SFT+）、response length、error taxonomy。

---

## 5. PPL Guardrails

### 5.1 Domain PPL

FineMath **document-isolated holdout**。

主要观察：

```text
Base → PEFT-CPT
Domain PPL ↓ ?
Math Accuracy ↑ ?
```

### 5.2 General Text PPL

增加很小、固定、版本化的通用文本 holdout。

用于敏感观察 domain-only CPT 的语言分布漂移。

### 5.3 解读

PPL 与 task accuracy 不是同一概念，必须一起报告。

---

## 6. Layer 3A — SFT Behavior

SFT+ 记录：

- instruction compliance；
- answer parse rate；
- format accuracy；
- empty/malformed rate；
- response length。

Base/CPT 可能“知道答案但不会按 instruction 输出”，因此能力与 instruction following 分开。

---

## 7. Answer Extraction / Metric Separation

统一：

```text
raw output
 ↓
boxed extractor
 ↓ (fallback)
final-answer extractor
 ↓
normalize
 ↓
math verifier
```

分别记录：

```text
parse_success
answer_correctness
format_correctness
```

答案正确但没有 boxed：

```text
accuracy=1
format_accuracy=0
```

---

## 8. Layer 3B — Preference Metrics

### 8.1 Algorithm-specific Training Metrics

DPO：DPO loss / DPO-specific margin。

SimPO-original：SimPO loss / SimPO-specific margin。

这些数值不直接跨算法比较。

### 8.2 Common Held-out Metrics

公共 score：

```text
score(response|prompt)
= length-normalized conditional log-probability
```

指标：

```text
policy_pair_accuracy
= mean(score(chosen) > score(rejected))

policy_pair_margin
= mean(score(chosen) - score(rejected))
```

SFT/DPO/SimPO 均可使用同一 evaluator。

Held-out pair 必须未参与 preference training，并遵守相同 pair quality gate。

### 8.3 Joint Interpretation

必须同时看 task accuracy，避免：

```text
Preference ↑
Math Accuracy ↓
```

仍被误判为整体更好。

---

## 9. Layer 3C — GRPO Metrics

统一记录：

- total reward / mean / std；
- accuracy reward；
- format reward；
- correct completion ratio；
- zero-reward ratio；
- parse failure rate；
- within-group reward variance；
- completion length；
- generated tokens；
- prompts/completions。

### KL

```text
GRPO-standard(beta=0.04): KL required
GRPO-lowmem(beta=0):      KL=N/A
```

所有结果携带 `grpo_profile`。

---

## 10. Optional RL Sampling Analysis

GRPO 可以额外报告：

```text
pass@2
pass@4
```

但仅作为采样/探索分析，不与其他 stage 的主 accuracy@1 混在一起。

---

## 11. Layer 4 — General Guardrail

```text
General Text PPL
+
selected non-math MMLU subsets
```

MMLU 主要检测明显通用任务回归；General PPL 减少 0.6B 模型 benchmark floor effect 带来的盲区。

主比较：Base / PEFT-CPT / SFT。

---

## 12. Cross-stage Fairness

### Track A — Capability

Base/CPT/SFT/Alignment 尽量使用统一、简单 task prompt、相同 answer parser 与 verifier。

### Track B — Instruction Following

只比较：SFT / DPO / SimPO / GRPO。

指标：format accuracy、instruction compliance、parse rate。

---

## 13. Prompt / Tokenizer Versioning

固定：

```text
training_prompt_version = qwen3-math-v1
eval_prompt_version     = qwen3-math-eval-v1
```

同时 pin：tokenizer repo/revision、rendered template hash、instruction hash。

改 prompt 即改 eval protocol，不能与旧结果直接混用。

---

## 14. Generation Configuration

官方 accuracy@1 固定：

- max_new_tokens；
- sampling on/off；
- temperature；
- top_p；
- stop condition；
- prompt version。

GRPO sampling analysis 使用另一份明确版本化的 generation config。

---

## 15. Error Taxonomy

```text
WRONG_ANSWER
PARSE_FAILURE
FORMAT_FAILURE
EMPTY_OUTPUT
TRUNCATED
REPETITION
INVALID_MATH
UNKNOWN
```

错误分布比单个 accuracy 更能解释模型变化。

---

## 16. Evaluation Frequency

### Training-time

低成本：eval loss（适用时）+ small internal subset。

### Stage-end

完整：GSM8K、MATH-500、Internal、stage-specific metrics、必要 guardrail。

Ablation 中间 checkpoint 不全量跑 benchmark，控制 4060 成本。

---

## 17. Stage Evaluation Matrix

| Metric | Base | PEFT-CPT | SFT | DPO | SimPO-original | GRPO |
|---|---:|---:|---:|---:|---:|---:|
| Algorithm Loss | - | ✓ | ✓ | ✓ | ✓ | ✓ |
| Domain PPL | ✓ | ✓ | optional | optional | optional | optional |
| General PPL | ✓ | ✓ | ✓ | rec. | rec. | rec. |
| GSM8K accuracy@1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| MATH-500 accuracy@1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Internal accuracy@1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Format / Parse | - | - | ✓ | ✓ | ✓ | ✓ |
| policy_pair_accuracy | opt. | opt. | ✓ | ✓ | ✓ | - |
| policy_pair_margin | opt. | opt. | ✓ | ✓ | ✓ | - |
| Reward Components | - | - | - | - | - | ✓ |
| KL | - | - | - | opt. | opt. | profile |
| MMLU Guardrail | ✓ | ✓ | ✓ | rec. | rec. | rec. |

---

## 18. Resource-aware Evaluation

所有对比尽量形成：

```text
Capability Gain
vs
GPU Cost
```

记录：peak VRAM、training/inference time、throughput。

---

## 19. Result Schema

```json
{
  "run_id":"...",
  "checkpoint":"...",
  "eval_protocol_version":"v1",
  "grpo_profile":null,
  "metrics":{
    "gsm8k_accuracy_at_1":null,
    "math500_accuracy_at_1":null,
    "internal_accuracy_at_1":null,
    "format_accuracy":null,
    "answer_parse_rate":null,
    "domain_ppl":null,
    "general_ppl":null,
    "policy_pair_accuracy":null,
    "policy_pair_margin":null,
    "reward_total":null,
    "reward_accuracy":null,
    "reward_format":null,
    "kl":null
  }
}
```

---

## 20. Variance

v0.1 大部分实验单 seed。

固定 robustness：

```text
SFT LR baseline/mid
seed 42 / 43 / 44
```

输出 mean/std；单 seed 报告必须声明统计限制。

---

## 21. Regression Rules

至少区分：

- task regression；
- format regression；
- parse regression；
- general PPL regression；
- cost regression。

不在 v0.1 预设硬门槛百分比，真实 baseline 后再定义。

---

## 22. Evaluation Acceptance Criteria

- [ ] accuracy@1 protocol frozen
- [ ] Domain PPL document-isolated
- [ ] General PPL implemented
- [ ] MMLU subset frozen
- [ ] prompt/tokenizer hashes pinned
- [ ] correctness/parse/format separated
- [ ] common policy_pair evaluator available
- [ ] algorithm-specific margins not cross-compared
- [ ] GRPO profile/KL rule applied
- [ ] resource metrics recorded
- [ ] error taxonomy output
- [ ] 3-seed representative result
- [ ] no fake official numbers
