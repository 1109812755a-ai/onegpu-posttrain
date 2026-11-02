# OneGPU-PostTrain — Experiment & Ablation Design

> **Design Version:** v0.1.1  
> **Target Release:** v0.1  
> **Freeze Date:** 2026-10-12  
> **Status:** FROZEN FOR IMPLEMENTATION


---

## 1. 核心方法

```text
Question
 ↓
Hypothesis
 ↓
Controlled Experiment
 ↓
Training
 ↓
Evaluation
 ↓
Comparison
 ↓
Interpretation
```

核心方法：**One Variable At A Time（OVAT）**。

v0.1 不做自动 HPO，因为目标是理解因果，不只是找到最好数字。

---

## 2. Experiment Definition

每个实验必须声明：

```text
Experiment ID
Question
Hypothesis
Parent Checkpoint + hash
Dataset Manifest + sample IDs
Control Variables
Independent Variable
Dependent Metrics
budget_type
Training Budget
Seed
Algorithm / Profile
Evaluation Protocol
```

`budget_type`：

```text
tokens | steps | epochs | completions
```

---

## 3. Control Variables

常见控制项：

- model / parent hash；
- dataset revision / sample IDs；
- seed；
- batch / accumulation；
- LoRA config；
- max length；
- optimizer / scheduler；
- prompt template；
- evaluation config；
- reward/generation config（RL）。

只有明确研究的变量允许变化。

---

## 4. Budget Fairness

### 4.1 Hyperparameter / Algorithm

优先：same sample IDs → same steps/tokens → comparable GPU cost。

### 4.2 Dataset Scaling

CPT size 研究“更多数据 + 更多 training tokens”，因此采用 nested subsets + one-pass/token budget，不固定 steps。

### 4.3 GRPO

必须报告：prompts、completions、generated tokens、optimizer steps、wall time。

不同算法无法绝对预算等价时，明确写出差异。

---

## 5. EXP-01 — CPT Dataset Size（Required）

### Question

更多 PEFT-CPT 数据如何影响领域适配、通用漂移和任务能力？

### Variable

```text
D5 ⊂ D10 ⊂ D20
5M / 10M / 20M tokens
```

必须 nested，不允许独立随机样本。

### Budget

```text
budget_type = tokens
one pass / increasing token budget
```

### Controls

Same Base、FineMath revision、sampling order、LR、optimizer、LoRA、max length、seed。

### Metrics

Domain PPL、General PPL、GSM8K/MATH/Internal accuracy@1、MMLU guardrail、time/VRAM。

---

## 6. EXP-02 — SFT Learning Rate（Required）

### Question

LR 如何影响收敛和最终下游能力？最低 train loss 是否意味着最高 accuracy？

### Variable

```text
low / mid / high
```

候选可以从 `1e-5 / 5e-5 / 1e-4` 开始 smoke，最终值在稳定性验证后冻结。

### Budget

```text
budget_type = steps
same sample IDs
same optimizer steps
```

### Robustness

mid/baseline 点：

```text
seed 42 / 43 / 44
```

### Metrics

实验内部 train/eval loss、accuracy@1、format、parse、VRAM、time。

---

## 7. EXP-03 — DPO vs SimPO-original（Required）

### Question

相同 M2、相同高质量 pairs 与相近预算下，两种 preference objective 的表现如何？

### Variable

```text
DPO
vs
SimPO-original
```

SimPO 冻结：

```yaml
beta: 2.0
simpo_gamma: 1.0
cpo_alpha: 0.0
```

### Controls

M2 hash、pair IDs、seed、LoRA、max length、optimizer、budget、eval protocol。

### Common Metrics

- policy_pair_accuracy；
- policy_pair_margin；
- GSM8K/MATH/Internal accuracy@1；
- format / parse；
- VRAM / time。

DPO-specific margin 与 SimPO-specific margin 只看各自训练动态，不能直接比较绝对值。

### Interpretation

只能说“在当前模型、数据、预算下 A 在 X 更高/B 在 Y 成本更低”，不宣布算法普遍优劣。

---

## 8. EXP-04 — GRPO Reward Design（Required）

### Question

Reward 组成改变后模型到底学到了什么？

### Variable

```text
R1 = accuracy
R2 = accuracy + 0.1 * format
```

### Controls

Same M2、prompt IDs、GRPO profile、G、generation config、optimizer、completion budget、seed。

### Metrics

accuracy@1、format、parse、reward components、completion length、zero-reward ratio、generated tokens、time/VRAM。

正确性与格式独立判定。

---

## 9. EXP-05 — DPO Beta（Recommended）

```text
0.05 / 0.1 / 0.2
```

Same M2、pairs、steps、LR、LoRA、seed。

看 common preference metrics + math capability，避免只优化 preference 而任务下降。

---

## 10. EXP-06 — GRPO Num Generations（Recommended）

```text
G=2 / G=4
```

固定：

```text
generation_batch_size = 4 completions
G=2 → 2 prompts × 2
G=4 → 1 prompt × 4
budget_type = completions
```

同一 GRPO profile / reward / M2 / LR / completion limit / seed。

记录 group variance、reward、accuracy、generated tokens、VRAM、wall time。

---

## 11. Optional — SFT Data Size

```text
1K ⊂ 5K ⊂ 10K
```

未来研究 scaling / marginal gain / overfitting，不属于 v0.1 Release Gate。

---

## 12. Optional — LoRA Rank

```text
r=4 / 8 / 16 / 32
```

观察 trainable params、VRAM、time、accuracy。

---

## 13. Run ID

推荐：

```text
<stage>-<experiment>-<value>-s<seed>
```

例如：

```text
cpt-data-5m-s42
sft-lr-mid-s42
dpo-beta-01-s42
pref-algo-dpo-s42
pref-algo-simpo-original-s42
grpo-reward-accfmt-s42
```

GRPO run metadata 额外标记 profile。

---

## 14. Run Artifact

```text
runs/<run_id>/
├── config.yaml
├── resolved_config.yaml
├── dataset_manifest.json
├── environment.json
├── lineage.json
├── protocol_versions.json
├── train_log.jsonl
├── metrics.json
├── eval.json
├── hashes.json
├── checkpoint/
└── report.md
```

---

## 15. Lineage

```json
{
  "run_id":"dpo-beta-01-s42",
  "parent_run_id":"sft-baseline-s42",
  "parent_checkpoint_hash":"...",
  "stage":"dpo",
  "dataset_manifest_hash":"..."
}
```

任何结果都可追溯到 M0。

---

## 16. Environment Metadata

至少：

```text
OS / Python
CUDA / Driver
GPU / VRAM
PyTorch / Transformers
ms-swift / EvalScope
math verifier version
Git commit
Model/Dataset/Tokenizer revisions
Prompt protocol
```

GRPO 额外：profile、beta、reference_enabled、use_vllm、G、generation_batch_size。

---

## 17. Results Registry

统一 `results.csv/jsonl` 字段：

```text
run_id
stage
algorithm/profile
model/dataset revisions
seed
budget_type / budget
lr / rank / beta / G / reward_config
train/eval loss (algorithm-specific)
domain_ppl / general_ppl
gsm8k/math500/internal accuracy@1
format / parse
policy_pair_accuracy / margin
reward components / KL
peak_vram / throughput / training_seconds
```

---

## 18. 可视化

每张图只回答一个问题：

- parameter → main metric；
- parameter → loss/reward dynamic；
- parameter → cost；
- step → learning curve。

不把不同 objective 的 loss 放在一张“谁更低”的图中。

---

## 19. Experiment Report Template

```markdown
# EXP-XXX

## Question
## Hypothesis
## Parent Checkpoint
## Dataset / Manifest
## Controls
## Independent Variable
## Budget Type / Budget
## Runs / Seeds
## Training Dynamics
## Evaluation Results
## Resource Usage
## Observation
## Interpretation
## Limitations
## Conclusion
```

---

## 20. 自动结论边界

工具可以自动计算：最好值、delta、mean/std、成本差异。

不自动生成未经证据支持的“为什么”。解释必须与当前实验范围绑定。

---

## 21. Seed Policy

大部分 v0.1：seed=42。

固定 3-seed：SFT LR baseline/mid point，42/43/44。

单 seed 报告必须写入 Limitation。

---

## 22. Experiment Validity Check

一个官方实验只有满足以下条件才进入结果：

- parent hash 一致；
- dataset IDs/revision 正确；
- 除 independent variable 外无意外 config drift；
- budget contract 满足；
- eval protocol 一致；
- run 未发生未记录的 OOM 降级；
- artifact 完整。

---

## 23. Official Result Set

Stage：

```text
Base
PEFT-CPT
SFT
DPO
SimPO-original
GRPO-standard or GRPO-lowmem (profile required)
```

Full-CPT 只作为 feasibility result。

Required Ablation：EXP-01/02/03/04。

Recommended：EXP-05/06。

---

## 24. Experiment Acceptance Criteria

- [ ] Question/Hypothesis/Controls 完整
- [ ] budget_type 明确
- [ ] CPT nested subsets
- [ ] SFT same steps / IDs
- [ ] SFT representative 3 seeds
- [ ] SimPO-original cpo_alpha=0
- [ ] common preference metrics
- [ ] GRPO completion budget fixed for G experiment
- [ ] accuracy/format reward separated
- [ ] run artifacts include hashes/protocol/lineage
- [ ] no cross-algorithm loss/margin misuse
- [ ] conclusions bounded to model/data/budget/seed
- [ ] no fake numbers
