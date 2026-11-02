# OneGPU-PostTrain — Training Recipe Design

> **Design Version:** v0.1.1  
> **Target Release:** v0.1  
> **Freeze Date:** 2026-10-12  
> **Status:** FROZEN FOR IMPLEMENTATION


---

## 1. 设计原则

训练层目标不是找“绝对最优参数”，而是：

- 4060 8GB 可运行；
- 默认参数保守、稳定；
- 每阶段语义清楚；
- 支持 OVAT；
- Checkpoint lineage 明确；
- 对未实测的 8GB 能力不做过度承诺。

---

## 2. Checkpoint Lineage

```text
M0 = Qwen3-0.6B-Base
 |
 | PEFT-CPT adapter
 v
M1 = merged parent
 |
 | fresh SFT adapter
 v
M2 = merged parent
 |
 +-- fresh DPO adapter            → M3-DPO
 +-- fresh SimPO-original adapter → M3-SimPO
 +-- fresh GRPO adapter           → M3-GRPO
```

独立 probe：

```text
M0 → Full-CPT Smoke
```

---

## 3. Stage Contract

输入：

```text
merged parent checkpoint
+ dataset manifest/hashes
+ model/training/eval config
+ tokenizer/template revision
+ seed
```

输出：

```text
stage adapter
merged parent checkpoint
training/resource metrics
lineage hashes
stage-end eval
```

Stage Boundary：

```text
fresh adapter
→ train
→ save adapter
→ merge into non-quantized parent
→ hash
→ next stage reloads parent
→ fresh adapter
```

---

## 4. Hardware Target

```text
GPU: NVIDIA RTX 4060
VRAM: 8GB
World size: 1
```

策略：

- per-device batch 优先 1；
- gradient checkpointing 默认开启；
- LoRA / QLoRA；
- sequence budget 由 Data Report 驱动；
- 同一张 8GB GPU 默认不用 vLLM rollout engine；
- 所有 OOM 降级必须可追踪，不能一次随机改多个变量。

---

## 5. Precision / PEFT Baseline

```text
BF16 if actually supported
otherwise FP16
QLoRA: 4-bit when needed
LoRA rank baseline: 8
```

LoRA alpha/dropout 在 smoke 后冻结为明确值；不以“implementation default”作为最终官方 recipe。

---

## 6. Sequence Length Contract

初始候选：

```text
PEFT-CPT 1024
SFT      1024
DPO      1024
SimPO    1024
GRPO prompt <= 512
GRPO completion <= 512
```

这些不是永久定值。最终冻结条件：

```text
Data eligibility retention
+
8GB peak VRAM
+
smoke stability
```

修改 max length 会改变 eligible dataset，因此必须生成新的 dataset manifest。

---

## 7. OOM 降级顺序

### 7.1 PEFT-CPT / SFT / Preference

```text
1. batch → 1
2. checkpointing ON
3. QLoRA / quantized parent
4. reduce max_length + rebuild eligible dataset
5. reduce LoRA rank
```

### 7.2 GRPO

```text
1. standard beta=0.04 → lowmem beta=0 if reference OOM
2. reduce completion length
3. reduce prompt length + rebuild dataset
4. baseline G=2
5. reduce LoRA rank
```

---

## 8. Stage 0 — Base Evaluation

M0 不训练。

至少记录：

- GSM8K/MATH/Internal accuracy@1；
- Domain PPL；
- General PPL；
- MMLU guardrail；
- inference resource baseline。

Base/CPT 不强求 instruction-format 指标与 SFT 公平比较。

---

## 9. Stage 1 — PEFT-CPT

### 9.1 Definition

```text
Full CPT = full parameter
PEFT-CPT = LoRA / QLoRA continued training
```

官方 8GB 路径：PEFT-CPT。

### 9.2 Mini Baseline

```yaml
data_tokens: 5_000_000
method: qlora_or_lora
per_device_batch: 1
gradient_checkpointing: true
max_length: resolved_after_smoke
lora_rank: 8
seed: 42
budget_type: tokens
```

LR smoke 后冻结。

### 9.3 Full-CPT Smoke Probe

仅跑少量 optimizer steps，验证 8GB 可行性。

结论枚举：

```text
SUPPORTED_ON_REFERENCE_8GB
CONDITIONALLY_SUPPORTED
UNSUPPORTED_ON_REFERENCE_8GB
```

Full-CPT smoke 不改变 M1 官方 lineage。

### 9.4 Output

```text
M1 = PEFT-CPT adapter merged into parent
```

Stage-end 看：Domain PPL、General PPL、math benchmarks、资源成本。

---

## 10. Stage 2 — SFT

### 10.1 Input

```text
M1 merged
+ 5K complete/correct/parseable/length-eligible traces
```

### 10.2 Objective

assistant-only supervised CE loss。

### 10.3 Baseline

```yaml
per_device_batch: 1
gradient_checkpointing: true
max_length: resolved_from_data_report
lora_rank: 8
seed: 42
budget_type: steps
```

LR 实验：low / mid / high，具体值 smoke 后冻结。

所有 LR run：same sample IDs + same optimizer steps。

中间 baseline/mid LR 额外：

```text
seed=42/43/44
```

### 10.4 Output

```text
M2 = SFT adapter merged into parent
```

M2 hash 是三个后训练分支的唯一共同 parent。

---

## 11. Stage 3A — DPO

Input：

```text
M2 merged
+ 2K verified length-controlled pairs
+ fresh adapter
```

Baseline：batch=1、checkpointing on、rank=8、max_length 由 pair eligibility 冻结。

官方 beta 实验：

```text
0.05 / 0.1 / 0.2
```

训练内部可看 DPO-specific reward/margin，但跨算法公共比较使用 `policy_pair_accuracy/margin`。

---

## 12. Stage 3B — SimPO-original

与 DPO 使用完全相同：M2、pair IDs、数据顺序/预算语义、PEFT 配置、eval protocol。

冻结算法配置：

```yaml
beta: 2.0
simpo_gamma: 1.0
cpo_alpha: 0.0
```

`cpo_alpha=1` 以后作为 `SimPO-stable` 单独实验，不混称原始 SimPO。

---

## 13. Stage 3C — GRPO

### 13.1 Data

```text
M2 merged
+ 2K verifiable prompts
+ ground truth for reward only
+ fresh adapter
```

### 13.2 Profiles

```text
standard:
  beta=0.04
  reference enabled
  KL required

lowmem:
  beta=0
  reference disabled
  KL=N/A
```

reference 8GB path：`use_vllm=false`。

### 13.3 Completion-Level Batch Contract

GRPO batch 按 completion 语义理解。

G 实验冻结逻辑：

```text
generation_batch_size = 4 completions

G=2 → 2 prompts × 2 completions
G=4 → 1 prompt  × 4 completions
```

可从以下 smoke 语义开始映射到 ms-swift：

```text
per_device_train_batch_size = 1 completion
gradient_accumulation_steps = 4
generation_batch_size = 4 completions
```

最终参数名/约束以 pinned ms-swift 实测为准，但不能改变上述实验语义。

### 13.4 Reward

```text
R1 = accuracy
R2 = accuracy + 0.1 * format
```

accuracy extractor 支持 boxed + fallback final answer。格式失败不自动等于答案错误。

### 13.5 Accounting

必须记录：

- prompts processed；
- completions generated；
- generated tokens；
- optimizer steps；
- total/accuracy/format reward；
- group reward variance；
- peak VRAM / wall time；
- KL（standard only）。

---

## 14. Reward Contract

概念接口：

```text
input:
  prompt
  completion
  ground_truth
  metadata

output:
  total_reward
  components
```

示例：

```json
{
  "total_reward":1.1,
  "components":{"accuracy":1.0,"format":0.1}
}
```

每个 component 单独记录。

---

## 15. Training Budget

跨实验必须记录：

```text
samples
tokens
optimizer steps
prompts
completions
generated tokens
wall time
GPU hours
```

Experiment Spec 必须声明：

```text
budget_type = tokens | steps | epochs | completions
```

CPT size scaling 与 SFT LR 的 budget 语义不同，不能机械统一。

---

## 16. Seed Policy

默认：`seed=42`。

固定 robustness：

```text
SFT LR baseline/mid point
seed=42/43/44
```

GRPO multi-seed 留到后续版本。

---

## 17. Checkpoint / Merge Policy

每个 stage 保存：

```text
adapter hash
merge config
merged parent hash
parent hash
```

官方 Stage Result 必须明确：final/best checkpoint 哪一个用于 stage-end comparison。

Repo 不要求提交模型权重，但必须能重建 lineage。

---

## 18. Cross-stage Interpretation

禁止：

```text
CPT loss < SFT loss
所以 CPT 模型更好
```

CPT CE、SFT CE、DPO objective、SimPO objective、GRPO objective 不在同一数值尺度。

可跨 stage 比较的是：

- accuracy@1；
- PPL（适用时）；
- format/parse；
- common preference metrics；
- reward components（GRPO 内）；
- resource cost。

---

## 19. Resource Metrics

所有 stage：

```text
peak_vram
wall_time
tokens_per_second / samples_per_second
optimizer_steps
```

GRPO 额外：completions/s、generated tokens、profile。

---

## 20. Failure Classification

Smoke/Run 失败至少区分：

```text
OOM
CUDA_ERROR
NAN_LOSS
DATALOADER_ERROR
VERIFIER_ERROR
GENERATION_ERROR
CONFIG_ERROR
UNKNOWN
```

失败 run 也保留 resolved config 与 environment。

---

## 21. Training Acceptance Criteria

- [ ] PEFT-CPT official path 跑通
- [ ] Full-CPT smoke 有明确 feasibility 结论
- [ ] adapter→merge→fresh adapter 边界可重复
- [ ] M1/M2 hashes 可追溯
- [ ] SFT 只用完整 eligible traces
- [ ] DPO 只用 verified length-controlled pairs
- [ ] SimPO-original `cpo_alpha=0`
- [ ] GRPO completion-batch contract 已 smoke
- [ ] GRPO-standard 已尝试
- [ ] GRPO-lowmem fallback 已验证
- [ ] same-GPU vLLM 不作为 8GB 默认路径
- [ ] reward correctness/format 解耦
- [ ] 资源指标完整
- [ ] SFT baseline point 3-seed 完成
