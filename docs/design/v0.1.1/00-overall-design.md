# OneGPU-PostTrain — Overall Design

> **Design Version:** v0.1.1  
> **Target Release:** v0.1  
> **Freeze Date:** 2026-10-12  
> **Status:** FROZEN FOR IMPLEMENTATION

> **Tagline:** One GPU. Full Post-Training. Reproducible Experiments.

---

## 1. 项目定位

OneGPU-PostTrain 不是新的 Trainer，而是一个面向消费级单卡用户的 **LLM Post-Training 可复现实验室**。

它把成熟训练框架之上的四件事连起来：

```text
Data → Training → Evaluation → Experiment
```

项目希望用户不仅能运行：

```text
Base → CPT → SFT → DPO / SimPO / GRPO
```

还能够回答：

- 每个阶段到底吃什么数据；
- 每个阶段优化什么目标；
- 模型在哪些能力上变好/变差；
- 修改一个参数后发生什么；
- train loss、reward 与真实任务能力是否一致；
- 在 RTX 4060 8GB 上这些方法的真实成本是什么。

---

## 2. v0.1 核心目标

1. 单 RTX 4060 8GB 跑通完整 post-training 教学链路；
2. 从 Base Model 开始观察 PEFT-CPT、SFT、Preference Optimization、RL 的阶段变化；
3. 给出每阶段可直接理解的数据 schema；
4. 用统一 evaluator 比较各 checkpoint；
5. 支持 One Variable At A Time（OVAT）消融；
6. 记录 GPU 显存、吞吐、时间等成本；
7. 所有官方结果能够回溯到数据、配置、版本、seed 和 checkpoint lineage；
8. 所有开源 README 结果均来自真实 run，不使用模拟数字。

---

## 3. 非目标

v0.1 不做：

- Web UI / FastAPI；
- Kubernetes / Ray / 多机多卡；
- 自动 HPO；
- Dataset/Model Registry 服务；
- PPO / Reward Model / PRM；
- Agent RL / Tool RL；
- 多模态；
- 多训练框架统一 SPI；
- Leaderboard 服务。

原则：**先成为一个可信的小型实验项目，再演进成平台。**

---

## 4. 技术基线

| 组件 | v0.1 冻结选择 |
|---|---|
| Base Model | Qwen3-0.6B-Base |
| Domain | Mathematical Reasoning |
| Reference GPU | RTX 4060 8GB / Single GPU |
| Training Backend | ms-swift |
| Evaluation | EvalScope + Custom Evaluators |
| CPT Data | FineMath-4+ |
| SFT/Alignment Data | OpenR1-Math-220k |
| Experiment Method | OVAT |
| Tracking | Local artifacts + TensorBoard；W&B optional |

所有 model/dataset/tokenizer/verifier/tooling 版本必须 pin。

---

## 5. 为什么选择数学推理

数学场景具有：

- 标准答案可客观验证；
- reasoning 数据丰富；
- preference pair 易构造；
- GRPO reward 可程序化；
- benchmark 清晰；
- 很适合观察 reward、格式、正确率之间的关系。

项目目的不是把 0.6B 模型训练成最强数学模型，而是建立完整实验链路。

---

## 6. 总体训练链路

```text
M0: Qwen3-0.6B-Base
        │
        │ PEFT-CPT（8GB 官方路径）
        │ + Full-CPT Smoke Probe（独立可行性测试）
        ▼
M1: merged parent
        │
        │ fresh SFT adapter
        ▼
M2: merged parent
        │
        ├───────────────┬────────────────────┐
        ▼               ▼                    ▼
   M3-DPO        M3-SimPO-original        M3-GRPO
                    cpo_alpha=0             │
                                            ├─ standard beta=0.04
                                            └─ lowmem   beta=0
```

### 6.1 Stage Boundary

官方阶段边界固定：

```text
merged parent
→ fresh stage adapter
→ train
→ save adapter + hash
→ merge into non-quantized parent
→ save merged checkpoint + hash
→ next stage reloads merged parent
→ fresh adapter
```

DPO、SimPO-original、GRPO 必须从同一个 M2 hash 平行分叉，不默认串联。

### 6.2 CPT 定义

必须区分：

```text
Full CPT = full-parameter continued pretraining
PEFT-CPT = LoRA/QLoRA continued training
```

v0.1 对 8GB 的官方承诺是 **PEFT-CPT**。Full-CPT 只做 smoke feasibility probe。

---

## 7. 总体数据链路

### 7.1 FineMath / CPT

```text
FineMath-4+
  ↓
document-level split
  ├── CPT Train Docs
  └── CPT Holdout Docs
  ↓
benchmark contamination verification
  ↓
chunk / pack separately
```

### 7.2 OpenR1 / SFT & Alignment

```text
OpenR1-Math-220k
  ↓
Normalize + Exact Dedup
  ↓
External Benchmark Decontamination
  ↓
Programmatically Verifiable Math Core
  ↓
Problem-level Split
  ├── SFT Pool
  │     └── complete + correct + parseable + length-eligible traces
  ├── Alignment Core
  │     ├── length-controlled Preference → DPO / SimPO-original
  │     └── prompt + verifiable GT      → GRPO
  └── Internal Holdout
```

关键规则：

- benchmark 去污染在 split 前完成；
- 不对长 CoT 静默 truncation；
- 长度在 pinned tokenizer + 最终 template 渲染后判断；
- GRPO 只使用程序可验证题型；
- 所有 gate 前后记录 retention 与分布变化。

---

## 8. 数据规模

| Recipe | CPT | SFT | Preference | GRPO | Internal Eval |
|---|---:|---:|---:|---:|---:|
| Smoke | 500K tokens | 200 | 200 pairs | 200 | 100 |
| Mini | 5M tokens | 5K | 2K pairs | 2K | 500 |
| Standard | 20M tokens | 10K | 5K pairs | 5K | 1K |

数字均指**最终 eligible 数据量**。

---

## 9. 4060 训练策略

```text
LoRA / QLoRA
per-device batch usually 1
Gradient Checkpointing ON
Sequence length data-driven
Same-GPU vLLM OFF
```

### 9.1 SimPO

官方算法比较冻结为：

```yaml
beta: 2.0
simpo_gamma: 1.0
cpo_alpha: 0.0
```

名称：**SimPO-original**。

### 9.2 GRPO Profiles

```text
GRPO-standard:
  beta=0.04
  reference enabled
  KL available

GRPO-lowmem:
  beta=0
  reference disabled
  KL=N/A
```

standard 先 smoke；如 reference model 在 8GB OOM，则参考 GPU 官方路径使用 lowmem，并在所有结果中显式标记 profile。

---

## 10. Evaluation 总体体系

### Layer 1 — Training Dynamics

- algorithm-specific loss；
- LR / grad norm；
- tokens/s、wall time、peak VRAM；
- tokens/samples/completions/generated tokens。

**不同算法的 loss 绝对值不跨阶段比较。**

### Layer 2 — Domain Capability

- GSM8K `accuracy@1`；
- MATH-500 `accuracy@1`；
- Internal Math Holdout `accuracy@1`；
- FineMath Domain PPL。

### Layer 3 — Alignment / RL

SFT+：format accuracy、parse rate、response length。

Preference 公共指标：

```text
policy_pair_accuracy
policy_pair_margin
```

score 使用 length-normalized conditional log-probability。

GRPO：reward components、group variance、zero-reward ratio、generated tokens、completion length、KL（standard only）。

### Layer 4 — General Guardrail

- General Text Holdout PPL；
- selected non-math MMLU subsets。

---

## 11. 指标分离规则

对于数学输出：

```text
answer correctness
parse success
format correctness
```

必须分开。

例如答案正确但未使用 `\boxed{}`：

```text
accuracy = 1
format_accuracy = 0
```

GRPO reward 同样遵守该规则。

---

## 12. v0.1 官方实验

### Required

1. **CPT Dataset Size**：nested `5M ⊂ 10M ⊂ 20M`；
2. **SFT Learning Rate**：same samples + same steps；中间 baseline 点 seed=42/43/44；
3. **DPO vs SimPO-original**：同 M2、同 pairs、同预算语义；
4. **GRPO Reward Design**：accuracy vs accuracy+format。

### Recommended

5. DPO Beta：0.05 / 0.1 / 0.2；
6. GRPO Num Generations：G=2 / G=4，固定 completion-level rollout budget。

每个 Experiment 必须显式记录 `budget_type = tokens | steps | epochs | completions`。

---

## 13. Run Artifact

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

## 14. Repository Structure

```text
onegpu-posttrain/
├── README.md
├── README_zh.md
├── LICENSE
├── CONTRIBUTING.md
├── THIRD_PARTY_LICENSES.md
├── pyproject.toml
├── docs/
│   ├── 00-overall-design.md
│   ├── 01-data-design.md
│   ├── 02-training-recipe-design.md
│   ├── 03-evaluation-design.md
│   ├── 04-experiment-design.md
│   ├── 05-design-review.md
│   └── DESIGN_FREEZE.md
├── data/
│   ├── examples/
│   ├── manifests/
│   ├── raw/
│   ├── processed/
│   └── reports/
├── recipes/
│   ├── smoke/
│   ├── mini/
│   └── standard/
├── adapters/ms_swift/
├── rewards/
├── evaluation/
├── experiments/
├── onegpu_posttrain/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── experiment/
│   └── reporting/
├── runs/
└── results/official/
```

---

## 15. Reproducibility Contract

每个官方 run 冻结：

- model repo + revision；
- dataset repo + config + revision；
- benchmark revision；
- tokenizer revision；
- math verifier version；
- ms-swift / EvalScope / PyTorch versions；
- exact/resolved config；
- prompt/reward/eval protocol versions；
- seed；
- git commit；
- GPU/CUDA/driver；
- source row hashes / processed SHA256；
- parent/adapter/merged checkpoint hashes。

---

## 16. Open-source Compliance

Release Gate 包含：

- Qwen/OpenR1 Apache-2.0 attribution；
- FineMath ODC-BY attribution；
- `THIRD_PARTY_LICENSES.md`；
- 不重新分发上游大数据集；
- derived examples 明确来源。

---

## 17. v0.1 Release Gate

### Pipeline

- [ ] Base Eval
- [ ] PEFT-CPT
- [ ] Full-CPT smoke 给出明确 feasibility 结论
- [ ] SFT
- [ ] DPO
- [ ] SimPO-original
- [ ] GRPO-standard smoke
- [ ] GRPO-lowmem fallback validated

### Data

- [ ] revisions pinned
- [ ] decontamination before split
- [ ] Verifiable Math Core
- [ ] Length Eligibility Gate
- [ ] length-controlled preference pairs
- [ ] verifiable GRPO prompts
- [ ] document-level CPT holdout
- [ ] overlap/near-duplicate reports
- [ ] artifact hashes

### Evaluation

- [ ] GSM8K/MATH/Internal accuracy@1
- [ ] Domain PPL / General PPL
- [ ] MMLU guardrail
- [ ] format / parse
- [ ] policy_pair metrics
- [ ] GRPO reward metrics
- [ ] KL profile rule

### Experiment

- [ ] 4 required experiments complete
- [ ] SFT representative 3-seed result
- [ ] no fake numbers

---

## 18. Design Freeze

以下核心项在实现阶段不再随意修改：

```text
Project: OneGPU-PostTrain
Model: Qwen3-0.6B-Base
Domain: Math
Backend: ms-swift
Official CPT: PEFT-CPT
SFT/Alignment: OpenR1 Verifiable Math Core
SimPO: original, cpo_alpha=0
GRPO: standard/lowmem profiles
Stage boundary: fresh adapter → merge → fresh adapter
Main metric: accuracy@1
Method: OVAT
```

只有 smoke test 证明 RTX 4060 8GB 存在硬性不可行时，允许调整实现可行性参数，并必须在 resolved recipe 中记录。
