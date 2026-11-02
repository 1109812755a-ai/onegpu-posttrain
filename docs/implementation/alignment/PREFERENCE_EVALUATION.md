# Common Preference Evaluation

`OneGPU-PostTrain` does not compare DPO and SimPO using trainer-specific reward values. Their objectives have different scales, so those numbers are useful for training diagnostics but are not a fair public cross-algorithm metric.

The common evaluator uses the frozen held-out file:

```text
data/processed/smoke/openr1/canonical/preference_eval.jsonl
```

This dataset is derived only from the top-level OpenR1 holdout pool. Its `problem_id` set has zero overlap with the DPO/SimPO training preference dataset.

## Score definition

For one prompt and response `y`:

```text
score(y | x) = mean token log p(y_t | x, y_<t)
```

Only response tokens are scored. The prompt is rendered with the model tokenizer's chat template plus generation prompt. No silent truncation is allowed.

For every pair:

```text
margin = score(chosen | prompt) - score(rejected | prompt)
```

Reported metrics:

```text
policy_pair_accuracy = fraction(margin > 0)
policy_pair_margin   = mean(margin)
```

Ties are reported separately and are not counted as wins.

## Why mean log-probability

Summed log-probability strongly favors shorter responses. The training data already controls chosen/rejected length ratios, and the evaluator additionally uses response-token mean log-probability so DPO and SimPO are compared using the same length-normalized policy score.

## Usage

```bash
onegpu-posttrain eval-preference \
  --model runs/smoke/sft-smoke-s42/merged/M2-SFT \
  --input data/processed/smoke/openr1/canonical/preference_eval.jsonl \
  --output runs/smoke/sft-smoke-s42/preference_evaluation
```

Repeat on merged `M3-DPO` and `M3-SimPO` models. The command writes:

```text
preference_metrics.json
preference_predictions.jsonl
```
