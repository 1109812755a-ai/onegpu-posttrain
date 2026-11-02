# ms-swift Adapter

OneGPU-PostTrain stores framework-neutral canonical datasets. Before training, `adapt-ms-swift` converts them into the current ms-swift standard format.

## Mappings

| Canonical stage | ms-swift native record |
|---|---|
| CPT | assistant-only `messages` |
| SFT | `messages` ending in assistant target |
| DPO / SimPO | chosen `messages` + `rejected_response` |
| GRPO | prompt-only `messages` + `solution` + trace IDs |

The canonical source file remains immutable. Every adapter output gets an adjacent `.adapter.json` containing record count and SHA-256 hashes for both input and output.

## Examples

```bash
onegpu-posttrain adapt-ms-swift \
  --stage cpt \
  --input artifacts/data/canonical/cpt.jsonl \
  --output artifacts/data/ms-swift/cpt.jsonl

onegpu-posttrain adapt-ms-swift \
  --stage dpo \
  --input artifacts/data/canonical/preference.jsonl \
  --output artifacts/data/ms-swift/preference.jsonl
```

GRPO retains `solution` as an extra column. It is never inserted into the model prompt; it is intended for the reward/verifier only.
