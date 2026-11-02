# Third-Party Licenses and Attribution

This file is part of the v0.1 release gate. Exact source revisions will be filled by the resolved manifests used for official runs.

| Component | Upstream | License / attribution status |
|---|---|---|
| Base model | Qwen/Qwen3-0.6B-Base | Apache-2.0 — revision must be pinned |
| SFT/alignment data | open-r1/OpenR1-Math-220k | Apache-2.0 — revision/config must be pinned |
| CPT data | HuggingFaceTB/finemath / finemath-4plus | ODC-BY — attribution required |
| Training engine | modelscope/ms-swift | Follow upstream package license |
| Evaluation | modelscope/evalscope | Apache-2.0 |
| Math verification | huggingface/Math-Verify (`math-verify==0.9.0`) | Apache-2.0 |

The repository must not redistribute full upstream datasets. Only small examples, manifests and preparation scripts are committed.
