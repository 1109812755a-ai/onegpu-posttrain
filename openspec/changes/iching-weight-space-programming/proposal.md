# Proposal

## Why

The experiment is now targeted at the paper's Qwen3.5-0.8B model family, using the pinned `Qwen/Qwen3.5-0.8B` checkpoint revision `05f056712f8c582c8708395156518d963b46fb29`. A WSL Docker run proved that this checkpoint loads with Transformers 5.8.0 and PyTorch 2.6.0+cu124 on the RTX 4060, and measured a 33/81 (40.74%) baseline on the protected I Ching holdout (`ds-eval-upload-54ca42ca021d295d8390`). The earlier 39/81 result belongs to Qwen3-0.6B-Base and is not the baseline for this target. The registered 300-question I Ching practice set (`ds-eval-upload-14aee30e5c7ec6ec0959`) remains for iteration feedback. A dedicated workbench can use the verified Bailian `qwen3.8-max` model to propose bounded weight edits while preserving the base checkpoint and separating practice feedback from final evidence.

## What Changes

- Add a dedicated local web workbench that sends bounded weight-inspection and edit-tool calls to Bailian `qwen3.8-max` and records each model/tool turn.
- Use the pinned Qwen3.5-0.8B checkpoint as the target model; keep Bailian `qwen3.8-max` explicitly identified as the edit-planning model.
- Run target-model inference in an isolated, digest-pinned WSL Docker image with the tested Transformers/PyTorch versions, leaving the project's existing Transformers `<5` environment unchanged.
- Use the 300-question set for iterative practice feedback and reserve the fixed 81-question set for explicit final evaluation.
- Apply validated edits only to isolated candidate checkpoints; record the operation, parent model, weight diff, and before/after digests.
- Keep the existing local provider credential in `.env.local`; add a dedicated Qwen3.8-Max profile without replacing the existing Qwen3.6 profile or exposing credentials to the browser.
- Show model, request budget, and estimated cost when known before a billable iteration; require user confirmation and clearly show when pricing is unknown.
- Identify the workflow as a paper-inspired domain adaptation: the target model family matches the paper, but the I Ching task, practice/holdout data, and objective-choice metric differ from the paper's experiment.

## Capabilities

### New Capabilities

- `weight-space-experiment`: Local, auditable, Bailian-powered iterative weight-space experiments with isolated candidates and protected final evaluation.

### Modified Capabilities

- `evaluation-lineage`: Associate each practice iteration and final holdout result with the exact candidate checkpoint, dataset role and digest, edit iteration, and evaluation protocol.
- `web-workspace-ux`: Make the new experiment workbench discoverable and usable, including provider status, cost disclosure, and clear incomplete/blocked/failed states.

## Impact

Builds on the existing React workspace navigation, local FastAPI UI, model/dataset registries, objective-choice evaluator, and OpenAI-compatible Bailian provider. Adds a dedicated Qwen3.8-Max profile, a bounded tool-call executor, a pinned Docker inference runtime for Qwen3.5-0.8B, and local experiment artifacts. The existing 81-question suite remains unchanged and is used only for final comparisons.
