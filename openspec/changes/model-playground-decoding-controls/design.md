# Design

## Context

See `proposal.md` for motivation. The current page stores only `max_new_tokens` and `temperature` in React state and sends those in both chat and comparison requests. The session manager already validates `top_p` and `repetition_penalty`, but defaults them to 0.9 and 1.0; it does not pass `top_k` or a thinking-mode template option. Successful responses already include a `parameters` object, so the change can extend the existing request/response contract without adding persistent storage or a new endpoint.

## Goals / Non-Goals

**Goals:**

- Keep one shared set of controls and defaults for chat and serial comparison.
- Enforce the same parameter schema server-side, independent of browser-side bounds.
- Make clear when sampling-only settings are inactive because Temperature is zero.
- Carry the actual normalized settings in each successful response for display and reproducibility.
- Pass thinking-mode control through the model's chat template only when support is confirmed.

**Non-Goals:**

- Do not change training or formal evaluation decoding.
- Do not persist user-selected values or chat content across page reloads.
- Do not claim a decoding preset can correct an undertrained or malformed checkpoint.

## Decisions

### 1. Extend the existing parameter object

Use the existing `parameters` object on chat and comparison requests and the existing response `parameters` evidence. Add `top_k` and `enable_thinking`; normalize and validate all fields at the session-manager boundary. Keep bounds aligned between UI and backend: `max_new_tokens` 1–1024, `temperature` 0–2, `top_p` 0.01–1, `top_k` 0–100 (0 disables top-k), and `repetition_penalty` 0.5–2. Reject unknown keys, booleans in numeric fields, non-finite values, and out-of-range values before generation. This keeps the API narrow and avoids parallel per-mode contracts.

### 2. Keep recommended defaults visible and editable

Initialize both modes with max new tokens 256, temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05, and thinking disabled. Provide a one-action reset. Keep max tokens and temperature in the primary controls; put top-p, top-k, repetition penalty, and thinking mode in an explicitly labelled advanced section to avoid overwhelming first-time users. Do not save these settings between page visits. The 1.05 repetition penalty is a conservative product heuristic rather than a Qwen recommendation; the UI should label the preset as a starting point, not a quality guarantee.

### 3. Treat zero temperature and thinking support explicitly

When temperature is zero, set deterministic decoding (`do_sample=false`) and omit sampling-only `top_p`/`top_k` from the generation call; retain their selected values as requested settings but report that they were inactive for that response. When sampling is enabled, pass top-p and top-k explicitly. For thinking mode, pass `enable_thinking` while rendering the chat template only if the tokenizer template demonstrably supports the option; if the user enabled it for an unsupported template, reject before generation with a clear model-specific error. This is safer than silently discarding a setting or assuming every Transformers tokenizer uses Qwen's template contract.

### 4. Show response-specific parameter evidence

Keep parameter evidence attached to each individual response, not just the current form state, because users can change settings between chat turns and comparison output may be partial. For comparison, each completed model result receives the normalized/effective parameter object already returned by the session manager. Render configured and inactive sampling fields distinctly when temperature is zero. Failed rows do not show a success parameter summary.

## Risks / Trade-offs

- [A repetition penalty can reduce legitimate repeated symbols or wording] → Use a mild default of 1.05, keep it editable, and state that parameters are exploratory rather than validated quality settings.
- [Chat templates expose thinking controls inconsistently] → Detect support from the tokenizer template contract; reject enabled-but-unsupported requests and cover both supported and unsupported templates in tests.
- [More controls can make the experience page feel dense] → Keep core controls prominent, group remaining settings under an accessible advanced disclosure, and provide reset/default labels.
- [Temperature zero makes top-p/top-k ineffective] → Mark those values inactive in per-response evidence rather than implying they affected generation.

## Migration Plan

No persistent data migration is needed. Existing clients that omit parameters receive the new recommended defaults from the server. Roll back by restoring the old defaults and hiding the added controls; no stored records or model artifacts need conversion.
