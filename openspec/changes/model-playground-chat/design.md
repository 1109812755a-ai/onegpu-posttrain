# Design

## Context

See `proposal.md` for the user problem and scope. The workspace already exposes verified model artifacts through run lineage and uses local Transformers generation in evaluation, but it has no free-form chat session or shared resource lease across all GPU task entry points. The current evaluation generator is single-prompt oriented; its model lifetime and task lifecycle are not sufficient for chat sessions or serial multi-model comparison.

## Goals / Non-Goals

**Goals:**

- Reuse model registry, hashes, source Run lineage, and local-only model loading rather than accepting browser-provided paths.
- Keep one local model loaded at a time, maintain bounded conversation context for single-model chat, and execute same-prompt comparison serially for up to three models.
- Make GPU admission, cancellation, unload, and failure cleanup explicit and coordinated with training and evaluation.
- Keep chat sessions distinct from formal `Run`/evaluation artifacts; surface model identity and decoding parameters beside every response.

**Non-Goals:**

- No remote Provider chat, arbitrary model paths, shell commands, model downloads, training, automatic dataset registration, benchmark scoring, or claims of factual correctness.
- No simultaneous multi-model loading, shared conversations across devices, or persistent chat history by default.
- No support for unverified adapter-only artifacts in the first version; expose only locally loadable base/merged models with verified identity and lineage.

## Decisions

### 1. Model catalog is derived from verified lineage, not caller paths

Add a server-side catalog adapter over model records and their producing/merge Runs. It returns only local, supported base or merged artifacts whose model ID, digest, path containment, existence, and source lineage validate. The API accepts model IDs; it resolves paths after validation. Adapter-only outputs remain unavailable until the loader can prove the base-model/adapter pair and load them safely.

This follows the existing verified handoff and stage-evaluation boundary. A directory browser or arbitrary `model_path` API would be more flexible but would bypass lineage and path-safety checks.

### 2. A model experience session is an ephemeral inference job, not a Run

The browser keeps conversation turns in page session state and submits bounded requests to a local inference job API. Server job state contains only a session/execution ID, model ID/hash, source Run, generation settings, progress and terminal status; it does not persist prompts or completions to general logs, Run lineage, evaluation results, or datasets. Explicit export is a separate user-confirmed action and is outside the default session lifecycle.

Single-model mode sends the current bounded history to the selected model. Comparison mode sends the same current user prompt independently to each selected model with no cross-model conversation state, then renders the responses side by side. This avoids comparing models with different histories and avoids accumulating a chat transcript across sequential models.

### 3. Serialize inference and coordinate all GPU users

Introduce one workspace-scoped GPU task lease shared by training, formal evaluation, and model experience. Each request obtains the lease before loading weights; a busy lease returns the active task identity and a link to its Run/task view. Comparison retains one lease for the comparison job, but loads, generates, and releases each model before proceeding to the next. Cancellation and every exception path unload the model and release the lease in `finally` cleanup. Startup recovery detects stale leases and reports them for safe reconciliation instead of silently assuming the GPU is free.

Separate supervisor-local `active` flags are insufficient: they prevent duplicates within one workflow but do not guarantee cross-workflow GPU exclusion. The lease is therefore the common admission boundary, with existing launch and evaluation starts updated to claim it.

### 4. Reuse local Transformers but add an explicit bounded session loader

Use the installed local inference stack with `local_files_only`, `trust_remote_code=False`, and the verified model path. Keep a single model/tokenizer pair alive for one single-model session; unload when the model changes or the session ends. Comparison creates one ephemeral loader at a time. Apply stage-independent, server-enforced bounds for prompt/history tokens, output tokens, and decoding settings; reject oversized context before GPU allocation. Do not fetch weights or execute model-supplied code.

Using the existing evaluation command per message would repeatedly start subprocesses and reload weights, making multi-turn chat slow. A new inference session manager provides clear model lifetime and cancellation without changing formal evaluation behavior.

### 5. Add a dedicated workspace subview and a Run handoff

Place “模型体验” under the existing “训练与评测” module. A successful Run's verified model card gets a “试用此模型” deep link carrying the model ID and return Run ID, never a path. The experience page allows one model for multi-turn chat or up to three models for same-question comparison. It labels every response as qualitative and unverified. Unsupported/missing models show a reason and no start action.

This is easier to discover than hiding chat inside Run details, while the deep link makes the run-to-model relationship immediate. It avoids adding a new top-level navigation module.

## Risks / Trade-offs

- [A global lease touches several launch paths and could block work after a crash] → Use atomic acquisition, owner/task metadata, heartbeat or process identity, and explicit stale-lease recovery; test release on cancellation, OOM, and exceptions.
- [Models may exceed local VRAM or context limits] → Admit only declared supported artifacts, enforce token/output caps before loading, unload between comparisons, and show an actionable OOM/resource state.
- [Sequential comparison takes longer than parallel generation] → Show per-model queue/load/generation status and retain completed responses if a later model fails; serial execution is the safe single-GPU trade-off.
- [Conversation prompts may contain sensitive material] → Keep prompt/response data out of server logs and persistent artifacts by default; provide an explicit, scoped export confirmation.
- [Verified merged-only scope excludes some stage adapters] → State the eligibility rule in the UI and add adapter support only after base/adapter lineage and loader behavior can be validated.

## Migration Plan

1. Implement and test model catalog resolution and shared GPU lease without changing the existing Run or evaluation artifact formats.
2. Integrate training/evaluation entry points with the shared lease; verify existing training and formal evaluation still serialize correctly.
3. Add the ephemeral inference job API and single-model chat, including bounded history, cancellation, unload, and safe state polling.
4. Add serial same-prompt comparison for up to three eligible models and partial-failure display.
5. Add the workspace subview, Run deep link, mobile layout, accessibility states, and API/UI tests.
6. Rollback by hiding the model-experience view and disabling its API routes; retain old Run/model/lineage records unchanged.

No data migration is required. Chat sessions are ephemeral unless explicitly exported.
