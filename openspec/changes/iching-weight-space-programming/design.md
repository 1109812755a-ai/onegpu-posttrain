# Design

## Context

The React workspace routes pages through `web/src/main.tsx` and `WorkspaceNavigation.tsx`; its local FastAPI server already exposes registered objective-choice evaluation. `StageEvaluationAdapter` accepts registered base models or produced stage models, and its objective-choice path currently requires `stage=eval` plus `dataset_role=evaluation_only`. The experiment therefore needs explicit practice/final roles and candidate model lineage rather than treating each round as an ordinary final evaluation. The target is the pinned `Qwen/Qwen3.5-0.8B` checkpoint, not the previously considered Qwen3-0.6B-Base. See proposal.md for the verified baseline and the spec deltas for user-visible contracts.

The existing local Bailian profile uses the OpenAI-compatible API and a local credential. A minimal request to `qwen3.8-max` succeeded through the configured Beijing DashScope endpoint. The existing profile is pinned to Qwen3.6 for other workflows, so the experiment needs its own fixed Qwen3.8-Max profile and must not rewrite or silently repurpose the existing one.

The project environment currently pins Transformers below 5, while Qwen3.5 requires the newer multimodal model implementation. A WSL Docker experiment validated a separate runtime using `hiyouga/llamafactory@sha256:2167de797ca005b1c9b013820e0264db17b799945588cdfe11e9d78cc42576a4` (PyTorch `2.6.0+cu124`, Transformers `5.8.0`) and `AutoProcessor` plus `AutoModelForMultimodalLM`. The tested checkpoint is `Qwen/Qwen3.5-0.8B` revision `05f056712f8c582c8708395156518d963b46fb29`. The target model and registered evaluation input are mounted read-only into the container; candidate artifacts are written to a separate persistent experiment location. This keeps the project's existing Python environment independent of the model runtime.

## Goals / Non-Goals

**Goals:**

- Make each practice feedback round and final 81-question evaluation attributable to an immutable model checkpoint, dataset digest, protocol, and experiment iteration.
- Call Bailian Qwen3.8-Max directly from the local backend and provide a bounded tool loop for tensor inspection and edit proposals.
- Keep API credentials on the local backend, disclose data sent to Bailian, and require explicit per-iteration confirmation when the request may be billable.
- Preserve the existing objective-choice scoring definitions and report per-type counts, overall counts, parse failures, and coverage.
- Pin the model revision, processor/chat-template behavior, generation settings, container image digest, and runtime versions as part of evaluation identity. Baseline and candidate runs must use the same inference wrapper.
- Let the user inspect, retain, or discard candidate branches and compare a selected candidate with the original baseline.

**Non-Goals:**

- Reproducing the paper's original task and data exactly; this uses the paper's Qwen3.5-0.8B target family on an I Ching objective-question adaptation.
- Training, quantization, merging, arbitrary Python execution, or unrestricted weight-file replacement through model tools.
- Making the 81-question holdout or its per-question feedback available to iterative model calls.

## Decisions

### Use a dedicated fixed Bailian profile

Add a profile for model ID `qwen3.8-max` using the existing configured compatible endpoint and local credential environment variable. Keep the endpoint fixed by the backend and reject endpoint/model overrides from browser requests. Read the credential from the existing local credential mechanism; never return it to the UI, persist it in experiment artifacts, or log request authorization headers. Leave the existing Qwen3.6 profile untouched.

The UI shows the active model, the maximum planned call/token budget, and pricing derived from configured profile rates when available. If either rate is missing or stale, cost is shown as unknown and the user must confirm before a request. Before the first call, disclose that practice prompts, model predictions, and requested tensor inspection results are sent to Bailian.

### Isolate target-model inference in a pinned Docker runtime

Run Qwen3.5 inference in WSL Docker using the digest-pinned Transformers 5.8.0/PyTorch 2.6.0+cu124 image and `AutoProcessor`/`AutoModelForMultimodalLM`. Do not upgrade or loosen the project's existing Transformers `<5` dependency for this experiment. Preflight must verify the locally available model snapshot revision and weight digest, GPU visibility, and the mounted practice/holdout dataset identities. Mount the registered starting checkpoint and evaluation inputs read-only; write only isolated candidate checkpoints and experiment artifacts to the designated output mount. If the container cannot access the model source, require a verified local cache or a configured reachable model source rather than silently substituting another checkpoint.

The reference baseline uses `Qwen/Qwen3.5-0.8B` revision `05f056712f8c582c8708395156518d963b46fb29`, Transformers' Qwen3.5 processor chat template, greedy generation (`do_sample=false`, `max_new_tokens=16`), and the existing `choice-accuracy-v1` parser/scorer. The verified 81-row input file SHA-256 is `19f02eab2e2d3894269eda8f41115c22c2f203cfe3c3cbb3604ab19488004ac2`. Its observed baseline is 33/81 (40.74%): single choice 16/27, true/false 17/30, multiple choice 0/24, with zero parse failures. The historical 39/81 Qwen3-0.6B-Base result is a separate model result and MUST NOT be used as the Qwen3.5 baseline.

### Implement a bounded function-call loop

Extend the provider request/response adapter to represent tool declarations and tool-call responses needed by the OpenAI-compatible interface. The experiment service owns a bounded multi-turn loop with maximum calls, elapsed time, and output-token limits. The model can inspect allowed tensor metadata or bounded slices and request supported edit operations; a local tool dispatcher validates each request and returns only the requested bounded result. Tool names and schemas are fixed server-side, not supplied by the browser or model.

Reject unsupported calls, unknown tensors, stale candidate digests, incompatible shapes/dtypes, non-finite values, and edits beyond the configured element/value budget. Record sanitized provider/tool events, token usage, and provider request IDs where available. A timeout, malformed response, or invalid operation cannot mutate a candidate.

### Keep an experiment manifest and immutable candidate branches

Create a versioned experiment record under `runs/weight-space/<experiment_id>/`. It pins the starting model ID and digest, practice/final dataset IDs and canonical digests, protocol version, paper-adaptation disclosure, iteration counter, and baseline reference. Each iteration stores its provider profile/model identity, bounded call log, practice feedback, edit operation, validation report, candidate checkpoint, and evaluation. Candidate records name a parent checkpoint; the registered base artifact is read-only. Applying an edit is transactional and writes a distinct candidate. A discarded candidate remains in history but is not eligible for final evaluation unless a separate restore action is recorded.

Prefer immutable candidate branches over in-place edits so experiments can compare, recover, and prove which weights produced a score. Use copy-on-write only if candidate writes cannot mutate or share writable storage with the base checkpoint.

### Register candidates and separate practice evaluation from final evaluation

Register each accepted candidate as a derived model with its parent checkpoint and edit artifact in lineage. Add a dedicated practice-evaluation path that reuses the objective-choice parser and exact-match scoring but tags output `practice`; do not weaken the existing `evaluation_only` guard globally. The final-evaluation action uses only the experiment's pinned 81-question dataset and existing `choice-accuracy-v1` protocol, recording candidate digest and `final_holdout` role.

The workbench displays practice metrics as iterative selection feedback and final metrics as held-out evidence. It reports changes descriptively and does not claim that an edit caused a general capability gain. Comparisons are valid only when model revision, container/runtime identity, chat template, generation parameters, dataset digest, sample coverage, and scoring protocol match.

### Add a dedicated local workspace page and API surface

Add a first-class workspace route for experiment setup, provider readiness, request review, iterative runs, candidate review, and final comparison. A focused backend adapter owns experiment validation and storage; it calls the existing model/dataset registries and evaluator instead of duplicating their parsing and scoring rules. Mutating experiment endpoints remain available only through the local experiment UI's loopback-only operating mode.

## Risks / Trade-offs

- [Provider tool-call shapes or parameters may differ from the current adapter's supported subset] → Add compatibility checks for the exact Qwen3.8-Max profile, validate every response, and keep all weight operations behind the local dispatcher.
- [Iterations send practice questions and bounded tensor information to a third party and may incur cost] → Disclose data egress and request limits, show configured cost or unknown status, and require per-iteration confirmation.
- [A valid edit may reduce accuracy or damage model behavior] → Preserve every parent checkpoint, require practice evaluation before final evaluation, and allow candidate discard without deleting provenance.
- [Practice and holdout registration metadata may currently share the evaluation-only role] → Add an explicit experiment role at the experiment record/evaluation artifact level and reject dataset identity/digest collisions; do not relabel the 81-question set as practice.
- [Repeatedly inspecting the holdout could overfit model selection] → The UI and tool dispatcher exclude holdout rows and per-question results from iteration requests and disclose any reuse after final scores are seen.

## Migration Plan

No existing checkpoint, provider profile, or dataset is rewritten. The workbench creates a new experiment record, dedicated provider profile, and candidate artifacts beside existing runs. Rollback consists of stopping iterations, discarding candidates, and returning to the pinned registered base model; evaluation history remains available. The registered 300-question corpus and 81-question suite retain their current IDs and content digests.
