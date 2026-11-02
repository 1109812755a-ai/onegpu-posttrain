# Design

## Context

See [proposal.md](proposal.md) for motivation and [spec.md](specs/local-experiment-launcher/spec.md) for observable behavior. The first supported entry is pinned to `recipes/mini/23-sft-1024-fallback.yaml`: parent `model-93611b3dbc4e10161349`, dataset `ds-sft-derived-1024-690ad3799b30c06d8d6f` (307 examples), 1,024 max length, seed 42. The same configuration has already completed an exploratory SFT and M2 merge on RTX 4060; sampled peak VRAM was 7,939 MiB of 8,188 MiB, so the UI must warn that headroom is narrow. Current FastAPI routes do not create processes, and `run-sft` currently creates its own unique `run_id`.

## Goals / Non-Goals

**Goals:**

- Deliver the smallest real user path from the browser to one known SFT run and back to its actual output evidence.
- Reuse existing CLI resolution, execution, status and lineage rather than implement a second trainer.
- Keep preflight and failed requests from launching or changing training artifacts.
- Keep task state/logs readable after browser refresh while the UI service remains alive.

**Non-Goals:**

- Supporting other stages, arbitrary dataset/model selection, recipe/parameter editing, evaluation launch, cancellation UI, queues, or parallel runs.
- Guaranteeing VRAM sufficiency from a snapshot or preventing a separate terminal process from using the GPU.
- Continuing a training process after the supervising UI service shuts down, auto-resuming interrupted training, or treating interrupted checkpoints as complete models.

## Decisions

### 1. Hard-code one reviewed recipe identity, not arbitrary paths

The capability catalog exposes a single stable key for `recipes/mini/23-sft-1024-fallback.yaml`. The recipe itself binds the registered Mini M1 and 307-example 1,024-token dataset. API payloads contain only the stable key and the exact preflight token; they do not carry paths, model/dataset overrides, CLI args, or environment. The backend resolves paths from the workspace and verifies the current recipe, manifest evidence, canonical content digest and model identity before preflight and immediately before start. This deliberately favors a usable end-to-end proof over a generic recipe browser; new stages can be added as separate reviewed catalog entries later.

### 2. Reuse SFT planning and execution contracts

Preflight uses the existing SFT planner/resolution logic in an isolated OS temporary directory and removes temporary plan files afterward; it creates no run/model/data artifact and performs no download. Extract only shared SFT resolution if needed to avoid drifting semantics. After explicit confirmation, call the existing `run-sft` entry with the fixed recipe/model/dataset/output and a server-generated `--run-id`. Make this argument optional in `run-sft` only, preserving auto-generated IDs for direct CLI users and other stages. Always invoke a fixed argv array with `shell=False`.

The existing CLI remains authoritative for training, checkpoint discovery, optional M2 merge, `run.json` and `lineage.json`. The launcher must report only artifacts found in those records; it must not synthesize model success from process exit alone.

### 3. Bind confirmation to actual inputs and serialize UI launches

Create an opaque, in-memory, one-use preflight token with a short expiry and config fingerprint. The fingerprint includes recipe bytes/resolved config, input IDs/content digests and generated output target. On start, consume the token and repeat critical checks. Generate the unique `run_id` after confirmation; the recipe's `experiment_id` and `run_label` remain fixed grouping/display metadata.

A process-local lock protects one UI-launched run. The UI is explicit that this is not a global GPU scheduler: it cannot stop the user from starting CLI work independently. Recheck available GPU memory immediately before start and show its timestamp/approximate nature; do not treat the snapshot as a guarantee or automatically alter the recipe.

### 4. Supervise one process and persist bounded status/log evidence

Use a small SFT-only manager that launches a process group and atomically writes a versioned state file plus bounded combined stdout/stderr log under a dedicated launcher state folder keyed by `run_id`. API log reads return a bounded tail and truncation indicator. The UI polls status while active; progress is parsed only when an explicit supported trainer log pattern is present, otherwise shown as unknown.

Closing/navigating the browser does not affect the server-owned task. On graceful UI service shutdown, the manager requests termination for the SFT process group, waits a bounded interval, then force-terminates any remaining child; it records `interrupted` unless a completed CLI run record confirms success. At service startup, prior launcher records still marked running become interrupted/unknown, not resumed. This simple supervisor model avoids orphan training but requires the local service to stay open until training finishes.

### 5. Add an independent local opt-in and keep execution routes narrow

Add `--enable-experiment-launcher` to `serve-ui`, independent of policy editing. The default service exposes capability status but no active training controls. Refuse launcher startup with remote access or non-loopback bind. Mutating APIs require a server-minted token, same-origin/loopback checks, bounded typed JSON, and no client-controlled command/environment/path. Never expose tokens or inherited environment values in capabilities, logs, or responses.

### 6. Make the UI show exactly what it will do

The Launcher page shows the one supported SFT recipe and its fixed inputs. The primary progression is:

1. Show source availability and one-click preflight.
2. Display blockers separately from warnings, including exploratory data status and tight historical VRAM headroom.
3. Require explicit confirmation of recipe/config hash and a clear note that closing the UI service interrupts training.
4. Show actual running/terminal status, elapsed time, logs and CLI-produced outputs/lineage; after success label unified evaluation as pending and link to the CLI guidance.

The page is absent/disabled in default read-only mode. Keyboard focus, responsive behavior, status text independent of color, and loading/empty/error/stale states are part of the first vertical slice.

## Risks / Trade-offs

- [The historical 1,024 SFT used nearly all sampled VRAM] → Keep this exact recipe exploratory, show the measured peak and warning, recheck current resource state, and preserve OOM rather than silently changing settings.
- [User can launch CLI training outside the UI] → The single-task lock covers only UI-launched runs; make this limitation explicit and treat GPU checks as advisory.
- [Stopping the supervising service interrupts training] → Warn before launch and in startup/docs; preserve partial CLI logs/checkpoints, mark interrupted, never auto-resume.
- [CLI's progress logs may vary] → Keep bounded raw logs and show progress as unknown unless parsing is verified; terminal status and lineage come from actual CLI records.
- [A stale or tampered local input could invalidate the preflight] → Recheck recipe and input digests just before process creation and require a new one-use confirmation when they differ.

## Migration Plan

1. Existing UI startup remains unchanged; add an independent launcher flag, disabled by default.
2. Add `--run-id` only to `run-sft`, optional for CLI use and supplied by the launcher for a new execution.
3. Document the single supported SFT recipe, exploratory caveats, loopback restriction and service-shutdown behavior in English and Chinese.
4. Roll back by starting without the launcher flag; existing CLI runs and their artifacts remain intact.
