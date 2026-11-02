# Design

## Context

The Launcher currently writes EvalScope's work directory below the training run (`runs/mini/ui-launcher/<training-run>/evaluations/<evaluation-run>/...`). EvalScope appends cache/retry identifiers to review filenames; on this checkout the attempted file path is 285 characters. The launcher already persists unique run state under `.onegpu-posttrain/launcher/runs/`, while most CLI training and evaluation artifacts use `run.json`, `evaluation_run.json`, and lineage sidecars under `runs/`. The main UI navigation still marks Runs as planned; only a current-run Launcher view is active.

## Goals / Non-Goals

**Goals:**

- Keep evaluation paths safely short on Windows without requiring the user to rename the repository.
- Preserve run identity and parent/producer links when evaluation output is no longer nested under its training output.
- Provide a read-only, local Runs index/detail view across supported run records, including old/incomplete evidence.
- Reuse the current filesystem artifacts and security constraints; preserve filters and accessible responsive interactions.

**Non-Goals:**

- Do not migrate or rewrite existing run artifacts.
- Do not expose arbitrary files, commands, recipes, or model weights through the Runs UI.
- Do not add persistence, a database, scheduling, multi-user support, or a generalized run launcher.
- Do not treat missing fields or validation warnings as successful provenance.

## Decisions

1. **Use a shallow canonical output root for new Launcher evaluations.** Store new evaluation work under `runs/evaluations/<evaluation_run_id>/`, not beneath the training run's directory. The evaluation state and `evaluation_lineage.json` retain the exact training `run_id`, model ID/hash, protocol and output path. API routes resolve output from the evaluation state, never from an assumed parent-directory layout. Readers continue to accept existing nested paths. Before launch, estimate the longest expected EvalScope cache path; keep an explicit safety margin below the Windows legacy path boundary and return an actionable preflight blocker if the workspace root alone makes that impossible. Shortening the checkout path is documented as a workaround, not a requirement.

   Alternative considered: turn on Windows long-path support or ask users to rename their checkout. Those approaches rely on machine policy or user setup and do not fix long paths for other file operations, so they are not the product-level solution.

2. **Build a bounded read-only run adapter over existing records.** Normalize supported CLI `run.json` and evaluation run records plus Launcher state into one response shape. Discover only configured workspace roots, deduplicate by unique `run_id`, preserve source references, and mark ambiguous/legacy records rather than inferring from directory names. List APIs return summaries; detail APIs read only allowlisted config, metric, log, artifact and lineage files with size/path/symlink checks. Avoid scanning model weight contents or recursively parsing unbounded artifacts.

   Alternative considered: create a generated SQLite registry. It would introduce synchronization and migration questions before the current local artifacts justify persistence; live filesystem discovery is sufficient for this single-user workspace slice.

3. **Make Runs a dedicated navigation route with URL-addressed state.** Add list/detail pages using the existing React shell and local API. Encode stage/status filters and selected run ID in query parameters so refresh/back work. Provide summary rows/cards, then a detail view with overview, config, logs, artifacts, evaluations and lineage sections; missing and unresolved evidence remains explicit. Link Launcher completion to the matching run detail instead of making Launcher the only run surface.

4. **Separate training and evaluation executions while connecting them.** Each evaluation remains its own execution with its own `run_id`/`eval_id`; the training detail links evaluations by the explicit parent training ID. No directory nesting or label is used as a lineage key. Failed evaluations remain visible with logs and no score.

## Risks / Trade-offs

- [EvalScope may create longer retry filenames than currently observed] → estimate worst-case known suffixes in preflight and test at the existing long checkout path; keep an actionable message for roots that still exceed the safe budget.
- [Historical records vary and can contain duplicate or incomplete metadata] → normalize conservatively, retain source links, show ambiguity/incompleteness, and never silently choose a conflicting record.
- [Filesystem discovery can grow with artifact count] → cap visited records and individual file sizes, cache only within a request or short refresh interval, and expose a partial-scan warning if limits are reached.
- [Moving evaluation outputs changes where new users find files] → expose the canonical relative path in Run Detail and keep every evaluation `run.json`/lineage self-describing; do not move old files.

## Migration Plan

1. Implement and test the shallow evaluation output path and state-based API resolution while retaining reads for legacy nested evaluations.
2. Add the read-only run discovery API and fixture coverage for current CLI, evaluation, Launcher, legacy, failed and malformed records.
3. Add Runs navigation/list/detail and document the new output location and browsing workflow in Chinese.
4. Validate from the current long checkout path, confirm successful 5+5 benchmark evaluation (GSM8K 5 total; MATH-500 one sample from each Level 1–5, 5 total), check evaluation lineage, and exercise the real Runs page plus 390px/keyboard behavior.
5. Rollback can disable the Runs route and restore the prior evaluator output resolver; existing and newly written artifacts remain intact. Do not rename or move the workspace as part of the code migration.
