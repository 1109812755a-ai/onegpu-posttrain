# Proposal

## Why

The local Launcher can finish training but its follow-up EvalScope run fails on this Windows workspace because the review-cache file path is 285 characters. Shortening the checkout path by 49 characters would reduce this instance to about 236 characters, but users should not have to rename or relocate a project to evaluate it. The UI also has no general Runs workspace for browsing historical executions, despite already emitting run, model, evaluation and lineage artifacts.

## What Changes

- Store new Launcher evaluation artifacts under a shallow, run-ID-keyed workspace path and preserve an explicit link to the producing training run; keep old nested evaluation outputs readable.
- Add a read-only Runs list and Run Detail workspace over existing local run/evaluation artifacts, including status, stage, configuration, logs, metrics, outputs and lineage where available.
- Add safe discovery, bounded reads and clear legacy, missing, partial and failed-evidence states; preserve filters and navigation context.
- Keep the existing explicit opt-in Launcher controls separate from browsing; add no database, scheduler, arbitrary command/path input or remote service.

## Capabilities

### New Capabilities

- `run-workspace`: Browse and inspect locally recorded training, merge and evaluation runs with their available provenance.

### Modified Capabilities

- `web-workspace-ux`: Activate Runs navigation and require usable, accessible list/detail experiences across supported viewports.

## Impact

- Evaluation output/state path handling and API routes in `src/onegpu_posttrain/ui/launcher.py` and `server.py`.
- Run artifact discovery and normalized responses for the local UI API.
- React navigation, Runs list/detail pages, styling, frontend/backend tests, and Chinese UI documentation.
- Existing artifacts remain immutable; new shallow evaluation output locations are referenced by their own run state and lineage rather than inferred from directory nesting.
