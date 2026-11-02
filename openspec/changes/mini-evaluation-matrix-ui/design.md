# Design

## Context

See `proposal.md` for motivation and the two delta specs for observable behavior. The workspace currently serves a bundled React page and exposes `/api/data`; its “阶段比较” navigation item is disabled. The matrix is already emitted as `evaluation_matrix.json` and contains model statuses, metrics, requested/observed sample counts, protocol consistency, eval IDs, model IDs and run references. Existing artifact APIs constrain files to workspace roots and avoid exposing arbitrary paths.

## Goals / Non-Goals

**Goals:**
- Present an existing matrix without invoking evaluation, training, or mutation.
- Allow a user to inspect one selected matrix, its comparison coverage and resolvable lineage.
- Make low sample coverage, PPL truncation, unresolved lineage and protocol mismatch apparent.
- Keep the current local-only, no-database deployment and support responsive/keyboard use.

**Non-Goals:**
- Matrix creation/editing, experiment launch/scheduling, benchmark execution or algorithm ranking.
- Changing metrics, computing statistical significance, assigning universal “better/worse” colors, or claiming causal data effects.
- Replacing Data, Runs, policy or targeted-data workflows.

## Decisions

1. **Read existing matrix artifacts through a narrow API.** Add a read-only index/detail API for `evaluation_matrix.json` under `runs/mini/*/evaluation-matrix/`; the server discovers paths itself, rejects symlinks/out-of-workspace paths, caps file size and result count, and never accepts a filesystem path from the client. Keep the current `/api/data` API unchanged. An absent matrix returns an explicit empty state; invalid JSON/schema produces a visible issue without exposing raw file contents.
   - Alternative: let the browser fetch local files or accept a path. Rejected because browser access is unavailable and arbitrary paths would bypass existing artifact containment rules.

2. **Use the existing “阶段比较” navigation as a dedicated read-only page.** Desktop uses a compact matrix table with M2 pinned as baseline; narrow viewports render per-model cards or an internally scrollable comparison region without page-level overflow. A matrix selector appears only when multiple artifacts exist.
   - Alternative: add comparison controls to the Data detail tab. Rejected because a model comparison is not specific to one input dataset, and it would bury the new multi-model view inside the existing data workflow.

3. **Render evidence, not interpretation.** Show status, raw metric and unit, `observed/requested` denominator, and numeric delta versus the declared baseline. Show comparability/protocol/config identity and coverage warnings above results. Do not infer metric direction or style a value as an improvement; retain matrix limitations such as tiny evaluation budgets and truncated PPL samples.
   - Alternative: plot model scores without denominators or use red/green “better” signals. Rejected because tiny samples and metrics with different directionality make that easy to misread.

4. **Resolve lineage by IDs only.** Use matrix model/evaluation IDs to attach available training model artifacts, parent IDs, run IDs and dataset IDs. Keep the matrix readable if a lineage reference cannot be resolved, and display that fact instead of guessing from labels or paths. Link to the existing detail/artifact view where a safe route exists.
   - Alternative: infer M3 parentage from names such as `M3-DPO`. Rejected because stable IDs and digests, not display names, are the project's lineage contract.

5. **Keep the front-end dependency set unchanged.** Use existing React/TypeScript and design primitives; implement page states for loading, no matrix, API error/retry, comparable, non-comparable and partial/failed results. Preserve URL navigation state so refresh/back works consistently with the current workspace.

## Risks / Trade-offs

- [Existing matrix producer formats may evolve] → Validate only required schema fields, return an explicit unsupported/malformed status, and add fixture tests for current and missing-field cases.
- [Scanning many runs can affect page load] → Restrict discovery to the Mini evaluation-matrix directory pattern, bound file count/bytes and avoid hashing model weights during page requests.
- [Lineage can be incomplete even when evaluation succeeded] → Keep score status separate from lineage status and show unresolved references without dropping valid metrics.
- [A dense comparison table is difficult on mobile] → Switch to per-model cards or a bounded scroll region at narrow widths; test at 390 CSS pixels.

## Migration Plan

No migration is required. Build the new page and API additively, keep all existing routes and artifacts compatible, verify empty-workspace and real-Mini behavior, then rebuild `web/dist`. Rollback is removing the page/API route while leaving matrix artifacts untouched.
