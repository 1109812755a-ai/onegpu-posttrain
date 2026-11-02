# Frontend Roadmap

**Status:** roadmap. The Data browser, policy-comparison editor and an early targeted-data workbench slice are implemented. Artifact browsing remains read-only; the workbench edits/exports a non-registered draft and reads only allowlisted summaries, with no provider or training execution entry point. The complete v0.1.1 UI and remaining pages stay planned as below. UX is a release gate at each stage.

Version labels in this roadmap refer to product releases. Product release v0.1.1 is distinct from frozen design revision v0.1.1, which targets product release v0.1.

中文版本：[前端路线图](FRONTEND_ROADMAP_zh.md)。

## Principles

- Consume artifacts first; do not make training depend on a web service.
- Preserve CLI as the stable execution interface, even after adding UI capabilities.
- Add each layer only when artifact schemas and provenance are mature enough to support it.
- No database or remote service until a concrete query/concurrency requirement justifies one.
- A UI must preserve scientific caveats: protocol compatibility, denominators, seeds and missing metrics.
- Data quality evidence is a first-class view: distinguish stage-specific measured evidence from estimates and model/recipe-dependent calibration; do not invent a universal score.
- Center interactions on user tasks: find a dataset/run, understand status and missing evidence, follow lineage to sources, and preserve filters/back-navigation context.
- Keyboard operation, visible focus/selection, readable contrast, responsive layouts, and loading/empty/error/stale states are acceptance criteria; color is never the only status signal.
- Charts and flow views need accessible table/list alternatives. Reveal advanced detail on demand without hiding denominators, units or limitations.

## Milestones

| Release | Scope | Exit gate |
|---|---|---|
| v0.1 | CLI-only smoke and Mini research baseline | Existing v0.1 release gates; UI remains an explicit non-goal. |
| v0.1.1 | Local Experiment UI: Overview, Runs, Run Detail, Stage Comparison, Data (lineage/quality/distribution/leakage), policy-comparison editor, targeted-data diagnosis/plan draft/generation/effect summaries, GRPO/Preference Analysis, hardware/resources and artifact viewer. Artifact browsing is read-only; policy preview and draft save are an explicitly enabled, constrained exception; targeted generation stays CLI-only. | Real artifacts render accurately; quality denominators, uncertainty, missing-data and compatibility warnings work; filesystem access is constrained; no provider/training process control; startup and safety documented; task-based usability walkthrough plus keyboard/narrow-screen checks pass. |
| v0.2 (first vertical slice) | Opt-in local Mini SFT launcher for the single fixed `recipes/mini/23-sft-1024-fallback.yaml` recipe (307 exploratory examples, 1,024-token limit); no parameter/path overrides or evaluation launch. | Loopback-only; read-only preflight + explicit config confirmation; one supervised process; truthful status/logs/lineage; browser close is harmless and service shutdown records interruption; fake-CLI end-to-end tests pass. Broader recipe selection/overrides require a separate design. |
| v0.3 | Experiment Studio: define OVAT runs, multi-run comparison, curves/tables/deltas, seed comparisons and report export | Experiment definitions are versioned/reproducible; comparison checks compatibility; exports include provenance and limitations. |
| v0.4+ | Dataset, model and experiment registry UI | Stable IDs/provenance and lifecycle semantics exist; registry is justified by real usage; persistence choice is separately reviewed. |
| v1.0 | Platform workspace: task queue/scheduling, resource and lifecycle views, dataset/model/run/evaluation workflows, deployment and operations entry points | Scheduling, interruption/recovery and resource feedback are validated against real workloads; security/permissions, deployment/upgrades and data export are verified; usability, accessibility and responsive task walkthroughs pass. |

The UI roadmap is additive to model/data/research roadmap work: the v0.2 first slice validates browser-to-SFT on the existing 0.6B Mini baseline; later recipe/parameter expansion can be coordinated with selected 1B/3B breadth, v0.3 combines Studio with data-centric experiments and may add a bounded local serial queue when OVAT use requires it, and v0.4+ builds registry views alongside alignment/RL breadth. v1.0 integrates validated scheduling, resource management and deployment operations into the platform workflow.

## v0.1.1 read-only architecture proposal

```text
React + Vite + TypeScript
          │ local artifact API + opt-in policy editor API
          ▼
FastAPI artifact adapter
          │ allowlisted reads
          ├── runs/
          └── docs/results/
```

Proposed UI libraries: Ant Design and ECharts. Storage remains the filesystem (`JSON`, `JSONL`, Markdown and logs); no DB. These are planning defaults, not yet frozen dependencies. Before implementation, add an ADR for the final stack and pin versions.

## Sequencing and gates

1. During dev7, implement only the explicitly reprioritized Data page; keep the rest of the UI behind the v0.1.1 gates and do not treat the preview as a complete UI release.
2. Stabilize a minimal artifact index/normalized read schema from actual run artifacts. Add schema-version and optional-field handling without rewriting historical files.
3. Keep artifact API read-only behind configured roots. Constrain the separately opt-in policy API to discovered source IDs, bounded previews and new draft directories. Add path traversal, malformed/large artifact and secret/model-weight exposure tests.
4. Validate against both complete and intentionally partial smoke runs, including current high clipping and low preference counts. Run task-based UX checks: can a new user find data sources/steps/quality reports, understand missing evidence and warnings, and return from filters without losing context?
5. Only after the read-only release is useful should a separate launcher design authorize process creation in v0.2.

## Explicit v0.1.1 exclusions

No arbitrary command execution, training process control, recipe mutation, model-weight serving, remote bind by default, user/role management, database, dataset/model registry writes, or automatic experiment scheduling. The constrained policy editor only writes to a dedicated new draft directory and uses the shared CLI data-policy contract.
