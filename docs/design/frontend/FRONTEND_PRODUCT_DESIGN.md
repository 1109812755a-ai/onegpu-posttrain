# Frontend Product Design

**Status:** The Data browser, opt-in local policy editor and targeted-data workbench preparation slice are implemented; the full UI remains in progress.  
**Full target:** v0.1.1 local Experiment UI, with read-only artifact browsing and constrained local editors; provider generation/training remain explicit CLI actions.

中文版本：[前端产品设计](FRONTEND_PRODUCT_DESIGN_zh.md)。

This UI is subordinate to the [project product vision](../PRODUCT_VISION.md): its job is to make experiment evidence easier to inspect, not to turn the project into a generic training platform.

## Product intent

Make the evidence already produced by OneGPU-PostTrain easy to inspect and compare without changing how experiments are built or run. The UI should help a single-GPU researcher answer:

- Which runs and model stages exist, and how are they related?
- Did a stage pass, OOM, or fail? What evidence supports that state?
- What changed in task quality, preference behavior, reward, runtime or VRAM?
- Which data gates reduced the usable sample count, and what artifacts can be inspected?

The artifact browser is local and read-only, not a training dashboard that implies live orchestration. A separate policy editor is available only when explicitly enabled and may preview local data or save a new policy/experiment draft. It cannot start training. The UI must distinguish missing metrics from zero, and smoke feasibility evidence from quality claims.

## Users and primary workflows

Primary user: the project author or contributor inspecting local experiments on a workstation.

1. Open Overview and see stage/run status, hardware context and notable warnings.
2. Filter Runs by stage, algorithm, seed and status; open a run.
3. Inspect its config, parent checkpoint, training/evaluation evidence, logs and output files.
4. Compare compatible checkpoints using the same datasets/protocol, while seeing missing or incomparable values explicitly.
5. Inspect data retention/rejection and GRPO/preference diagnostics to identify the next experiment or hardening task.

For the evaluation-driven targeted-data workflow, the UI adds a bounded path from scenario evidence and denominator → provider/prompt/output draft → generation/validation summary → model/evaluation effect summary. The UI exports a non-registered configuration draft and reads only allowlisted summaries. It must not accept secret values, call a provider, start training, or infer model improvement from candidate count/loss/reward.

## Release boundaries

### v0.1.1: local Experiment UI

- Consume existing files under configured `runs/` and `docs/results/` roots.
- Provide Overview, Runs, Run Detail, Stage Comparison, Data and GRPO/Preference Analysis.
- Do not launch/stop processes, edit configs, write into run directories, or claim that polling an artifact means a run is live.
- No database, account system, remote service or scheduler.

### Later releases

- v0.2 may launch a constrained allowlist of existing recipes and display process/log status.
- v0.3 may define OVAT matrices and compare runs/seeds, then export reports.
- v0.4+ may add dataset/model/experiment registry views after provenance schemas stabilize.

See [Frontend Roadmap](FRONTEND_ROADMAP.md) for gates and sequencing.

## Proposed implementation shape

Proposed default, subject to an implementation ADR before dependencies are added:

- Frontend: React + Vite + TypeScript; Ant Design for consistent tables/forms; ECharts for linked metric/time-series charts.
- Local API: FastAPI as a thin read-only adapter over filesystem artifacts, plus separately gated policy preview/draft endpoints.
- Storage: existing JSON, JSONL, Markdown and trainer logs in `runs/` and `docs/results/`; no database for v0.1.1.
- Deployment: local development process, bound to loopback by default.

The backend normalizes different artifact versions to a small UI-facing schema. It should not make training/evaluation semantics or silently rewrite source artifacts.

The v0.1.1 UI should render lineage from typed manifests as a read-only graph in Overview and Run Detail. A standalone registry-style graph browser is not required for the first UI release. The optional policy editor writes only to its dedicated new-draft directory; it does not rewrite source datasets, runs or evaluations.

## Trust, safety and usability principles

- **Task-first, not dashboard-first:** organize around the user's questions (what ran, what changed, why, and where is the evidence), not around a wall of charts.
- **Progressive disclosure:** show status and key takeaways first; reveal protocol, denominator, config, lineage and raw artifact details on demand without losing context.
- **Explain the science in place:** provide concise definitions/tooltips for metrics and stage-specific terms; make direction, units, sample counts and limitations easy to find.
- **Keep navigation coherent:** preserve filters and comparison context when opening a run; provide clear back links and source links from every derived result.
- **Make lineage explorable:** selecting a node should reveal its parents/children and the artifacts behind it; also provide a keyboard-usable list/table alternative to the graph.
- **Design for decisions, not decoration:** comparisons should make baseline, delta, compatibility and missing evidence legible; avoid visual emphasis that implies unsupported winners.
- **Test with real user tasks:** validate that a first-time user can find a run's parent/data/evaluation, understand why a metric is missing, and move from a comparison to its evidence without relying on chat or hidden knowledge.
- **Treat interaction quality as a release gate:** primary tasks must be discoverable; filters, detail, back navigation and source links preserve context; long content is progressively disclosed; loading, empty, error, stale and access-denied states explain what happened and what the user can do next.
- **Accessibility and responsive behavior:** core actions work by keyboard with visible focus/selection; essential information and actions remain usable on narrow screens; charts have table/list alternatives; status never relies on color alone.
- Read-only means no filesystem mutation and no process control in v0.1.1.
- Restrict resolved paths to explicitly configured artifact roots; reject traversal and symlinks that escape those roots.
- Never serve model weights, credentials, environment dumps or arbitrary files simply because they are below a run folder. Expose an allowlist of known artifact types.
- Render Markdown as sanitized text/limited safe Markdown; never execute embedded HTML or scripts.
- Bound JSONL/log reads with pagination/tailing and size limits; show truncation and parse errors.
- Missing, stale, malformed and not-yet-produced data have distinct states. Do not coerce missing metrics to zero.
- Show run ID, stage, seed, parent lineage, data/eval provenance and source file for every displayed result where available.
- Keep comparison honest: show protocol/dataset mismatch and do not calculate deltas across incompatible evaluations.
- Keyboard navigation, readable contrast, responsive layout and empty/error states are part of acceptance.

## Success criteria for the first release

- The six v0.1.1 pages can render representative smoke artifacts without modifying them.
- Every shown run/status/metric can be traced to an artifact or is labeled as derived.
- Missing/legacy fields, malformed files, large logs and absent optional artifacts fail visibly and safely.
- A comparison flags incompatible or incomplete evaluation sets instead of presenting misleading rankings.
- File access remains within configured roots; model checkpoints and secrets are not exposed.
- Startup and shutdown are documented; CLI workflows remain usable with the UI stopped.
- Task-based usability review confirms that key evidence and caveats are discoverable, navigation context is preserved, and charts do not obscure the underlying artifacts.
- A user can complete an unassisted walkthrough: find a dataset's processing steps and quality evidence, recognize missing evidence/warnings, follow it to a related run, then filter, open details and return without losing context.

## Explicit non-goals

No training scheduler, remote multi-user deployment, RBAC, database-backed registry, automatic HPO, distributed monitoring, artifact upload, or replacement for CLI in v0.1.1.
