# Frontend Page Specification

**Scope:** The UI is organized into Data, Training & Evaluation, Experiment Runs, and Analysis & Improvement. Requirements prioritize reproducible object identities, truthful states, and clear next actions. A stage that is not implemented must be marked as planned or blocked; it must never appear launchable.

The data-processing trace and data-to-model effects follow the [detailed processing/effect design](../DATA_PROCESSING_EFFECT_DESIGN.md).

## 1. Data

### Dataset catalog and sources

- Show registered dataset name, stage, source, count/unit, quality state, license state, and available actions; filter by stage and source.
- Source and training stage are independent dimensions. Unknown source/license details may be registered, but schema validation must not upgrade their state.
- Local imports require an explicit stage and validation against that stage's canonical schema. Model generation exposes only implemented stages and modes.
- Preview bounded samples from local imports, open-source builds, and model generation when evidence allows it. Verify record identity and content digest; return only bounded fields and sample counts.

### Dataset detail

Every entry point for the same `dataset_id` opens the same immutable version detail, containing at least:

- **Version and source:** stage, source/revision, license declaration, count/unit, and content digest; raw paths and digests are collapsed by default.
- **Samples:** bounded preview. A successful preview proves only that bytes match the registered digest; it does not prove semantic quality or license compliance.
- **Quality:** metrics, units, numerators/denominators, states, limitations, and report references by protocol. Missing values are never shown as zero, and unsupported composite scores are not computed.
- **Processing lineage:** ordered steps from source to final dataset, rule versions, input/output counts, rejection reasons, and report references. Missing evidence is reported rather than inferred.
- **Model/Run lineage:** explicitly registered parent datasets, consuming Runs, produced models, and evaluations. Relations are shown only when supported by real lineage references.

Sample, quality, and lineage detail may load on demand. A failure in one detail endpoint stays within that detail and offers retry; it must not hide other datasets or Runs.

### Processing and rules

- Processing & Quality shows existing build flows, quality, distributions, contamination, and overlap reports. Counts identify whether they mean raw rows, documents, examples, pairs, prompts, or tokens.
- Rule editing and same-source filtering previews belong in Data. Policy and experiment identities remain stable so executed results can link back from Analysis.
- Model/recipe-dependent calibration is distinct from dataset-only quality reports.

## 2. Training & Evaluation

- Organize the training entry by CPT, SFT, DPO, SimPO, and GRPO profiles that are implemented and approved. Each stage selects compatible registered data, parent-model rules, and a fixed profile.
- Preflight displays dataset ID/digest, source/license and quality evidence, parent model, profile, resource hints, and blockers. Starting requires explicit confirmation.
- Do not accept arbitrary shell commands, model/data paths, or unapproved hyperparameter overrides. Only one local GPU task may be active.
- Show stage and parent-model relationships. If a parent is missing, explain the upstream work the user needs to complete; never launch it implicitly.
- After a successful training Run and verified model identity, offer a separate preflight and explicit launch for supported evaluation. Evaluation does not overwrite training state or reuse training data as if it were a generalization result.
- Preserve actual states for unsupported, unmeasured, blocked, partial, and failed operations.

## 3. Experiment Runs

- Filter Runs by `run_id`, stage, algorithm, status, time, and available resource data. Unknown time/VRAM remains unknown, not zero.
- Opening the same `run_id` from Launcher, dataset detail, or Analysis must show the same Run detail.
- Run detail shows actual status, live/final logs, config summary, parent model, consumed dataset ID/hash, model outputs, evaluations, and lineage.
- Evaluation is a separate Run; evaluation failure does not alter successful training state. A failed, OOM, interrupted, or merge-failed training Run must not fabricate a model or offer evaluation that depends on a missing model.
- Advanced config, paths, hashes, and raw JSON are available on demand. Logs are bounded and scroll independently.

## 4. Analysis & Improvement

- **Stage comparison:** Calculate deltas only when benchmark version, sample scope, decoding/parsing protocol, and denominator match. Otherwise show the compatibility difference and do not declare a default winner.
- **Policy effects:** Show real executed baseline/control rules, data-quality changes, and model-evaluation deltas. Mark model effects unmeasured when no controlled evaluation exists.
- **Weak scenarios:** Explain metric meaning and denominator, and identify the source model/Run/evaluation and scenario. Sending a target for data generation transfers its context to Data.
- Targeted data must keep training seeds separate from final holdout. Holdout examples must not be sent to a generation provider or used for training.

## Shared interaction requirements

- Keep four stable primary areas. Place functionality in its responsible area instead of making unrelated page copies of one object.
- Render loading, empty, error, partial, and unsupported states separately with local recovery actions.
- Give metrics a readable meaning, unit, numerator/denominator, protocol, and evidence source. Never rely on color alone for state.
- Keep form drafts, preflight tokens, and started task identities separate. Page/mode changes and reloads must not discard active tasks or retry them automatically.
- Cross-area actions transfer `dataset_id`, `run_id`, `model_id`, `eval_id`, or an explicit scenario identity and preserve the return path where possible; users should not copy internal IDs.
- Support desktop, 390 CSS pixel narrow screens, and keyboard navigation. Use a single-column layout or horizontally scrollable secondary navigation on narrow screens; long logs/tables scroll within their own areas.
- Make the next valid primary action clear for the current state. Show raw technical information on demand.

## Currently implemented scope

- The dataset workbench supports source capability listing, local canonical JSONL registration, bounded sample preview across supported source types, quality summary, and links to processing reports/related Runs.
- Rule comparison, targeted data workbench, and ordinary SFT generation continue to use their existing views/APIs. Shared generation UI and cross-page task recovery are later tasks.
- The Training Launcher currently supports only the fixed Mini SFT profile and its applicable exploratory evaluation. General CPT, DPO, SimPO, and GRPO launch flows are later stage-workbench work.
