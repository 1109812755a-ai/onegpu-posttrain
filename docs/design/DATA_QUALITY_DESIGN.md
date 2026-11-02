# Data Quality Design

**Version:** Data Quality v1 (design proposal)  
**Status:** `stage-data-quality-v1` implements the stage metric catalog, evidence coverage, and offline reassessment; quality thresholds and semantic approval remain unfrozen  
**Scope:** stage-specific quality evidence for data builds and pre-run suitability checks

中文版本：[数据质量设计](DATA_QUALITY_DESIGN_zh.md)。

For the detailed source-processing trace, artifact schemas, controlled model-effect studies and UI workflow, see [Data Processing, Quality Change, and Model Effect Design](DATA_PROCESSING_EFFECT_DESIGN.md).

## Purpose and place in the product

Data quality is a first-class project line alongside lineage, training, evaluation and controlled experiments. Counts alone cannot establish that an experiment's inputs are valid, representative, uncontaminated or suitable for the intended objective.

```text
Data
  ├── Lineage: where it came from and how it changed
  ├── Quality: whether records satisfy stage-specific expectations
  ├── Distribution: what examples and lengths it contains
  └── Leakage / contamination: what it may overlap with
       ↓ validated inputs
Training → Evaluation → Controlled Experiments
```

This is additive to the frozen v0.1.1 data contract. It does not silently change existing eligibility rules or rewrite historical results. Any change to a frozen gate requires a versioned design revision and rebuilt artifacts.

## Principles

1. **Quality is stage-specific.** Do not apply one universal score to CPT, SFT, preference, GRPO and evaluation data.
2. **Separate evidence from judgment.** Store raw counts/rates and protocol; show a gate result only when a versioned rule supports it.
3. **No opaque composite score.** Do not collapse correctness, coverage, leakage and usability into one number.
4. **Make uncertainty visible.** Each metric is `measured`, `estimated`, `not_measured`, or `not_applicable`; preserve numerator, denominator, method/version, source and limitations.
5. **Do not claim semantic certainty from a heuristic.** For example, model-based relevance is an estimate; verifier-backed correctness depends on verifier coverage and reliability.
6. **Keep v0.1 thresholds conservative.** Initially emit reports and warnings. Calibrate non-universal thresholds from Mini evidence before promoting them to blocking gates. Safety invariants such as split separation remain enforceable independently.
7. **Keep all checks reproducible and inspectable.** Quality protocol/config and checker/verifier versions are part of artifact provenance.

### Current implementation boundary (v1)

New `QualityReport` schema `1.1` records the `quality_protocol`, checker name/version, per-metric status, and `evidence_coverage`. Historical `1.0` reports remain readable with coverage interpreted as `legacy_unknown`. `complete` means every applicable catalog metric has an explicit status; it does not mean quality passed. Missing verifier, parser, or scoring evidence remains `not_measured`. Mini reassessment uses `assess-mini-data-quality` to write an additive digest-bound bundle without rewriting datasets, manifests, or historical reports. See the [Mini readiness CLI guide](../implementation/data/MINI_READINESS_CLI_zh.md) for commands and constraints.

## Three quality layers

| Layer | Questions | Typical evidence |
|---|---|---|
| Structural | Is the record parseable and within declared schema/length constraints? | schema validity, null/empty fields, parse rate, exact duplicates, length distribution |
| Semantic | Does the content make sense for its task and label? | domain relevance estimate, answer/verifier correctness, reasoning completeness, pair ordering |
| Experimental suitability | Can it support the planned comparison without invalidating interpretation? | split/benchmark overlap, source/difficulty coverage, verifiability, and explicitly model/recipe-dependent calibration signals |

These layers are reported separately. A dataset can pass structural checks while having weak semantic evidence or unsuitable experimental coverage.

## Stage-specific assessment

| Stage | Core questions | Suggested metrics (only emit when supported) |
|---|---|---|
| CPT | Is the corpus usable text, relevant to the declared domain, sufficiently diverse, and separated from evaluation material? | document and token counts; schema/encoding validity; exact and near-duplicate rates with method; language/source/task distribution; document- and token-weighted length distributions; domain-relevance estimate with method and audit coverage; benchmark contamination/overlap with detection limits |
| SFT | Are prompts and targets complete, valid, correct enough for the declared task, and trainable within the recipe budget? | schema validity; empty/malformed prompt/answer rate; verifier-backed correctness **within verifier coverage**; final-answer parseability; format validity; tokenizer/recipe-specific prompt/response length and truncation risk; duplicate-problem rate; source/difficulty/task distribution; reasoning completeness only as an explicitly estimated/audited signal |
| DPO / SimPO | Is each pair traceable, unambiguous and consistent with the declared preference-label semantics, and are splits separated? | same-prompt pairing; malformed/tied/ambiguous pair rate; label consistency; verifier-backed chosen/rejected task correctness **only where labels are defined by correctness**, with verifier coverage; length and chosen-vs-rejected length-bias diagnostics; duplicate/same-problem consistency; train/eval overlap. Pair yield is a **build-efficiency metric**, not a quality score; model margin/hard-negative scores are model-dependent diagnostics |
| GRPO | Are prompts/references structurally usable and supported by the verifier? Separately, does a specified model/recipe produce a usable rollout signal? | **Static data:** reference parseability, verifier support/coverage, prompt length, task/difficulty mix, duplicate rate, expected answer type. **Calibration:** baseline pass rate, reward non-zero rate/group variance, completion length and truncation/clipping under a named model, reward/verifier, seed and generation recipe |
| Evaluation | Does the set measure the intended capability with stable scoring and sufficiently independent, relevant examples? | source/revision and protocol; item/task/difficulty coverage and sample counts; answer/parser/verifier coverage; exact/near-duplicate and ancestor/problem-ID overlap with training; contamination method and limits; denominator/exclusions; parse/scoring failure rate. Model accuracy is an evaluation outcome, not a dataset-quality metric |

The table is a metric menu, not a requirement to invent or compute every metric. Reports must explain unavailable metrics rather than substitute zero.

### Representativeness and metric classes

Representativeness is always relative to the declared use: a corpus is representative of its intended training/evaluation population only to the extent that its task, source, difficulty, language and length coverage match that target. Report the intended population and observed coverage; do not claim that source balance or a single relevance score proves representativeness. Data-centric evaluation research likewise emphasizes breadth, difficulty and faithfulness of the underlying problems, rather than cleanliness alone ([DataPerf](https://proceedings.neurips.cc/paper_files/paper/2023/hash/112db88215e25b3ae2750e9eefcded94-Abstract-Datasets_and_Benchmarks.html)).

Every metric must be labeled as one of:

- **Dataset quality:** validity, correctness evidence, duplication, coverage, leakage and provenance of the artifact.
- **Build diagnostic:** requested/accepted counts, rejection reasons and pair yield. These explain builder efficiency and target feasibility; they do not establish that accepted examples are good or bad.
- **Model/recipe suitability:** pass rate, reward signal and completion truncation observed in a calibration rollout. These are conditional on model, evaluator/reward and generation settings, not intrinsic dataset properties.

For MVP prioritization, structural validity, IDs/splits, source/version provenance, supported-verifier coverage, answer parseability and overlap checks are core. Length/distribution, duplicate and rejection/yield summaries are reported diagnostics. Semantic estimates, near-duplicate detection, manual audits and model-margin/hard-negative analysis are optional until their method and cost are justified. Do not make an unvalidated estimated metric a blocking gate.

For each ratio, retain numerator, denominator, unit and exclusions; for sampled or estimated measures, retain sampling method, sample size and uncertainty/audit coverage where feasible. Separate **check result** (`PASS`/`FAIL`/`NOT_EVALUATED`) from **consequence** (`error`/block vs. `warning`/continue). A failed advisory check may warn without blocking a run.

## Preference yield and GRPO calibration

Preference yield is an important capacity-planning/build-efficiency signal, not a quality score. Define the scan unit and denominator (for example, valid pairs per scanned unique problem), preserve rejection counters, and report actual target attainment separately from pair validity and label correctness. This makes it possible to estimate how much source material a Mini target may require without relaxing correctness criteria to fill a quota. “Chosen correct / rejected incorrect” is appropriate only when the dataset's declared preference semantics are correctness ordering (as in the project's verifiable math pairs); it is not a universal DPO/SimPO definition.

GRPO static prompt checks cannot predict generation behavior completely. A separate **pre-training calibration report** may run a bounded baseline rollout from the intended parent model and record prompt subset/seed, generation settings, verifier/reward versions, completion-length percentiles, truncation rate, reward non-zero rate and group reward variance. These are model-and-recipe-dependent suitability measurements, not immutable properties of the dataset. v0.1 reports them as warnings; a hard gate and threshold require evidence, a declared calibration protocol, and review after Mini.

## Processing trace and before/after quality

Record the actual source-to-stage processing order, including normalization, deduplication, contamination checks, problem-level split, verification, length gates, pair construction and token-budget selection where applicable. Extend the existing versioned data-build report with an ordered `processing_steps` list; keep the final dataset's `quality_report.json` linked from its manifest. Intermediate datasets need their own IDs/manifests only when materialized and reused, not for every in-memory operation.

The [implemented smoke build order](../implementation/data/SMOKE_DATA_BUILD.md) is the starting reference: OpenR1 normalizes/deduplicates, checks contamination, verifies the math core, splits by problem, then applies stage-specific length/transform rules; FineMath assigns a document-level split before secondary contamination checking and chunking. A proposed diagram must not silently reorder these gates. If the order or eligibility semantics change, version the processing protocol and rebuild affected artifacts.

Each step records a stable step ID and order, input/output artifact or logical-pool references, transform/checker name and version, config digest, input/output counts **with units**, first-rejection reason counts, and before/after metric references when measured. The trace also records skipped/unknown steps and why. When record-level ancestry is available, keep IDs for accepted examples and rejected examples in a separate optional, license-aware sidecar; aggregate reports do not embed source text.

```json
{
  "schema_version": "1.0",
  "build_id": "build-openr1-mini-v1",
  "source_dataset_id": "ds-openr1-source-v1",
  "processing_steps": [
    {
      "step_id": "verify-math-v1",
      "order": 4,
      "input_ref": "pool-after-decontamination",
      "output_ref": "pool-verified",
      "transform": {"name": "verify_math", "version": "1", "config_sha256": "<digest-or-null>"},
      "input_count": {"value": 1000, "unit": "examples"},
      "output_count": {"value": 820, "unit": "examples"},
      "first_rejection_reasons": {"unsupported_answer": 80, "incorrect_answer": 100},
      "quality_before_ref": "quality/decontaminated.json",
      "quality_after_ref": "quality/verified.json"
    }
  ]
}
```

The example values are illustrative. A retention percentage is valid only when input/output have the same unit and the output is a subset of the input; pairing, chunking and tokenization may change units. Rejection counts refer to the first failing rule under a declared rule order; they do not estimate each rule's independent contribution. Compare a quality metric across a step only with compatible definition, checker/version, denominator and sampling protocol. Report coverage loss alongside quality gains, including source/task/difficulty mix and any reduction in verifier-supported or rare examples. Raw-source quality can be assessed on a bounded sample, with sample size and method shown; unmeasured properties stay `not_measured`.

### Connecting processing to model outcomes

A before/after quality change describes the data; it does not establish an effect on the trained model. To test that effect, register a data-selection experiment with the source pool, the baseline and cleaned dataset IDs/digests, the single changed processing rule or policy, parent-model hash, recipe, seeds, budget, evaluation datasets/protocols and downstream model/eval IDs. The baseline must satisfy minimum schema, licensing and contamination safeguards; "raw" does not imply training on malformed or leaking records.

Distinguish two experiment questions:

| Comparison | Question answered | Required interpretation |
|---|---|---|
| **Pipeline effect** | What happened when the complete cleaning policy was applied in practice? | Report the actual pre/post counts, token volume, distributions, quality evidence, resource cost and model evaluations. Dataset size and composition may both change, so the model delta is an end-to-end pipeline result. |
| **Controlled quality ablation** | What changed when one declared selection/processing policy changed at a matched training budget? | Hold the eligible source pool and independent evaluation set fixed; match examples or tokens, optimizer steps, parent, recipe and seeds where feasible. Report remaining distribution differences and uncertainty across seeds. This supports a narrower interpretation of the chosen policy's effect. |

Both comparisons report data quality/coverage differences, model-metric deltas and compute cost. Training loss or reward improvement alone cannot prove a task-capability gain. A single run is descriptive; stronger effect claims require repeated seeds or an explicit uncertainty/replication limit. Do not assume a verifier-filtered dataset's conditional correctness rate proves the original labels were all wrong, or that a quality score predicts benchmark gain.

Candidate studies include deduplication or verifier selection for SFT, preference negative selection (random wrong vs. length-matched vs. hard negative) under the same problem IDs/M2/algorithm and pair budget, and GRPO prompt selection based on bounded rollout calibration. For GRPO, keep the generation recipe fixed when testing prompt selection; changing `max_completion_length` is a separate recipe experiment. Compare downstream task metrics alongside format/parser behavior, preference accuracy/margin or GRPO reward variance/truncation as applicable, with evaluation denominators and protocols visible.

One small, optional matched-size study may follow Mini once the baseline and required OVAT contracts are stable. It is not a v0.1 release gate or a fifth required OVAT; reusable data-quality studies and their comparison UI belong to the later data-centric/Experiment Studio phase. The v0.1.1 read-only UI may display available processing steps and linked model comparisons, clearly marking absent controlled evidence as `not evaluated`.

## Quality report contract

Each built dataset should link to a `quality_report.json` (or an equivalent versioned report) beside its manifest. The report is an assessment of a particular dataset artifact under a named protocol; changing data or checker protocol produces a new report identity/version.

```json
{
  "schema_version": "1.0",
  "dataset_id": "ds-openr1-pref-mini-v1",
  "quality_report_id": "dq-openr1-pref-mini-v1",
  "quality_protocol": "preference-quality-v1",
  "assessment_scope": "pre_training_data",
  "created_at": "<RFC3339 timestamp>",
  "checker": {"name": "preference_validator", "version": "1"},
  "metrics": {
    "scanned_problems": {"value": 5000, "unit": "unique_problems", "status": "measured", "class": "build_diagnostic"},
    "valid_pairs": {"value": 2032, "unit": "preference_pairs", "status": "measured", "class": "build_diagnostic"},
    "pair_yield": {"value": 0.4064, "numerator": 2032, "denominator": 5000, "unit": "ratio", "status": "measured", "class": "build_diagnostic"},
    "chosen_correct_rate": {"value": 1.0, "numerator": 2032, "denominator": 2032, "status": "measured", "method": "math_verify@<version>", "class": "dataset_quality"},
    "train_eval_problem_overlap": {"value": 0, "unit": "unique_problems", "status": "measured", "class": "dataset_quality"}
  },
  "gate": {"summary_status": "WARNING", "policy_id": "preference-gates-v1", "checks": [{"check_id": "chosen_correctness", "result": "PASS", "severity": "error"}, {"check_id": "pair_yield_target", "result": "FAIL", "severity": "warning", "reason": "Thresholds are provisional"}]},
  "limitations": ["Verifier coverage does not establish correctness for unsupported answer forms"]
}
```

Required report behavior:

- Link the dataset ID/content digest and the manifest; the manifest links back to the report when available.
- Preserve counts, units, numerator/denominator, method, checker version, scope and limitations.
- Each metric has a class (`dataset_quality`, `build_diagnostic`, or `model_recipe_suitability`) so operational yield is not mistaken for quality.
- Gate check result (`PASS`/`FAIL`/`NOT_EVALUATED`) and consequence (`error`/block or `warning`/continue) are separate. A summary status may be `PASS`, `WARNING`, `FAIL`, or `NOT_EVALUATED`; without a reviewed policy, do not imply a pass.
- Distinguish immutable build-time quality from model/recipe-dependent calibration reports.
- Keep sample-level diagnostics optional and privacy/license aware; do not copy restricted source text into public reports.

## Open-source framework strategy

The project owns the canonical quality protocol, stage semantics, report schema, and reproducibility guarantees. General-purpose frameworks may be used behind optional adapters, but no external framework is required for the core project to run. Reuse the project's canonical record schemas for JSONL validation; add a tabular validation dependency only if a concrete DataFrame workflow justifies it.

| Candidate | Potential role | Adoption stance |
|---|---|---|
| [DataTrove](https://github.com/huggingface/datatrove) | CPT corpus filtering, deduplication and scalable text processing | Run a bounded PoC against the current CPT inputs. Keep optional; compare quality evidence, memory/runtime, installation cost, and preservation of sample IDs and lineage before adopting an adapter. |
| [Data-Juicer](https://github.com/datajuicer/data-juicer) | SFT-oriented filtering and diagnostic/scoring operators | Run a targeted PoC only for signals absent from built-in checks. Treat model/LLM scores as `estimated` diagnostics, not correctness labels or automatic rejection gates. |
| [Pandera](https://pandera.readthedocs.io/en/stable/dataframe_schemas.html) | Schema and constraint checks on tabular/DataFrame representations | Not a core dependency for canonical JSONL records. Reconsider if DataFrame-based analysis becomes a maintained project interface. |
| [Argilla](https://docs.argilla.io/latest/) | Human sampling, review and annotation workflows | Optional external review tool for ambiguous or subjective examples; the core report must remain usable without a hosted service. |
| [Cleanlab](https://docs.cleanlab.ai/stable/tutorials/datalab/text.html) | Model-assisted label issue, outlier and near-duplicate audits | Optional audit adapter; its findings are model/data-dependent signals and should retain method, model and coverage metadata. |
| [NeMo Curator](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication) | Advanced corpus-scale filtering and deduplication | Defer to a later, explicitly optional integration if corpus scale or required operators exceed the lightweight pipeline. |
| [Distilabel](https://github.com/lightonai/distilabel) | Synthetic-data and AI-feedback pipelines | Not a general-purpose quality checker and not a core dependency. Consider only when synthetic-data generation/AI-feedback workflows enter project scope; verify maintenance and compatibility at that time. |

### Framework PoC acceptance criteria

PoCs are non-blocking experiments, not production integrations. Use representative, license-compatible samples from the actual project formats and compare each candidate with the built-in baseline on: (1) a clearly identified missing capability; (2) signal usefulness and known limitations; (3) memory, runtime and installation footprint on the supported single-GPU/development environment; (4) deterministic reruns and version capture; (5) stable record IDs, source references and manifest/lineage preservation; and (6) export into the versioned `quality_report.json` without losing metric class or uncertainty status. Promote a tool to an optional adapter only when it adds material value; do not make the core install depend on it.

## Gate policy and versioning

| Stage | v0.1 behavior | Potential invariant / future gate candidate |
|---|---|---|
| CPT | Report structural checks, dedup, length and contamination evidence; warn on unknowns | Valid encoding/schema; explicit source/license; no known prohibited contamination |
| SFT | Report correctness coverage, parse/format, lengths and distributions | No split overlap; declared schema and minimum verifier coverage; reviewed task-specific correctness/length thresholds |
| DPO / SimPO | Report pair eligibility, yield, correctness evidence, length and overlap | Chosen/rejected ordering verified under supported verifier; no train/eval overlap; reviewed length-ratio policy |
| GRPO | Report reference/verifier coverage; calibration warning for clipping/reward signal | Supported verifiable references; no split overlap; reviewed calibration bounds tied to a recipe/model profile |
| Evaluation | Report protocol, coverage, parser failures and overlap | Fixed/versioned evaluation protocol; no known train overlap; scoring coverage meets reviewed protocol requirements |

Do not freeze illustrative values such as “100% correctness” or a universal truncation/yield cutoff without verifier-coverage semantics and observed distributions. Mini evidence informs proposed thresholds; any adopted threshold is versioned and its effect on accepted/rejected counts is reported. A failed gate blocks only the affected formal run path, with an explicit override policy if one is later designed; it must never silently discard the warning.

## UI and experiment use

The read-only Data experience should provide Dataset Overview, Lineage, Quality, Distribution and Leakage sections (these can be tabs within the Data page, not five top-level pages). Show the gate state, metric definition, numerator/denominator, status/confidence, protocol, source artifact and limitations. Default to summaries; sample inspection is opt-in and access-controlled. The Overview may surface actionable quality warnings such as low preference yield or high GRPO calibration clipping, linking to evidence.

Data-quality ablations belong after the core Mini and required OVAT contracts are stable. Examples include high-quality vs. random matched-size SFT data, hard vs. random negatives, or calibrated vs. unfiltered GRPO prompts. Such studies must hold data volume and other controls constant where possible, version selection rules, use the same parent/evaluation protocol and report the quality differences themselves—not merely label one set “high quality.”

## Adoption sequence

1. **Design now:** define stage-specific metrics, status semantics and report linkage; keep the frozen v0.1.1 contract intact.
2. **dev7 / Mini preparation:** expose existing counts, rejection reasons, pair yield, split/overlap, length and clipping evidence; mark unsupported metrics unavailable. Run bounded DataTrove (CPT) and Data-Juicer (SFT) PoCs only where they address a documented gap; do not add them as core dependencies.
3. **Mini:** gather real distributions and checker coverage; review candidate thresholds without relaxing correctness to hit sample targets. Decide whether any PoC merits an optional adapter using the acceptance criteria above.
4. **v0.1 release:** publish quality reports for official datasets/runs and document gates/limitations; no opaque quality score.
5. **v0.1.1 UI:** visualize quality evidence with accessible tables, drill-down explanations and provenance links.
6. **Later data-centric releases:** conduct controlled data-quality ablations and only then consider automated data CI/flywheel behavior.

### Targeted synthetic-data slice (early, bounded)

The `targeted-synthetic-data` OpenSpec change brings forward a narrow, optional research path: diagnose a labeled weak scenario from per-example evaluation evidence; prepare a versioned plan and provider/prompt/output draft; explicitly generate math SFT candidates or verified same-question Preference pairs; report candidate checks, selection, provenance, request budget and unknown/estimated cost; then train and re-evaluate through separate existing CLI workflows. A local fake adapter supports offline fixtures, and the optional OpenAI-compatible adapter is never called by the UI. The browser only reads allowlisted summaries and exports a non-registered draft; it has no provider execution or training control.

Generated count and answer verification are not a universal data-quality score. The supported answer checker does not certify reasoning quality; new questions require an independent reference or verifiable construction. Missing labels/denominators remain unavailable, costs remain unknown without evidence, final holdout content is isolated, and `no_improvement` or regression are valid study outcomes. This early slice does not implement an autonomous data flywheel, arbitrary task synthesis, CPT corpus generation, or GRPO prompt/reference synthesis.

## Out of scope

- One universal quality score or an opaque model-generated “quality grade”;
- claiming all semantic correctness can be automatically verified;
- blocking smoke iteration based on thresholds not yet validated;
- changing frozen v0.1.1 eligibility rules without a versioned revision;
- storing/rehosting external datasets or sensitive sample text in reports;
- treating rollout calibration as an intrinsic, timeless property of a dataset.
