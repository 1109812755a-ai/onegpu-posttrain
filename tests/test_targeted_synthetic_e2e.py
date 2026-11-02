from __future__ import annotations

import json
from pathlib import Path

import yaml
from test_synthetic_generation import _prepare, _private_output

from onegpu_posttrain.data.decontaminate import BenchmarkIndex
from onegpu_posttrain.data.policy import DataPolicy, PolicyTokenizer
from onegpu_posttrain.data.quality_assessment import create_quality_assessment_bundle
from onegpu_posttrain.data.quality_review import (
    create_quality_review_bundle,
    load_quality_review_bundle,
)
from onegpu_posttrain.data.targeted import ScenarioDiagnosis
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.lineage import (
    ArtifactRef,
    Digest,
    EvaluationLineage,
    ModelLineage,
    RunRecord,
    new_run_id,
    sha256_json,
)
from onegpu_posttrain.lineage_validation import validate_lineage_root
from onegpu_posttrain.synthetic.artifacts import write_targeted_artifacts
from onegpu_posttrain.synthetic.candidates import validate_candidate_response
from onegpu_posttrain.synthetic.effects import (
    Guardrail,
    build_before_after_report,
    save_effect_report,
)
from onegpu_posttrain.synthetic.generation import execute_targeted_generation
from onegpu_posttrain.synthetic.policy_export import (
    CandidatePolicyInput,
    export_candidates_with_policy,
)
from onegpu_posttrain.synthetic.providers import FakeProvider


def test_cpu_e2e_generation_export_training_eval_lineage_and_regression(tmp_path):
    _, setup = _prepare(tmp_path)
    selected = setup.selection.records
    response_by_problem = {
        "seed-problem-001": '{"problem_id":"seed-problem-001","solution":"4 groups of 3 tiles is 12. Final \\\\boxed{12}","final_answer":"12"}',
        "seed-problem-002": '{"problem_id":"seed-problem-002","solution":"5 times 6 is 30. Final \\\\boxed{30}","final_answer":"30"}',
    }
    provider = FakeProvider([response_by_problem[row["metadata"]["problem_id"]] for row in selected])
    private_output = _private_output(tmp_path, setup, "e2e-responses.jsonl")
    generated = execute_targeted_generation(setup, output_path=private_output, provider=provider)
    assert generated["network_used"] is False
    assert generated["request_count"] == 2
    assert generated["successful_candidates"] == 2
    assert generated["shortfall"] == 1

    # A second execution reuses complete cache entries without spending the plan budget.
    resumed_provider = FakeProvider()
    resumed = execute_targeted_generation(
        setup, output_path=_private_output(tmp_path, setup, "e2e-resumed.jsonl"), provider=resumed_provider
    )
    assert resumed["request_count"] == 0
    assert resumed["cache_reused_count"] == 2
    assert resumed["ledger"]["request_attempts"] == 2

    raw_responses = [json.loads(line) for line in private_output.read_text(encoding="utf-8").splitlines()]
    seeds = {row["id"]: row for row in selected}
    candidate_inputs = []
    for raw in raw_responses:
        seed = seeds[raw["record_id"]]
        candidate = validate_candidate_response(
            raw["response"],
            source_record_id=raw["record_id"],
            expected_problem_id=raw["problem_id"],
            candidate_slot=raw["candidate_slot"],
            request_sha256=raw["request_sha256"],
            finish_reason=raw["finish_reason"],
            question=seed["raw_problem"],
            protocol=setup.protocol,
            tokenizer=WhitespaceTokenizer(),
            max_training_tokens=setup.plan.budget.max_output_tokens,
        )
        assert candidate.validation_status == "candidate_ready_for_quality_review"
        candidate_inputs.append(CandidatePolicyInput(
            candidate=candidate,
            question=seed["raw_problem"],
            reference_answer=seed["metadata"]["reference"],
        ))

    policy = DataPolicy(
        stage="sft",
        tokenizer=PolicyTokenizer(test_only=True),
        resources={"max_length": 128, "target_count": 10},
    )
    export = export_candidates_with_policy(
        candidate_inputs,
        policy=policy,
        tokenizer=WhitespaceTokenizer(),
        registry=setup.registry,
        protected_index=BenchmarkIndex([("unrelated-holdout", "An unrelated project-owned holdout question.")]),
        protected_index_id="fixture-protected-v1",
        protected_index_sha256="b" * 64,
    )
    assert export["selected_count"] == 2
    assert export["assessment_counts"] == {"correct": 2}

    contaminated = export_candidates_with_policy(
        candidate_inputs,
        policy=policy,
        tokenizer=WhitespaceTokenizer(),
        registry=setup.registry,
        protected_index=BenchmarkIndex([("contaminated-copy", selected[0]["raw_problem"])]),
        protected_index_id="fixture-contaminated-v1",
        protected_index_sha256="c" * 64,
    )
    assert contaminated["selected_count"] < export["selected_count"]
    assert contaminated["rejected"]

    root = tmp_path / "e2e"
    artifacts = write_targeted_artifacts(
        output_dir=root / "build-001",
        plan=setup.plan,
        profile=setup.profile,
        tokenizer=WhitespaceTokenizer(),
        candidates=[item.candidate for item in candidate_inputs],
        policy_export=export,
        generation_summary=generated,
        validation_summary={"candidate_count": 2, "valid_format": 2},
    )
    manifest = yaml.safe_load(Path(artifacts["training_manifest"]).read_text(encoding="utf-8"))
    training_digest = manifest["hashes"]["canonical_records_sha256"]

    mini_profile = tmp_path / "targeted-sft-profile.yaml"
    mini_profile.write_text(yaml.safe_dump({
        "schema_version": "1.0", "profile_id": "targeted-fixture-mini", "profile_version": "1",
        "description": "Targeted SFT fixture", "datasets": [{
            "dataset_key": "openr1.sft_train", "stage": "sft", "target": 1,
            "unit": "examples", "count_source": "processing_report.outputs.sft.budget_count",
        }],
    }, sort_keys=False), encoding="utf-8")
    _, quality_index = create_quality_assessment_bundle(root / "build-001", mini_profile)
    _, review_index_path = create_quality_review_bundle(
        root / "build-001", mini_profile, quality_index,
        Path(__file__).parents[1] / "configs/data/quality-review/mini-review-v1.yaml",
    )
    _, review_index = load_quality_review_bundle(review_index_path)
    reviewed = review_index.datasets[0]
    assert reviewed.dataset_id == artifacts["training_dataset_id"]
    assert reviewed.content_sha256 == training_digest
    assert reviewed.usage == "exploratory"
    assert reviewed.outcome == "pending"
    assert manifest["lineage"]["parents"] == [artifacts["candidate_dataset_id"]]

    # These lineage refs are deliberately simulated: no weights, optimizer or model eval was run.
    model_before = setup.plan.starting_model.model_id
    model_after = "model-fixture-targeted-after"
    model_dir = root / "model_artifacts"
    model_dir.mkdir(parents=True)
    fixture_digest = Digest(status="not_computed", reason="CPU fixture; no trained weights")
    for model_id, parents in ((model_before, []), (model_after, [model_before])):
        record = ModelLineage(
            model_id=model_id,
            kind="merged",
            digest=fixture_digest,
            parent_model_ids=parents,
            produced_by_run_id=None,
            stage="fixture-targeted-sft",
        )
        (model_dir / f"{model_id}.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")

    train_run_id = new_run_id("train", label="targeted-e2e-fixture")
    train_record = RunRecord(
        run_id=train_run_id,
        run_type="train",
        status="completed",
        run_label="simulated-targeted-training",
        started_at="2026-10-18T00:00:00Z",
        completed_at="2026-10-18T00:00:01Z",
        recipe_path="fixture/simulated-targeted-training",
        resolved_config_sha256=sha256_json({"fixture": "simulated-targeted-training"}),
        inputs=[
            ArtifactRef(
                id=artifacts["training_dataset_id"],
                kind="dataset",
                role="train",
                digest=Digest(value=training_digest, status="available"),
            ),
            ArtifactRef(id=model_before, kind="model", role="parent", digest=fixture_digest),
        ],
        outputs=[ArtifactRef(id=model_after, kind="model", role="checkpoint", digest=fixture_digest)],
    )
    after_record_path = model_dir / f"{model_after}.json"
    after_record = ModelLineage.model_validate_json(after_record_path.read_text(encoding="utf-8"))
    after_record.produced_by_run_id = train_run_id
    after_record_path.write_text(after_record.model_dump_json(indent=2), encoding="utf-8")
    train_dir = root / "runs" / train_run_id
    train_dir.mkdir(parents=True)
    (train_dir / "lineage.json").write_text(train_record.model_dump_json(indent=2), encoding="utf-8")

    before = setup.report
    after = before.model_copy(update={
        "eval_id": "eval-fixture-targeted-after",
        "model_id": model_after,
        "scenarios": [
            ScenarioDiagnosis(
                scenario_id="algebra", sample_count=2, denominator=2,
                correct=2, incorrect=0, accuracy=1.0, status="available",
            ),
            ScenarioDiagnosis(
                scenario_id="arithmetic", sample_count=1, denominator=1,
                correct=0, incorrect=1, accuracy=0.0, status="insufficient_evidence",
            ),
        ],
    })
    eval_run_id = new_run_id("evaluate", label="targeted-e2e-fixture")
    eval_record = RunRecord(
        run_id=eval_run_id,
        run_type="evaluate",
        status="completed",
        run_label="simulated-targeted-evaluation",
        started_at="2026-10-18T00:00:02Z",
        completed_at="2026-10-18T00:00:03Z",
        inputs=[ArtifactRef(id=model_after, kind="model", digest=fixture_digest),
                ArtifactRef(id=after.dataset_id, kind="dataset")],
        outputs=[ArtifactRef(id=after.eval_id, kind="evaluation")],
    )
    eval_dir = root / "runs" / eval_run_id
    eval_dir.mkdir(parents=True)
    (eval_dir / "lineage.json").write_text(eval_record.model_dump_json(indent=2), encoding="utf-8")
    evaluation = EvaluationLineage(
        eval_id=after.eval_id,
        produced_by_run_id=eval_run_id,
        model_id=model_after,
        model_produced_by_run_id=train_run_id,
        dataset_ids=[after.dataset_id],
        protocol_version=after.protocol_version,
        config_sha256=sha256_json({"fixture": "simulated-evaluation"}),
        status="completed",
        metrics={"algebra": {"accuracy": 1.0, "denominator": 2}, "evidence_kind": {"value": "fixture"}},
    )
    (eval_dir / "evaluation_lineage.json").write_text(evaluation.model_dump_json(indent=2), encoding="utf-8")

    effect = build_before_after_report(
        before=before,
        after=after,
        target_scenario_id="algebra",
        target=setup.plan.target,
        guardrails=[Guardrail(name="arithmetic_accuracy", direction="higher", max_regression=0.05)],
        before_guard_metrics={"arithmetic_accuracy": 1.0},
        after_guard_metrics={"arithmetic_accuracy": 0.0},
        training_dataset_id=artifacts["training_dataset_id"],
        training_dataset_sha256=training_digest,
        actual_added_records=export["selected_count"],
        planned_training_budget=setup.plan.training_budget,
        actual_training_budget=export["selected_count"],
        cost_status="unknown",
    )
    save_effect_report(effect, root / "build-001" / "reports" / "remediation_report.json")
    assert effect.status == "guardrail_regression"
    assert effect.target.delta == 0.5
    assert effect.guardrails["arithmetic_accuracy"].regressed is True
    assert effect.automatic_next_round_started is False

    lineage = validate_lineage_root(root)
    assert lineage["valid"], lineage["errors"]
    assert any(item["code"] == "evaluation_model_digest_mismatch" for item in lineage["warnings"])
    assert lineage["summary"]["runs"] >= 4  # build, quality assessment, simulated train and eval
    assert provider.calls == 2
