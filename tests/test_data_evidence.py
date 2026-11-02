import pytest
from pydantic import ValidationError

from onegpu_posttrain.data.evidence import QualityMetric, assess_dataset, ratio, step
from onegpu_posttrain.data.manifest import DatasetManifest
from onegpu_posttrain.data.quality_protocol import (
    QUALITY_PROTOCOL_ID,
    STAGE_METRIC_CATALOG,
    metric_catalog_for,
)


def test_metric_validates_missing_and_ratio_semantics():
    assert ratio(0, 0, "fixture").value is None
    assert ratio(1, 2, "fixture").model_dump(by_alias=True)["class"] == "dataset_quality"
    with pytest.raises(ValidationError, match="positive denominator"):
        QualityMetric(value=0, unit="ratio", status="measured", metric_class="dataset_quality",
                      method="fixture", numerator=0, denominator=0)
    with pytest.raises(ValidationError, match="require null"):
        QualityMetric(value=0, unit="ratio", status="not_measured", metric_class="dataset_quality", method="fixture")


def test_step_rejects_inconsistent_subset_accounting():
    with pytest.raises(ValidationError, match="reconcile"):
        step(1, "filter", "filter", "in", "out", 10, 8, "examples", rejected={"bad": 1})
    skipped = step(1, "check", "verify", "in", "out", 10, 10, "examples", skipped=True, limitations=["No checker"])
    assert skipped["status"] == "skipped"


def test_legacy_manifest_defaults_new_report_references():
    old = DatasetManifest.model_validate({"dataset_id": "legacy", "stage": "sft", "sample_count": 1,
                                         "source": {"repo": "fixture", "revision": "r1"}})
    assert old.processing_report is None
    assert old.quality_report is None


def test_stage_quality_catalog_is_versioned_and_covers_all_mini_stages():
    assert QUALITY_PROTOCOL_ID == "stage-data-quality-v1"
    assert set(STAGE_METRIC_CATALOG) == {
        "cpt", "cpt_holdout", "sft", "preference", "preference_eval", "grpo", "eval",
    }
    for stage, definitions in STAGE_METRIC_CATALOG.items():
        keys = [definition.metric_id for definition in definitions]
        assert len(keys) == len(set(keys)), stage
        assert {"schema_valid_rate", "duplicate_record_ids", "train_eval_overlap", "near_duplicate_rate"} <= set(keys)
        assert all(definition.metric_class and definition.unit and definition.method and definition.description
                   for definition in definitions)
    assert "upstream_label_eligibility_rate" in {m.metric_id for m in metric_catalog_for("sft")}
    assert "upstream_label_eligibility_rate" in {m.metric_id for m in metric_catalog_for("preference")}
    assert "verifier_support_coverage" in {m.metric_id for m in metric_catalog_for("grpo")}
    assert "scoring_coverage" in {m.metric_id for m in metric_catalog_for("eval")}
    with pytest.raises(ValueError, match="Unsupported quality assessment stage"):
        metric_catalog_for("unknown")


@pytest.mark.parametrize(
    ("stage", "rows", "label_metric", "expected"),
    [
        ("cpt", [{"id": "c1", "text": "proof", "source": "fixture", "metadata": {"tokens": 3}}], None, None),
        ("cpt_holdout", [{"id": "c2", "text": "proof", "source": "fixture", "metadata": {"tokens": 2}}], None, None),
        ("sft", [
            {"id": "s1", "raw_problem": "1+1", "messages": [
                {"role": "user", "content": "1+1"}, {"role": "assistant", "content": "2"}],
             "prompt_template_id": "v1", "metadata": {"correctness_math_verify": True, "rendered_tokens": 5}},
            {"id": "s2", "raw_problem": "2+2", "messages": [
                {"role": "user", "content": "2+2"}, {"role": "assistant", "content": "4"}],
             "prompt_template_id": "v1", "metadata": {"correctness_math_verify": None, "rendered_tokens": 5}},
        ], "upstream_label_eligibility_rate", (1, 1)),
        ("preference", [
            {"id": "p1", "raw_problem": "1+1", "prompt": [{"role": "user", "content": "1+1"}],
             "chosen": "2", "rejected": "3", "prompt_template_id": "v1",
             "metadata": {"chosen_correct": True, "rejected_correct": False, "chosen_tokens": 2, "rejected_tokens": 2}},
        ], "upstream_label_eligibility_rate", (1, 1)),
        ("preference_eval", [
            {"id": "p2", "raw_problem": "1+1", "prompt": [{"role": "user", "content": "1+1"}],
             "chosen": "2", "rejected": "3", "prompt_template_id": "v1",
             "metadata": {"chosen_correct": True, "rejected_correct": False}},
        ], "upstream_label_eligibility_rate", (1, 1)),
        ("grpo", [{"id": "g1", "raw_problem": "1+1", "prompt": [{"role": "user", "content": "1+1"}],
                   "answer": "2", "prompt_template_id": "v1", "metadata": {"prompt_tokens": 4, "verifier": "math_verify"}}],
         "verifier_support_coverage", None),
        ("eval", [{"id": "e1", "problem_id": "e1", "raw_problem": "1+1", "prompt": "1+1",
                   "answer": "2", "eval_prompt_version": "v1"}], "scoring_coverage", None),
    ],
)
def test_assessment_emits_stage_catalog_and_uses_known_label_denominators(stage, rows, label_metric, expected):
    report = assess_dataset(rows, stage=stage, dataset_id=f"ds-{stage}", content_sha256="a" * 64,
                            build_id="build-fixture", tokenizer_name="fixture", overlap=0)
    assert set(report.metrics) == {definition.metric_id for definition in metric_catalog_for(stage)}
    if expected is not None:
        metric = report.metrics[label_metric]
        assert (metric.numerator, metric.denominator) == expected
        assert metric.limitations
    elif label_metric:
        assert report.metrics[label_metric].status == "not_measured"
    assert report.metrics["semantic_correctness"].status == "not_measured" if "semantic_correctness" in report.metrics else True
    assert report.metrics["near_duplicate_rate"].value is None
    assert report.evidence_coverage.status == "complete"
    assert report.evidence_coverage.expected_metrics == len(report.metrics)
    assert report.gate.summary_status != "PASS"


def test_quality_report_coverage_does_not_imply_quality_approval_and_reads_legacy_v1():
    report = assess_dataset(
        [{"id": "s1", "raw_problem": "1+1", "messages": [
            {"role": "user", "content": "1+1"}, {"role": "assistant", "content": "2"}],
         "prompt_template_id": "v1", "metadata": {"correctness_math_verify": True}}],
        stage="sft", dataset_id="ds-sft", content_sha256="a" * 64,
        build_id="build", tokenizer_name="fixture", overlap=0,
    )
    assert report.schema_version == "1.1"
    assert report.evidence_coverage.status == "complete"
    assert report.evidence_coverage.not_measured_metrics > 0
    assert report.gate.summary_status == "NOT_EVALUATED"
    assert report.checker.name == "onegpu-posttrain-stage-quality-assessor"
    assert report.checker.version == "1.0.0"

    legacy_payload = report.model_dump(mode="json", by_alias=True)
    legacy_payload["schema_version"] = "1.0"
    legacy_payload.pop("evidence_coverage")
    legacy = type(report).model_validate(legacy_payload)
    assert legacy.schema_version == "1.0"
    assert legacy.evidence_coverage is None
