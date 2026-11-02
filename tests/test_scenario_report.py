from __future__ import annotations

import json

import pytest
import yaml

from onegpu_posttrain.cli import build_parser
from onegpu_posttrain.data.io import write_jsonl
from onegpu_posttrain.data.targeted import ScenarioDefinition, ScenarioLabel
from onegpu_posttrain.evaluation.scenarios import (
    build_scenario_report,
    write_scenario_report,
)
from onegpu_posttrain.lineage import sha256_file


def eval_record(record_id: str, problem_id: str, metadata: dict):
    return {
        "id": record_id,
        "problem_id": problem_id,
        "raw_problem": f"problem {problem_id}",
        "prompt": f"solve {problem_id}",
        "answer": "1",
        "eval_prompt_version": "test-v1",
        "metadata": metadata,
    }


def test_scenario_report_joins_by_exact_eval_id_and_preserves_unknowns(tmp_path):
    dataset = tmp_path / "eval.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        eval_record("r1", "p1", {"topic": "algebra"}),
        eval_record("r2", "p2", {"topic": "algebra"}),
        eval_record("r3", "p3", {"topic": "geometry"}),
        eval_record("r4", "p4", {"topic": ["algebra", "linear"]}),
        eval_record("r5", "p5", {"topic": "algebra"}),
        eval_record("r6", "p6", {}),
    ]
    pred_rows = [
        {"id": "r1", "problem_id": "p1", "parse_success": True, "correct": True},
        {"id": "r2", "problem_id": "p2", "parse_success": True, "correct": False},
        {"id": "r3", "problem_id": "p3", "parse_success": False, "correct": False},
        {"id": "r4", "problem_id": "p4", "parse_success": True, "correct": None, "supported": False},
        {"id": "external", "problem_id": "other", "parse_success": True, "correct": True},
    ]
    write_jsonl(dataset, rows)
    write_jsonl(predictions, pred_rows)
    dataset_hash, predictions_hash = sha256_file(dataset), sha256_file(predictions)
    definition = ScenarioDefinition(
        taxonomy_id="math", taxonomy_version="v1", dataset_id="eval-math-dev", dataset_sha256=dataset_hash,
        label_source="metadata", label_field="topic", scenarios=[
            ScenarioLabel(scenario_id="algebra", title="代数", values=["algebra"]),
            ScenarioLabel(scenario_id="linear", title="线性", values=["linear"]),
            ScenarioLabel(scenario_id="geometry", title="几何", values=["geometry"]),
        ],
    )

    report = build_scenario_report(
        definition=definition, dataset_path=dataset, predictions_path=predictions,
        eval_id="eval-1", model_id="model-sft", protocol_version="internal-math-v1",
        threshold_source="test registration", minimum_samples_required=3,
    )

    by_id = {item.scenario_id: item for item in report.scenarios}
    assert by_id["algebra"].sample_count == 4
    assert by_id["algebra"].denominator == 2
    assert by_id["algebra"].accuracy == 0.5
    assert by_id["algebra"].status == "insufficient_evidence"
    assert by_id["algebra"].unsupported == 1
    assert by_id["algebra"].missing_prediction == 1
    assert by_id["linear"].status == "not_evaluated"
    assert by_id["linear"].accuracy is None
    assert by_id["geometry"].parse_failed == 1
    assert report.unknown_label_count == 2
    assert report.overlapping_sample_count == 1
    assert report.unmatched_prediction_count == 1
    assert report.missing_prediction_count == 2
    assert sha256_file(dataset) == dataset_hash
    assert sha256_file(predictions) == predictions_hash

    output = tmp_path / "reports" / "scenario_report.json"
    write_scenario_report(report, output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["report_id"] == report.report_id
    assert saved["scenarios"][0]["accuracy"] == 0.5


def test_sidecar_must_match_declared_hash(tmp_path):
    dataset = tmp_path / "eval.jsonl"
    sidecar = tmp_path / "labels.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(dataset, [eval_record("r1", "p1", {})])
    write_jsonl(sidecar, [{"problem_id": "p1", "topic": "algebra"}])
    write_jsonl(predictions, [{"id": "r1", "problem_id": "p1", "parse_success": True, "correct": True}])
    definition = ScenarioDefinition(
        taxonomy_id="math", taxonomy_version="v1", dataset_id="eval-math-dev",
        dataset_sha256=sha256_file(dataset), label_source="sidecar", label_field="topic",
        sidecar_sha256="0" * 64,
        scenarios=[ScenarioLabel(scenario_id="algebra", title="代数", values=["algebra"])],
    )
    with pytest.raises(ValueError, match="sidecar_sha256"):
        build_scenario_report(
            definition=definition, dataset_path=dataset, predictions_path=predictions,
            eval_id="eval-1", model_id="model-sft", protocol_version="internal-math-v1",
            threshold_source="test registration", minimum_samples_required=1, sidecar_path=sidecar,
        )


def test_aggregate_score_cannot_replace_per_example_predictions(tmp_path):
    dataset = tmp_path / "eval.jsonl"
    predictions = tmp_path / "eval.json"
    write_jsonl(dataset, [eval_record("r1", "p1", {"topic": "algebra"})])
    predictions.write_text('{"accuracy": 0.9}\n', encoding="utf-8")
    definition = ScenarioDefinition(
        taxonomy_id="math", taxonomy_version="v1", dataset_id="eval-math-dev",
        dataset_sha256=sha256_file(dataset), label_source="metadata", label_field="topic",
        scenarios=[ScenarioLabel(scenario_id="algebra", title="代数", values=["algebra"])],
    )
    with pytest.raises(TypeError, match="id 和 problem_id"):
        build_scenario_report(
            definition=definition, dataset_path=dataset, predictions_path=predictions,
            eval_id="eval-1", model_id="model-sft", protocol_version="internal-math-v1",
            threshold_source="test registration", minimum_samples_required=1,
        )


def test_scenario_report_cli_writes_content_addressed_json_and_refuses_overwrite(tmp_path, capsys):
    dataset = tmp_path / "eval.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    definition_path = tmp_path / "scenario.yaml"
    output = tmp_path / "scenario-report.json"
    write_jsonl(dataset, [eval_record("r1", "p1", {"topic": "algebra"})])
    write_jsonl(predictions, [{"id": "r1", "problem_id": "p1", "parse_success": True, "correct": True}])
    definition = ScenarioDefinition(
        taxonomy_id="math", taxonomy_version="v1", dataset_id="eval-math-dev",
        dataset_sha256=sha256_file(dataset), label_source="metadata", label_field="topic",
        scenarios=[ScenarioLabel(scenario_id="algebra", title="代数", values=["algebra"])],
    )
    definition_path.write_text(yaml.safe_dump(definition.model_dump(mode="json"), allow_unicode=True), encoding="utf-8")
    args = build_parser().parse_args([
        "build-scenario-report", "--definition", str(definition_path), "--dataset", str(dataset),
        "--predictions", str(predictions), "--eval-id", "eval-1", "--model-id", "model-sft",
        "--protocol-version", "internal-math-v1", "--threshold-source", "test registration",
        "--minimum-samples", "1", "--output", str(output),
    ])
    assert args.func(args) == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["report_id"].startswith("scenario-report-")
    assert saved["scenarios"][0]["status"] == "available"
    with pytest.raises(FileExistsError):
        args.func(args)

