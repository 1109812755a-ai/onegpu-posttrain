"""Aggregate per-example evaluation evidence into explicit scenario reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..data.io import read_jsonl
from ..data.targeted import (
    ScenarioDefinition,
    ScenarioDiagnosis,
    ScenarioReport,
    label_values,
)
from ..lineage import sha256_file
from ..schemas import EvalRecord


def load_scenario_definition(path: str | Path) -> ScenarioDefinition:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("场景定义必须是 YAML 对象")
    return ScenarioDefinition.model_validate(payload)


def _read_labels(path: Path, definition: ScenarioDefinition) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    for row in read_jsonl(path):
        problem_id = row.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise ValueError("标签 sidecar 的每行都必须包含非空 problem_id")
        if problem_id in labels:
            raise ValueError(f"标签 sidecar 中 problem_id 重复: {problem_id}")
        labels[problem_id] = label_values(row.get(definition.label_field))
    return labels


def _prediction_outcome(prediction: dict[str, Any] | None) -> str:
    if prediction is None:
        return "missing_prediction"
    if prediction.get("excluded") is True:
        return "excluded"
    if prediction.get("supported") is False or prediction.get("score_status") == "unsupported":
        return "unsupported"
    if prediction.get("parse_success") is False:
        return "parse_failed"
    correct = prediction.get("correct")
    if type(correct) is bool:
        return "correct" if correct else "incorrect"
    return "unsupported"


def build_scenario_report(
    *,
    definition: ScenarioDefinition,
    dataset_path: str | Path,
    predictions_path: str | Path,
    eval_id: str,
    model_id: str,
    protocol_version: str,
    threshold_source: str,
    minimum_samples_required: int,
    sidecar_path: str | Path | None = None,
) -> ScenarioReport:
    """Join prediction rows to exact eval-record IDs; emit counts, never sample text."""

    dataset_path = Path(dataset_path)
    predictions_path = Path(predictions_path)
    dataset_sha256 = sha256_file(dataset_path)
    if definition.dataset_sha256 != dataset_sha256:
        raise ValueError("场景定义的数据集摘要与输入 JSONL 不匹配")
    if sidecar_path is not None and definition.label_source != "sidecar":
        raise ValueError("metadata 场景定义不能额外传 sidecar")
    if definition.label_source == "sidecar" and sidecar_path is None:
        raise ValueError("场景定义要求 sidecar，但没有提供 --sidecar")

    if definition.label_source == "sidecar":
        assert sidecar_path is not None
        label_source_sha256 = sha256_file(sidecar_path)
        if definition.sidecar_sha256 != label_source_sha256:
            raise ValueError("场景定义的 sidecar_sha256 与输入文件不匹配")
        sidecar_labels = _read_labels(Path(sidecar_path), definition)
    else:
        label_source_sha256 = dataset_sha256
        sidecar_labels = {}

    dataset_by_id: dict[str, EvalRecord] = {}
    for row in read_jsonl(dataset_path):
        try:
            record = EvalRecord.model_validate(row)
        except ValidationError as exc:
            raise ValueError(f"评测数据不是 canonical EvalRecord: {exc}") from exc
        if record.id in dataset_by_id:
            raise ValueError(f"评测数据 id 重复: {record.id}")
        dataset_by_id[record.id] = record

    predictions: dict[str, dict[str, Any]] = {}
    unmatched_prediction_count = 0
    for row in read_jsonl(predictions_path):
        prediction_id = row.get("id")
        problem_id = row.get("problem_id")
        if not isinstance(prediction_id, str) or not isinstance(problem_id, str):
            raise TypeError("预测记录必须包含字符串 id 和 problem_id")
        if prediction_id in predictions:
            raise ValueError(f"预测记录 id 重复: {prediction_id}")
        predictions[prediction_id] = row
        source = dataset_by_id.get(prediction_id)
        if source is None:
            unmatched_prediction_count += 1
        elif source.problem_id != problem_id:
            raise ValueError(f"预测 problem_id 与输入记录不一致: {prediction_id}")

    counters: dict[str, dict[str, int]] = {
        scenario.scenario_id: {
            "correct": 0, "incorrect": 0, "parse_failed": 0, "unsupported": 0,
            "excluded": 0, "missing_prediction": 0,
        }
        for scenario in definition.scenarios
    }
    scenario_values = {scenario.scenario_id: set(scenario.values) for scenario in definition.scenarios}
    unknown_label_count = unmatched_prediction_count
    overlapping_sample_count = 0
    matched_prediction_ids: set[str] = set()

    for record in dataset_by_id.values():
        if definition.label_source == "metadata":
            raw_labels = record.metadata.get(definition.label_field)
            labels = label_values(raw_labels)
        else:
            labels = sidecar_labels.get(record.problem_id, set())
        matched_scenarios = [sid for sid, values in scenario_values.items() if labels & values]
        if not matched_scenarios:
            unknown_label_count += 1
            continue
        if len(matched_scenarios) > 1:
            overlapping_sample_count += 1
        prediction = predictions.get(record.id)
        if prediction is not None:
            matched_prediction_ids.add(record.id)
        outcome = _prediction_outcome(prediction)
        for scenario_id in matched_scenarios:
            counters[scenario_id][outcome] += 1

    missing_prediction_count = len(dataset_by_id) - len(matched_prediction_ids)
    diagnoses: list[ScenarioDiagnosis] = []
    for scenario in definition.scenarios:
        counts = counters[scenario.scenario_id]
        sample_count = sum(counts.values())
        denominator = counts["correct"] + counts["incorrect"]
        if denominator == 0:
            status = "not_evaluated"
            accuracy = None
        else:
            status = "available" if denominator >= minimum_samples_required else "insufficient_evidence"
            accuracy = counts["correct"] / denominator
        diagnoses.append(ScenarioDiagnosis(
            scenario_id=scenario.scenario_id,
            sample_count=sample_count,
            denominator=denominator,
            correct=counts["correct"],
            incorrect=counts["incorrect"],
            parse_failed=counts["parse_failed"],
            unsupported=counts["unsupported"],
            excluded=counts["excluded"],
            missing_prediction=counts["missing_prediction"],
            accuracy=accuracy,
            status=status,
        ))

    return ScenarioReport(
        eval_id=eval_id,
        model_id=model_id,
        dataset_id=definition.dataset_id,
        dataset_sha256=dataset_sha256,
        protocol_version=protocol_version,
        predictions_sha256=sha256_file(predictions_path),
        scenario_definition_id=definition.definition_id,
        scenario_definition_sha256=definition.config_sha256,
        label_source_sha256=label_source_sha256,
        threshold_source=threshold_source,
        minimum_samples_required=minimum_samples_required,
        input_record_count=len(dataset_by_id),
        prediction_count=len(predictions),
        unmatched_prediction_count=unmatched_prediction_count,
        missing_prediction_count=missing_prediction_count,
        unknown_label_count=unknown_label_count,
        overlapping_sample_count=overlapping_sample_count,
        scenarios=diagnoses,
    )


def write_scenario_report(report: ScenarioReport, output_path: str | Path) -> dict[str, str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report_id": report.report_id,
        "report_sha256": report.report_sha256,
        **report.model_dump(mode="json"),
    }
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"report_id": report.report_id, "report_sha256": report.report_sha256, "output": str(output_path)}
