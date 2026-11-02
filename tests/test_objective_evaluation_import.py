import hashlib
import json
from io import BytesIO

from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.manifest import load_manifest, validate_manifest_evidence
from onegpu_posttrain.data.objective_evaluation_import import (
    OBJECTIVE_EVALUATION_FORMAT, convert_objective_evaluation_jsonl,
)
from onegpu_posttrain.data.registry import DataSource


def _row(kind="single_choice", answer="B", options=None):
    return {"questionType": kind, "question": "which answer?",
            "options": json.dumps(options or ["one", "two", "three"], ensure_ascii=False),
            "correctAnswer": answer, "tags": ""}


def test_objective_converter_supports_mixed_types_and_sixth_option():
    rows = [
        _row("single_choice", "B"),
        _row("multiple_choice", '["A", "F"]', ["a", "b", "c", "d", "e", "f"]),
        {"questionType": "true_false", "question": "claim?", "options": "", "correctAnswer": "✅"},
    ]
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows).encode() + b"\n"
    result = convert_objective_evaluation_jsonl(BytesIO(payload))
    assert result.valid
    assert result.source_record_count == result.output_record_count == 3
    normalized = [json.loads(line) for line in result.canonical_bytes.splitlines()]
    assert normalized[0]["answer"] == "B"
    assert normalized[1]["answer"] == "A,F"
    assert normalized[1]["options"][-1] == "f"
    assert normalized[2]["answer"] == "true"
    assert all(item["metadata"]["dataset_role"] == "evaluation_only" for item in normalized)


def test_objective_converter_reports_and_excludes_missing_answer_rows():
    good = _row()
    missing = _row()
    del missing["correctAnswer"]
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in (good, missing)).encode() + b"\n"
    result = convert_objective_evaluation_jsonl(BytesIO(payload))
    report = result.report()
    assert result.valid
    assert report["source_record_count"] == 2
    assert report["output_record_count"] == 1
    assert report["rejected_record_count"] == 1
    assert report["errors"][0]["line"] == 2
    assert report["errors"][0]["field"] == "correctAnswer"


def test_objective_import_registration_retains_source_and_eval_only_role(tmp_path):
    row = _row("multiple_choice", '["A", "F"]', ["a", "b", "c", "d", "e", "f"])
    raw = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    conversion = convert_objective_evaluation_jsonl(BytesIO(raw))
    origin = DataSource(source_id="upload-pending", kind="local_upload", name="考试题集",
                        revision_status="not_provided", license_status="not_provided", declared_by="user")
    result = register_jsonl(
        BytesIO(conversion.canonical_bytes), workspace=tmp_path,
        output_root=tmp_path / "data" / "processed" / "user-upload", stage="eval", origin=origin,
        raw_source_bytes=raw, source_format=OBJECTIVE_EVALUATION_FORMAT,
        converter_config={"source_format": OBJECTIVE_EVALUATION_FORMAT, "converter_version": "objective-choice-converter-v1",
                          "source_record_count": 1, "output_record_count": 1},
        registration_metadata={"dataset_role": "evaluation_only", "evaluation_protocol": "choice-accuracy-v1"},
    )
    manifest_path = result.path / "manifests" / "eval-user-upload-v1.yaml"
    manifest = load_manifest(manifest_path)
    assert manifest.stage == "eval"
    assert manifest.metadata["dataset_role"] == "evaluation_only"
    assert manifest.metadata["evaluation_protocol"] == "choice-accuracy-v1"
    assert manifest.metadata["source_path"] == "raw/source.jsonl"
    assert (result.path / "raw" / "source.jsonl").read_bytes() == raw
    assert hashlib.sha256((result.path / "canonical" / "eval.jsonl").read_bytes()).hexdigest() == result.canonical_sha256
    assert validate_manifest_evidence(manifest_path) == []
