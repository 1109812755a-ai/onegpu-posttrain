import json
import re

import pytest
from pydantic import ValidationError

from onegpu_posttrain.lineage import (
    DatasetLineage,
    Digest,
    EvaluationLineage,
    RunRecord,
    assert_unique_ids,
    load_versioned_payload,
    new_run_id,
    resolve_local_reference,
    sha256_directory,
    sha256_file,
    sha256_json,
)


def test_run_ids_are_unique_and_labels_are_not_identity(tmp_path):
    first = new_run_id("train", label="sft-mini-s42")
    second = new_run_id("train", label="sft-mini-s42")
    assert first != second
    assert first.startswith("run-sft-mini-s42-")
    assert_unique_ids([{"run_id": first}, {"run_id": second}])
    with pytest.raises(ValueError, match="duplicate run_id"):
        assert_unique_ids([{"run_id": first}, {"run_id": first}])


def test_versioned_run_schema_and_legacy_detection(tmp_path):
    run = RunRecord(
        run_id=new_run_id("train"), run_type="train", status="completed",
        started_at="2026-10-17T00:00:00Z",
    )
    assert run.lineage_schema_version == "1.0"
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"run_id": "old-label", "stage": "sft"}), encoding="utf-8")
    assert load_versioned_payload(legacy, legacy_keys={"run_id"})[1] == "legacy"
    legacy.write_text(json.dumps({"lineage_schema_version": "2.0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported lineage"):
        load_versioned_payload(legacy, legacy_keys={"run_id"})


def test_targeted_plan_identity_is_explicit_and_paired_on_run_and_evaluation():
    plan_id = "plan-targeted-0123456789abcdefabcd"
    run = RunRecord(
        run_id="run-training-fixture", run_type="train", status="completed",
        started_at="2026-10-17T00:00:00Z", targeted_plan_id=plan_id,
        targeted_plan_version=2,
    )
    evaluation = EvaluationLineage(
        eval_id="eval-fixture", produced_by_run_id="run-eval-fixture", model_id="model-fixture",
        targeted_plan_id=plan_id, targeted_plan_version=2,
        protocol_version="protocol-v1", status="completed",
    )
    assert run.model_dump(mode="json")["targeted_plan_id"] == evaluation.targeted_plan_id
    with pytest.raises(ValidationError, match="supplied together"):
        RunRecord(run_id="run-bad", run_type="train", status="failed", started_at="now",
                  targeted_plan_id=plan_id)


def test_dataset_lineage_requires_real_producing_run_and_digest():
    with pytest.raises(ValidationError):
        DatasetLineage(
            produced_by_run_id="not-a-run", transform={}, source_evidence={},
            record_provenance="dataset_only", count_unit="examples", content_sha256="bad",
        )
    assert Digest(status="unavailable", reason="remote file").value is None
    with pytest.raises(ValidationError):
        Digest(status="available")


def test_file_and_directory_hashes_are_stable_and_content_sensitive(tmp_path):
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"weights")
    first = sha256_directory(artifact)
    assert sha256_file(artifact / "weights.bin")
    (artifact / "lineage.json").write_text("metadata changes are excluded", encoding="utf-8")
    assert sha256_directory(artifact) == first
    (artifact / "weights.bin").write_bytes(b"different weights")
    assert sha256_directory(artifact) != first
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


def test_local_reference_rejects_parent_traversal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    inside = root / "inside.txt"
    inside.write_text("ok", encoding="utf-8")
    assert resolve_local_reference(root, "inside.txt") == inside.resolve()
    with pytest.raises(ValueError, match="escapes"):
        resolve_local_reference(root, "../outside.txt")


def test_chinese_lineage_design_example_parses_and_smoke_run_stays_legacy():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    document = (root / "docs/design/LINEAGE_DESIGN_zh.md").read_text(encoding="utf-8")
    example = re.search(r"```json\s*(\{.*?\})\s*```", document, re.DOTALL)
    assert example
    RunRecord.model_validate_json(example.group(1))
    from onegpu_posttrain.lineage import EvaluationLineage
    evaluation_example = next(
        block for block in re.findall(r"```json\s*(\{.*?\})\s*```", document, re.DOTALL)
        if '"eval_id"' in block
    )
    evaluation = EvaluationLineage.model_validate_json(evaluation_example)
    assert evaluation.model_id == "model-sft-mini-9c44c15e1a2b"
    assert evaluation.dataset_ids == ["benchmark-gsm8k", "ds-math-holdout-v1"]
    smoke = root / "runs/smoke/simpo-original-smoke-s42/run.json"
    if smoke.is_file():
        assert load_versioned_payload(smoke, legacy_keys={"run_id"})[1] == "legacy"
