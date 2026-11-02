from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from onegpu_posttrain.adapters.ms_swift import adapt_sft
from onegpu_posttrain.data.calibration_registration import register_sft_calibration_candidate
from onegpu_posttrain.data.manifest import load_manifest, validate_manifest_evidence
from onegpu_posttrain.data.sft_derivation import derive_sft_length_subset
from onegpu_posttrain.lineage_validation import validate_lineage_root
from onegpu_posttrain.cli import (
    _dataset_lineage_ref,
    _model_lineage_ref,
    _policy_training_context,
    _write_training_run,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(tmp_path: Path) -> Path:
    revision = "a1b2c3d4"
    source = tmp_path / "hf" / "snapshots" / revision / "data"
    source.mkdir(parents=True)
    artifact = tmp_path / "length-1536"
    (artifact / "canonical").mkdir(parents=True)
    (artifact / "native").mkdir()
    canonical = artifact / "canonical" / "sft.jsonl"
    native = artifact / "native" / "sft.jsonl"
    rows = [
        {"id": "row-1", "raw_problem": "1+1?", "messages": [
            {"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"}],
         "prompt_template_id": "qwen3-math-v1", "metadata": {"rendered_tokens": 800, "source_record_id": "p1"}},
        {"id": "row-2", "raw_problem": "2+2?", "messages": [
            {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}],
         "prompt_template_id": "qwen3-math-v1", "metadata": {"rendered_tokens": 1200, "source_record_id": "p2"}},
    ]
    canonical.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    native_rows = [adapt_sft(row) for row in rows]
    native.write_text("".join(json.dumps(row) + "\n" for row in native_rows), encoding="utf-8")
    payload = {
        "schema_version": "1.1",
        "calibration_id": "sft-length-test-s42-exact",
        "dataset_id": "ds-sft-length-test-42-exact",
        "status": "exploratory",
        "stage": "sft",
        "source": {
            "name": "zwhe99/DeepMath-103K",
            "local_path": str(source),
            "revision": revision,
            "snapshot_sha256": "b" * 64,
        },
        "tokenizer": {
            "repo": "Qwen/Qwen3-0.6B-Base",
            "revision": "c3d4e5f6",
            "chat_template_sha256": "c" * 64,
        },
        "sampling": {"seed": 42, "source_row_ids_sha256": "d" * 64},
        "eligibility": {"max_rendered_tokens": 1536, "answer_verification_mode": "exact"},
        "counts": {"verification_selected_training_examples": 2},
        "hashes": {"canonical_sha256": _sha(canonical), "native_sha256": _sha(native)},
        "paths": {"canonical": "canonical/sft.jsonl", "native": "native/sft.jsonl"},
        "quality_evidence": {"quality_approval": "not_evaluated"},
    }
    manifest = artifact / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


def test_register_calibration_emits_complete_dataset_and_build_lineage(tmp_path):
    calibration = _candidate(tmp_path)
    canonical_before = _sha(calibration.parent / "canonical" / "sft.jsonl")
    native_before = _sha(calibration.parent / "native" / "sft.jsonl")

    result = register_sft_calibration_candidate(calibration)

    registered = Path(result["manifest"])
    dataset = load_manifest(registered)
    assert result["status"] == "registered"
    assert dataset.dataset_id == "ds-sft-length-test-42-exact"
    assert dataset.sample_count == 2
    assert dataset.lineage is not None
    assert dataset.lineage.produced_by_run_id == result["run_id"]
    assert dataset.lineage.record_provenance == "dataset_only"
    assert dataset.metadata["usage"] == "exploratory"
    assert dataset.metadata["quality_evidence"]["quality_approval"] == "not_evaluated"
    assert validate_manifest_evidence(registered) == []
    assert _sha(calibration.parent / "canonical" / "sft.jsonl") == canonical_before
    assert _sha(calibration.parent / "native" / "sft.jsonl") == native_before

    validation = validate_lineage_root(calibration.parent, require_complete=True)
    assert validation["valid"] is True
    assert validation["summary"]["errors"] == 0


def test_register_calibration_is_idempotent_for_same_identity_and_content(tmp_path):
    calibration = _candidate(tmp_path)
    first = register_sft_calibration_candidate(calibration)
    second = register_sft_calibration_candidate(calibration)

    assert first["status"] == "registered"
    assert second["status"] == "reused"
    assert second["run_id"] == first["run_id"]


def test_register_calibration_fails_closed_on_native_digest_mismatch(tmp_path):
    calibration = _candidate(tmp_path)
    native = calibration.parent / "native" / "sft.jsonl"
    native.write_text(native.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="native_sha256 mismatch"):
        register_sft_calibration_candidate(calibration)
    assert not (calibration.parent / "manifests" / "sft-calibration-v1.yaml").exists()


def test_register_calibration_fails_closed_on_source_revision_mismatch(tmp_path):
    calibration = _candidate(tmp_path)
    payload = json.loads(calibration.read_text(encoding="utf-8"))
    payload["source"]["revision"] = "wrong-revision"
    calibration.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source revision mismatch"):
        register_sft_calibration_candidate(calibration)
    assert not (calibration.parent / "manifests" / "sft-calibration-v1.yaml").exists()


def test_training_lineage_resolves_native_bytes_to_registered_dataset_id(tmp_path):
    calibration = _candidate(tmp_path)
    register_sft_calibration_candidate(calibration)
    native = calibration.parent / "native" / "sft.jsonl"

    reference, complete, warnings = _dataset_lineage_ref(native)

    assert complete is True
    assert reference.kind == "dataset"
    assert reference.id == "ds-sft-length-test-42-exact"
    assert reference.digest is not None
    assert reference.digest.value == _sha(calibration.parent / "canonical" / "sft.jsonl")
    assert warnings == []


def test_training_lineage_rejects_native_bytes_changed_after_registration(tmp_path):
    calibration = _candidate(tmp_path)
    register_sft_calibration_candidate(calibration)
    native = calibration.parent / "native" / "sft.jsonl"
    native.write_text(native.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset content does not match its manifest"):
        _dataset_lineage_ref(native)


def test_calibration_candidate_preflight_preserves_exploratory_limits(tmp_path):
    calibration = _candidate(tmp_path)
    register_sft_calibration_candidate(calibration)
    native = calibration.parent / "native" / "sft.jsonl"

    context = _policy_training_context(native, {"max_length": 1536}, "sft")

    assert context["usage"] == "exploratory"
    assert context["official_mini_acceptance"] == "not_assessed"
    assert context.get("policy_id") is None
    assert "语义推理正确性仍未评估" in context["warnings"][0]


def test_calibration_candidate_preflight_rejects_implicit_truncation(tmp_path):
    calibration = _candidate(tmp_path)
    register_sft_calibration_candidate(calibration)
    native = calibration.parent / "native" / "sft.jsonl"

    with pytest.raises(ValueError, match="不能静默截断"):
        _policy_training_context(native, {"max_length": 1024}, "sft")


def test_training_run_records_calibration_context_without_fake_policy_reference(tmp_path):
    calibration = _candidate(tmp_path)
    register_sft_calibration_candidate(calibration)
    native = calibration.parent / "native" / "sft.jsonl"
    context = _policy_training_context(native, {"max_length": 1536}, "sft")
    model = tmp_path / "parent-model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()

    _write_training_run(
        output=output,
        result={"train": {"status": "PASS"}, "data_policy_context": context},
        run_id="run-test-calibration-sft-123456789abc",
        run_label="test-calibration-sft",
        experiment_id="test-calibration",
        stage="sft",
        model=model,
        dataset=native,
        config_path="recipe.yaml",
        resolved_config={"training": {"max_length": 1536}},
        started_at="2026-10-20T00:00:00+00:00",
    )

    run = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    assert run["lineage_status"] == "complete"
    assert any(item["id"] == "ds-sft-length-test-42-exact" for item in run["inputs"])
    assert not any(item.get("role") == "data_policy" for item in run["inputs"])
    assert "语义推理正确性仍未评估" in run["lineage_warnings"][0]


def test_derive_shorter_cohort_records_parent_filter_counts_and_lineage(tmp_path):
    calibration = _candidate(tmp_path)
    register_sft_calibration_candidate(calibration)
    parent_manifest = calibration.parent / "manifests" / "sft-calibration-v1.yaml"
    output = tmp_path / "derived-1024"

    result = derive_sft_length_subset(parent_manifest, max_length=1024, output=output)

    child_manifest_path = Path(result["manifest"])
    child = load_manifest(child_manifest_path)
    processing = json.loads(Path(result["processing_report"]).read_text(encoding="utf-8"))
    assert result["parent_count"] == 2
    assert result["sample_count"] == 1
    assert result["length_rejected"] == 1
    assert child.lineage is not None
    assert child.lineage.parents == ["ds-sft-length-test-42-exact"]
    assert child.lineage.record_provenance == "record"
    assert processing["processing_steps"][0]["rejected_count"] == 1
    assert processing["processing_steps"][0]["first_rejection_reasons"] == {
        "rendered_tokens_exceeds_limit": 1
    }
    reference, complete, warnings = _dataset_lineage_ref(result["native"])
    assert complete is True
    assert reference.id == result["dataset_id"]
    assert warnings == []
    context = _policy_training_context(Path(result["native"]), {"max_length": 1024}, "sft")
    assert context["actual_record_count"] == 1
    assert context["official_mini_acceptance"] == "not_assessed"

    validation = validate_lineage_root(tmp_path, require_complete=True)
    assert validation["valid"] is True
    assert validation["summary"]["errors"] == 0


def test_derive_shorter_cohort_fails_on_parent_digest_mismatch(tmp_path):
    calibration = _candidate(tmp_path)
    register_sft_calibration_candidate(calibration)
    parent_manifest = calibration.parent / "manifests" / "sft-calibration-v1.yaml"
    canonical = calibration.parent / "canonical" / "sft.jsonl"
    canonical.write_text(canonical.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="parent canonical dataset digest mismatch"):
        derive_sft_length_subset(parent_manifest, max_length=1024, output=tmp_path / "derived-1024")
    assert not (tmp_path / "derived-1024").exists()
