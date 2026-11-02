import hashlib
import json

import pytest

from onegpu_posttrain.adapters.registered_dataset import adapt_registered_dataset
from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.lineage_validation import validate_lineage_root


def _origin():
    return DataSource(source_id="pending", kind="local_upload", name="fixture upload",
                      license_status="unknown", revision_status="unknown", declared_by="user")


def _record(stage):
    row = {
        "cpt": {"id": "cpt-id", "text": "source text", "source": "fixture", "metadata": {"source_row_id": "raw-row-1"}},
        "sft": {"id": "sft-id", "raw_problem": "question", "messages": [
            {"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}],
            "prompt_template_id": "chat-v1", "metadata": {"source_record_id": "source-sft-1", "problem_id": "p1"}},
        "preference": {"id": "pref-id", "raw_problem": "question", "prompt": [
            {"role": "user", "content": "question"}], "chosen": "good", "rejected": "bad",
            "prompt_template_id": "chat-v1", "metadata": {"parent_record_ids": ["chosen-1", "rejected-1"]}},
        "grpo": {"id": "grpo-id", "raw_problem": "question", "prompt": [
            {"role": "user", "content": "question"}], "answer": "4", "prompt_template_id": "chat-v1",
            "metadata": {"problem_id": "p1", "source_uuid": "uuid-1"}},
    }[stage]
    return (json.dumps(row, ensure_ascii=False) + "\n").encode()


@pytest.mark.parametrize("stage", ["cpt", "sft", "preference", "grpo"])
def test_registered_adapter_creates_trainer_file_and_record_lineage(tmp_path, stage):
    registered_root = tmp_path / "data" / "processed" / "user-upload"
    registered = register_jsonl(
        _stream(_record(stage)), workspace=tmp_path, output_root=registered_root,
        stage=stage, origin=_origin(),
    )
    manifest = registered.path / "manifests" / f"{stage}-user-upload-v1.yaml"
    source_files = {path: path.read_bytes() for path in registered.path.rglob("*") if path.is_file()}

    result = adapt_registered_dataset(manifest, tmp_path / "data" / "adapted" / registered.dataset_id)

    native_path = tmp_path / "data" / "adapted" / registered.dataset_id / result["native_path"]
    provenance_path = tmp_path / "data" / "adapted" / registered.dataset_id / result["provenance_path"]
    native = json.loads(native_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert result["dataset_id"] == registered.dataset_id
    assert result["canonical_sha256"] == registered.canonical_sha256
    assert result["record_count"] == 1
    assert result["native_sha256"] == hashlib.sha256(native_path.read_bytes()).hexdigest()
    from onegpu_posttrain.cli import _dataset_lineage_ref
    dataset_ref, lineage_complete, lineage_warnings = _dataset_lineage_ref(native_path)
    assert dataset_ref.id == registered.dataset_id
    assert lineage_complete is True
    assert lineage_warnings == []
    assert provenance["dataset_id"] == registered.dataset_id
    expected_ids = {"cpt": "cpt-id", "sft": "sft-id", "preference": "pref-id", "grpo": "grpo-id"}
    assert provenance["canonical_record_id"] == expected_ids[stage]
    assert "question" not in json.dumps(provenance)
    assert "source_record_id" in provenance["metadata"] or "source_row_id" in provenance["metadata"] or "problem_id" in provenance["metadata"] or "parent_record_ids" in provenance["metadata"]
    assert {path: path.read_bytes() for path in registered.path.rglob("*") if path.is_file()} == source_files
    lineage = validate_lineage_root(tmp_path, require_complete=True)
    assert lineage["valid"], lineage


def _stream(raw):
    from io import BytesIO
    return BytesIO(raw)


def test_registered_adapter_refuses_tampered_canonical_content(tmp_path):
    registered = register_jsonl(
        _stream(_record("sft")), workspace=tmp_path, output_root=tmp_path / "data/processed/user-upload",
        stage="sft", origin=_origin(),
    )
    canonical = registered.path / "canonical/sft.jsonl"
    canonical.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical data hash"):
        adapt_registered_dataset(
            registered.path / "manifests/sft-user-upload-v1.yaml", tmp_path / "native-output",
        )
    assert not (tmp_path / "native-output").exists()


def test_registered_adapter_refuses_to_overwrite_existing_output(tmp_path):
    registered = register_jsonl(
        _stream(_record("sft")), workspace=tmp_path, output_root=tmp_path / "data/processed/user-upload",
        stage="sft", origin=_origin(),
    )
    output = tmp_path / "native-output"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        adapt_registered_dataset(registered.path / "manifests/sft-user-upload-v1.yaml", output)
    assert marker.read_text(encoding="utf-8") == "keep"
