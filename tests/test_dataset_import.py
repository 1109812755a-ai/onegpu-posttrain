import hashlib
import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
import yaml

from onegpu_posttrain.data.manifest import validate_manifest_evidence
from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.lineage_validation import validate_lineage_root


def _origin(**updates):
    values = {
        "source_id": "upload-pending",
        "kind": "local_upload",
        "name": "用户上传",
        "revision_status": "unknown",
        "license_status": "unknown",
        "declared_by": "user",
    }
    values.update(updates)
    return DataSource(**values)


def _sft_jsonl(*, reverse_keys=False):
    record = {
        "id": "example-1",
        "raw_problem": "1+1?",
        "messages": [
            {"role": "user", "content": "1+1?"},
            {"role": "assistant", "content": "2"},
        ],
        "prompt_template_id": "chat-v1",
    }
    if reverse_keys:
        record = dict(reversed(list(record.items())))
    return (json.dumps(record, ensure_ascii=False) + "\n").encode()


def _stage_jsonl(stage):
    records = {
        "cpt": {"id": "cpt-1", "text": "A short source document.", "source": "upload"},
        "sft": json.loads(_sft_jsonl()),
        "preference": {
            "id": "pref-1", "raw_problem": "Choose the correct answer.",
            "prompt": [{"role": "user", "content": "Choose the correct answer."}],
            "chosen": "The correct choice.", "rejected": "An incorrect choice.",
            "prompt_template_id": "chat-v1",
        },
        "grpo": {
            "id": "grpo-1", "raw_problem": "What is 1+1?",
            "prompt": [{"role": "user", "content": "What is 1+1?"}],
            "answer": "2", "prompt_template_id": "chat-v1",
        },
    }
    return (json.dumps(records[stage], ensure_ascii=False) + "\n").encode()


def test_registration_records_raw_hash_and_deterministic_canonical_hash(tmp_path):
    raw = _sft_jsonl(reverse_keys=True)
    result = register_jsonl(
        BytesIO(raw), workspace=tmp_path, output_root=tmp_path / "data" / "processed" / "user-upload",
        stage="sft", origin=_origin(),
    )

    assert result.original_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.canonical_sha256 == hashlib.sha256((result.path / "canonical" / "sft.jsonl").read_bytes()).hexdigest()
    assert result.dataset_id == f"ds-sft-upload-{result.canonical_sha256[:20]}"
    metadata = json.loads((result.path / "registration.json").read_text(encoding="utf-8"))
    assert metadata["source"]["content_sha256"] == result.original_sha256
    assert metadata["source"]["license_status"] == "unknown"
    assert metadata["source"]["revision_status"] == "unknown"


def test_registration_reuses_same_normalized_content_without_overwriting(tmp_path):
    output_root = tmp_path / "data" / "processed" / "user-upload"
    first = register_jsonl(
        BytesIO(_sft_jsonl()), workspace=tmp_path, output_root=output_root,
        stage="sft", origin=_origin(name="first"),
    )
    marker = first.path / "user-marker.txt"
    marker.write_text("keep", encoding="utf-8")
    second = register_jsonl(
        BytesIO(_sft_jsonl(reverse_keys=True)), workspace=tmp_path, output_root=output_root,
        stage="sft", origin=_origin(name="second"),
    )

    assert first.dataset_id == second.dataset_id
    assert second.reused is True
    assert marker.read_text(encoding="utf-8") == "keep"
    assert len(list(output_root.glob("ds-*"))) == 1


def test_invalid_import_leaves_no_registered_dataset(tmp_path):
    output_root = tmp_path / "data" / "processed" / "user-upload"
    with pytest.raises(ValueError, match="schema_validation"):
        register_jsonl(
            BytesIO(b'{"id":"missing-required-fields"}\n'), workspace=tmp_path,
            output_root=output_root, stage="sft", origin=_origin(),
        )

    assert list(output_root.iterdir()) == []


def test_output_root_must_stay_within_workspace(tmp_path):
    with pytest.raises(ValueError, match="escapes workspace"):
        register_jsonl(
            BytesIO(_sft_jsonl()), workspace=tmp_path, output_root=tmp_path.parent / "outside",
            stage="sft", origin=_origin(),
        )


def test_registration_requires_upload_source_kind(tmp_path):
    with pytest.raises(ValueError, match="local_upload"):
        register_jsonl(
            BytesIO(_sft_jsonl()), workspace=tmp_path, output_root=tmp_path / "data",
            stage="sft", origin=_origin(kind="model_generated"),
        )


def test_failed_commit_cleans_temporary_staging(tmp_path, monkeypatch):
    import onegpu_posttrain.data.dataset_import as dataset_import

    output_root = tmp_path / "data" / "processed" / "user-upload"

    def fail_replace(*_args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(dataset_import.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated disk failure"):
        register_jsonl(
            BytesIO(_sft_jsonl()), workspace=tmp_path, output_root=output_root,
            stage="sft", origin=_origin(),
        )

    assert list(output_root.iterdir()) == []


def test_symlinked_output_root_is_rejected_when_supported(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace / "linked-output"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(ValueError, match="symbolic links"):
        register_jsonl(
            BytesIO(_sft_jsonl()), workspace=workspace, output_root=link,
            stage="sft", origin=_origin(),
        )


@pytest.mark.parametrize("stage", ["cpt", "sft", "preference", "grpo"])
def test_registered_import_has_complete_manifest_quality_and_lineage(tmp_path, stage):
    output_root = tmp_path / "data" / "processed" / "user-upload"
    result = register_jsonl(
        BytesIO(_stage_jsonl(stage)), workspace=tmp_path, output_root=output_root,
        stage=stage, origin=_origin(name=f"{stage} upload"),
    )
    manifest_path = result.path / "manifests" / f"{stage}-user-upload-v1.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    quality_path = result.path / "reports" / "quality" / f"{stage}.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))

    assert validate_manifest_evidence(manifest_path) == []
    assert manifest["dataset_id"] == result.dataset_id
    assert manifest["lineage"]["content_sha256"] == result.canonical_sha256
    assert quality["run_id"]
    assert quality["gate"]["summary_status"] == "NOT_EVALUATED"
    assert quality["metrics"]["train_eval_overlap"]["status"] == "not_measured"
    if stage == "cpt":
        canonical = json.loads((result.path / "canonical/cpt.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert "data_role" not in canonical.get("metadata", {})
    lineage_result = validate_lineage_root(result.path, require_complete=True)
    assert lineage_result["valid"], lineage_result
    child_env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    existing_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_root, existing_pythonpath) if item
    )
    cli_result = subprocess.run(
        [sys.executable, "-m", "onegpu_posttrain", "validate-lineage", "--root", str(result.path), "--require-complete"],
        capture_output=True, text=True, check=False, env=child_env,
    )
    assert cli_result.returncode == 0, cli_result.stdout + cli_result.stderr
