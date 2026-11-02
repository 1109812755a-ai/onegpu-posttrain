from __future__ import annotations

import json

import pytest

from onegpu_posttrain.data.evidence import ReportReference
from onegpu_posttrain.data.manifest import DatasetManifest, load_manifest, save_manifest
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.data.manifest import sha256_file
from onegpu_posttrain.ui.dataset_preview import PreviewUnavailable, preview_dataset

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from onegpu_posttrain.ui.server import create_app


def _manifest(workspace, *, dataset_id, stage="sft", kind="local_upload", content=None,
              relative_root="data/processed"):
    root = workspace.joinpath(*relative_root.split("/"), dataset_id)
    manifests = root / "manifests"
    manifests.mkdir(parents=True)
    canonical = root / "canonical" / f"{stage}.jsonl"
    canonical.parent.mkdir()
    default_rows = {
        "cpt": {"id": f"{dataset_id}-1", "text": "Fractions describe equal parts.", "source": "fixture",
                "metadata": {"private_note": "do-not-return"}},
        "sft": {"id": f"{dataset_id}-1", "raw_problem": "2 + 2?", "messages": [
            {"role": "user", "content": "2 + 2?"}, {"role": "assistant", "content": "4"}],
            "prompt_template_id": "chat-v1", "metadata": {"private_note": "do-not-return"}},
        "preference": {"id": f"{dataset_id}-1", "raw_problem": "2 + 2?", "prompt": [
            {"role": "user", "content": "2 + 2?"}], "chosen": "4", "rejected": "5",
            "prompt_template_id": "chat-v1", "metadata": {"private_note": "do-not-return"}},
        "grpo": {"id": f"{dataset_id}-1", "raw_problem": "2 + 2?", "prompt": [
            {"role": "user", "content": "2 + 2?"}], "answer": "4", "prompt_template_id": "chat-v1",
            "metadata": {"private_note": "do-not-return"}},
    }
    default_row = default_rows[stage]
    rows = content if isinstance(content, list) else [content or default_row]
    canonical.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    digest = sha256_file(canonical)
    source = DataSource(source_id=f"source-{dataset_id}", kind=kind, name=f"{kind} fixture", declared_by="user")
    processing = None
    metadata = {"canonical_path": f"canonical/{stage}.jsonl"}
    if kind != "local_upload":
        report = {"build_id": f"build-{dataset_id}", "outputs": {stage: {
            "dataset_id": dataset_id, "path": f"canonical/{stage}.jsonl", "content_sha256": digest,
        }}}
        report_path = root / "reports" / "processing_report.json"
        report_path.parent.mkdir()
        report_path.write_text(json.dumps(report), encoding="utf-8")
        processing = ReportReference(path="../reports/processing_report.json", sha256=sha256_file(report_path),
                                     report_id=report["build_id"])
    manifest = DatasetManifest(dataset_id=dataset_id, stage=stage, origin=source, sample_count=len(rows),
                               hashes={"canonical_records_sha256": digest}, processing_report=processing,
                               metadata=metadata)
    path = manifests / f"{stage}-v1.yaml"
    save_manifest(path, manifest)
    return path, canonical


@pytest.mark.parametrize("kind", ["local_upload", "open_source", "model_generated"])
def test_preview_resolves_registered_canonical_for_each_source_kind(tmp_path, kind):
    _manifest(tmp_path, dataset_id=f"ds-{kind}", kind=kind)

    result = preview_dataset(tmp_path, f"ds-{kind}")

    assert result["available"] is True
    assert result["integrity"] == "verified"
    assert result["stage"] == "sft"
    assert result["total_records"] == 1
    assert result["samples"][0]["fields"]["messages"][1]["content"] == "4"
    assert "metadata" not in result["samples"][0]["fields"]
    assert "private_note" not in json.dumps(result)


@pytest.mark.parametrize("stage,expected_fields", [
    ("cpt", {"text"}),
    ("sft", {"raw_problem", "messages", "prompt_template_id"}),
    ("preference", {"raw_problem", "prompt", "chosen", "rejected", "prompt_template_id"}),
    ("grpo", {"raw_problem", "prompt", "answer", "prompt_template_id"}),
])
def test_preview_api_uses_the_declared_stage_field_allowlist(tmp_path, stage, expected_fields):
    _manifest(tmp_path, dataset_id=f"ds-{stage}", stage=stage)

    with TestClient(create_app(tmp_path)) as client:
        response = client.get(f"/api/data/datasets/ds-{stage}/preview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["stage"] == stage
    assert set(payload["samples"][0]["fields"]) == expected_fields


def test_preview_rejects_manifest_path_traversal(tmp_path):
    manifest_path, _ = _manifest(tmp_path, dataset_id="ds-path")
    manifest = load_manifest(manifest_path)
    manifest.metadata["canonical_path"] = "../../outside.jsonl"
    with pytest.raises(ValueError, match="canonical_path"):
        save_manifest(manifest_path, manifest)
    # Corrupt legacy fixtures are written directly to exercise the read-side fail-closed check.
    import yaml
    manifest_path.write_text(yaml.safe_dump(manifest.model_dump(mode="json")), encoding="utf-8")

    with pytest.raises(PreviewUnavailable, match="路径") as error:
        preview_dataset(tmp_path, "ds-path")
    assert error.value.code == "canonical_path_invalid"


def test_preview_rejects_inconsistent_duplicate_dataset_identity(tmp_path):
    _manifest(tmp_path, dataset_id="ds-duplicate")
    _manifest(tmp_path, dataset_id="ds-duplicate", relative_root="artifacts/synthetic", content={
        "id": "different", "raw_problem": "3 + 3?", "messages": [
            {"role": "user", "content": "3 + 3?"}, {"role": "assistant", "content": "6"}],
        "prompt_template_id": "chat-v1",
    })

    with pytest.raises(PreviewUnavailable) as error:
        preview_dataset(tmp_path, "ds-duplicate")
    assert error.value.code == "identity_conflict"


def test_preview_reports_legacy_manifest_without_canonical_evidence(tmp_path):
    root = tmp_path / "data/processed/legacy"
    path = root / "manifests/sft-v1.yaml"
    path.parent.mkdir(parents=True)
    save_manifest(path, DatasetManifest(dataset_id="legacy-dataset", stage="sft", sample_count=1,
                                        hashes={"canonical_records_sha256": "a" * 64}), allow_legacy=True)

    with pytest.raises(PreviewUnavailable) as error:
        preview_dataset(tmp_path, "legacy-dataset")
    assert error.value.code == "evidence_missing"


def test_preview_does_not_follow_symlink_outside_artifact_root(tmp_path):
    manifest_path, _ = _manifest(tmp_path, dataset_id="ds-symlink")
    artifact_root = manifest_path.parent.parent
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    link = artifact_root / "canonical" / "external.jsonl"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this Windows environment")
    manifest = load_manifest(manifest_path)
    manifest.metadata["canonical_path"] = "canonical/external.jsonl"
    import yaml
    manifest_path.write_text(yaml.safe_dump(manifest.model_dump(mode="json")), encoding="utf-8")

    with pytest.raises(PreviewUnavailable) as error:
        preview_dataset(tmp_path, "ds-symlink")
    assert error.value.code == "canonical_path_invalid"


def test_preview_api_is_bounded_allowlisted_and_read_only(tmp_path):
    long_problem = "x" * 2500
    manifest_path, canonical = _manifest(tmp_path, dataset_id="ds-api", content=[
        {"id": f"record-{index}", "raw_problem": long_problem if index == 0 else f"problem-{index}",
         "messages": [{"role": "user", "content": "question"},
                      {"role": "assistant", "content": f"answer-{index}"}],
         "prompt_template_id": "chat-v1", "metadata": {"private_note": "do-not-return"}}
        for index in range(12)
    ])
    before = {path: sha256_file(path) for path in (manifest_path, canonical)}

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/data/datasets/ds-api/preview")
        assert response.status_code == 200
        payload = response.json()
        assert payload["returned_records"] == 3
        assert payload["total_records"] == 12
        assert payload["truncated"] is True
        assert payload["integrity"] == "verified"
        assert payload["samples"][0]["fields"]["raw_problem"].endswith("…")
        assert "raw_problem" in payload["samples"][0]["truncated_fields"][0]
        assert "metadata" not in json.dumps(payload)
        assert client.get("/api/data/datasets/ds-api/preview?limit=10").json()["returned_records"] == 10
        assert client.get("/api/data/datasets/ds-api/preview?limit=11").status_code == 422
        assert client.get("/api/data/datasets/ds-api/preview?limit=0").status_code == 422
        assert client.get("/api/data/datasets/not-registered/preview").status_code == 404

    assert before == {path: sha256_file(path) for path in before}


def test_preview_api_suppresses_sample_on_canonical_digest_mismatch(tmp_path):
    _, canonical = _manifest(tmp_path, dataset_id="ds-tampered")
    canonical.write_text(canonical.read_text(encoding="utf-8").replace('"4"', '"tampered"'), encoding="utf-8")

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/data/datasets/ds-tampered/preview")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "digest_mismatch"
    assert "samples" not in response.text


def test_dataset_status_is_stable_read_only_and_exposes_missing_evidence_not_samples(tmp_path):
    manifest_path, canonical = _manifest(tmp_path, dataset_id="ds-status")
    before = (manifest_path.read_bytes(), canonical.read_bytes())
    with TestClient(create_app(tmp_path)) as client:
        first = client.get("/api/data/datasets/ds-status/status")
        second = client.get("/api/data/datasets/ds-status/status")
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        payload = first.json()
        assert payload["schema_version"] == "dataset-status-v1"
        assert payload["read_only"] is True
        assert payload["integrity"] == "verified"
        assert "quality_report" in payload["missing_evidence"]
        assert "dataset_lineage" in payload["missing_evidence"]
        assert "samples" not in payload
        assert "raw_problem" not in first.text
    assert (manifest_path.read_bytes(), canonical.read_bytes()) == before
