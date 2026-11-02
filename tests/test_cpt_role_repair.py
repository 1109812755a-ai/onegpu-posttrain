from __future__ import annotations

import hashlib
import io
import json

import pytest

from onegpu_posttrain.data.build import build_finemath_artifacts
from onegpu_posttrain.data.cpt_corpus import validate_cpt_corpus_jsonl
from onegpu_posttrain.data.cpt_role_repair import inspect_legacy_cpt_role_repair, repair_legacy_cpt_roles
from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.manifest import load_manifest, save_manifest, validate_manifest_evidence
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.data.finemath import FineMathBuildConfig
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.cli import build_parser
from onegpu_posttrain.data.cpt_training import inspect_registered_cpt_dataset
from onegpu_posttrain.ui.cpt_stage_adapter import CPTStageAdapter
from onegpu_posttrain.ui.server import create_app
from fastapi.testclient import TestClient


def _rows():
    return [
        {"url": f"https://fixture.invalid/{i}", "text": f"Legacy CPT mathematical exposition {i}." * 8}
        for i in range(6)
    ]


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _legacy_finemath_artifact(root):
    build_finemath_artifacts(
        _rows(), output_dir=root, tokenizer=WhitespaceTokenizer(),
        config=FineMathBuildConfig(train_token_budget=200, holdout_token_budget=0,
                                   max_seq_tokens=32, holdout_permille=0),
        source_repo="fixture-finemath", source_config=None, source_revision="revision-1",
        tokenizer_repo="fixture-tokenizer", tokenizer_revision="tokenizer-v1",
    )
    manifest_path = root / "manifests/cpt-smoke-v1.yaml"
    manifest = load_manifest(manifest_path)
    canonical = root / manifest.metadata["canonical_path"]
    records = [json.loads(line) for line in canonical.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["metadata"].pop("data_role", None)
    canonical.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
                                  for row in records), encoding="utf-8")
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    manifest.hashes["canonical_records_sha256"] = digest
    manifest.lineage.content_sha256 = digest
    manifest.metadata["build_protocol"] = "finemath-build-v2"
    manifest.protocol_versions.pop("data_role", None)

    report_path = root / "reports/processing_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["processing_protocol"] = "finemath-build-v2"
    report["outputs"]["cpt"]["content_sha256"] = digest
    report["outputs"]["cpt"].pop("data_role_counts", None)
    _write_json(report_path, report)
    manifest.processing_report.sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()

    quality_path = root / manifest.quality_report.path.replace("../", "")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["content_sha256"] = digest
    _write_json(quality_path, quality)
    manifest.quality_report.sha256 = hashlib.sha256(quality_path.read_bytes()).hexdigest()

    build_run_path = root / "runs" / manifest.lineage.produced_by_run_id / "lineage.json"
    build_run = json.loads(build_run_path.read_text(encoding="utf-8"))
    output_ref = next(item for item in build_run["outputs"] if item["id"] == manifest.dataset_id)
    output_ref["digest"]["value"] = digest
    _write_json(build_run_path, build_run)
    save_manifest(manifest_path, manifest)
    assert validate_manifest_evidence(manifest_path) == []
    return manifest_path


def test_legacy_repair_preview_is_read_only_and_requires_verified_evidence(tmp_path):
    manifest_path = _legacy_finemath_artifact(tmp_path / "legacy")
    files = [manifest_path, manifest_path.parent.parent / "canonical/cpt_train.jsonl",
             manifest_path.parent.parent / "reports/processing_report.json",
             manifest_path.parent.parent / "reports/quality/cpt.json"]
    before = {path: path.read_bytes() for path in files}

    preview = inspect_legacy_cpt_role_repair(manifest_path)

    assert preview["status"] == "eligible"
    assert preview["role_assignment"]["data_role"] == "train_seed"
    assert preview["role_assignment"]["record_count"] == load_manifest(manifest_path).sample_count
    assert preview["execution_started"] is False
    assert {path: path.read_bytes() for path in files} == before

    report = manifest_path.parent.parent / "reports/processing_report.json"
    report.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch|digest"):
        inspect_legacy_cpt_role_repair(manifest_path)


def test_repair_cli_defaults_to_read_only_preview(tmp_path, capsys):
    workspace = tmp_path.resolve()
    manifest_path = _legacy_finemath_artifact(workspace / "legacy")
    args = build_parser().parse_args([
        "repair-cpt-roles", "--workspace", str(workspace), "--manifest", str(manifest_path),
    ])

    assert args.apply is False
    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["read_only"] is True
    assert not (workspace / "data/processed/cpt-role-repair").exists()


def test_legacy_repair_registers_immutable_child_with_parent_lineage(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    manifest_path = _legacy_finemath_artifact(workspace / "legacy")
    root = manifest_path.parent.parent
    old_files = [manifest_path, root / "canonical/cpt_train.jsonl", root / "reports/processing_report.json",
                 root / "reports/quality/cpt.json"]
    before = {path: path.read_bytes() for path in old_files}

    preview = repair_legacy_cpt_roles(manifest_path, workspace=workspace,
                                      output_root=workspace / "data/processed/cpt-role-repair")
    assert preview["status"] == "eligible"
    assert not (workspace / "data/processed/cpt-role-repair").exists()

    result = repair_legacy_cpt_roles(manifest_path, workspace=workspace,
                                     output_root=workspace / "data/processed/cpt-role-repair", apply=True)

    assert result["status"] == "registered"
    assert result["dataset_id"] != result["parent_dataset_id"]
    assert result["old_artifacts_modified"] is False
    assert {path: path.read_bytes() for path in old_files} == before
    child_manifest_path = workspace / result["manifest_path"]
    child = load_manifest(child_manifest_path)
    assert child.lineage.parents == [result["parent_dataset_id"]]
    assert child.metadata["role_source_protocol"] == "finemath-build-v2"
    assert child.metadata["data_role_counts"] == {"train_seed": child.sample_count}
    assert validate_manifest_evidence(child_manifest_path) == []
    with (child_manifest_path.parent.parent / child.metadata["canonical_path"]).open("rb") as stream:
        assert validate_cpt_corpus_jsonl(stream, expected_role="train_seed").valid
    with pytest.raises(ValueError, match="role/schema invalid"):
        inspect_registered_cpt_dataset(manifest_path)

    recipe_path = workspace / "recipes/smoke/10-peft-cpt.yaml"
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(
        "stage: peft_cpt\nmodel_config: configs/model/qwen3-0.6b-base.yaml\n"
        "training:\n  tuner_type: lora\n  max_length: 1024\n  max_steps: 1\n"
        "  learning_rate: 0.0001\n  gradient_accumulation_steps: 8\n", encoding="utf-8")
    model_root = workspace / "models/base"
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (model_root / "model.safetensors").write_bytes(b"fixture model weights")
    model_config_path = workspace / "configs/model/qwen3-0.6b-base.yaml"
    model_config_path.parent.mkdir(parents=True)
    model_config_path.write_text(
        f"model:\n  repo: {model_root.as_posix()}\n  revision: fixture-revision\n", encoding="utf-8")
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter.console_command_available", lambda _: True)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter._gpu_snapshot", lambda: {"available": False})
    direct = CPTStageAdapter(workspace).preflight(result["dataset_id"])
    app = create_app(workspace, enable_experiment_launcher=True)
    with TestClient(app) as client:
        response = client.post("/api/training/stage/preflight", headers={
            "X-Local-Launcher-Token": app.state.experiment_launcher_token,
        }, json={"stage": "cpt", "profile_id": "cpt-smoke-v1", "dataset_ids": [result["dataset_id"]]})
    assert direct["status"] == "ready_for_cli"
    assert next(item for item in direct["checks"] if item["check_id"] == "dataset:registered_cpt")["status"] == "pass"
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "ready_for_confirmation"
    assert response.json()["execution_started"] is False
    assert not (workspace / "runs").exists()


def test_uploaded_cpt_without_trusted_build_evidence_is_never_repaired(tmp_path):
    workspace = tmp_path.resolve()
    row = {"id": "upload-1", "text": "Uploaded corpus row.", "source": "user"}
    registration = register_jsonl(
        io.BytesIO((json.dumps(row) + "\n").encode()), workspace=workspace,
        output_root=workspace / "data/processed/upload", stage="cpt",
        origin=DataSource(source_id="upload-source", kind="local_upload", name="upload",
                          license_status="user_declared_unverified", declared_by="user"),
    )
    manifest_path = registration.path / "manifests/cpt-user-upload-v1.yaml"
    output_root = workspace / "data/processed/cpt-role-repair"

    with pytest.raises(ValueError, match="不会自动补角色"):
        repair_legacy_cpt_roles(manifest_path, workspace=workspace, output_root=output_root, apply=True)
    assert not output_root.exists()
