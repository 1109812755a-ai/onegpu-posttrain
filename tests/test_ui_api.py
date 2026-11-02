from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from onegpu_posttrain.data.build import build_openr1_artifacts
from onegpu_posttrain.data.manifest import DatasetManifest, SourceSpec, save_manifest
from onegpu_posttrain.data.openr1 import OpenR1BuildConfig
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.lineage import ArtifactRef, Digest, RunRecord, new_run_id
from onegpu_posttrain.ui.catalog import MAX_BYTES, ArtifactCatalog, ArtifactError
from onegpu_posttrain.ui.server import create_app


@pytest.fixture
def reports(tmp_path):
    output = tmp_path / "data/processed/test/openr1"
    rows = [{"problem": f"Question {i}: find x in x + {i} = {i + 2}.", "answer": "2", "uuid": f"row-{i}",
             "source": "fixture", "generations": [r"Reasoning \boxed{2}.", r"Reasoning \boxed{4}.", "unfinished"],
             "is_reasoning_complete": [True, True, False], "finish_reasons": ["stop", "stop", "length"],
             "correctness_math_verify": [True, False, False]} for i in range(100)]
    build_openr1_artifacts(rows, output_dir=output, tokenizer=WhitespaceTokenizer(),
                           config=OpenR1BuildConfig(sft_target=3, alignment_target=3, holdout_target=3,
                                                   preference_target=2, preference_eval_target=2, grpo_target=2),
                           source_repo="open-r1/OpenR1-Math-220k", source_config=None, source_revision="fixture-v1",
                           tokenizer_repo="fixture", tokenizer_revision="v1")
    return output


def test_data_page_reads_modern_reports_without_mutation(tmp_path, reports):
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in reports.rglob("*") if p.is_file()}
    with TestClient(create_app(tmp_path)) as client:
        index = client.get("/api/data").json()
        assert index["read_only"] is True
        assert len(index["builds"]) == 1
        detail = client.get(f"/api/data/{index['builds'][0]['id']}").json()
        assert len(detail["steps"]) == 11
        assert detail["model_effect"]["status"] == "not_evaluated"
        assert detail["related_runs"] == []
        assert len(detail["datasets"]) == 5
        for stage in ("sft", "preference", "grpo"):
            dataset = next(item for item in detail["datasets"] if item["stage"] == stage)
            assert dataset["source"]["source_id"] == "openr1"
            assert dataset["source"]["kind"] == "open_source"
            assert dataset["registry_artifact"]["canonical_sha256"] == dataset["content_sha256"]
            status_response = client.get(f"/api/data/datasets/{dataset['dataset_id']}/status")
            assert status_response.status_code == 200
            status = status_response.json()
            assert status["read_only"] is True
            assert status["dataset_id"] == dataset["dataset_id"]
            assert status["canonical_sha256"] == dataset["content_sha256"]
            assert status["processing"]["status"] == "verified"
            assert status["quality"]["report_id"] == dataset["quality"]["quality_report_id"]
            assert status["lineage"]["produced_by_run_id"]
            assert "samples" not in status and "generations" not in json.dumps(status)
            assert client.get(f"/api/data/datasets/{dataset['dataset_id']}/status").json() == status
        assert detail["datasets"][0]["quality"]["metrics"]["semantic_correctness"]["value"] is None
        artifact = client.get(f"/api/artifacts/{detail['artifacts'][0]['id']}")
        assert artifact.status_code == 200
        assert artifact.headers["x-content-type-options"] == "nosniff"
        assert client.post("/api/data", json={}).status_code == 405
        assert client.get("/api/artifacts/unknown").status_code == 404
        assert client.get("/api/data/unknown").status_code == 404
        assert client.get("/api/data/datasets/not-registered/status").status_code == 404
        capabilities = client.get("/api/data/capabilities").json()["capabilities"]
        assert next(item for item in capabilities if item["source_id"] == "upload")["supported_stages"] == [
            "cpt", "sft", "preference", "grpo",
        ]
        unsupported = next(item for item in capabilities if item["source_id"] == "llm-generation-grpo")
        assert unsupported["status"] == "unsupported"
        assert "reward/reference" in unsupported["reason"]
        assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 400
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in reports.rglob("*") if p.is_file()}


def test_remote_host_validation_is_enabled_only_with_remote_opt_in(tmp_path):
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    local_app = create_app(tmp_path)
    remote_app = create_app(tmp_path, allow_remote=True)
    local_host_middleware = next(mw for mw in local_app.user_middleware if mw.cls is TrustedHostMiddleware)
    remote_host_middleware = next(mw for mw in remote_app.user_middleware if mw.cls is TrustedHostMiddleware)

    assert local_host_middleware.kwargs["allowed_hosts"] == ["localhost", "127.0.0.1", "[::1]", "testserver"]
    assert remote_host_middleware.kwargs["allowed_hosts"] == ["*"]


def test_legacy_reports_are_labeled_and_missing_quality_is_not_zero(tmp_path):
    root = tmp_path / "data/processed/smoke/openr1"
    (root / "reports").mkdir(parents=True)
    (root / "reports/openr1_data_report.json").write_text(json.dumps({
        "protocol": "openr1-build-v1", "counters": {"raw_rows": 100, "normalized_unique": 95, "verifiable_core": 80},
        "targets": {"sft": 10}, "selected": {"sft": 3}, "overlap": {"train_holdout": 0},
    }), encoding="utf-8")
    save_manifest(root / "manifests/sft-smoke-v1.yaml", DatasetManifest(
        dataset_id="legacy-sft", stage="sft", sample_count=3, source=SourceSpec(repo="fixture", revision="v1"),
        metadata={"api_key": "DO_NOT_EXPOSE", "text": "RAW_SAMPLE_MUST_NOT_APPEAR"},
    ), allow_legacy=True)
    with TestClient(create_app(tmp_path)) as client:
        index = client.get("/api/data").json()
        detail = client.get(f"/api/data/{index['builds'][0]['id']}").json()
        assert detail["mode"] == "legacy"
        assert detail["datasets"][0]["target"] == 10
        assert detail["datasets"][0]["count"] == 3
        assert detail["datasets"][0]["quality"] is None
        assert detail["steps"][1]["output_count"]["value"] is None
        assert detail["warnings"]
        preview = client.get(f"/api/artifacts/{detail['datasets'][0]['manifest']['id']}").text
        assert "DO_NOT_EXPOSE" not in preview
        assert "RAW_SAMPLE_MUST_NOT_APPEAR" not in preview


def test_artifact_reader_enforces_limits_and_containment(tmp_path, reports):
    with TestClient(create_app(tmp_path)) as client:
        index = client.get("/api/data").json()
        detail = client.get(f"/api/data/{index['builds'][0]['id']}").json()
        token = detail["artifacts"][0]["id"]
        (reports / "reports/processing_report.json").write_bytes(b" " * (MAX_BYTES + 1))
        assert client.get(f"/api/artifacts/{token}").status_code == 422
        refreshed = client.get("/api/data").json()
        assert refreshed["issues"]
        assert client.get("/api/artifacts/..%2F..%2F.env").status_code == 404
    catalog = ArtifactCatalog(tmp_path)
    with pytest.raises(ArtifactError):
        catalog._read(tmp_path / "secret.json")


def test_corrupt_quality_hash_is_visible_and_not_rendered(tmp_path, reports):
    quality = reports / "reports/quality/sft.json"
    payload = json.loads(quality.read_text(encoding="utf-8"))
    payload["metrics"]["schema_valid_rate"]["value"] = 0.123
    quality.write_text(json.dumps(payload), encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        index = client.get("/api/data").json()
        detail = client.get(f"/api/data/{index['builds'][0]['id']}").json()
        assert any("sft" in warning for warning in detail["warnings"])
        assert not any(d["stage"] == "sft" for d in detail["datasets"])


def test_legacy_open_source_manifest_without_registry_evidence_does_not_break_catalog(tmp_path, reports):
    import yaml

    manifest_path = next((reports / "manifests").glob("*sft*.yaml"))
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["lineage"] = None
    payload["processing_report"] = None
    manifest_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/data")
        assert response.status_code == 200
        build = response.json()["builds"][0]
        detail = client.get(f"/api/data/{build['id']}").json()
        sft = next(dataset for dataset in detail["datasets"] if dataset["stage"] == "sft")
        assert sft["registry_artifact"] is None
        assert sft["registry_status"] == "evidence_incomplete"
        summary_sft = next(dataset for dataset in response.json()["builds"][0]["datasets"]
                           if dataset["stage"] == "sft")
        assert summary_sft["mixture_eligible"] is False
        assert "processing report" in summary_sft["mixture_block_reason"]
        assert any("不能作为已登记数据集消费" in warning for warning in detail["warnings"])


def test_run_links_require_explicit_dataset_identity(tmp_path, reports):
    run = tmp_path / "runs/test/linked"
    run.mkdir(parents=True)
    unrelated = tmp_path / "runs/test/same-stage"
    unrelated.mkdir()
    (unrelated / "run.json").write_text(json.dumps({"run_id": "guess-by-stage", "stage": "sft"}), encoding="utf-8")
    catalog = ArtifactCatalog(tmp_path)
    summary = catalog.scan()["builds"][0]
    dataset_id = next(d["dataset_id"] for d in summary["datasets"] if d["stage"] == "sft")
    (run / "lineage.json").write_text(json.dumps({"run_id": "linked", "stage": "sft", "inputs": {"datasets": [{"dataset_id": dataset_id}]}}), encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        summary = client.get("/api/data").json()["builds"][0]
        detail = client.get(f"/api/data/{summary['id']}").json()
        assert [r["run_id"] for r in detail["related_runs"]] == ["linked"]
        assert detail["model_effect"]["status"] == "not_evaluated"


def test_v1_run_link_verifies_dataset_digest_and_rejects_out_of_root_path(tmp_path, reports):
    manifest_path = next((reports / "manifests").glob("*sft*.yaml"))
    import yaml
    manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    dataset_id = manifest_data["dataset_id"]
    data_path = reports / "canonical" / "sft.jsonl"
    content_digest = manifest_data["hashes"]["canonical_records_sha256"]
    run_dir = tmp_path / "runs/v1-linked"
    run_dir.mkdir(parents=True)
    record = RunRecord(
        run_id=new_run_id("train", label="sft-lineage"), run_type="train", status="completed",
        started_at="2026-10-17T00:00:00Z", inputs=[ArtifactRef(
            id=dataset_id, kind="dataset", path=str(data_path),
            digest=Digest(value=content_digest, status="available"), role="train",
        )],
    )
    (run_dir / "lineage.json").write_text(record.model_dump_json(), encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        summary = client.get("/api/data").json()["builds"][0]
        detail = client.get(f"/api/data/{summary['id']}").json()
        match = next(item for item in detail["related_runs"] if item["run_id"] == record.run_id)
        assert match["verification"] == "content_sha256"
        assert match["status"] == "completed"

    private = tmp_path / "private.jsonl"
    private.write_text("must not be read", encoding="utf-8")
    record.inputs[0].path = str(private)
    (run_dir / "lineage.json").write_text(record.model_dump_json(), encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        summary = client.get("/api/data").json()["builds"][0]
        detail = client.get(f"/api/data/{summary['id']}").json()
        assert not any(item["run_id"] == record.run_id for item in detail["related_runs"])
        index = client.get("/api/data").json()
        assert any(issue["path"].endswith("lineage.json") for issue in index["issues"])


def test_malformed_v1_lineage_is_not_rendered(tmp_path, reports):
    manifest_path = next((reports / "manifests").glob("*sft*.yaml"))
    import yaml
    dataset_id = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))["dataset_id"]
    run_dir = tmp_path / "runs/malformed"
    run_dir.mkdir(parents=True)
    (run_dir / "lineage.json").write_text(json.dumps({
        "lineage_schema_version": "1.0", "run_id": "bad", "inputs": [{"id": dataset_id, "kind": "dataset"}],
    }), encoding="utf-8")
    catalog = ArtifactCatalog(tmp_path)
    scan = catalog.scan()
    detail = catalog.detail(scan["builds"][0]["id"])
    assert detail["related_runs"] == []
    assert any("无效" in issue["message"] for issue in catalog.issues)


def test_symlink_outside_roots_is_not_served(tmp_path):
    root = tmp_path / "data/processed/leak/reports"
    root.mkdir(parents=True)
    secret = tmp_path / "private.json"
    secret.write_text('{"secret":"never serve"}', encoding="utf-8")
    try:
        (root / "processing_report.json").symlink_to(secret)
    except OSError:
        pytest.skip("Symlink creation is not permitted on this Windows host")
    with TestClient(create_app(tmp_path)) as client:
        result = client.get("/api/data").json()
        assert result["builds"] == []
        assert result["issues"]


def test_empty_workspace_and_unbuilt_frontend_are_explicit(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/data").json()["builds"] == []
        assert client.get("/").status_code == 503
