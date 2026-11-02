from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from onegpu_posttrain.ui.evaluation_matrix import MAX_MATRIX_BYTES
from onegpu_posttrain.ui.server import create_app


def _write_matrix(root, *, name="fixture", payload=None):
    path = root / "runs" / "mini" / name / "evaluation-matrix" / "evaluation_matrix.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload or {"protocol": "fixture-v1", "models": []}), encoding="utf-8")
    return path


def test_matrix_index_and_detail_are_discovered_without_client_paths(tmp_path):
    path = _write_matrix(tmp_path)
    v2 = tmp_path / "runs" / "mini" / "fixture" / "evaluation-matrix-v2" / "evaluation_matrix.json"
    v2.parent.mkdir(parents=True)
    v2.write_text(json.dumps({"protocol": "fixture-v2", "models": []}), encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        index = client.get("/api/evaluation-matrices").json()
        assert len(index["matrices"]) == 2
        item = next(entry for entry in index["matrices"] if entry["protocol"] == "fixture-v1")
        assert item["status"] == "available"
        assert "path" not in item
        detail = client.get(f"/api/evaluation-matrices/{item['matrix_id']}")
        assert detail.status_code == 200
        assert detail.json()["models"] == []
        assert client.get("/api/evaluation-matrices/..%2F..%2Fsecret").status_code == 404
        assert client.post("/api/evaluation-matrices", json={}).status_code == 405
    assert path.is_file()


def test_matrix_api_returns_explicit_empty_and_invalid_states(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/evaluation-matrices").json()["matrices"] == []
    path = _write_matrix(tmp_path, payload={"not": "a matrix"})
    with TestClient(create_app(tmp_path)) as client:
        entry = client.get("/api/evaluation-matrices").json()["matrices"][0]
        assert entry["status"] == "invalid"
        assert entry["issue"]
        assert client.get(f"/api/evaluation-matrices/{entry['matrix_id']}").status_code == 404
    path.write_bytes(b"{" * (MAX_MATRIX_BYTES + 1))
    with TestClient(create_app(tmp_path)) as client:
        entry = client.get("/api/evaluation-matrices").json()["matrices"][0]
        assert entry["status"] == "invalid"
        assert "上限" in entry["issue"]


def test_matrix_discovery_rejects_symlinked_experiment_directory(tmp_path):
    external = tmp_path / "outside"
    external.mkdir()
    _write_matrix(external)
    mini = tmp_path / "runs" / "mini"
    mini.mkdir(parents=True)
    try:
        (mini / "linked").symlink_to(external / "runs" / "mini" / "fixture", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/evaluation-matrices").json()["matrices"] == []


def test_matrix_detail_preserves_coverage_status_comparability_and_stable_lineage(tmp_path):
    payload = {
        "protocol": "unified-model-evaluation-matrix-v1",
        "baseline_model_id": "M2-SFT",
        "expected_metrics": ["gsm8k", "math_500", "internal_acc", "domain_ppl"],
        "requested_limits": {"gsm8k": 2, "math_500_total": 10, "internal_holdout": 5, "domain_ppl": 5},
        "comparability": {"status": "not_comparable", "protocol_consistency": {
            "expected_protocol_version": "math-eval-v1", "protocol_version_consistent": False,
            "config_hash_consistent": True, "dataset_ids_consistent": False,
            "protocol_versions": ["math-eval-v1", "other-v1"], "config_hashes": ["a" * 64],
            "dataset_id_sets": [["benchmark-a"], ["benchmark-b"]]}, "limitations": ["tiny sample"]},
        "models": [
            {"model_id": "M2-SFT", "status": "completed", "metrics": {"gsm8k": 0.0, "math_500": 0.1},
             "samples": {"gsm8k": {"requested": 2, "observed": 2}, "math_500": {"requested": 10, "observed": 8}},
             "provenance": {"status": "verified", "issues": []},
             "delta_vs_baseline": {"gsm8k": 0.0}, "lineage": {"eval_id": "eval-base", "produced_by_run_id": "run-eval-base",
             "model_id": "model-base", "dataset_ids": ["benchmark-a"], "protocol_version": "math-eval-v1"}},
            {"model_id": "M3-DPO", "status": "failed", "metrics": {"gsm8k": None}, "samples": {},
             "errors": ["private raw error should not be exposed"], "lineage": {"model_id": "../private"}},
            {"model_id": "M3-Partial", "status": "partial", "metrics": {"domain_ppl": 5.0},
             "samples": {"domain_ppl": {"requested": 5, "observed": 3}}, "lineage": {}},
        ],
    }
    _write_matrix(tmp_path, payload=payload)
    with TestClient(create_app(tmp_path)) as client:
        item = client.get("/api/evaluation-matrices").json()["matrices"][0]
        data = client.get(f"/api/evaluation-matrices/{item['matrix_id']}").json()
    base, failed, partial = data["models"]
    assert data["comparability"]["status"] == "not_comparable"
    assert data["comparability"]["dataset_ids_consistent"] is False
    assert data["comparability"]["limitations"] == ["tiny sample"]
    assert base["metrics"]["gsm8k"] == {"value": 0.0, "unit": "ratio", "status": "available",
        "requested": 2, "observed": 2, "delta": 0.0}
    assert base["metrics"]["math_500"]["observed"] == 8
    assert base["provenance"] == {"status": "verified", "issues": []}
    assert failed["status"] == "failed" and failed["metrics"]["gsm8k"]["value"] is None
    assert failed["metrics"]["gsm8k"]["status"] == "unavailable"
    assert failed["lineage"]["model_artifact_id"] is None
    assert failed["lineage"]["training_resolution"]["resolved"] is False
    assert failed["lineage"]["evaluation_resolution"]["resolved"] is False
    assert failed["errors_count"] == 1
    assert partial["metrics"]["domain_ppl"]["status"] == "partial"
    assert partial["metrics"]["domain_ppl"]["observed"] == 3
    rendered = json.dumps(data)
    assert "private raw error" not in rendered and "eval_path" not in rendered and "model_path" not in rendered


def test_matrix_detail_exposes_provenance_warnings_without_raw_errors(tmp_path):
    payload = {"protocol": "matrix-v1", "expected_metrics": ["gsm8k"], "comparability": {"status": "partial_or_inconsistent"},
        "models": [{"model_id": "M3-GRPO", "status": "partial", "metrics": {"gsm8k": 0.5},
            "samples": {"gsm8k": {"requested": 2, "observed": 2}}, "provenance": {"status": "mismatch",
                "issues": [{"code": "run_id_mismatch", "message": "评测 Run 与 lineage 不匹配"},
                           {"code": "invalid/code", "message": "D:\\private\\secrets"}]}, "lineage": {}}]}
    _write_matrix(tmp_path, payload=payload)
    with TestClient(create_app(tmp_path)) as client:
        item = client.get("/api/evaluation-matrices").json()["matrices"][0]
        model = client.get(f"/api/evaluation-matrices/{item['matrix_id']}").json()["models"][0]
    assert model["metrics"]["gsm8k"]["value"] == 0.5
    assert model["provenance"] == {"status": "mismatch", "issues": [
        {"code": "run_id_mismatch", "message": "评测 Run 与 lineage 不匹配"}]}


def test_matrix_lineage_resolution_uses_exact_model_eval_and_run_ids(tmp_path):
    payload = {"protocol": "matrix-v1", "baseline_model_id": "friendly-label-does-not-resolve",
        "expected_metrics": ["gsm8k"], "comparability": {"status": "comparable"},
        "models": [{"model_id": "friendly-label-does-not-resolve", "status": "completed", "metrics": {"gsm8k": 0.5},
            "samples": {"gsm8k": {"requested": 2, "observed": 2}},
            "lineage": {"eval_id": "eval-stable-1", "produced_by_run_id": "run-eval-stable-1",
                "model_id": "model-stable-1", "dataset_ids": ["eval-dataset-1"],
                "protocol_version": "eval-v1", "config_sha256": "a" * 64}}]}
    _write_matrix(tmp_path, payload=payload)
    training_dir = tmp_path / "runs" / "mini" / "train-run"
    evaluation_dir = tmp_path / "runs" / "mini" / "eval-run"
    training_dir.mkdir(parents=True)
    evaluation_dir.mkdir(parents=True)
    (training_dir / "lineage.json").write_text(json.dumps({"lineage_schema_version": "1.0", "run_id": "run-train-stable-1",
        "run_type": "train", "status": "completed", "inputs": [
            {"id": "parent-model-stable", "kind": "model", "role": "parent"},
            {"id": "train-dataset-stable", "kind": "dataset", "role": "train"}],
        "outputs": [{"id": "model-stable-1", "kind": "model", "role": "merged"}]}), encoding="utf-8")
    (evaluation_dir / "evaluation_run.json").write_text(json.dumps({"lineage_schema_version": "1.0", "run_id": "run-eval-stable-1",
        "run_type": "evaluate", "status": "completed", "inputs": [], "outputs": []}), encoding="utf-8")
    (evaluation_dir / "evaluation_lineage.json").write_text(json.dumps({"eval_id": "eval-stable-1",
        "produced_by_run_id": "run-eval-stable-1", "model_id": "model-stable-1", "dataset_ids": ["eval-dataset-1"],
        "protocol_version": "eval-v1", "config_sha256": "a" * 64, "status": "completed"}), encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        index = client.get("/api/evaluation-matrices").json()["matrices"][0]
        model = client.get(f"/api/evaluation-matrices/{index['matrix_id']}").json()["models"][0]
    assert model["metrics"]["gsm8k"]["value"] == 0.5
    assert model["lineage"]["training_resolution"] == {"run_id": "run-train-stable-1", "run_type": "train",
        "status": "completed", "parent_model_ids": ["parent-model-stable"], "dataset_ids": ["train-dataset-stable"], "resolved": True}
    assert model["lineage"]["evaluation_resolution"]["resolved"] is True
    assert model["lineage"]["evaluation_resolution"]["dataset_ids"] == ["eval-dataset-1"]


def test_matrix_resolves_evaluation_artifacts_under_runs_outside_mini(tmp_path):
    payload = {"protocol": "matrix-v1", "baseline_model_id": "M2", "expected_metrics": ["gsm8k"],
        "comparability": {"status": "comparable"}, "models": [{"model_id": "M2", "status": "completed",
            "metrics": {"gsm8k": 0.5}, "samples": {"gsm8k": {"requested": 2, "observed": 2}},
            "lineage": {"eval_id": "eval-outside-mini", "produced_by_run_id": "run-outside-mini",
                "model_id": "model-m2", "dataset_ids": [], "protocol_version": "eval-v1", "config_sha256": "a" * 64}}]}
    _write_matrix(tmp_path, payload=payload)
    evaluation_dir = tmp_path / "runs" / "e" / "grpo-s"
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "evaluation_run.json").write_text(json.dumps({"lineage_schema_version": "1.0",
        "run_id": "run-outside-mini", "run_type": "evaluate", "status": "completed", "inputs": [], "outputs": []}), encoding="utf-8")
    (evaluation_dir / "evaluation_lineage.json").write_text(json.dumps({"eval_id": "eval-outside-mini",
        "produced_by_run_id": "run-outside-mini", "model_id": "model-m2", "dataset_ids": [],
        "protocol_version": "eval-v1", "config_sha256": "a" * 64, "status": "completed"}), encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        index = client.get("/api/evaluation-matrices").json()["matrices"][0]
        model = client.get(f"/api/evaluation-matrices/{index['matrix_id']}").json()["models"][0]
    assert model["lineage"]["evaluation_resolution"]["resolved"] is True
    assert model["lineage"]["evaluation_resolution"]["run_id"] == "run-outside-mini"


def test_repository_mini_matrix_can_be_read_without_mutating_or_exposing_paths():
    workspace = Path(__file__).resolve().parents[1]
    matrices = list((workspace / "runs" / "mini").glob("*/evaluation-matrix*/evaluation_matrix.json"))
    if not matrices:
        pytest.skip("no real Mini evaluation matrix in this checkout")
    target = matrices[0]
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    with TestClient(create_app(workspace)) as client:
        index_response = client.get("/api/evaluation-matrices")
        assert index_response.status_code == 200
        entry = next(item for item in index_response.json()["matrices"] if "矩阵-v2" in item["name"])
        detail_response = client.get(f"/api/evaluation-matrices/{entry['matrix_id']}")
        assert detail_response.status_code == 200
        matrix = detail_response.json()
        assert matrix["models"]
        assert matrix["baseline_model_id"]
        assert all("evaluation_resolution" in item["lineage"] for item in matrix["models"])
        resolutions = {item["model_id"]: item["lineage"]["evaluation_resolution"]["resolved"] for item in matrix["models"]}
        assert resolutions["M2-SFT"] and resolutions["M3-DPO"] and resolutions["M3-SimPO"]
        assert resolutions["M3-GRPO-standard"] and resolutions["M3-GRPO-lowmem"]
        assert all(item["provenance"]["status"] == "verified" for item in matrix["models"])
        assert matrix["models"][3]["metrics"]["math_500"]["value"] is not None
        assert client.post("/api/evaluation-matrices", json={}).status_code == 405
    encoded = json.dumps(matrix)
    assert "model_path" not in encoded and "eval_path" not in encoded
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before


def test_matrix_rejects_unknown_schema_and_never_fills_missing_metrics(tmp_path):
    _write_matrix(tmp_path, payload={"models": [{"model_id": "M2", "status": "completed"}],
                                     "expected_metrics": ["gsm8k"]})
    with TestClient(create_app(tmp_path)) as client:
        item = client.get("/api/evaluation-matrices").json()["matrices"][0]
        result = client.get(f"/api/evaluation-matrices/{item['matrix_id']}").json()
    metric = result["models"][0]["metrics"]["gsm8k"]
    assert metric["value"] is None
    assert metric["status"] == "unavailable"
    assert metric["requested"] is None and metric["observed"] is None and metric["delta"] is None
