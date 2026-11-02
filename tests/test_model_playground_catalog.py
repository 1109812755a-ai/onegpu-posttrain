import json
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from onegpu_posttrain.lineage import sha256_directory
from onegpu_posttrain.ui.model_playground_catalog import ModelPlaygroundCatalog
from onegpu_posttrain.ui.server import create_app


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def register_merged(root, *, model_id="model-sft-good", kind="merged", model_path=None,
                    source_run_id="run-sft-source", merge_run_id="run-sft-merge",
                    completed_at=None):
    source_id, merge_id = source_run_id, merge_run_id
    adapter_id = "model-sft-adapter" if source_id == "run-sft-source" else f"model-adapter-{source_id.removeprefix('run-train-')}"
    source_output = {"id": adapter_id, "kind": "model", "role": "adapter",
                     "digest": {"algorithm": "sha256", "status": "available", "value": "a" * 64}}
    put(root / "runs/training" / source_id / "run.json",
        {"run_id": source_id, "status": "completed", "stage": "sft", "completed_at": completed_at})
    put(root / "runs/training" / source_id / "lineage.json",
        {"run_id": source_id, "run_type": "train", "status": "completed", "lineage_status": "complete",
         "inputs": [], "outputs": [source_output]})
    path = model_path or (root / "runs/merges" / merge_id / "merged-model")
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type":"tiny"}', encoding="utf-8")
    (path / "weights.safetensors").write_bytes(b"weights")
    digest = sha256_directory(path)
    merge_output = {"id": model_id, "kind": "model", "role": "merged", "path": str(path),
                    "digest": {"algorithm": "sha256", "status": "available", "value": digest}}
    put(root / "runs/merges" / merge_id / "lineage.json",
        {"run_id": merge_id, "run_type": "merge", "status": "completed",
         "inputs": [source_output], "outputs": [merge_output]})
    put(root / "runs/merges" / merge_id / "model_artifacts" / f"{model_id}.json",
        {"model_id": model_id, "kind": kind, "stage": "sft", "path": str(path),
         "produced_by_run_id": merge_id, "parent_model_ids": [source_output["id"]],
         "adapter_id": source_output["id"],
         "digest": {"algorithm": "sha256", "status": "available", "value": digest}})
    return path


def catalog_response(client):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get("/api/model-playground/models")
        assert response.status_code == 200
        if response.json()["status"] != "loading":
            return response.json()
        time.sleep(0.01)
    raise AssertionError("model catalog audit did not finish")


def ready_snapshot(catalog):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        snapshot = catalog.snapshot()
        if snapshot["status"] == "ready":
            return snapshot
        time.sleep(0.01)
    raise AssertionError("model catalog audit did not finish")


def test_model_catalog_lists_only_verified_merged_artifacts(tmp_path):
    path = register_merged(tmp_path)

    catalog = ModelPlaygroundCatalog(tmp_path).list_models()

    assert len(catalog) == 1
    assert catalog[0]["model_id"] == "model-sft-good"
    assert catalog[0]["stage"] == "sft"
    assert catalog[0]["source_run_id"] == "run-sft-source"
    assert catalog[0]["produced_by_run_id"] == "run-sft-merge"
    assert catalog[0]["training_completed_at"] is None
    assert catalog[0]["available"] is True
    assert catalog[0]["_resolved_path"] == str(path.resolve())


def test_catalog_snapshot_returns_while_large_digest_audit_runs_in_background(tmp_path, monkeypatch):
    catalog = ModelPlaygroundCatalog(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def slow_list():
        entered.set()
        release.wait(2)
        return [{"model_id": "model-ready", "available": True}]

    monkeypatch.setattr(catalog, "list_models", slow_list)
    first = catalog.snapshot()
    assert first["status"] == "loading"
    assert entered.wait(1)
    release.set()
    deadline = time.monotonic() + 2
    while catalog.snapshot()["status"] == "loading" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert catalog.snapshot()["models"] == [{"model_id": "model-ready", "available": True}]


def test_model_catalog_marks_missing_artifact_unavailable(tmp_path):
    path = register_merged(tmp_path)
    (path / "weights.safetensors").unlink()

    model = ModelPlaygroundCatalog(tmp_path).list_models()[0]

    assert model["available"] is False
    assert model["unavailable_reason"] == "缺少 Transformers 配置或权重文件"
    assert model["_resolved_path"] is None


def test_model_catalog_marks_mutated_artifact_unavailable(tmp_path):
    path = register_merged(tmp_path)
    (path / "weights.safetensors").write_bytes(b"mutated")

    catalog = ModelPlaygroundCatalog(tmp_path)
    model = catalog.list_models()[0]

    assert model["available"] is True
    assert model["unavailable_reason"] is None
    ready_snapshot(catalog)
    with pytest.raises(RuntimeError, match="模型内容摘要与登记值不匹配，已阻止加载"):
        catalog.resolve(model["model_id"])


def test_model_catalog_excludes_unsupported_adapter_artifacts(tmp_path):
    register_merged(tmp_path, model_id="model-adapter-only", kind="adapter")

    assert ModelPlaygroundCatalog(tmp_path).list_models() == []


def test_model_catalog_rejects_missing_registry_path_and_non_transformers_folder(tmp_path):
    register_merged(tmp_path)
    registry = tmp_path / "runs/merges/run-sft-merge/model_artifacts/model-sft-good.json"
    record = json.loads(registry.read_text(encoding="utf-8"))
    record["path"] = str(tmp_path / "absent")
    put(registry, record)

    model = ModelPlaygroundCatalog(tmp_path).list_models()[0]

    assert model["available"] is False
    assert model["unavailable_reason"] == "模型路径缺失、越界或不可访问"


def test_model_catalog_api_returns_allowlisted_metadata_without_local_paths(tmp_path):
    register_merged(tmp_path, completed_at="2026-10-31T08:30:00Z")

    with TestClient(create_app(tmp_path)) as client:
        catalog = catalog_response(client)
        model = catalog["models"][0]

    assert model == {
        "model_id": "model-sft-good", "kind": "merged", "stage": "sft",
        "digest": model["digest"], "source_run_id": "run-sft-source",
        "produced_by_run_id": "run-sft-merge", "training_completed_at": "2026-10-31T08:30:00Z",
        "display_name": None, "available": True,
        "unavailable_reason": None,
    }
    assert "path" not in model and "_resolved_path" not in model


def test_model_catalog_api_reports_unavailable_reason_without_path(tmp_path):
    path = register_merged(tmp_path)
    (path / "weights.safetensors").unlink()

    with TestClient(create_app(tmp_path)) as client:
        model = catalog_response(client)["models"][0]

    assert model["available"] is False
    assert model["unavailable_reason"] == "缺少 Transformers 配置或权重文件"
    assert "path" not in model


def test_model_catalog_api_rejects_non_loopback_host(tmp_path):
    with TestClient(create_app(tmp_path), base_url="http://example.com") as client:
        response = client.get("/api/model-playground/models")
    assert response.status_code == 400


def test_model_catalog_sorts_by_verified_training_completion_and_keeps_merge_run_distinct(tmp_path):
    register_merged(tmp_path, model_id="model-older", source_run_id="run-train-old",
                    merge_run_id="run-merge-old", completed_at="2026-10-29T10:00:00Z")
    register_merged(tmp_path, model_id="model-newer", source_run_id="run-train-new",
                    merge_run_id="run-merge-new", completed_at="2026-10-31T10:00:00+00:00")
    register_merged(tmp_path, model_id="model-unknown", source_run_id="run-train-unknown",
                    merge_run_id="run-merge-unknown", completed_at="not-a-time")

    models = ModelPlaygroundCatalog(tmp_path).list_models()

    assert [model["model_id"] for model in models] == ["model-newer", "model-older", "model-unknown"]
    assert models[0]["source_run_id"] == "run-train-new"
    assert models[0]["produced_by_run_id"] == "run-merge-new"
    assert models[0]["training_completed_at"] == "2026-10-31T10:00:00+00:00"
    assert all(model["available"] for model in models)


def test_model_catalog_orders_equal_completion_time_stably_and_missing_times_last(tmp_path):
    register_merged(tmp_path, model_id="model-z", source_run_id="run-train-z",
                    merge_run_id="run-merge-z", completed_at="2026-10-31T10:00:00Z")
    register_merged(tmp_path, model_id="model-a", source_run_id="run-train-a",
                    merge_run_id="run-merge-a", completed_at="2026-10-31T10:00:00+00:00")
    register_merged(tmp_path, model_id="model-no-time", source_run_id="run-train-no-time",
                    merge_run_id="run-merge-no-time")

    models = ModelPlaygroundCatalog(tmp_path).list_models()

    assert [model["model_id"] for model in models] == ["model-a", "model-z", "model-no-time"]


def test_model_catalog_read_is_metadata_only_and_does_not_resolve_or_load_model(tmp_path, monkeypatch):
    register_merged(tmp_path, completed_at="2026-10-31T08:30:00Z")
    resolve_calls = []

    def unexpected_resolve(self, model_id):
        resolve_calls.append(model_id)
        raise AssertionError("catalog read must not resolve or load a model")

    monkeypatch.setattr(ModelPlaygroundCatalog, "resolve", unexpected_resolve)

    with TestClient(create_app(tmp_path)) as client:
        response = catalog_response(client)

    assert response["models"][0]["model_id"] == "model-sft-good"
    assert resolve_calls == []


def test_model_display_name_api_updates_catalog_and_clears_name(tmp_path):
    register_merged(tmp_path)
    app = create_app(tmp_path, enable_experiment_launcher=True)
    headers = {"X-Local-Launcher-Token": app.state.experiment_launcher_token}

    with TestClient(app) as client:
        catalog_response(client)
        updated = client.put("/api/model-playground/models/model-sft-good/display-name",
                             json={"display_name": "  SFT 数学助手  "}, headers=headers)
        assert updated.status_code == 200
        assert updated.json() == {"model_id": "model-sft-good", "display_name": "SFT 数学助手"}
        assert catalog_response(client)["models"][0]["display_name"] == "SFT 数学助手"

        cleared = client.delete("/api/model-playground/models/model-sft-good/display-name", headers=headers)
        assert cleared.status_code == 204
        assert catalog_response(client)["models"][0]["display_name"] is None


def test_run_views_join_alias_by_model_id_without_rewriting_run_or_lineage(tmp_path):
    register_merged(tmp_path)
    state_path = tmp_path / ".onegpu-posttrain/launcher/runs/run-sft-source.json"
    state = {"run_id": "run-sft-source", "run_kind": "training", "stage": "sft",
             "status": "completed", "model_artifacts": [{"model_id": "model-sft-good",
             "kind": "merged", "stage": "sft", "produced_by_run_id": "run-sft-merge"}]}
    put(state_path, state)
    run_record = (tmp_path / "runs/training/run-sft-source/run.json").read_bytes()
    lineage_record = (tmp_path / "runs/training/run-sft-source/lineage.json").read_bytes()
    app = create_app(tmp_path, enable_experiment_launcher=True)
    headers = {"X-Local-Launcher-Token": app.state.experiment_launcher_token}

    with TestClient(app) as client:
        catalog_response(client)
        saved = client.put("/api/model-playground/models/model-sft-good/display-name",
                           json={"display_name": "SFT 试验模型"}, headers=headers)
        assert saved.status_code == 200
        launcher_run = client.get("/api/launcher/runs/run-sft-source")
        assert launcher_run.status_code == 200
        assert launcher_run.json()["model_artifacts"][0]["display_name"] == "SFT 试验模型"
        run_detail = client.get("/api/runs/run-sft-source")
        assert run_detail.status_code == 200
        assert run_detail.json()["handoff_models"][0]["display_name"] == "SFT 试验模型"
        assert run_detail.json()["handoff_models"][0]["model_id"] == "model-sft-good"

    assert json.loads(state_path.read_text(encoding="utf-8")) == state
    assert (tmp_path / "runs/training/run-sft-source/run.json").read_bytes() == run_record
    assert (tmp_path / "runs/training/run-sft-source/lineage.json").read_bytes() == lineage_record


def test_model_display_name_api_requires_token_registered_id_and_valid_plain_text(tmp_path):
    register_merged(tmp_path)
    app = create_app(tmp_path, enable_experiment_launcher=True)
    headers = {"X-Local-Launcher-Token": app.state.experiment_launcher_token}

    with TestClient(app) as client:
        catalog_response(client)
        path = "/api/model-playground/models/model-sft-good/display-name"
        assert client.put(path, json={"display_name": "Valid"}).status_code == 403
        assert client.put("/api/model-playground/models/model-not-registered/display-name",
                          json={"display_name": "Valid"}, headers=headers).status_code == 404
        assert client.put(path, json={"display_name": "   "}, headers=headers).status_code == 422
        assert client.put(path, json={"display_name": "line1\nline2"}, headers=headers).status_code == 422
        assert client.put(path, json={"display_name": "x" * 81}, headers=headers).status_code == 422
        assert client.put(path, json={"display_name": "Valid", "model_path": "C:/private"},
                          headers=headers).status_code == 422
        assert client.put("/api/model-playground/models/model-%2E%2E%2Fprivate/display-name",
                          json={"display_name": "Valid"}, headers=headers).status_code in {404, 422}
