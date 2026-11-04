"""The local API requires a fresh explicit confirmation before editor work."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi.testclient import TestClient

from onegpu_posttrain.ui.server import create_app


def test_weight_space_api_confirmation_and_remote_boundary(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    app = create_app(root, enable_experiment_launcher=True)
    service = app.state.weight_space_workbench
    monkeypatch.setattr(service, "snapshot", lambda: {
        "ready": True, "provider": {"model": "qwen3.8-max", "credential_ready": True},
        "practice_predictions_available": True,
    })
    called = threading.Event()
    monkeypatch.setattr(service, "_run_iteration", lambda: called.set() or {"status": "completed"})
    with TestClient(app) as client:
        assert client.post("/api/weight-space/iteration/start", json={"confirmation_token": "wrong"}).status_code == 409
        assert not called.is_set()
        preview = client.post("/api/weight-space/iteration/preflight").json()
        assert preview["pricing"] == "unknown"
        assert not called.is_set()
        started = client.post("/api/weight-space/iteration/start", json={
            "confirmation_token": preview["confirmation_token"]})
        assert started.status_code == 202
        assert called.wait(2)
        assert client.post("/api/weight-space/iteration/start", json={
            "confirmation_token": preview["confirmation_token"]}).status_code == 409
    with TestClient(create_app(root, allow_remote=True)) as remote:
        assert remote.get("/api/weight-space/status").status_code == 403
        assert remote.post("/api/weight-space/iteration/preflight").status_code == 403
