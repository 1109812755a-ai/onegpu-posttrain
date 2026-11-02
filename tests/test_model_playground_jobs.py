import threading
import time

import pytest

from onegpu_posttrain.ui.model_playground_jobs import ModelPlaygroundJobs


class FakeCatalog:
    def resolve(self, model_id):
        if model_id not in {"model-one", "model-two", "model-three"}:
            return None
        return {"model_id": model_id, "available": True, "digest": "a" * 64}


class FakeManager:
    catalog = FakeCatalog()

    def __init__(self):
        self.active = None
        self.cancelled = threading.Event()
        self.entered = threading.Event()

    def open(self, model_id, *, session_id):
        self.active = {"session_id": session_id, "model_id": model_id, "status": "ready"}
        return {"session_id": session_id, "model_id": model_id, "status": "ready"}

    def chat(self, session_id, prompt, parameters):
        self.entered.set()
        while not self.cancelled.is_set():
            time.sleep(0.002)
        self.active = None
        raise RuntimeError("生成已取消")

    def cancel(self, session_id):
        self.cancelled.set()
        return True

    @property
    def active_session(self):
        return self.active

    def close(self, session_id=None):
        self.active = None
        return True

    def compare(self, model_ids, prompt, parameters):
        return {"status": "partial", "results": [
            {"status": "completed", "model_id": model_ids[0], "text": "private answer"},
            {"status": "failed", "model_id": model_ids[1], "detail": "model load failed"},
        ], "parameters": parameters, "qualitative_unverified": True}


def wait_for(predicate, timeout=2):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("asynchronous model-playground job did not reach expected state")


def test_session_and_message_jobs_keep_prompt_out_of_job_metadata(tmp_path, caplog):
    manager = FakeManager()
    jobs = ModelPlaygroundJobs(tmp_path, manager.catalog, session_manager=manager)
    try:
        session = jobs.start_session("model-one")
        ready = wait_for(lambda: (state if (state := jobs.session_status(session["session_id"]))["status"] != "loading" else None))
        assert ready["status"] == "ready"
        prompt = "private user question"
        job = jobs.submit_message(session["session_id"], prompt, {"temperature": 0})
        wait_for(manager.entered.is_set)
        state = jobs.job_status(job["job_id"])
        internal = jobs._jobs[job["job_id"]]
        assert prompt not in repr(internal)
        assert "prompt" not in state
        assert prompt not in caplog.text
        assert jobs.cancel(job["job_id"])["status"] in {"running", "cancelled"}
        final = wait_for(lambda: (state if (state := jobs.job_status(job["job_id"]))["status"] != "running" else None))
        assert final["status"] == "cancelled"
        assert jobs.session_status(session["session_id"])["status"] == "closed"
        assert not (tmp_path / "runs").exists()
        assert not (tmp_path / ".onegpu-posttrain").exists()
    finally:
        jobs.shutdown()


def test_model_playground_api_requires_local_launcher_and_rejects_paths(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from onegpu_posttrain.ui.server import create_app

    with TestClient(create_app(tmp_path)) as client:
        unavailable = client.post("/api/model-playground/sessions", json={"model_id": "model-one"})
        assert unavailable.status_code == 403
        assert client.get("/api/model-playground/models").json()["models"] == []

    app = create_app(tmp_path, enable_experiment_launcher=True)
    token = app.state.experiment_launcher_token
    with TestClient(app) as client:
        bad = client.post("/api/model-playground/sessions", json={"model_id": "model-one", "model_path": "C:/private"},
                          headers={"X-Local-Launcher-Token": token})
        assert bad.status_code == 422
        unauthenticated_compare = client.post("/api/model-playground/comparisons", json={
            "model_ids": ["model-one", "model-two"], "message": "hello",
        })
        assert unauthenticated_compare.status_code == 403
        path_injection = client.post("/api/model-playground/comparisons", json={
            "model_ids": ["model-one", "model-two"], "message": "hello", "model_path": "C:/private",
        }, headers={"X-Local-Launcher-Token": token})
        assert path_injection.status_code == 422


def test_completed_chat_output_is_memory_only_and_not_logged(tmp_path, caplog):
    class SuccessfulManager(FakeManager):
        def chat(self, session_id, prompt, parameters):
            return {"text": "private generated completion", "model_id": "model-one",
                    "digest": "a" * 64, "stage": "sft", "source_run_id": "run-source",
                    "parameters": parameters, "qualitative_unverified": True}

    manager = SuccessfulManager()
    jobs = ModelPlaygroundJobs(tmp_path, manager.catalog, session_manager=manager)
    try:
        session = jobs.start_session("model-one")
        wait_for(lambda: (state if (state := jobs.session_status(session["session_id"]))["status"] == "ready" else None))
        prompt = "private prompt string"
        job = jobs.submit_message(session["session_id"], prompt)
        result = wait_for(lambda: (state if (state := jobs.job_status(job["job_id"]))["status"] == "completed" else None))
        assert result["result"]["text"] == "private generated completion"
        assert prompt not in caplog.text
        assert "private generated completion" not in caplog.text
        assert not (tmp_path / "runs").exists()
        assert not (tmp_path / ".onegpu-posttrain").exists()
    finally:
        jobs.shutdown()


def test_comparison_job_preserves_partial_results_without_retaining_prompt(tmp_path, caplog):
    manager = FakeManager()
    jobs = ModelPlaygroundJobs(tmp_path, manager.catalog, session_manager=manager)
    try:
        prompt = "same private prompt"
        job = jobs.submit_comparison(["model-one", "model-two"], prompt)
        final = wait_for(lambda: (state if (state := jobs.job_status(job["job_id"]))["status"] == "partial" else None))
        assert final["result"]["results"][0]["text"] == "private answer"
        assert final["result"]["status"] == "partial"
        assert prompt not in repr(jobs._jobs[job["job_id"]])
        assert prompt not in caplog.text and "private answer" not in caplog.text
        assert not (tmp_path / "runs").exists()
        assert not (tmp_path / ".onegpu-posttrain").exists()
    finally:
        jobs.shutdown()


def test_comparison_job_rejects_bad_selection_and_models(tmp_path):
    manager = FakeManager()
    jobs = ModelPlaygroundJobs(tmp_path, manager.catalog, session_manager=manager)
    try:
        with pytest.raises(ValueError, match="2 到 3"):
            jobs.submit_comparison(["model-one"], "hello")
        with pytest.raises(ValueError, match="不能重复"):
            jobs.submit_comparison(["model-one", "model-one"], "hello")
        with pytest.raises(ValueError, match="不存在"):
            jobs.submit_comparison(["model-one", "missing"], "hello")
    finally:
        jobs.shutdown()


def test_single_session_cannot_queue_during_model_comparison(tmp_path):
    class SlowComparisonManager(FakeManager):
        def compare(self, model_ids, prompt, parameters):
            self.entered.set()
            self.cancelled.wait(2)
            return {"status": "completed", "results": [], "parameters": parameters}

    manager = SlowComparisonManager()
    jobs = ModelPlaygroundJobs(tmp_path, manager.catalog, session_manager=manager)
    try:
        job = jobs.submit_comparison(["model-one", "model-two"], "hello")
        assert manager.entered.wait(2)
        with pytest.raises(RuntimeError, match="模型对照任务"):
            jobs.start_session("model-one")
        manager.cancelled.set()
        assert wait_for(lambda: (state if (state := jobs.job_status(job["job_id"]))["status"] == "completed" else None))
    finally:
        manager.cancelled.set()
        jobs.shutdown()
