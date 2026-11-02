from __future__ import annotations

import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from onegpu_posttrain.lineage import sha256_directory, sha256_file, sha256_json
from onegpu_posttrain.ui import launcher as launcher_module
from onegpu_posttrain.ui.server import create_app


def _write_sft_recipe(path: Path) -> None:
    path.write_text(
        "schema_version: '1.0'\nstage: sft\nseed: 42\ntraining:\n"
        "  max_length: 1024\n  num_train_epochs: 1\n  learning_rate: 0.0001\n"
        "  gradient_accumulation_steps: 8\n",
        encoding="utf-8",
    )


def test_launcher_is_disabled_by_default_and_rejects_launch_requests(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/launcher/capabilities").json()["enabled"] is False
        assert client.post("/api/launcher/preflight", json={}).status_code == 403
        assert client.post("/api/launcher/start", json={}).status_code == 403


def test_launcher_cannot_be_enabled_for_remote_access(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        create_app(tmp_path, enable_experiment_launcher=True, allow_remote=True)


def test_get_run_trusts_active_durable_supervisor_over_missing_windows_popen(tmp_path):
    launcher = launcher_module.SFTLauncher(tmp_path)
    run_id = "run-grpo-supervisor-recovery-test"
    launcher._write_state({
        "run_id": run_id, "run_kind": "stage-training", "stage": "grpo",
        "profile_id": "grpo-lowmem-smoke-v1", "status": "interrupted",
        "detail": "监督进程不可用；不会自动续训", "completed_at": None,
        "output": f"runs/grpo/grpo-lowmem-smoke-v1/{run_id}",
        "log_path": f".onegpu-posttrain/launcher/logs/{run_id}.log",
        "outputs": [], "model_ids": [],
    })
    supervisor = launcher_module.LocalRunSupervisor(tmp_path)
    supervisor.create(run_id=run_id, stage="grpo", profile_id="grpo-lowmem-smoke-v1",
                      output_location=f"runs/grpo/grpo-lowmem-smoke-v1/{run_id}")
    supervisor.transition(run_id, "starting")
    supervisor.transition(run_id, "running")
    launcher._run_supervisor = supervisor

    observed = launcher.get_run(run_id)

    assert observed["status"] == "running"
    assert observed["detail"] is None


def test_launcher_api_requires_token_same_origin_and_rejects_overrides(tmp_path, monkeypatch):
    app = create_app(tmp_path, enable_experiment_launcher=True)
    app.state.experiment_launcher.preflight = lambda dataset_id=launcher_module.DATASET_ID, parameter_overrides=None, memory_mode_id=None: {"ok": False, "blockers": ["fixture"]}
    with TestClient(app) as client:
        caps = client.get("/api/launcher/capabilities").json()
        token = caps["token"]
        assert token and caps["local_only"] is True
        assert caps["parent_model_id"] == launcher_module.MODEL_ID
        assert client.post("/api/launcher/preflight", json={}).status_code == 403
        headers = {"X-Local-Launcher-Token": token}
        assert client.post("/api/launcher/preflight", json={"recipe": "other"}, headers=headers).status_code == 422
        assert client.post("/api/launcher/preflight", json={"dataset_id": "bad", "recipe": "other"}, headers=headers).status_code == 422
        assert client.post("/api/launcher/preflight", json={}, headers={**headers, "Origin": "https://evil.example"}).status_code == 403
        response = client.post("/api/launcher/preflight", json={}, headers=headers)
        assert response.status_code == 200
        assert response.json()["blockers"] == ["fixture"]
        assert client.post("/api/launcher/start", json={"confirmation_token": "x", "command": "whoami"}, headers=headers).status_code == 422
        assert client.post("/api/launcher/preflight", content=b" " * 5000, headers={**headers, "Content-Type": "application/json"}).status_code == 413


def test_preflight_uses_temp_plan_and_creates_no_training_artifacts(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    recipe = workspace / launcher_module.RECIPE
    recipe.parent.mkdir(parents=True)
    _write_sft_recipe(recipe)
    model = workspace / "model"
    model.mkdir()
    dataset = workspace / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    inputs = {
        "recipe_path": recipe, "recipe_sha256": "a" * 64, "model_path": model,
        "model_sha256": "b" * 64, "model_id": launcher_module.MODEL_ID,
        "manifest_sha256": "c" * 64, "canonical_sha256": "d" * 64,
        "native_path": dataset, "native_sha256": "e" * 64,
        "dataset_id": launcher_module.DATASET_ID,
    }
    monkeypatch.setattr(launcher_module.SFTLauncher, "_resolve_fixed_inputs", lambda self, dataset_id=launcher_module.DATASET_ID: inputs)
    monkeypatch.setattr(launcher_module, "_gpu_snapshot", lambda: {"available": False, "status": "unknown"})
    monkeypatch.setattr(launcher_module, "_sft_runtime_available", lambda: True)

    def fake_plan(args):
        recipe_data = __import__("yaml").safe_load(Path(args.config).read_text(encoding="utf-8"))
        Path(args.output, "train_plan.json").write_text(json.dumps({"stage": "sft", "training": recipe_data["training"]}), encoding="utf-8")
        return 0

    from onegpu_posttrain import cli
    monkeypatch.setattr(cli, "cmd_plan_sft", fake_plan)
    launcher = launcher_module.SFTLauncher(workspace)
    result = launcher.preflight(parameter_overrides={"max_length": 768, "num_train_epochs": 2})
    assert result["ok"] is True, result
    assert result["dataset"]["count"] == 307
    assert result["resolved_config"]["training"]["max_length"] == 768
    assert result["resolved_config"]["training"]["num_train_epochs"] == 2
    assert result["training_parameters"]["overrides"] == {"max_length": 768, "num_train_epochs": 2}
    assert len(result["training_parameters"]["config_sha256"]) == 64
    assert result["training_memory_mode"]["mode_id"] == "lora"
    assert "quant_bits" not in result["resolved_config"]["training"]
    monkeypatch.setattr(launcher_module, "inspect_training_memory_mode",
                        lambda mode_id: {"ok": True, "mode_id": mode_id, "dependencies": []})
    qlora = launcher.preflight(parameter_overrides={"max_length": 768, "num_train_epochs": 2},
                               memory_mode_id="qlora-4bit-nf4")
    assert qlora["ok"] is True
    assert qlora["training_memory_mode"]["mode_id"] == "qlora-4bit-nf4"
    assert qlora["resolved_config"]["training"]["quant_bits"] == 4
    assert qlora["training_parameters"]["config_sha256"] != result["training_parameters"]["config_sha256"]
    with pytest.raises(ValueError, match="预检已过期或已使用"):
        launcher.start(result["confirmation_token"])
    assert list((workspace / "runs").glob("**/run.json")) == []
    assert not (workspace / ".onegpu-posttrain").exists()


def test_preflight_blocks_missing_training_runtime_without_creating_run_artifacts(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    recipe = workspace / launcher_module.RECIPE
    recipe.parent.mkdir(parents=True)
    _write_sft_recipe(recipe)
    model = workspace / "model"
    model.mkdir()
    dataset = workspace / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    inputs = {"recipe_path": recipe, "recipe_sha256": "a" * 64, "model_path": model,
              "model_sha256": "b" * 64, "model_id": launcher_module.MODEL_ID,
              "manifest_sha256": "c" * 64, "canonical_sha256": "d" * 64,
              "native_path": dataset, "native_sha256": "e" * 64,
              "dataset_id": launcher_module.DATASET_ID}
    monkeypatch.setattr(launcher_module.SFTLauncher, "_resolve_fixed_inputs", lambda self, dataset_id=launcher_module.DATASET_ID: inputs)
    monkeypatch.setattr(launcher_module, "_gpu_snapshot", lambda: {"available": False})
    monkeypatch.setattr(launcher_module, "_sft_runtime_available", lambda: False)
    from onegpu_posttrain import cli
    monkeypatch.setattr(cli, "cmd_plan_sft", lambda args: Path(args.output, "train_plan.json").write_text("{}", encoding="utf-8"))
    result = launcher_module.SFTLauncher(workspace).preflight()
    assert result["ok"] is False
    assert "swift 命令" in result["blockers"][0]
    assert not (workspace / "runs").exists()


def test_fixed_catalog_checks_recipe_model_manifest_and_content_digests(tmp_path):
    import yaml

    workspace = tmp_path.resolve()
    recipe_path = workspace / launcher_module.RECIPE
    recipe_path.parent.mkdir(parents=True)
    model_path = workspace / "models/m1"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    model_digest = sha256_directory(model_path)
    registry = workspace / "runs/mini/cpt-finemath-5m-s42-v3/model_artifacts" / f"{launcher_module.MODEL_ID}.json"
    registry.parent.mkdir(parents=True)
    producer_run_id = "run-fixed-cpt-merge"
    registry.write_text(json.dumps({"model_id": launcher_module.MODEL_ID, "kind": "merged", "stage": "cpt",
                                    "path": "models/m1", "produced_by_run_id": producer_run_id,
                                    "parent_model_ids": ["model-base"],
                                    "digest": {"value": model_digest, "status": "available"}}), encoding="utf-8")
    merge_lineage = registry.parent.parent / "merge_runs" / producer_run_id / "lineage.json"
    merge_lineage.parent.mkdir(parents=True)
    merge_lineage.write_text(json.dumps({"run_id": producer_run_id, "run_type": "merge", "status": "completed",
                                         "started_at": "2026-01-01T00:00:00Z",
                                         "completed_at": "2026-01-01T00:01:00Z",
                                         "outputs": [{"id": launcher_module.MODEL_ID, "kind": "model",
                                                      "digest": {"value": model_digest, "status": "available"}}]}),
                             encoding="utf-8")
    canonical = workspace / "data/canonical.jsonl"
    native = workspace / "data/native.jsonl"
    manifest_path = workspace / "data/manifest.yaml"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("canonical\n", encoding="utf-8")
    native.write_text("native\n", encoding="utf-8")
    manifest_path.write_text(yaml.safe_dump({
        "dataset_id": launcher_module.DATASET_ID, "stage": "sft", "sample_count": 307,
        "hashes": {"canonical_records_sha256": sha256_file(canonical)},
        "metadata": {"native_sha256": sha256_file(native), "usage": "exploratory"},
    }), encoding="utf-8")
    recipe_path.write_text(yaml.safe_dump({
        "stage": "sft", "parent_model_id": launcher_module.MODEL_ID, "parent_model_path": "models/m1",
        "seed": 42, "training": {"max_length": 1024},
        "dataset": {"dataset_id": launcher_module.DATASET_ID, "manifest": "data/manifest.yaml",
                    "canonical": "data/canonical.jsonl", "native": "data/native.jsonl",
                    "expected_count": 307, "usage": "exploratory"},
    }), encoding="utf-8")
    launcher = launcher_module.SFTLauncher(workspace)
    assert launcher._resolve_fixed_inputs()["dataset_id"] == launcher_module.DATASET_ID
    native.write_text("modified\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        launcher._resolve_fixed_inputs()


def test_cli_run_id_option_is_limited_to_run_sft():
    from onegpu_posttrain.cli import build_parser

    sft = build_parser().parse_args(["run-sft", "--model", "m", "--dataset", "d", "--output", "o", "--run-id", "run-ui-42"])
    assert sft.run_id == "run-ui-42"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run-dpo", "--model", "m", "--dataset", "d", "--output", "o", "--run-id", "run-ui-42"])


def test_stage_log_reader_is_bounded_and_parses_observed_swift_metrics(tmp_path):
    service = launcher_module.SFTLauncher(tmp_path)
    output = tmp_path / "runs/mini/ui-launcher/run-test"
    log = output / "logs/train/stdout.log"
    log.parent.mkdir(parents=True)
    log.write_text("prefix\n{'loss': 0.42, 'token_acc': 0.8, 'global_step/max_steps': '10/38', 'remaining_time': '2m'}\n", encoding="utf-8")
    state = {"output": output.relative_to(tmp_path).as_posix()}
    tail = service._stage_log_tail(state)
    assert tail and "global_step/max_steps" in tail["text"]
    assert service._parse_training_progress(tail["text"]) == {"step": 10, "total": 38}
    assert service._parse_training_metrics(tail["text"]) == {"loss": 0.42, "token_acc": 0.8, "remaining_time": "2m"}
    log.write_text("x" * (launcher_module.MAX_LOG_TAIL * 2), encoding="utf-8")
    bounded = service._stage_log_tail(state)
    assert bounded and len(bounded["text"].encode("utf-8")) <= launcher_module.MAX_LOG_TAIL
    assert bounded["truncated"] is True


def test_stage_log_reader_ignores_missing_symlink_and_out_of_workspace_paths(tmp_path):
    service = launcher_module.SFTLauncher(tmp_path)
    output = tmp_path / "runs/mini/ui-launcher/run-test"
    state = {"output": output.relative_to(tmp_path).as_posix()}
    assert service._stage_log_tail(state) is None
    output.mkdir(parents=True)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.log"
    outside.write_text("secret outside workspace", encoding="utf-8")
    try:
        (output / "logs").mkdir()
        (output / "logs" / "train").symlink_to(outside.parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    assert service._stage_log_tail(state) is None
    assert service._stage_log_tail({"output": "../outside"}) is None


def _prepare_eval_preflight_fixture(workspace: Path, monkeypatch, *, evalscope=True):
    import yaml

    train_id = "run-mini-sft-fixture"
    output = workspace / "runs/mini/ui-launcher" / train_id
    model = output / "merged/M2-SFT"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    artifacts = output / "model_artifacts"
    artifacts.mkdir()
    digest = "a" * 64
    model_id = f"model-{digest[:20]}"
    (artifacts / f"{model_id}.json").write_text(json.dumps({
        "model_id": model_id, "kind": "merged", "path": str(model),
        "digest": {"status": "available", "value": digest}, "produced_by_run_id": "run-sft-merge-fixture", "stage": "sft",
    }), encoding="utf-8")
    service = launcher_module.SFTLauncher(workspace)
    monkeypatch.setattr(service, "_has_cli_success", lambda state: True)
    monkeypatch.setattr(launcher_module, "sha256_directory", lambda _: digest)
    monkeypatch.setattr(launcher_module, "_gpu_snapshot", lambda: {"available": False, "status": "unknown"})
    recipe = workspace / launcher_module.EVAL_RECIPE
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text(yaml.safe_dump({"stage": "sft_eval", "evaluation_config": launcher_module.EVAL_CONFIG, "seed": 42}), encoding="utf-8")
    config = workspace / launcher_module.EVAL_CONFIG
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump({"protocol_version": "math-eval-v1", "benchmarks": ["gsm8k", "math_500"],
        "benchmark_execution": [
            {"dataset": "gsm8k", "limit": 5},
            {"dataset": "math_500", "limit": 1, "dataset_args": {"subset_list": ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]}},
        ]}), encoding="utf-8")
    service._write_state({"run_id": train_id, "status": "completed", "output": output.relative_to(workspace).as_posix(),
                          "model_ids": [model_id], "started_at": "2026-10-20T00:00:00+00:00", "outputs": []})
    monkeypatch.setattr(launcher_module.shutil, "which", lambda command, path=None: "evalscope.exe" if evalscope and command == "evalscope" else None)
    return service, train_id, output, model_id, digest


def test_evaluation_preflight_is_read_only_pinned_and_requires_verified_merged_m2(tmp_path, monkeypatch):
    service, train_id, _output, model_id, digest = _prepare_eval_preflight_fixture(tmp_path.resolve(), monkeypatch)
    result = service.evaluation_preflight(train_id)
    assert result["ok"] is True, result
    assert result["protocol_version"] == "math-eval-v1"
    assert result["benchmarks"] == [{"name": "GSM8K", "samples": 5},
                                     {"name": "MATH-500", "samples": 5, "sampling": "Level 1–5 each 1"}]
    assert result["model"]["model_id"] == model_id and result["model"]["sha256"] == digest
    assert result["model"]["produced_by_run_id"] == "run-sft-merge-fixture"
    assert result["output"].startswith("runs/evaluations/run-")
    assert result["path_budget"]["supported"] is True
    assert not (service.workspace / result["output"]).exists()
    assert not (service.records_root / f"{result['evaluation_run_id']}.json").exists()


def test_evaluation_preflight_blocks_missing_evalscope_without_side_effects(tmp_path, monkeypatch):
    service, train_id, output, _, _ = _prepare_eval_preflight_fixture(tmp_path.resolve(), monkeypatch, evalscope=False)
    result = service.evaluation_preflight(train_id)
    assert result["ok"] is False and "EvalScope" in result["blockers"][0]
    assert not (output / "evaluations").exists()


def test_evaluation_preflight_refuses_failed_training_run(tmp_path, monkeypatch):
    service, train_id, _, _, _ = _prepare_eval_preflight_fixture(tmp_path.resolve(), monkeypatch)
    state = service._read_state(train_id)
    state["status"] = "oom"
    service._write_state(state)
    result = service.evaluation_preflight(train_id)
    assert result["ok"] is False
    assert "成功完成" in result["blockers"][0]


def test_evaluation_run_uses_shared_lock_one_use_token_and_persists_interruption(tmp_path, monkeypatch):
    service, train_id, _, _, _ = _prepare_eval_preflight_fixture(tmp_path.resolve(), monkeypatch)
    import threading

    class SlowEvalProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.StringIO("eval started\n")
            self.returncode = None
            self.ready = threading.Event()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if not self.ready.wait(timeout):
                raise launcher_module.subprocess.TimeoutExpired("eval", timeout)
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = -2
            self.ready.set()

        def kill(self):
            self.returncode = -9
            self.ready.set()

    monkeypatch.setattr(launcher_module.subprocess, "Popen", lambda *args, **kwargs: SlowEvalProcess())
    preflight = service.evaluation_preflight(train_id)
    started = service.start_evaluation(preflight["confirmation_token"])
    assert started["status"] == "running"
    parent_during_eval = service.get_run(train_id)
    assert parent_during_eval["evaluation_status"] == "running"
    assert parent_during_eval["status"] == "completed"
    with pytest.raises(ValueError, match="已使用"):
        service.start_evaluation(preflight["confirmation_token"])
    service._confirmations["train-token"] = {"expires": time.time() + 60, "run_id": "run-other", "fingerprint": "unused"}
    with pytest.raises(RuntimeError, match="已有 SFT"):
        service.start("train-token")
    second = service.evaluation_preflight(train_id)
    with pytest.raises(RuntimeError, match="已有任务"):
        service.start_evaluation(second["confirmation_token"])
    service.shutdown(grace_seconds=0.1)
    if service._monitor_thread:
        service._monitor_thread.join(timeout=1)
    for _ in range(100):
        result = service._read_state(started["run_id"])
        if result and result["status"] == "interrupted":
            break
        time.sleep(0.01)
    assert result["status"] == "interrupted"
    assert service.get_run(train_id)["evaluation_status"] == "interrupted"


def test_evaluation_success_persists_separate_lineage_and_real_sample_denominators(tmp_path, monkeypatch):
    service, train_id, _, model_id, _ = _prepare_eval_preflight_fixture(tmp_path.resolve(), monkeypatch)

    class SuccessfulEvalProcess:
        returncode = 0

        def __init__(self, argv, **_kwargs):
            self.stdout = io.StringIO("benchmark progress 1/5\nbenchmark progress 5/5\n")
            output = Path(argv[argv.index("--output") + 1])
            run_id = argv[argv.index("--run-id") + 1]
            (output / "eval.json").write_text(json.dumps({
                "resolved_evaluation": {"protocol_version": "math-eval-v1"},
                "evalscope_reports": {"gsm8k": {"score": 0.6, "samples": 5}, "math_500": {"score": 0.4, "samples": 4}},
            }), encoding="utf-8")
            (output / "evaluation_lineage.json").write_text(json.dumps({"eval_id": "eval-independent", "produced_by_run_id": run_id,
                "model_id": model_id, "model_produced_by_run_id": "run-sft-merge-fixture", "status": "completed"}), encoding="utf-8")
            (output / "evaluation_run.json").write_text(json.dumps({"run_id": run_id, "status": "completed"}), encoding="utf-8")

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(launcher_module.subprocess, "Popen", SuccessfulEvalProcess)
    preflight = service.evaluation_preflight(train_id)
    started = service.start_evaluation(preflight["confirmation_token"])
    assert started["run_id"] != train_id
    for _ in range(100):
        parent = service.get_run(train_id)
        if parent and parent["evaluation_status"] == "completed":
            break
        time.sleep(0.01)
    assert parent["evaluation_run_id"] == started["run_id"]
    assert parent["evaluation_metrics"]["metrics"] == [
        {"name": "GSM8K", "metric": "accuracy@1", "value": 0.6, "denominator": 5},
        {"name": "MATH-500", "metric": "accuracy@1", "value": 0.4, "denominator": 4},
    ]
    result = service._read_state(started["run_id"])
    assert result["run_kind"] == "evaluation" and result["status"] == "completed"
    assert service.get_run(train_id)["status"] == "completed"
    with TestClient(create_app(tmp_path.resolve(), enable_experiment_launcher=True)) as client:
        restored = client.get("/api/launcher/run").json()["run"]
        assert restored["run_id"] == train_id
        assert restored["evaluation_run_id"] == started["run_id"]
        assert restored["evaluation_status"] == "completed"
        assert restored["evaluation_metrics"]["metrics"][1]["denominator"] == 4


def test_evaluation_api_requires_local_token_and_only_accepts_fixed_request_shapes(tmp_path, monkeypatch):
    app = create_app(tmp_path, enable_experiment_launcher=True)
    launcher = app.state.experiment_launcher
    launcher.evaluation_preflight = lambda training_run_id: {"ok": True, "training_run_id": training_run_id}
    launcher.start_evaluation = lambda confirmation_token: {"run_id": "run-eval-fixture", "status": "running"}
    with TestClient(app) as client:
        token = client.get("/api/launcher/capabilities").json()["token"]
        headers = {"X-Local-Launcher-Token": token}
        assert client.post("/api/launcher/evaluation/preflight", json={"training_run_id": "run-ok"}).status_code == 403
        assert client.post("/api/launcher/evaluation/preflight", json={"training_run_id": "run-ok", "path": "C:/"}, headers=headers).status_code == 422
        response = client.post("/api/launcher/evaluation/preflight", json={"training_run_id": "run-ok"}, headers=headers)
        assert response.status_code == 200 and response.json()["ok"] is True
        assert client.post("/api/launcher/evaluation/start", json={"confirmation_token": "x", "command": "whoami"}, headers=headers).status_code == 422
        started = client.post("/api/launcher/evaluation/start", json={"confirmation_token": "x"}, headers=headers)
        assert started.status_code == 200 and started.json()["status"] == "running"


def test_stage_evaluation_api_requires_token_and_only_accepts_profile_identities(tmp_path):
    app = create_app(tmp_path, enable_experiment_launcher=True)
    launcher = app.state.experiment_launcher
    launcher.stage_evaluation_preflight = lambda payload: {"ok": True, "profile_id": payload["profile_id"]}
    launcher.start_stage_evaluation = lambda token: {"run_id": "run-eval-profile", "status": "running"}
    with TestClient(app) as client:
        token = client.get("/api/launcher/capabilities").json()["token"]
        headers = {"X-Local-Launcher-Token": token}
        payload = {"training_run_id": "run-train", "profile_id": "sft-math-eval-v1"}
        assert client.post("/api/training/evaluation/preflight", json=payload).status_code == 403
        assert client.post("/api/training/evaluation/preflight", json=payload | {"command": "whoami"}, headers=headers).status_code == 422
        response = client.post("/api/training/evaluation/preflight", json=payload, headers=headers)
        assert response.status_code == 200 and response.json()["ok"] is True
        malformed = client.post("/api/training/evaluation/preflight", json={**payload, "training_run_id": "bad/id"}, headers=headers)
        assert malformed.status_code == 422
        assert client.post("/api/training/evaluation/start", json={"confirmation_token": "t", "model_path": "C:/"}, headers=headers).status_code == 422
        started = client.post("/api/training/evaluation/start", json={"confirmation_token": "t"}, headers=headers)
        assert started.status_code == 200 and started.json()["status"] == "running"


def test_completed_legacy_run_discovers_registered_merged_model_for_evaluation_cta(tmp_path, monkeypatch):
    service, train_id, _, model_id, _ = _prepare_eval_preflight_fixture(tmp_path.resolve(), monkeypatch)
    state = service._read_state(train_id)
    state.pop("model_artifacts", None)
    service._write_state(state)
    refreshed = service.get_run(train_id)
    assert any(item["model_id"] == model_id and item["kind"] == "merged" and item["stage"] == "sft"
               for item in refreshed["model_artifacts"])


def test_evaluation_preflight_blocks_output_collision_without_overwriting(tmp_path, monkeypatch):
    service, train_id, _output, _, _ = _prepare_eval_preflight_fixture(tmp_path.resolve(), monkeypatch)
    collision_id = "run-evaluate-collision"
    monkeypatch.setattr(launcher_module, "new_run_id", lambda *args, **kwargs: collision_id)
    target = service.workspace / "runs" / "evaluations" / collision_id
    target.mkdir(parents=True)
    marker = target / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    result = service.evaluation_preflight(train_id)
    assert result["ok"] is False and "冲突" in result["blockers"][0]
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_evaluation_path_budget_accepts_current_checkout_and_blocks_impossible_windows_root(tmp_path):
    checkout = launcher_module._evaluation_path_budget(Path.cwd(), "run-evaluate-current", windows=True)
    assert checkout["supported"] is True
    current = launcher_module._evaluation_path_budget(tmp_path.resolve(), "run-evaluate-current", windows=True)
    assert current["supported"] is True
    impossible_root = Path("C:/") / ("deep" * 70)
    impossible = launcher_module._evaluation_path_budget(impossible_root, "run-evaluate-long", windows=True)
    assert impossible["supported"] is False
    assert impossible["path_length"] > impossible["safe_limit"]
    # The safety check is platform-aware; POSIX is not rejected for a Windows-only limit.
    assert launcher_module._evaluation_path_budget(impossible_root, "run-evaluate-long", windows=False)["supported"] is True


def test_evaluation_preflight_path_budget_blocker_is_actionable_and_has_no_token(tmp_path, monkeypatch):
    service, train_id, _, _, _ = _prepare_eval_preflight_fixture(tmp_path.resolve(), monkeypatch)
    monkeypatch.setattr(launcher_module, "_evaluation_path_budget", lambda *_args: {
        "supported": False, "path_length": 300, "safe_limit": 240, "windows_rules": True})
    monkeypatch.setattr(launcher_module.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("preflight must never create a process"))
    result = service.evaluation_preflight(train_id)
    assert result["ok"] is False
    assert "安全上限" in result["blockers"][0]
    assert "confirmation_token" not in result
    assert len(list(service.records_root.glob("*.json"))) == 1


@pytest.mark.parametrize("output", [
    "runs/evaluations/run-eval-shallow",
    "runs/mini/ui-launcher/run-train-legacy/evaluations/run-eval-legacy",
])
def test_evaluation_result_and_lineage_endpoints_resolve_shallow_and_legacy_paths(tmp_path, output):
    workspace = tmp_path.resolve()
    app = create_app(workspace, enable_experiment_launcher=True)
    launcher = app.state.experiment_launcher
    eval_id = "run-eval-shallow" if "shallow" in output else "run-eval-legacy"
    eval_dir = workspace / output
    eval_dir.mkdir(parents=True)
    (eval_dir / "eval.json").write_text(json.dumps({"actual_samples": 5}), encoding="utf-8")
    (eval_dir / "evaluation_lineage.json").write_text(json.dumps({"produced_by_run_id": eval_id}), encoding="utf-8")
    launcher._write_state({"run_id": eval_id, "run_kind": "evaluation", "status": "completed", "output": output})
    with TestClient(app) as client:
        assert client.get(f"/api/launcher/evaluations/{eval_id}/result").json() == {"actual_samples": 5}
        assert client.get(f"/api/launcher/evaluations/{eval_id}/lineage").json()["produced_by_run_id"] == eval_id


def test_service_reload_marks_inflight_evaluation_and_parent_interrupted(tmp_path):
    workspace = tmp_path.resolve()
    first = launcher_module.SFTLauncher(workspace)
    parent_id, eval_id = "run-sft-parent", "run-eval-child"
    first._write_state({"run_id": parent_id, "status": "completed", "output": "runs/parent",
                        "evaluation_status": "running", "evaluation_run_id": eval_id})
    first._write_state({"run_id": eval_id, "run_kind": "evaluation", "parent_run_id": parent_id,
                        "status": "running", "output": "runs/evaluation"})
    recovered = launcher_module.SFTLauncher(workspace)
    assert recovered._read_state(eval_id)["status"] == "interrupted"
    assert recovered._read_state(eval_id)["evaluation_status"] == "interrupted"
    assert recovered._read_state(eval_id)["completed_at"]
    assert recovered._read_state(parent_id)["evaluation_status"] == "interrupted"


def test_service_reload_repairs_interrupted_evaluation_status_mismatch(tmp_path):
    workspace = tmp_path.resolve()
    first = launcher_module.SFTLauncher(workspace)
    parent_id, eval_id = "run-sft-parent", "run-eval-child"
    first._write_state({"run_id": parent_id, "status": "completed", "output": "runs/parent",
                        "evaluation_status": "running", "evaluation_run_id": eval_id})
    first._write_state({"run_id": eval_id, "run_kind": "evaluation", "parent_run_id": parent_id,
                        "status": "interrupted", "evaluation_status": "running", "output": "runs/evaluation"})
    recovered = launcher_module.SFTLauncher(workspace)
    assert recovered._read_state(eval_id)["status"] == "interrupted"
    assert recovered._read_state(eval_id)["evaluation_status"] == "interrupted"
    assert recovered._read_state(eval_id)["completed_at"]
    assert recovered._read_state(parent_id)["evaluation_status"] == "interrupted"


def test_service_reload_recovers_terminal_evaluation_artifacts_and_parent_state(tmp_path):
    workspace = tmp_path.resolve()
    first = launcher_module.SFTLauncher(workspace)
    parent_id, eval_id = "run-sft-parent", "run-eval-recoverable"
    output = workspace / "runs/evaluations" / eval_id
    output.mkdir(parents=True)
    (output / "evaluation_run.json").write_text(json.dumps({
        "run_id": eval_id, "run_type": "evaluate", "status": "completed",
        "completed_at": "2026-10-25T01:00:00+00:00",
    }), encoding="utf-8")
    (output / "evaluation_lineage.json").write_text(json.dumps({
        "produced_by_run_id": eval_id, "status": "completed",
    }), encoding="utf-8")
    first._write_state({"run_id": parent_id, "status": "completed", "output": "runs/parent",
                        "evaluation_status": "running", "evaluation_run_id": eval_id})
    first._write_state({"run_id": eval_id, "run_kind": "evaluation", "parent_run_id": parent_id,
                        "status": "running", "evaluation_status": "running",
                        "output": output.relative_to(workspace).as_posix()})

    recovered = launcher_module.SFTLauncher(workspace)
    eval_state = recovered._read_state(eval_id)
    assert eval_state["status"] == "completed"
    assert eval_state["evaluation_status"] == "completed"
    assert eval_state["evaluation_lineage_path"].endswith("evaluation_lineage.json")
    assert recovered._read_state(parent_id)["evaluation_status"] == "completed"


def test_interrupted_evaluation_is_reconciled_when_cli_artifacts_arrive_later(tmp_path):
    workspace = tmp_path.resolve()
    service = launcher_module.SFTLauncher(workspace)
    parent_id, eval_id = "run-sft-parent", "run-eval-late-artifacts"
    output = workspace / "runs/evaluations" / eval_id
    output.mkdir(parents=True)
    service._write_state({"run_id": parent_id, "status": "completed", "output": "runs/parent",
                          "evaluation_status": "running", "evaluation_run_id": eval_id})
    service._write_state({"run_id": eval_id, "run_kind": "evaluation", "parent_run_id": parent_id,
                          "status": "running", "evaluation_status": "running",
                          "output": output.relative_to(workspace).as_posix()})

    assert service.get_run(eval_id)["status"] == "interrupted"
    (output / "evaluation_run.json").write_text(json.dumps({
        "run_id": eval_id, "run_type": "evaluate", "status": "completed",
    }), encoding="utf-8")
    (output / "evaluation_lineage.json").write_text(json.dumps({
        "produced_by_run_id": eval_id, "status": "completed",
    }), encoding="utf-8")

    recovered = service.get_run(parent_id)
    assert recovered["evaluation_status"] == "completed"
    assert service._read_state(eval_id)["status"] == "completed"


def test_evaluation_supervisor_preserves_oom_terminal_state(tmp_path):
    service = launcher_module.SFTLauncher(tmp_path.resolve())
    output = tmp_path / "runs/evaluation"
    output.mkdir(parents=True)
    (output / "evaluation_run.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    state = {"run_id": "run-eval-oom", "run_kind": "evaluation", "parent_run_id": "run-parent",
             "output": output.relative_to(tmp_path).as_posix(), "status": "running"}

    class OOMProcess:
        def wait(self): return 1

    service._write_state({"run_id": "run-parent", "status": "completed", "output": "runs/parent"})
    (output / "launcher.log").write_text("CUDA out of memory\n", encoding="utf-8")
    service._monitor_evaluation(OOMProcess(), state, output / "launcher.log")
    assert state["status"] == "oom"
    assert service._read_state("run-parent")["evaluation_status"] == "oom"


def test_partial_evaluation_without_scores_stays_partial_and_exposes_no_result(tmp_path):
    workspace = tmp_path.resolve()
    service = launcher_module.SFTLauncher(workspace)
    output = workspace / "runs/evaluations/run-eval-partial"
    output.mkdir(parents=True)
    (output / "evaluation_run.json").write_text(json.dumps({"run_id": "run-eval-partial", "status": "partial"}), encoding="utf-8")
    parent = {"run_id": "run-parent", "status": "completed", "output": "runs/parent"}
    service._write_state(parent)
    state = {"run_id": "run-eval-partial", "run_kind": "evaluation", "parent_run_id": "run-parent",
             "output": output.relative_to(workspace).as_posix(), "status": "running"}

    class PartialProcess:
        stdout = io.StringIO("benchmark produced no report\n")
        def wait(self): return 0

    service._monitor_evaluation(PartialProcess(), state, output / "launcher.log")
    assert state["status"] == "partial"
    assert not (output / "eval.json").exists()
    assert service._read_state("run-parent")["evaluation_status"] == "partial"
    with TestClient(create_app(workspace, enable_experiment_launcher=True)) as client:
        result = client.get("/api/launcher/evaluations/run-eval-partial/result")
        assert result.status_code == 404


def test_completed_stage_run_keeps_configuration_snapshot_for_refresh(tmp_path):
    service = launcher_module.SFTLauncher(tmp_path)
    parameters = {
        "defaults": {"max_length": 1024, "num_train_epochs": 1},
        "overrides": {"max_length": 768},
        "effective": {"max_length": 768, "num_train_epochs": 1},
        "config_sha256": "a" * 64,
    }
    service._write_state({
        "run_id": "run-stage-config-snapshot", "run_kind": "stage-training", "stage": "sft",
        "profile_id": "sft-mini-1024-v1", "status": "completed", "dataset_id": "ds-registered",
        "parent_model_id": "model-parent", "training_parameters": parameters,
        "effective_config_sha256": parameters["config_sha256"], "outputs": [], "model_ids": [],
        "model_artifacts": [],
    })

    restored = service.get_run("run-stage-config-snapshot")

    assert restored is not None
    assert restored["profile_id"] == "sft-mini-1024-v1"
    assert restored["dataset_id"] == "ds-registered"
    assert restored["parent_model_id"] == "model-parent"
    assert restored["training_parameters"] == parameters
    assert restored["effective_config_sha256"] == "a" * 64


def test_sft_export_failure_keeps_qlora_run_ineligible_for_evaluation_handoff(tmp_path):
    service = launcher_module.SFTLauncher(tmp_path)
    run_id = "run-qlora-export-failed"
    output = tmp_path / "runs" / "mini" / "ui-launcher" / run_id
    output.mkdir(parents=True)
    (output / "run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (output / "lineage.json").write_text(json.dumps({"run_id": run_id, "status": "completed"}), encoding="utf-8")
    service._write_state({"run_id": run_id, "status": "completed", "output": output.relative_to(tmp_path).as_posix(),
                          "training_memory_mode": {"mode_id": "qlora-4bit-nf4"}, "outputs": [], "model_ids": []})

    preflight = service.evaluation_preflight(run_id)

    assert preflight["ok"] is False
    assert "未找到已合并的 M2-SFT 模型" in preflight["blockers"][0]
    assert not (output / "merged" / "M2-SFT").exists()


def test_sft_cli_persists_launcher_supplied_run_id(tmp_path, monkeypatch):
    from onegpu_posttrain import cli

    dataset = tmp_path / "data.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    config = tmp_path / "recipe.yaml"
    config.write_text(
        "stage: sft\ntraining:\n  max_steps: 1\n  max_length: 768\n"
        "_onegpu_ui:\n  profile_id: sft-mini-1024-v1\n  parameter_overrides: {max_length: 768}\n"
        "  training_memory_mode: {mode_id: qlora-4bit-nf4, label_zh: QLoRA, settings: {quant_bits: 4}}\n"
        f"  effective_config_sha256: {'a' * 64}\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    output = tmp_path / "run"
    monkeypatch.setattr(cli, "run_monitored_command", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(cli, "discover_latest_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(cli, "capture_environment", lambda: {"python": "test"})
    args = SimpleNamespace(config=str(config), model=str(model), dataset=str(dataset), output=str(output),
                           merge=False, dry_run=False, run_id="run-ui-42")
    assert cli.cmd_run_sft(args) == 0
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    lineage = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    assert run["run_id"] == lineage["run_id"] == "run-ui-42"
    assert run["training"]["max_length"] == resolved["training"]["max_length"] == 768
    assert run["parameter_overrides"] == resolved["parameter_overrides"] == {"max_length": 768}
    assert run["training_memory_mode"]["mode_id"] == resolved["training_memory_mode"]["mode_id"] == "qlora-4bit-nf4"
    assert lineage["training_memory_mode"]["mode_id"] == "qlora-4bit-nf4"
    assert run["effective_config_sha256"] == resolved["effective_config_sha256"] == "a" * 64
    assert lineage["resolved_config_sha256"] == sha256_json(resolved)


@pytest.mark.parametrize(("exit_code", "log_line", "expected"), [
    (0, "global_step=4/4\n", "completed"),
    (1, "training failed\n", "failed"),
    (1, "CUDA out of memory\n", "oom"),
])
def test_fake_cli_supervisor_reports_real_terminal_state_and_one_use_token(tmp_path, monkeypatch, exit_code, log_line, expected):
    workspace = tmp_path.resolve()
    recipe = workspace / launcher_module.RECIPE
    recipe.parent.mkdir(parents=True)
    _write_sft_recipe(recipe)
    model = workspace / "model"
    model.mkdir()
    dataset = workspace / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    inputs = {"recipe_path": recipe, "recipe_sha256": "a" * 64, "model_path": model,
              "model_sha256": "b" * 64, "model_id": launcher_module.MODEL_ID,
              "manifest_sha256": "c" * 64, "canonical_sha256": "d" * 64,
              "native_path": dataset, "native_sha256": "e" * 64,
              "dataset_id": launcher_module.DATASET_ID}
    monkeypatch.setattr(launcher_module.SFTLauncher, "_resolve_fixed_inputs", lambda self, dataset_id=launcher_module.DATASET_ID: inputs)
    monkeypatch.setattr(launcher_module, "_gpu_snapshot", lambda: {"available": False})
    monkeypatch.setattr(launcher_module, "_sft_runtime_available", lambda: True)
    monkeypatch.setattr(launcher_module, "inspect_training_memory_mode",
                        lambda mode_id: {"ok": True, "mode_id": mode_id, "dependencies": []})
    from onegpu_posttrain import cli
    monkeypatch.setattr(cli, "cmd_plan_sft", lambda args: Path(args.output, "train_plan.json").write_text("{}", encoding="utf-8"))

    class FakeProcess:
        def __init__(self, argv, cwd):
            assert "--merge" in argv
            config_path = Path(argv[argv.index("--config") + 1])
            if not config_path.is_absolute():
                config_path = Path(cwd) / config_path
            resolved_recipe = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
            assert resolved_recipe["training"]["max_length"] == 768
            assert resolved_recipe["training"]["num_train_epochs"] == 2
            assert resolved_recipe["training"]["quant_bits"] == 4
            self.stdout = io.StringIO(log_line)
            self.returncode = exit_code
            output = Path(argv[argv.index("--output") + 1])
            run_id = argv[argv.index("--run-id") + 1]
            if exit_code == 0:
                (output / "run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
                (output / "lineage.json").write_text(json.dumps({"run_id": run_id, "status": "completed", "outputs": []}), encoding="utf-8")

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(launcher_module.subprocess, "Popen", lambda argv, **kwargs: FakeProcess(argv, kwargs["cwd"]))
    service = launcher_module.SFTLauncher(workspace)
    preflight = service.preflight(parameter_overrides={"max_length": 768, "num_train_epochs": 2},
                                  memory_mode_id="qlora-4bit-nf4")
    started = service.start(preflight["confirmation_token"])
    assert started["status"] == "running"
    if expected == "completed":
        service.shutdown()
    with pytest.raises(ValueError, match="过期或已使用"):
        service.start(preflight["confirmation_token"])
    for _ in range(100):
        result = service.get_run(started["run_id"])
        if result and result["status"] not in {"starting", "running"}:
            break
        time.sleep(0.01)
    assert result["status"] == expected
    assert result["training_memory_mode"]["mode_id"] == "qlora-4bit-nf4"
    assert result["training_memory_mode"]["settings"]["quant_bits"] == 4
    lifecycle = service._run_supervisor.get(started["run_id"])
    assert lifecycle["status"] == expected
    assert lifecycle["log_bytes"] > 0
    if expected == "completed":
        assert result["progress"] == {"step": 4, "total": 4}
        assert lifecycle["progress"] == {"step": 4, "total": 4}
        assert result["evaluation_status"] == "pending"
    assert log_line.strip() in result["log_tail"]


def test_duplicate_launcher_run_is_rejected_and_service_shutdown_interrupts(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    recipe = workspace / launcher_module.RECIPE
    recipe.parent.mkdir(parents=True)
    _write_sft_recipe(recipe)
    model = workspace / "model"
    model.mkdir()
    dataset = workspace / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    inputs = {"recipe_path": recipe, "recipe_sha256": "a" * 64, "model_path": model,
              "model_sha256": "b" * 64, "model_id": launcher_module.MODEL_ID,
              "manifest_sha256": "c" * 64, "canonical_sha256": "d" * 64,
              "native_path": dataset, "native_sha256": "e" * 64,
              "dataset_id": launcher_module.DATASET_ID}
    monkeypatch.setattr(launcher_module.SFTLauncher, "_resolve_fixed_inputs", lambda self, dataset_id=launcher_module.DATASET_ID: inputs)
    monkeypatch.setattr(launcher_module, "_gpu_snapshot", lambda: {"available": False})
    monkeypatch.setattr(launcher_module, "_sft_runtime_available", lambda: True)
    from onegpu_posttrain import cli
    monkeypatch.setattr(cli, "cmd_plan_sft", lambda args: Path(args.output, "train_plan.json").write_text("{}", encoding="utf-8"))

    class SlowProcess:
        def __init__(self, argv):
            self.stdout = io.StringIO("")
            self.returncode = None
            self.ready = __import__("threading").Event()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if not self.ready.wait(timeout):
                raise launcher_module.subprocess.TimeoutExpired("fake", timeout)
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = -2
            self.ready.set()

        def kill(self):
            self.returncode = -9
            self.ready.set()

    monkeypatch.setattr(launcher_module.subprocess, "Popen", lambda argv, **kwargs: SlowProcess(argv))
    service = launcher_module.SFTLauncher(workspace)
    first = service.preflight()
    active = service.start(first["confirmation_token"])
    second = service.preflight()
    with pytest.raises(RuntimeError, match="已有 SFT"):
        service.start(second["confirmation_token"])
    service.shutdown(grace_seconds=1)
    for _ in range(100):
        result = service.get_run(active["run_id"])
        if result and result["status"] == "interrupted":
            break
        time.sleep(0.01)
    assert result["status"] == "interrupted"


def test_fake_cli_api_vertical_slice_exposes_actual_lineage_and_model_artifact(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    recipe = workspace / launcher_module.RECIPE
    recipe.parent.mkdir(parents=True)
    _write_sft_recipe(recipe)
    model = workspace / "model"
    model.mkdir()
    dataset = workspace / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    inputs = {"recipe_path": recipe, "recipe_sha256": "a" * 64, "model_path": model,
              "model_sha256": "b" * 64, "model_id": launcher_module.MODEL_ID,
              "manifest_sha256": "c" * 64, "canonical_sha256": "d" * 64,
              "native_path": dataset, "native_sha256": "e" * 64,
              "dataset_id": launcher_module.DATASET_ID}
    monkeypatch.setattr(launcher_module.SFTLauncher, "_resolve_fixed_inputs", lambda self, dataset_id=launcher_module.DATASET_ID: inputs)
    monkeypatch.setattr(launcher_module, "_gpu_snapshot", lambda: {"available": False})
    monkeypatch.setattr(launcher_module, "_sft_runtime_available", lambda: True)
    from onegpu_posttrain import cli
    monkeypatch.setattr(cli, "cmd_plan_sft", lambda args: Path(args.output, "train_plan.json").write_text("{}", encoding="utf-8"))

    class FakeProcess:
        def __init__(self, argv):
            self.stdout = io.StringIO("step=2/2\n")
            self.returncode = 0
            output = Path(argv[argv.index("--output") + 1])
            run_id = argv[argv.index("--run-id") + 1]
            model_id = "model-0123456789abcdef0123"
            (output / "run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
            (output / "lineage.json").write_text(json.dumps({"run_id": run_id, "status": "completed",
                "outputs": [{"id": model_id, "kind": "model", "role": "adapter"}]}), encoding="utf-8")
            artifacts = output / "model_artifacts"
            artifacts.mkdir()
            (artifacts / f"{model_id}.json").write_text(json.dumps({"model_id": model_id, "kind": "adapter"}), encoding="utf-8")

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(launcher_module.subprocess, "Popen", lambda argv, **kwargs: FakeProcess(argv))
    with TestClient(create_app(workspace, enable_experiment_launcher=True)) as client:
        capabilities = client.get("/api/launcher/capabilities").json()
        token = capabilities["token"]
        headers = {"X-Local-Launcher-Token": token}
        preflight = client.post("/api/launcher/preflight", json={}, headers=headers).json()
        assert preflight["ok"] is True
        started = client.post("/api/launcher/start", json={"confirmation_token": preflight["confirmation_token"]}, headers=headers)
        assert started.status_code == 200
        run_id = started.json()["run_id"]
        for _ in range(100):
            run = client.get(f"/api/launcher/runs/{run_id}").json()
            if run["status"] == "completed":
                break
            time.sleep(0.01)
        assert run["evaluation_status"] == "pending"
        assert isinstance(run.get("updated_at"), str) and run["updated_at"].endswith("+00:00")
        assert run["model_ids"] == ["model-0123456789abcdef0123"]
        lineage = client.get(f"/api/launcher/runs/{run_id}/lineage")
        assert lineage.status_code == 200 and lineage.json()["run_id"] == run_id
        model = client.get(f"/api/launcher/runs/{run_id}/models/model-0123456789abcdef0123")
        assert model.status_code == 200 and model.json()["kind"] == "adapter"


def test_stale_preflight_and_output_collision_do_not_launch_or_overwrite(tmp_path, monkeypatch):
    workspace = tmp_path.resolve()
    recipe = workspace / launcher_module.RECIPE
    recipe.parent.mkdir(parents=True)
    _write_sft_recipe(recipe)
    model = workspace / "model"
    model.mkdir()
    dataset = workspace / "dataset.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    inputs = {"recipe_path": recipe, "recipe_sha256": "a" * 64, "model_path": model,
              "model_sha256": "b" * 64, "model_id": launcher_module.MODEL_ID,
              "manifest_sha256": "c" * 64, "canonical_sha256": "d" * 64,
              "native_path": dataset, "native_sha256": "e" * 64,
              "dataset_id": launcher_module.DATASET_ID}
    monkeypatch.setattr(launcher_module.SFTLauncher, "_resolve_fixed_inputs", lambda self, dataset_id=launcher_module.DATASET_ID: inputs)
    monkeypatch.setattr(launcher_module, "_gpu_snapshot", lambda: {"available": False})
    monkeypatch.setattr(launcher_module, "_sft_runtime_available", lambda: True)
    from onegpu_posttrain import cli
    monkeypatch.setattr(cli, "cmd_plan_sft", lambda args: Path(args.output, "train_plan.json").write_text("{}", encoding="utf-8"))
    monkeypatch.setattr(launcher_module, "new_run_id", lambda *args, **kwargs: "run-stable-fixture")
    monkeypatch.setattr(launcher_module.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("stale preflight must not launch"))
    service = launcher_module.SFTLauncher(workspace)
    stale = service.preflight()
    inputs["native_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="摘要已变化"):
        service.start(stale["confirmation_token"])
    assert not (workspace / stale["output"]).exists()

    inputs["native_sha256"] = "e" * 64
    collision = service.preflight()
    target = workspace / collision["output"]
    target.mkdir(parents=True)
    marker = target / "keep.txt"
    marker.write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="输出目录冲突"):
        service.start(collision["confirmation_token"])
    assert marker.read_text(encoding="utf-8") == "user data"

