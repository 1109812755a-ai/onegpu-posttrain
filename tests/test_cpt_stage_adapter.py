import io
import json

import pytest
from fastapi.testclient import TestClient

from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.ui.cpt_stage_adapter import CPTStageAdapter
from onegpu_posttrain.ui.server import create_app


def _workspace(tmp_path, *, data_role="train_seed", include_weights=True):
    root = tmp_path.resolve()
    recipe = root / "recipes/smoke/10-peft-cpt.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        "stage: peft_cpt\nmodel_config: configs/model/qwen3-0.6b-base.yaml\n"
        "training:\n  tuner_type: lora\n  max_length: 1024\n  max_steps: 1\n"
        "  learning_rate: 0.0001\n  gradient_accumulation_steps: 8\n", encoding="utf-8")
    model = root / "models/base"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    if include_weights:
        (model / "model.safetensors").write_bytes(b"fixture model weights")
    config = root / "configs/model/qwen3-0.6b-base.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(f"model:\n  repo: {model.as_posix()}\n  revision: fixture-revision\n", encoding="utf-8")
    row = {"id": "cpt-1", "text": "A mathematical statement.", "source": "fixture",
           "metadata": {"data_role": data_role}}
    registered = register_jsonl(
        io.BytesIO((json.dumps(row) + "\n").encode()), workspace=root,
        output_root=root / "data/processed/registered-cpt", stage="cpt",
        origin=DataSource(source_id="fixture-cpt", kind="local_upload", name="fixture",
                          license_status="user_declared_unverified", declared_by="user"),
    )
    return root, registered.dataset_id, model


def test_cpt_adapter_preflight_and_command_are_allowlisted_and_offline(tmp_path, monkeypatch):
    workspace, dataset_id, model = _workspace(tmp_path)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter.console_command_available", lambda _: True)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter._gpu_snapshot",
                        lambda: {"available": True, "status": "advisory", "name": "fixture GPU"})
    adapter = CPTStageAdapter(workspace)

    preflight = adapter.preflight(dataset_id)
    command = adapter.build_command(dataset_id, "runs/cpt/cpt-smoke-v1/run-cpt-test-1")

    assert preflight["status"] == "ready_for_cli"
    assert preflight["parent_model"]["local_weights_available"] is True
    assert preflight["parent_model"]["downloads_performed"] is False
    assert any("许可状态" in warning for warning in preflight["warnings"])
    assert command[command.index("run-registered-cpt") + 1] == "--workspace"
    assert command[command.index("--manifest") + 1].startswith("data/processed/")
    assert __import__("pathlib").Path(command[command.index("--model") + 1]) == model
    assert "--merge" in command
    assert command[command.index("--run-id") + 1] == "run-cpt-test-1"
    with pytest.raises(ValueError, match="輸出|输出位置"):
        adapter.build_command(dataset_id, "../runs/escape")


def test_cpt_adapter_blocks_missing_base_weights_and_holdout_role(tmp_path, monkeypatch):
    workspace, dataset_id, _ = _workspace(tmp_path, include_weights=False)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter.console_command_available", lambda _: True)
    adapter = CPTStageAdapter(workspace)
    missing_parent = adapter.preflight(dataset_id)
    assert missing_parent["status"] == "blocked"
    assert any("权重" in blocker for blocker in missing_parent["blockers"])

    other, holdout_id, _ = _workspace(tmp_path / "holdout", data_role="final_holdout")
    holdout = CPTStageAdapter(other).preflight(holdout_id)
    assert holdout["status"] == "blocked"
    assert any("role/schema" in blocker or "train_seed" in blocker for blocker in holdout["blockers"])


def test_cpt_preflight_api_uses_only_registered_dataset_id(tmp_path, monkeypatch):
    workspace, dataset_id, _ = _workspace(tmp_path)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter.console_command_available", lambda _: True)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter._gpu_snapshot", lambda: {"available": False})
    client = TestClient(create_app(workspace))

    response = client.post("/api/training/cpt/preflight", json={"dataset_id": dataset_id})
    rejected = client.post("/api/training/cpt/preflight", json={
        "dataset_id": dataset_id, "manifest_path": "arbitrary.yaml",
    })

    assert response.status_code == 200
    assert response.json()["stage"] == "cpt"
    assert response.json()["dataset"]["dataset_id"] == dataset_id
    assert response.json()["execution_started"] is False
    assert response.json()["read_only"] is True
    assert response.json()["schema_version"] == "1.0"
    assert {item["check_id"] for item in response.json()["checks"]} >= {
        "dataset:registered_cpt", "parent_model:local_base", "runtime:ms_swift",
        "hardware:gpu", "launcher:execution_adapter",
    }
    assert all(item["status"] in {"pass", "warning", "fail"} for item in response.json()["checks"])
    assert rejected.status_code == 422


def test_cpt_preflight_reports_data_swift_and_model_blockers_independently(tmp_path, monkeypatch):
    workspace, dataset_id, _ = _workspace(tmp_path, data_role=None, include_weights=False)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter.console_command_available", lambda _: False)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter._gpu_snapshot",
                        lambda: {"available": False, "status": "unknown", "reason": "fixture GPU unavailable"})

    result = CPTStageAdapter(workspace).preflight(dataset_id)

    assert result["status"] == "blocked"
    assert result["execution_started"] is False
    checks = {check["check_id"]: check for check in result["checks"]}
    assert checks["dataset:registered_cpt"]["status"] == "fail"
    assert checks["runtime:ms_swift"]["status"] == "fail"
    assert checks["parent_model:local_base"]["status"] == "fail"
    assert "权重" in checks["parent_model:local_base"]["detail"]
    assert result["parent_model"]["local_weights_available"] is False
    assert any("swift" in item for item in result["blockers"])
    assert any("数据集/登记证据" in item for item in result["blockers"])
    assert any("父模型本地文件" in item for item in result["blockers"])
    assert checks["hardware:gpu"]["status"] == "warning"
    assert checks["dataset:registered_cpt"].get("repair_hint")
    assert checks["runtime:ms_swift"].get("repair_hint")
    assert checks["parent_model:local_base"].get("repair_hint")
    assert result["parent_model"]["downloads_performed"] is False


def test_cpt_preflight_is_read_only_and_never_installs_downloads_or_starts_training(tmp_path, monkeypatch):
    workspace, dataset_id, _ = _workspace(tmp_path)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter.console_command_available", lambda _: True)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter._gpu_snapshot", lambda: {"available": True})
    import subprocess
    import urllib.request
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("preflight must not spawn a process"))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("preflight must not make HTTP requests"))

    result = CPTStageAdapter(workspace).preflight(dataset_id)

    assert result["read_only"] is True
    assert result["execution_started"] is False
    assert result["parent_model"]["downloads_performed"] is False
    assert result["status"] == "ready_for_cli"


def test_missing_model_config_is_a_readable_independent_blocker(tmp_path, monkeypatch):
    workspace, dataset_id, _ = _workspace(tmp_path)
    (workspace / "configs/model/qwen3-0.6b-base.yaml").unlink()
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter.console_command_available", lambda _: False)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter._gpu_snapshot", lambda: {"available": False})

    result = CPTStageAdapter(workspace).preflight(dataset_id)

    checks = {check["check_id"]: check for check in result["checks"]}
    assert checks["dataset:registered_cpt"]["status"] == "pass"
    assert checks["parent_model:local_base"]["status"] == "fail"
    assert "配置不存在" in checks["parent_model:local_base"]["detail"]
    assert checks["runtime:ms_swift"]["status"] == "fail"
    assert checks["hardware:gpu"]["status"] == "warning"
    assert result["execution_started"] is False


def test_registered_cpt_stage_preflight_api_preserves_checklist_and_blocks_confirmation(tmp_path, monkeypatch):
    workspace, dataset_id, _ = _workspace(tmp_path, include_weights=False)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter.console_command_available", lambda _: False)
    monkeypatch.setattr("onegpu_posttrain.ui.cpt_stage_adapter._gpu_snapshot", lambda: {"available": False})
    app = create_app(workspace, enable_experiment_launcher=True)
    with TestClient(app) as client:
        headers = {"X-Local-Launcher-Token": app.state.experiment_launcher_token}
        payload = {"stage": "cpt", "profile_id": "cpt-smoke-v1", "dataset_ids": [dataset_id]}
        preflight = client.post("/api/training/stage/preflight", json=payload, headers=headers)
        confirm = client.post("/api/training/stage/confirm", json=payload, headers=headers)

    assert preflight.status_code == 200, preflight.text
    checks = {item["check_id"]: item for item in preflight.json()["checks"]}
    assert checks["dataset:registered_cpt"]["status"] == "pass"
    assert checks["parent_model:local_base"]["status"] == "fail"
    assert checks["runtime:ms_swift"]["status"] == "fail"
    assert preflight.json()["status"] == "blocked"
    assert preflight.json()["execution_started"] is False
    assert confirm.status_code == 409
    assert confirm.json()["detail"]["status"] == "blocked"
    assert "confirmation_token" not in confirm.json()["detail"]
    assert not list(workspace.glob("runs/**/run.json"))
