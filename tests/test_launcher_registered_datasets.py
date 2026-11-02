import io
import json
import time
from pathlib import Path

import pytest
import yaml

from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.lineage import sha256_directory, sha256_file
from onegpu_posttrain.ui import launcher as launcher_module


def _sft_record():
    return (json.dumps({"id": "upload-sft-1", "raw_problem": "1+1?", "messages": [
        {"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"}],
        "prompt_template_id": "chat-v1", "metadata": {"source_record_id": "raw-1"}}) + "\n").encode()


def _registered(tmp_path, stage="sft", registration_metadata=None):
    import io as io_module
    row = _sft_record() if stage == "sft" else (json.dumps({"id": "cpt-1", "text": "math text", "source": "fixture"}) + "\n").encode()
    origin = DataSource(source_id="local-upload", kind="local_upload", name="My data",
                        license_status="unknown", revision_status="unknown", declared_by="user")
    return register_jsonl(io_module.BytesIO(row), workspace=tmp_path,
        output_root=tmp_path / "data/processed/user-upload", stage=stage, origin=origin,
        registration_metadata=registration_metadata)


def _fixed_workspace(tmp_path):
    model = tmp_path / "models/m1"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    registry = tmp_path / "runs/mini/cpt-finemath-5m-s42-v3/model_artifacts" / f"{launcher_module.MODEL_ID}.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"model_id": launcher_module.MODEL_ID, "kind": "merged",
                                    "digest": {"value": sha256_directory(model)}}), encoding="utf-8")
    default = tmp_path / "default"
    default.mkdir()
    canonical, native, manifest = default / "sft.jsonl", default / "native.jsonl", default / "manifest.yaml"
    canonical.write_text("{}\n", encoding="utf-8")
    native.write_text("{}\n", encoding="utf-8")
    manifest.write_text(yaml.safe_dump({"dataset_id": launcher_module.DATASET_ID, "stage": "sft", "sample_count": 307,
        "hashes": {"canonical_records_sha256": sha256_file(canonical)},
        "metadata": {"native_sha256": sha256_file(native), "usage": "exploratory"}}), encoding="utf-8")
    recipe = {"stage": "sft", "parent_model_id": launcher_module.MODEL_ID, "parent_model_path": "models/m1",
        "seed": 42, "training": {"max_length": 1024}, "dataset": {"dataset_id": launcher_module.DATASET_ID,
        "manifest": str(manifest.relative_to(tmp_path)), "canonical": str(canonical.relative_to(tmp_path)),
        "native": str(native.relative_to(tmp_path)), "expected_count": 307, "usage": "exploratory"}}
    recipe_path = tmp_path / launcher_module.RECIPE
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_text(yaml.safe_dump(recipe), encoding="utf-8")


def test_launcher_lists_registered_sft_and_blocks_other_stages(tmp_path):
    _fixed_workspace(tmp_path)
    sft = _registered(tmp_path)
    cpt = _registered(tmp_path, "cpt")
    launcher = launcher_module.SFTLauncher(tmp_path)
    assert any(item["dataset_id"] == sft.dataset_id and item["license_status"] == "unknown"
               for item in launcher.registered_sft_datasets())
    resolved = launcher._resolve_fixed_inputs(sft.dataset_id)
    assert resolved["dataset_count"] == 1
    assert resolved["model_id"] == launcher_module.MODEL_ID
    assert resolved["recipe"]["parent_model_id"] == launcher_module.MODEL_ID
    with pytest.raises(ValueError, match="只接受 SFT"):
        launcher._resolve_fixed_inputs(cpt.dataset_id)


def test_unknown_license_confirmation_is_bound_and_written_to_training_lineage(tmp_path, monkeypatch):
    _fixed_workspace(tmp_path)
    registered = _registered(tmp_path)
    monkeypatch.setattr(launcher_module, "_gpu_snapshot", lambda: {"available": False})
    monkeypatch.setattr(launcher_module, "_sft_runtime_available", lambda: True)
    from onegpu_posttrain import cli
    monkeypatch.setattr(cli, "cmd_plan_sft", lambda args: Path(args.output, "train_plan.json").write_text("{}", encoding="utf-8"))

    class FakeProcess:
        def __init__(self, argv, **kwargs):
            output = Path(argv[argv.index("--output") + 1])
            run_id = argv[argv.index("--run-id") + 1]
            (output / "run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
            (output / "lineage.json").write_text(json.dumps({"run_id": run_id, "status": "completed", "lineage_warnings": []}), encoding="utf-8")
            self.stdout = io.StringIO("step=1/1\n")
            self.returncode = 0

        def poll(self): return self.returncode
        def wait(self, timeout=None): return self.returncode

    monkeypatch.setattr(launcher_module.subprocess, "Popen", FakeProcess)
    launcher = launcher_module.SFTLauncher(tmp_path)
    preflight = launcher.preflight(registered.dataset_id)
    assert preflight["ok"] is True, preflight
    assert preflight["dataset"]["quality_status"]
    assert preflight["dataset"]["license_status"] == "unknown"
    with pytest.raises(ValueError, match="dataset_id"):
        launcher.start(preflight["confirmation_token"])
    result = launcher.start(preflight["confirmation_token"], acknowledged_dataset_id=registered.dataset_id)
    assert result["status"] == "running"
    for _ in range(40):
        state = launcher.get_run(result["run_id"])
        if state and state["status"] == "completed":
            break
        time.sleep(0.02)
    lineage = json.loads((tmp_path / state["output"] / "lineage.json").read_text(encoding="utf-8"))
    assert any(registered.dataset_id in warning and "unverified" in warning for warning in lineage["lineage_warnings"])
    native = tmp_path / state["output"] / "dataset-adapter/native/sft.jsonl"
    sidecar = json.loads(native.with_suffix(native.suffix + ".adapter.json").read_text(encoding="utf-8"))
    assert sidecar["input_sha256"] == registered.canonical_sha256


def test_sft_launcher_preserves_targeted_plan_identity_in_job_and_cli_args(tmp_path, monkeypatch):
    _fixed_workspace(tmp_path)
    from test_targeted_stage_plan import stage_plan

    from onegpu_posttrain.data.targeted_stage_plan import register_targeted_stage_plan

    plan_registration = register_targeted_stage_plan(tmp_path, stage_plan("sft", version=3))
    plan_id = plan_registration["plan_id"]
    registered = _registered(tmp_path, registration_metadata={
        "targeted_stage_plan_id": plan_id, "targeted_stage_plan_version": 3,
    })
    from onegpu_posttrain import cli
    registered_manifest = launcher_module.SFTLauncher(tmp_path)._registered_sft_manifest(registered.dataset_id)
    canonical = registered_manifest.parent.parent / "canonical" / "sft.jsonl"
    assert cli._dataset_targeted_plan(canonical) == (plan_id, 3)
    monkeypatch.setattr(launcher_module, "_gpu_snapshot", lambda: {"available": False})
    monkeypatch.setattr(launcher_module, "_sft_runtime_available", lambda: True)
    monkeypatch.setattr(cli, "cmd_plan_sft", lambda args: Path(args.output, "train_plan.json").write_text("{}", encoding="utf-8"))
    launched = {}

    class FakeProcess:
        def __init__(self, argv, **kwargs):
            launched["argv"] = argv
            output = Path(argv[argv.index("--output") + 1])
            run_id = argv[argv.index("--run-id") + 1]
            (output / "run.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
            (output / "lineage.json").write_text(json.dumps({"run_id": run_id, "status": "completed", "lineage_warnings": []}), encoding="utf-8")
            self.stdout = io.StringIO("step=1/1\n")
            self.returncode = 0

        def poll(self): return self.returncode
        def wait(self, timeout=None): return self.returncode

    monkeypatch.setattr(launcher_module.subprocess, "Popen", FakeProcess)
    launcher = launcher_module.SFTLauncher(tmp_path)
    preflight = launcher.preflight(registered.dataset_id)
    assert preflight["ok"] is True, preflight
    result = launcher.start(preflight["confirmation_token"], acknowledged_dataset_id=registered.dataset_id)
    assert result["status"] == "running"
    assert launcher.get_run(result["run_id"])["targeted_stage_plan_id"] == plan_id
    argv = launched["argv"]
    assert argv[argv.index("--targeted-stage-plan-id") + 1] == plan_id
    assert argv[argv.index("--targeted-stage-plan-version") + 1] == "3"


def test_run_sft_writes_targeted_plan_identity_into_resolved_config(tmp_path, monkeypatch):
    from argparse import Namespace

    from onegpu_posttrain import cli

    config = tmp_path / "recipe.yaml"
    config.write_text("stage: sft\ntraining: {}\n", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(cli, "run_monitored_command", lambda *args, **kwargs: {"status": "PASS"})
    monkeypatch.setattr(cli, "discover_latest_checkpoint", lambda _root: tmp_path / "checkpoint")
    monkeypatch.setattr(cli, "_write_training_run", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(cli, "capture_environment", dict)
    args = Namespace(config=str(config), model="model", dataset="dataset", output=str(tmp_path / "out"),
                     run_id=None, dry_run=True, merge=False, targeted_stage_plan_id="plan-targeted-0123456789abcdefabcd",
                     targeted_stage_plan_version=3)
    assert cli.cmd_run_sft(args) == 0
    assert captured["resolved_config"]["targeted_stage_plan"] == {
        "plan_id": args.targeted_stage_plan_id, "plan_version": 3,
    }
