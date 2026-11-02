import io
import json
from argparse import Namespace

import pytest

from onegpu_posttrain import cli
from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.lineage import (
    ArtifactRef,
    ModelLineage,
    RunRecord,
    sha256_json,
    sha256_directory,
)
from onegpu_posttrain.ui.preference_stage_adapter import (
    PreferenceStageAdapter,
    PreferenceStageRequest,
)


def _workspace(tmp_path):
    recipes = {
        "recipes/smoke/30-dpo.yaml": {"run_label": "dpo-adapter-test", "stage": "dpo", "seed": 42,
                                      "training": {"max_steps": 1}},
        "recipes/smoke/31-simpo-original.yaml": {"run_label": "simpo-adapter-test", "stage": "simpo", "seed": 42,
                                                 "training": {"max_steps": 1, "cpo_alpha": 0}},
        "recipes/smoke/10-peft-cpt.yaml": {}, "recipes/mini/23-sft-1024-fallback.yaml": {},
        "recipes/smoke/40-grpo-standard.yaml": {}, "recipes/smoke/41-grpo-lowmem.yaml": {},
    }
    for relative, payload in recipes.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        import yaml
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins/grpo_rewards.py").write_text("# fixture", encoding="utf-8")
    row = {"id": "pref-1", "raw_problem": "1+1?", "prompt": [{"role": "user", "content": "1+1?"}],
           "chosen": "2", "rejected": "3", "prompt_template_id": "math-v1",
           "metadata": {"dataset_role": "train_pairs", "chosen_correct": True, "rejected_correct": False}}
    registered = register_jsonl(
        io.BytesIO((json.dumps(row) + "\n").encode()), workspace=tmp_path,
        output_root=tmp_path / "data/processed/preference-fixture", stage="preference",
        origin=DataSource(source_id="test-source", kind="local_upload", name="fixture",
                          license_status="not_provided", declared_by="user"),
    )
    model_path = tmp_path / "models/sft-merged"
    model_path.mkdir(parents=True)
    (model_path / "weights.bin").write_bytes(b"registered-sft-parent")
    digest = sha256_directory(model_path)
    model_id = f"model-{digest[:20]}"
    producing_run_id = "run-sft-parent-fixture"
    run_dir = tmp_path / "runs" / producing_run_id
    (run_dir / "model_artifacts").mkdir(parents=True)
    model = ModelLineage(model_id=model_id, kind="merged", path=str(model_path),
                         digest={"value": digest, "status": "available"},
                         parent_model_ids=["model-cpt-parent"], produced_by_run_id=producing_run_id, stage="sft")
    (run_dir / "model_artifacts" / f"{model_id}.json").write_text(model.model_dump_json(), encoding="utf-8")
    run = RunRecord(run_id=producing_run_id, run_type="train", status="completed",
                    started_at="2026-01-01T00:00:00Z", completed_at="2026-01-01T00:01:00Z",
                    outputs=[ArtifactRef(id=model_id, kind="model", path=str(model_path), role="merged",
                                         digest={"value": digest, "status": "available"})])
    (run_dir / "lineage.json").write_text(run.model_dump_json(), encoding="utf-8")
    return registered.dataset_id, model_id, model_path


def test_preflight_and_commands_keep_dpo_simpo_independent_and_bind_same_sft_parent(tmp_path, monkeypatch):
    dataset_id, model_id, _ = _workspace(tmp_path)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.shutil.which", lambda _: "swift")
    adapter = PreferenceStageAdapter(tmp_path)

    dpo = adapter.preflight(PreferenceStageRequest(profile_id="dpo-smoke-v1", dataset_id=dataset_id,
                                                   parent_model_id=model_id))
    simpo = adapter.preflight(PreferenceStageRequest(profile_id="simpo-smoke-v1", dataset_id=dataset_id,
                                                     parent_model_id=model_id))
    dpo_command = adapter.build_command(PreferenceStageRequest(profile_id="dpo-smoke-v1", dataset_id=dataset_id,
                                                               parent_model_id=model_id), run_id="run-dpo-fixture")
    simpo_command = adapter.build_command(PreferenceStageRequest(profile_id="simpo-smoke-v1", dataset_id=dataset_id,
                                                                 parent_model_id=model_id), run_id="run-simpo-fixture")

    assert dpo["status"] == simpo["status"] == "ready_for_cli"
    assert dpo["parent_model"]["model_id"] == simpo["parent_model"]["model_id"] == model_id
    assert dpo["datasets"][0]["ordering_evidence_status"] == "verified"
    assert dpo_command != simpo_command
    assert dpo_command[dpo_command.index("--profile-id") + 1] == "dpo-smoke-v1"
    assert simpo_command[simpo_command.index("--profile-id") + 1] == "simpo-smoke-v1"
    assert dpo_command[dpo_command.index("--parent-model-id") + 1] == model_id
    assert "--config" not in dpo_command


def test_paired_dpo_simpo_runs_retain_shared_data_parent_and_separate_config_lineage(tmp_path, monkeypatch):
    dataset_id, model_id, _model_path = _workspace(tmp_path)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.shutil.which", lambda _: "swift")
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "fake-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.bin").write_bytes(b"fake-adapter")

    monkeypatch.setattr(cli, "run_monitored_command", lambda *args, **kwargs: {"status": "PASS", "returncode": 0})
    monkeypatch.setattr(cli, "discover_latest_checkpoint", lambda _: checkpoint)

    def fake_merge(command, *, log_dir=None):
        target = Path(command[command.index("--output_dir") + 1])
        target.mkdir(parents=True)
        (target / "weights.bin").write_bytes(target.name.encode())
        return {"status": "PASS", "returncode": 0}

    from pathlib import Path
    monkeypatch.setattr(cli, "run_command", fake_merge)
    artifacts = {}
    for stage, profile_id, run_id in (
        ("dpo", "dpo-smoke-v1", "run-dpo-paired"),
        ("simpo", "simpo-smoke-v1", "run-simpo-paired"),
    ):
        output = f"runs/{stage}/{profile_id}/{run_id}"
        args = Namespace(workspace=str(tmp_path), profile_id=profile_id, dataset_id=dataset_id,
                         parent_model_id=model_id, run_id=run_id, output=output)
        assert cli.cmd_run_registered_preference(args) == 0
        run_root = tmp_path / output
        artifacts[stage] = {
            "run": json.loads((run_root / "run.json").read_text(encoding="utf-8")),
            "lineage": json.loads((run_root / "lineage.json").read_text(encoding="utf-8")),
        }

    dpo, simpo = artifacts["dpo"], artifacts["simpo"]
    assert dpo["lineage"]["run_id"] != simpo["lineage"]["run_id"]
    assert [item["id"] for item in dpo["lineage"]["inputs"]] == [model_id, dataset_id]
    assert [item["id"] for item in simpo["lineage"]["inputs"]] == [model_id, dataset_id]
    assert [item["digest"] for item in dpo["lineage"]["inputs"]] == [item["digest"] for item in simpo["lineage"]["inputs"]]
    assert dpo["lineage"]["resolved_config_sha256"] != simpo["lineage"]["resolved_config_sha256"]
    assert dpo["lineage"]["recipe_path"].endswith("30-dpo.yaml")
    assert simpo["lineage"]["recipe_path"].endswith("31-simpo-original.yaml")
    for stage, profile_id, run_id in (
        ("dpo", "dpo-smoke-v1", "run-dpo-paired"),
        ("simpo", "simpo-smoke-v1", "run-simpo-paired"),
    ):
        run_root = tmp_path / "runs" / stage / profile_id / run_id
        resolved = json.loads((run_root / "resolved_config.json").read_text(encoding="utf-8"))
        assert sha256_json(resolved) == artifacts[stage]["lineage"]["resolved_config_sha256"]


@pytest.mark.parametrize(("profile_id", "expected_stage", "merged_name"), [
    ("dpo-smoke-v1", "dpo", "M3-DPO"),
    ("simpo-smoke-v1", "simpo", "M3-SimPO"),
])
def test_registered_preference_cli_fake_training_and_merge_record_real_lineage(
    tmp_path, monkeypatch, profile_id, expected_stage, merged_name,
):
    dataset_id, model_id, _model_path = _workspace(tmp_path)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.shutil.which", lambda _: "swift")
    monkeypatch.chdir(tmp_path)
    checkpoint_path = tmp_path / "fake-checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "adapter.bin").write_bytes(b"fake-adapter")

    def fake_train(command, *, log_dir, dry_run=False, allow_failure=False):
        assert allow_failure is True and dry_run is False
        return {"status": "PASS", "returncode": 0}

    def fake_merge(command, *, log_dir=None):
        target = Path(command[command.index("--output_dir") + 1])
        target.mkdir(parents=True)
        (target / "weights.bin").write_bytes(b"fake-merged-model")
        return {"status": "PASS", "returncode": 0}

    from pathlib import Path
    monkeypatch.setattr(cli, "run_monitored_command", fake_train)
    monkeypatch.setattr(cli, "discover_latest_checkpoint", lambda _: checkpoint_path)
    monkeypatch.setattr(cli, "run_command", fake_merge)
    args = Namespace(workspace=str(tmp_path), profile_id=profile_id, dataset_id=dataset_id,
                     parent_model_id=model_id, run_id=f"run-{expected_stage}-fake-cli",
                     output=f"runs/{expected_stage}/{profile_id}/run-{expected_stage}-fake-cli")

    assert cli.cmd_run_registered_preference(args) == 0
    output = tmp_path / args.output
    result = json.loads((output / "run.json").read_text(encoding="utf-8"))
    lineage = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    assert result["run_id"] == lineage["run_id"] == args.run_id
    assert lineage["status"] == "completed"
    assert result["stage"] == expected_stage
    assert lineage["inputs"][0]["id"] == model_id
    assert lineage["inputs"][1]["id"] == dataset_id
    assert lineage["outputs"]
    assert (output / "merged" / merged_name).is_dir()
    assert (output / "merge_runs").is_dir()


def test_registered_preference_merge_failure_records_no_fake_merged_model(tmp_path, monkeypatch):
    dataset_id, model_id, _ = _workspace(tmp_path)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.shutil.which", lambda _: "swift")
    monkeypatch.chdir(tmp_path)
    checkpoint_path = tmp_path / "fake-checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "adapter.bin").write_bytes(b"fake-adapter")
    monkeypatch.setattr(cli, "run_monitored_command", lambda *args, **kwargs: {"status": "PASS", "returncode": 0})
    monkeypatch.setattr(cli, "discover_latest_checkpoint", lambda _: checkpoint_path)
    monkeypatch.setattr(cli, "run_command", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("merge failed")))
    args = Namespace(workspace=str(tmp_path), profile_id="dpo-smoke-v1", dataset_id=dataset_id,
                     parent_model_id=model_id, run_id="run-dpo-merge-failed", output="runs/dpo/dpo-smoke-v1/run-dpo-merge-failed")

    assert cli.cmd_run_registered_preference(args) == 4
    output = tmp_path / args.output
    lineage = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    merge_lineages = list((output / "merge_runs").glob("*/lineage.json"))
    merge = json.loads(merge_lineages[0].read_text(encoding="utf-8"))
    assert lineage["status"] == "completed"  # training really passed; merge is its own failed Run
    assert len(lineage["outputs"]) == 1 and lineage["outputs"][0]["role"] == "adapter"
    assert merge["status"] == "failed"
    assert merge["outputs"] == []
    assert not (output / "merged/M3-DPO").exists()


def test_registered_preference_failed_train_does_not_claim_models(tmp_path, monkeypatch):
    dataset_id, model_id, _ = _workspace(tmp_path)
    monkeypatch.setattr("onegpu_posttrain.ui.training_preflight.shutil.which", lambda _: "swift")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_monitored_command", lambda *args, **kwargs: {"status": "FAILED", "returncode": 1})
    monkeypatch.setattr(cli, "discover_latest_checkpoint", lambda _: (_ for _ in ()).throw(AssertionError("no checkpoint lookup after failure")))
    args = Namespace(workspace=str(tmp_path), profile_id="simpo-smoke-v1", dataset_id=dataset_id,
                     parent_model_id=model_id, run_id="run-simpo-failed", output="runs/simpo/simpo-smoke-v1/run-simpo-failed")

    assert cli.cmd_run_registered_preference(args) == 4
    output = tmp_path / args.output
    lineage = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    assert lineage["status"] == "failed"
    assert lineage["outputs"] == []
    assert not (output / "merged/M3-SimPO").exists()

