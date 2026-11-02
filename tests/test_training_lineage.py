import json
from argparse import Namespace

import pytest

from onegpu_posttrain.cli import (
    _assert_new_run_directory,
    _dataset_lineage_ref,
    _model_lineage_ref,
    _write_training_run,
)
from onegpu_posttrain.data.manifest import (
    DatasetManifest,
    SourceSpec,
    save_manifest,
    sha256_file,
)
from onegpu_posttrain.lineage import new_run_id


def _write_run(output, dataset, label):
    run_id = new_run_id("train", label=label)
    result = {"train": {"status": "PASS"}, "checkpoint": None}
    _write_training_run(
        output=output, result=result, run_id=run_id, run_label=label,
        experiment_id="exp-test", stage="sft", model="remote/model", dataset=dataset,
        config_path="recipe.yaml", resolved_config={"lr": 1e-4}, started_at="2026-10-17T00:00:00Z",
    )
    return run_id, json.loads((output / "lineage.json").read_text(encoding="utf-8")), result


def test_run_label_is_reused_but_each_execution_gets_unique_run_id(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"text":"fixture"}\n', encoding="utf-8")
    first, lineage1, result1 = _write_run(tmp_path / "first", dataset, "sft-seed-42")
    second, lineage2, result2 = _write_run(tmp_path / "second", dataset, "sft-seed-42")
    assert first != second
    assert result1["run_label"] == result2["run_label"] == "sft-seed-42"
    assert result1["run_id"] == lineage1["run_id"] == first
    assert lineage2["run_id"] == second
    assert lineage1["lineage_status"] == "incomplete"


def test_existing_execution_record_blocks_reuse_but_dry_run_is_allowed(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="fresh output"):
        _assert_new_run_directory(output, dry_run=False)
    _assert_new_run_directory(output, dry_run=True)


def test_training_resolves_manifest_and_rejects_content_mismatch(tmp_path):
    root = tmp_path / "data"
    canonical = root / "canonical" / "sft.jsonl"
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"record":"v1"}\n', encoding="utf-8")
    manifest_dir = root / "manifests"
    manifest = DatasetManifest(
        dataset_id="ds-sft-v1", stage="sft", source=SourceSpec(repo="fixture", revision="r1"),
        sample_count=1, hashes={"canonical_records_sha256": sha256_file(canonical)},
        metadata={"canonical_path": "canonical/sft.jsonl"},
    )
    save_manifest(manifest_dir / "sft-v1.yaml", manifest)
    native = tmp_path / "native.jsonl"
    native.write_text('{"native":true}\n', encoding="utf-8")
    native.with_suffix(".jsonl.adapter.json").write_text(
        json.dumps({"input": str(canonical), "input_sha256": sha256_file(canonical)}), encoding="utf-8")
    ref, complete, warnings = _dataset_lineage_ref(native)
    assert complete and ref.kind == "dataset" and ref.id == "ds-sft-v1"
    assert warnings == []
    canonical.write_text('{"record":"changed"}\n', encoding="utf-8")
    native.with_suffix(".jsonl.adapter.json").write_text(
        json.dumps({"input": str(canonical), "input_sha256": sha256_file(canonical)}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        _dataset_lineage_ref(native)


def test_training_resolves_native_file_through_manifest_digest_metadata(tmp_path):
    artifact = tmp_path / "deepmath-sft"
    canonical = artifact / "canonical" / "sft.jsonl"
    native = artifact / "native" / "sft.jsonl"
    canonical.parent.mkdir(parents=True)
    native.parent.mkdir(parents=True)
    canonical.write_text('{"id":"canonical"}\n', encoding="utf-8")
    native.write_text('{"messages":[]}\n', encoding="utf-8")
    save_manifest(
        artifact / "manifests" / "sft.yaml",
        DatasetManifest(
            dataset_id="ds-deepmath-v1", stage="sft", source=SourceSpec(repo="deepmath", revision="r1"),
            sample_count=1,
            hashes={"canonical_records_sha256": sha256_file(canonical)},
            metadata={"native_path": "native/sft.jsonl", "native_sha256": sha256_file(native),
                      "canonical_path": "canonical/sft.jsonl", "usage": "exploratory"},
        ),
    )
    ref, complete, warnings = _dataset_lineage_ref(native)
    assert complete is True
    assert ref.id == "ds-deepmath-v1"
    assert ref.path == str(canonical.resolve())
    assert ref.digest.value == sha256_file(canonical)
    assert warnings == []


@pytest.mark.parametrize("train_status,expected_run_status,expected_exit", [
    ("PASS", "completed", 0), ("OOM", "oom", 4), ("FAILED", "failed", 4),
])
def test_sft_cli_accepts_unmanifested_jsonl_and_records_terminal_state(
    tmp_path, monkeypatch, train_status, expected_run_status, expected_exit,
):
    from onegpu_posttrain import cli

    data = tmp_path / "plain.jsonl"
    data.write_text('{"messages":[{"role":"user","content":"x"}]}\n', encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "run-output"
    config = tmp_path / "recipe.yaml"
    config.write_text("run_label: sft-fixture\ntraining:\n  max_steps: 1\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    monkeypatch.setattr(cli, "run_monitored_command", lambda *a, **k: {"status": train_status})
    monkeypatch.setattr(cli, "discover_latest_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(cli, "capture_environment", lambda: {"python": "test"})
    args = Namespace(config=str(config), model=str(model), dataset=str(data), output=str(output),
                     merge=False, dry_run=False)
    assert cli.cmd_run_sft(args) == expected_exit
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    lineage = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    assert run["run_id"] == lineage["run_id"]
    assert lineage["lineage_status"] == "incomplete"
    assert lineage["status"] == expected_run_status
    assert any("no dataset manifest" in item for item in lineage["lineage_warnings"])


def test_parallel_alignment_runs_share_parent_model_identity_and_digest(tmp_path):
    parent = tmp_path / "m2"
    parent.mkdir()
    (parent / "config.json").write_text('{"model":"fixture"}', encoding="utf-8")
    dpo = _model_lineage_ref(parent)
    simpo = _model_lineage_ref(parent)
    assert dpo.id == simpo.id
    assert dpo.digest.model_dump() == simpo.digest.model_dump()
    assert dpo.digest.status == "available"


def test_grpo_oom_before_checkpoint_records_no_model_output(tmp_path, monkeypatch):
    from onegpu_posttrain import cli

    data = tmp_path / "plain.jsonl"
    data.write_text('{"messages":[]}\n', encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    config = tmp_path / "grpo.yaml"
    config.write_text("run_label: grpo-fixture\nprofile: lowmem\ntraining:\n  max_steps: 1\n", encoding="utf-8")
    output = tmp_path / "grpo-output"
    monkeypatch.setattr(cli, "run_monitored_command", lambda *a, **k: {"status": "OOM"})
    monkeypatch.setattr(cli, "capture_environment", lambda: {"python": "test"})
    args = Namespace(config=str(config), model=str(model), dataset=str(data), output=str(output),
                     external_plugin="plugin.py", merge=False, dry_run=False)
    assert cli.cmd_run_grpo(args) == 4
    lineage = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    assert lineage["status"] == "oom"
    assert lineage["outputs"] == []


def test_merge_is_a_separate_run_and_records_only_real_output(tmp_path, monkeypatch):
    from onegpu_posttrain import cli

    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "config.json").write_text("{}", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter.bin").write_bytes(b"adapter")
    merged = tmp_path / "merged"
    output = tmp_path / "experiment"

    def successful_merge(*args, **kwargs):
        merged.mkdir()
        (merged / "model.bin").write_bytes(b"merged")
        return {"returncode": 0}

    monkeypatch.setattr(cli, "run_command", successful_merge)
    result = cli._run_merge_with_lineage(
        output=output, checkpoint=adapter, parent_model=parent, merged_dir=merged,
        run_label="sft-merge", experiment_id="exp-1", stage="sft",
    )
    run = json.loads((output / "merge_runs" / result["run_id"] / "lineage.json").read_text(encoding="utf-8"))
    model = json.loads((output / "model_artifacts" / f"{run['outputs'][0]['id']}.json").read_text(encoding="utf-8"))
    assert run["run_type"] == "merge" and run["status"] == "completed"
    assert len(run["inputs"]) == 2 and len(run["outputs"]) == 1
    assert model["parent_model_ids"] == [run["inputs"][0]["id"]]
    assert model["adapter_id"] == run["inputs"][1]["id"]


def test_failed_merge_run_has_no_merged_model_output(tmp_path, monkeypatch):
    from onegpu_posttrain import cli

    parent, adapter = tmp_path / "parent", tmp_path / "adapter"
    parent.mkdir()
    adapter.mkdir()
    monkeypatch.setattr(cli, "run_command", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("merge failed")))
    result = cli._run_merge_with_lineage(
        output=tmp_path / "experiment", checkpoint=adapter, parent_model=parent,
        merged_dir=tmp_path / "merged", run_label="merge-fails", experiment_id=None, stage="sft",
    )
    run = json.loads((tmp_path / "experiment" / "merge_runs" / result["run_id"] / "lineage.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["outputs"] == []
