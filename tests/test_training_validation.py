import json
import hashlib
from pathlib import Path

import pytest

from onegpu_posttrain.training.ms_swift_runner import (
    build_grpo_command, build_peft_cpt_command, build_preference_rlhf_command, build_sft_command,
)
from onegpu_posttrain.lineage import ArtifactRef, Digest
from onegpu_posttrain import cli
from onegpu_posttrain.ui.launcher import SFTLauncher
from onegpu_posttrain.ui.training_validation import (
    SPLIT_ALGORITHM, TrainingValidationConfig, apply_native_partition, plan_random_split,
)


def _rows():
    return [
        {"id": f"r-{index}", "prompt": f"question-{index // 2}", "answer": str(index),
         "metadata": {"problem_id": f"problem-{index // 2}"}}
        for index in range(20)
    ]


def test_random_split_is_reproducible_and_keeps_semantic_groups_atomic():
    rows = _rows()
    first = plan_random_split(rows, ratio=0.2, seed=42)
    second = plan_random_split(rows, ratio=0.2, seed=42)
    assert first == second
    assert first["algorithm"] == SPLIT_ALGORITHM
    train, validation = set(first["train_indices"]), set(first["validation_indices"])
    assert train.isdisjoint(validation)
    assert train | validation == set(range(len(rows)))
    for left in range(len(rows)):
        for right in range(len(rows)):
            if rows[left]["metadata"]["problem_id"] == rows[right]["metadata"]["problem_id"]:
                assert (left in validation) == (right in validation)


def test_random_split_blocks_when_semantic_grouping_cannot_make_two_partitions():
    with pytest.raises(ValueError, match="分组不足"):
        plan_random_split([{"prompt": "duplicate"}, {"prompt": "duplicate"}], ratio=0.2, seed=42)


def test_apply_native_partition_writes_separate_files_without_mutating_input(tmp_path):
    source = tmp_path / "native.jsonl"
    rows = _rows()
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = source.read_bytes()
    plan = plan_random_split(rows, ratio=0.2, seed=7)
    result = apply_native_partition(source, plan, tmp_path / "materialized")
    assert source.read_bytes() == before
    assert len(Path(result["train_path"]).read_text(encoding="utf-8").splitlines()) == plan["train_count"]
    assert len(Path(result["validation_path"]).read_text(encoding="utf-8").splitlines()) == plan["validation_count"]
    assert result["membership_manifest_sha256"] == plan["membership_manifest_sha256"]
    members = json.loads(Path(result["membership_manifest_path"]).read_text(encoding="utf-8"))
    assert len(members["train_members"]) == plan["train_count"]
    assert len(members["validation_members"]) == plan["validation_count"]


def test_validation_config_modes_are_mutually_exclusive_and_bounded():
    assert TrainingValidationConfig().mode == "none"
    assert TrainingValidationConfig(mode="registered", dataset_id="eval-v1").dataset_id == "eval-v1"
    assert TrainingValidationConfig(mode="random_split", ratio=0.1, seed=42).seed == 42
    with pytest.raises(ValueError):
        TrainingValidationConfig(mode="random_split", ratio=0.01, seed=42)
    with pytest.raises(ValueError):
        TrainingValidationConfig(mode="registered", dataset_id="eval-v1", seed=42)


def test_ms_swift_receives_val_dataset_and_never_combines_implicit_split():
    training = {"split_dataset_ratio": 0, "eval_strategy": "steps", "eval_steps": 5}
    command = build_sft_command(model="m", dataset="train.jsonl", output_dir="out", training=training,
                                validation_dataset="validation.jsonl")
    assert command[command.index("--val_dataset") + 1] == "validation.jsonl"
    assert command[command.index("--eval_strategy") + 1] == "steps"
    assert command[command.index("--eval_steps") + 1] == "5"
    with pytest.raises(ValueError, match="split_dataset_ratio"):
        build_preference_rlhf_command(algorithm="dpo", model="m", dataset="train.jsonl", output_dir="out",
                                      training={"split_dataset_ratio": 0.1}, validation_dataset="val.jsonl")
    cpt = build_peft_cpt_command(model="m", model_revision="rev", dataset="train.jsonl", output_dir="out",
                                 training={"split_dataset_ratio": 0}, validation_dataset="val.jsonl")
    grpo = build_grpo_command(model="m", dataset="train.jsonl", output_dir="out", training={"split_dataset_ratio": 0},
                              reward={"accuracy": 1.0, "boxed_format": 0.0}, validation_dataset="val.jsonl")
    assert cpt[cpt.index("--val_dataset") + 1] == "val.jsonl"
    assert grpo[grpo.index("--val_dataset") + 1] == "val.jsonl"


def test_training_lineage_records_validation_file_and_source_dataset(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    derived = tmp_path / "derived"
    derived.mkdir()
    validation_file = derived / "validation.jsonl"
    validation_file.write_text("{}\n", encoding="utf-8")
    membership_file = derived / "split_manifest.json"
    membership_file.write_text('{"schema_version":"training-validation-members-v1"}\n', encoding="utf-8")
    model = ArtifactRef(id="model-test", kind="model", digest=Digest(value="a" * 64, status="available"),
                        role="parent_model")

    def dataset_ref(path, *, role="train"):
        digest = "b" * 64 if role == "train" else "c" * 64
        return ArtifactRef(id=f"artifact-{role}", kind="file", path=str(path),
                           digest=Digest(value=digest, status="available"), role=role), True, []

    monkeypatch.setattr(cli, "_dataset_lineage_ref", dataset_ref)
    monkeypatch.setattr(cli, "_dataset_targeted_plan", lambda _path: (None, None))
    validation = {
        "mode": "random_split", "algorithm": SPLIT_ALGORITHM, "ratio": 0.2, "seed": 42,
        "split_identity_sha256": "d" * 64, "source_dataset_id": "sft-training-data",
        "source_content_sha256": "e" * 64, "train_members_sha256": "f" * 64,
        "validation_members_sha256": "1" * 64,
        "membership_manifest_path": str(membership_file),
        "membership_manifest_sha256": hashlib.sha256(membership_file.read_bytes()).hexdigest(),
    }
    cli._write_training_run(
        output=output, result={"train": {"status": "PASS"}}, run_id="run-validation-lineage-test",
        run_label="validation-lineage-test", experiment_id=None, stage="sft", model="model-path",
        dataset="train.jsonl", validation_dataset=validation_file, validation_metadata=validation,
        config_path="recipe.yaml", resolved_config={"model": "model-path"},
        started_at="2026-11-01T00:00:00+00:00", model_ref=model,
    )

    lineage = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    inputs = lineage["inputs"]
    assert any(item["role"] == "training_validation" and item["path"] == str(validation_file.resolve()) for item in inputs)
    assert any(item["role"] == "training_validation_membership_manifest"
               and item["path"] == str(membership_file.resolve()) for item in inputs)
    assert any(item["role"] == "training_source_before_validation_split"
               and item["id"] == "sft-training-data" and item["digest"]["value"] == "e" * 64 for item in inputs)
    resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["training_validation"]["split_identity_sha256"] == "d" * 64


def test_training_and_validation_log_metrics_are_not_conflated():
    log = "2026-11-01 - INFO - {'loss': 0.5, 'eval_loss': 0.25, 'eval_accuracy': 0.75}\n"
    assert SFTLauncher._parse_training_metrics(log) == {"loss": 0.5}
    assert SFTLauncher._parse_validation_metrics(log) == {"eval_loss": 0.25, "eval_accuracy": 0.75}
