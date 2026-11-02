from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from onegpu_posttrain.synthetic.workflow_bundle import (
    load_sft_workflow_bundle,
    register_sft_workflow_bundle,
)
from test_synthetic_generation import _prepare_files


def _inputs(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    generated = _prepare_files(tmp_path / "generated")
    workspace = tmp_path / "workspace"
    inputs_dir = workspace / "inputs"
    inputs_dir.mkdir(parents=True)
    names = {
        "plan": generated["plan"],
        "profile": generated["profile"],
        "template": generated["template"],
        "protocol": generated["protocol"],
        "definition": generated["definition"],
        "report": generated["report"],
        "registry": generated["registry"],
        "train_seeds": generated["train"],
        "diagnostic": generated["diagnostic"],
        "holdout": generated["holdout"],
        "policy": Path(__file__).resolve().parents[1] / "configs/data/policies/sft-example.yaml",
    }
    relocated = {}
    for name, source in names.items():
        target = inputs_dir / f"{name}{source.suffix}"
        shutil.copyfile(source, target)
        relocated[name] = target
    return workspace, relocated


def test_bundle_registers_and_reloads_exact_workspace_inputs(tmp_path):
    workspace, inputs = _inputs(tmp_path)
    bundle = register_sft_workflow_bundle(workspace, inputs=inputs)
    loaded, resolved = load_sft_workflow_bundle(workspace, bundle.bundle_id)

    assert loaded.bundle_sha256 == bundle.bundle_sha256
    assert set(resolved) == set(inputs)
    assert all(path.is_relative_to(workspace) for path in resolved.values())
    assert (workspace / "artifacts/synthetic" / bundle.bundle_id / "workflow_bundle.json").is_file()


def test_bundle_registration_is_immutable(tmp_path):
    workspace, inputs = _inputs(tmp_path)
    bundle = register_sft_workflow_bundle(workspace, inputs=inputs)
    with pytest.raises(FileExistsError):
        register_sft_workflow_bundle(workspace, inputs=inputs)
    assert load_sft_workflow_bundle(workspace, bundle.bundle_id)[0] == bundle


def test_bundle_detects_changed_input(tmp_path):
    workspace, inputs = _inputs(tmp_path)
    bundle = register_sft_workflow_bundle(workspace, inputs=inputs)
    inputs["train_seeds"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after registration"):
        load_sft_workflow_bundle(workspace, bundle.bundle_id)


def test_bundle_rejects_path_escape_and_symlink(tmp_path):
    workspace, inputs = _inputs(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    inputs["train_seeds"] = outside
    with pytest.raises(ValueError, match="inside workspace|escapes workspace"):
        register_sft_workflow_bundle(workspace, inputs=inputs)

    workspace, inputs = _inputs(tmp_path / "second")
    link = workspace / "inputs" / "linked.jsonl"
    try:
        link.symlink_to(inputs["train_seeds"])
    except OSError as exc:
        pytest.skip(f"Windows symlink privilege unavailable: {exc}")
    inputs["train_seeds"] = link
    with pytest.raises(ValueError, match="symbolic links"):
        register_sft_workflow_bundle(workspace, inputs=inputs)
