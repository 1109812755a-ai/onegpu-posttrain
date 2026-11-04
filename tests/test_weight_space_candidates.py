"""Published candidate edits preserve the parent and record exact lineage."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from onegpu_posttrain.lineage import Digest, ModelLineage, sha256_directory
from onegpu_posttrain.ui.model_playground_catalog import ModelPlaygroundCatalog
from onegpu_posttrain.weight_space.candidate_registry import register_candidate
from onegpu_posttrain.weight_space.candidates import (
    apply_candidate_edit,
    candidate_state,
    require_finalizable_candidate,
    transition_candidate,
)
from onegpu_posttrain.weight_space.tool_provider import ToolCall

TENSOR = "model.language_model.layers.0.mlp.down_proj.weight"


def _parent(tmp_path):
    model = tmp_path / "base"
    model.mkdir()
    save_file({TENSOR: torch.arange(4, dtype=torch.float32).reshape(2, 2)}, model / "model.safetensors")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {TENSOR: "model.safetensors"}}), encoding="utf-8")
    (model / "config.json").write_text("{}", encoding="utf-8")
    return model


def _call(parent_sha, value):
    return ToolCall(id="tool-1", name="propose_edit", arguments={
        "tensor_name": TENSOR, "offset": 1, "values": [value], "parent_sha256": parent_sha})


def test_valid_edit_publishes_distinct_candidate_and_lineage(tmp_path):
    model = _parent(tmp_path)
    original = sha256_directory(model)
    result = apply_candidate_edit(tmp_path, "exp-weight-space-fixture01", model, original,
                                  _call(original, 1.05))
    assert sha256_directory(model) == original
    assert result.model_path != model and sha256_directory(result.model_path) == result.candidate_sha256
    assert load_file(str(result.model_path / "model.safetensors"))[TENSOR].flatten()[1].item() == pytest.approx(1.05)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["parent_sha256"] == original and metadata["changed_elements"] == 1


def test_rejected_edit_does_not_create_candidate(tmp_path):
    model = _parent(tmp_path)
    original = sha256_directory(model)
    with pytest.raises(ValueError, match="change budget"):
        apply_candidate_edit(tmp_path, "exp-weight-space-fixture01", model, original,
                             _call(original, 100.0))
    assert sha256_directory(model) == original
    assert not (tmp_path / "runs" / "weight-space").exists()


def test_experiment_id_cannot_escape_workspace(tmp_path):
    model = _parent(tmp_path)
    original = sha256_directory(model)
    with pytest.raises(ValueError, match="invalid experiment"):
        apply_candidate_edit(tmp_path, "exp-weight-space-../outside", model, original,
                             _call(original, 1.05))


def test_discard_blocks_final_until_explicit_restore(tmp_path):
    model = _parent(tmp_path)
    original = sha256_directory(model)
    experiment_id = "exp-weight-space-fixture01"
    candidate = apply_candidate_edit(tmp_path, experiment_id, model, original, _call(original, 1.05))
    assert require_finalizable_candidate(tmp_path, experiment_id, candidate.candidate_id) == candidate
    transition_candidate(tmp_path, experiment_id, candidate.candidate_id, "discard")
    with pytest.raises(ValueError, match="discarded"):
        require_finalizable_candidate(tmp_path, experiment_id, candidate.candidate_id)
    with pytest.raises(ValueError, match="explicit restore"):
        transition_candidate(tmp_path, experiment_id, candidate.candidate_id, "retain")
    transition_candidate(tmp_path, experiment_id, candidate.candidate_id, "restore")
    assert require_finalizable_candidate(tmp_path, experiment_id, candidate.candidate_id) == candidate
    assert [item["action"] for item in candidate_state(tmp_path, experiment_id, candidate.candidate_id)["history"]] == ["discard", "restore"]
    assert sha256_directory(model) == original


def test_candidate_registers_as_derived_discoverable_model(tmp_path):
    model = _parent(tmp_path)
    original = sha256_directory(model)
    candidate = apply_candidate_edit(tmp_path, "exp-weight-space-fixture01", model, original,
                                     _call(original, 1.05))
    parent_registry = tmp_path / "runs" / "base-models" / "model_artifacts"
    parent_registry.mkdir(parents=True)
    (parent_registry / "model-parent.json").write_text(ModelLineage(
        model_id="model-parent", kind="base", path=str(model),
        digest=Digest(status="available", value=original), stage="base",
    ).model_dump_json(), encoding="utf-8")
    record = register_candidate(tmp_path, candidate, "model-parent")
    assert record.parent_model_ids == ["model-parent"]
    assert record.edit_artifact_path == str(candidate.metadata_path)
    discovered = ModelPlaygroundCatalog(tmp_path).list_models()
    selected = next(item for item in discovered if item["model_id"] == record.model_id)
    assert selected["available"] is True and selected["digest"] == candidate.candidate_sha256
