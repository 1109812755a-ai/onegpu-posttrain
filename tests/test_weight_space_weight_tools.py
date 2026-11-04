"""Read and edit proposals stay within declared tensor and value budgets."""

import json

import pytest
import torch
from safetensors.torch import save_file

from onegpu_posttrain.weight_space.tool_provider import ToolCall
from onegpu_posttrain.weight_space.weight_tools import (
    dispatch_weight_tool,
    inspect_tensor,
)

TENSOR = "model.language_model.layers.0.mlp.down_proj.weight"


def _model(tmp_path):
    weights = tmp_path / "model.safetensors"
    save_file({TENSOR: torch.arange(12, dtype=torch.float32).reshape(3, 4)}, weights)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {TENSOR: weights.name}}), encoding="utf-8")
    return tmp_path


def test_inspect_reads_bounded_slice(tmp_path):
    model = _model(tmp_path)
    result = inspect_tensor(model, TENSOR, 3, 3)
    assert result["shape"] == [3, 4]
    assert result["values"] == [3.0, 4.0, 5.0]


def test_inspect_rejects_unknown_tensor_and_large_read(tmp_path):
    model = _model(tmp_path)
    with pytest.raises(ValueError, match="allowed"):
        inspect_tensor(model, "model.visual.layers.0.weight", 0, 1)
    with pytest.raises(ValueError, match="length"):
        inspect_tensor(model, TENSOR, 0, 33)
    with pytest.raises(ValueError, match="shape"):
        inspect_tensor(model, TENSOR, 11, 2)


def test_edit_proposal_rejects_stale_or_large_delta(tmp_path):
    model = _model(tmp_path)
    arguments = {"tensor_name": TENSOR, "offset": 0, "values": [0.05], "parent_sha256": "a" * 64}
    assert dispatch_weight_tool(ToolCall(id="1", name="propose_edit", arguments=arguments), model,
                                "a" * 64)["status"] == "validated"
    with pytest.raises(ValueError, match="stale"):
        dispatch_weight_tool(ToolCall(id="1", name="propose_edit", arguments=arguments), model, "b" * 64)
    arguments["values"] = [1000.0]
    with pytest.raises(ValueError, match="change budget"):
        dispatch_weight_tool(ToolCall(id="1", name="propose_edit", arguments=arguments), model, "a" * 64)
