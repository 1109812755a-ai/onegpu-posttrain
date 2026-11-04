"""Fixed tensor read and edit-proposal tools with small, numeric outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .tool_provider import ToolCall

MAX_ELEMENTS = 32
MAX_ABSOLUTE_DELTA = 0.1
ALLOWED_PREFIX = "model.language_model.layers."


def _tensor_file(model_path: Path, name: str) -> Path:
    if not isinstance(name, str) or not name.startswith(ALLOWED_PREFIX) or len(name) > 200:
        raise ValueError("tensor is not in the allowed language-model layers")
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    filename = index.get("weight_map", {}).get(name)
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".safetensors"):
        raise ValueError("tensor is unknown or its weight file is invalid")
    path = model_path / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError("tensor weight file is unavailable")
    return path


def _coordinates(index: int, shape: list[int]) -> tuple[int, ...]:
    coords = []
    for dimension in reversed(shape):
        coords.append(index % dimension)
        index //= dimension
    return tuple(reversed(coords))


def inspect_tensor(model_path: str | Path, tensor_name: str, offset: int, length: int) -> dict[str, Any]:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("tensor offset is invalid")
    if not isinstance(length, int) or isinstance(length, bool) or not 1 <= length <= MAX_ELEMENTS:
        raise ValueError("tensor read length exceeds limit")
    root = Path(model_path).resolve(strict=True)
    path = _tensor_file(root, tensor_name)
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        tensor_slice = reader.get_slice(tensor_name)
        shape = tensor_slice.get_shape()
        count = math.prod(shape)
        if not shape or not count or offset + length > count:
            raise ValueError("tensor read range exceeds shape")
        values = []
        dtype = None
        for index in range(offset, offset + length):
            coords = _coordinates(index, shape)
            item = tensor_slice[tuple(slice(coord, coord + 1) for coord in coords)]
            dtype = str(item.dtype)
            value = float(item.item())
            if not math.isfinite(value):
                raise ValueError("tensor contains a non-finite value")
            values.append(value)
        return {"tensor_name": tensor_name, "shape": shape, "dtype": dtype,
                "offset": offset, "values": values}


def dispatch_weight_tool(call: ToolCall, model_path: str | Path, parent_sha256: str) -> dict[str, Any]:
    args = call.arguments
    if call.name == "inspect_tensor":
        if set(args) != {"tensor_name", "offset", "length"}:
            raise ValueError("invalid inspect arguments")
        return inspect_tensor(model_path, args["tensor_name"], args["offset"], args["length"])
    if call.name == "propose_edit":
        if set(args) != {"tensor_name", "offset", "values", "parent_sha256"}:
            raise ValueError("invalid edit arguments")
        if args["parent_sha256"] != parent_sha256:
            raise ValueError("stale parent checkpoint digest")
        values = args["values"]
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_ELEMENTS:
            raise ValueError("edit element count exceeds limit")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not math.isfinite(value) for value in values):
            raise ValueError("edit values must be finite numbers")
        before = inspect_tensor(model_path, args["tensor_name"], args["offset"], len(values))
        if before["dtype"] not in {"torch.float16", "torch.bfloat16", "torch.float32"}:
            raise ValueError("only floating-point model weights can be edited")
        if any(abs(float(new) - old) > MAX_ABSOLUTE_DELTA
               for new, old in zip(values, before["values"], strict=True)):
            raise ValueError("edit exceeds per-element change budget")
        return {"status": "validated", "tensor_name": args["tensor_name"],
                "offset": args["offset"], "element_count": len(values),
                "parent_sha256": parent_sha256, "before": before["values"],
                "after": [float(value) for value in values]}
    raise ValueError("unknown weight tool")
