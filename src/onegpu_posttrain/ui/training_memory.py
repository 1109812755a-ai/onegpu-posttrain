"""Local, read-only compatibility checks for approved memory presets."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import importlib
from typing import Any


def inspect_training_memory_mode(mode_id: str) -> dict[str, Any]:
    """Inspect QLoRA prerequisites without installing packages or changing state."""
    if mode_id != "qlora-4bit-nf4":
        return {"ok": True, "mode_id": mode_id, "dependencies": []}

    dependencies: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        version = importlib.metadata.version("bitsandbytes")
        if importlib.util.find_spec("bitsandbytes") is None:
            raise ImportError("bitsandbytes module is not importable")
        importlib.import_module("bitsandbytes")
        dependencies.append({"name": "bitsandbytes", "available": True, "version": version})
    except importlib.metadata.PackageNotFoundError:
        dependencies.append({"name": "bitsandbytes", "available": False, "version": None})
        blockers.append("QLoRA 需要当前项目 Python 环境安装 bitsandbytes；预检不会自动安装依赖。")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        dependencies.append({"name": "bitsandbytes", "available": False, "version": None})
        blockers.append(f"当前 bitsandbytes 无法导入或与运行环境不兼容：{str(exc)[:160]}")

    try:
        torch_version = importlib.metadata.version("torch")
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            raise ModuleNotFoundError
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_runtime = getattr(getattr(torch, "version", None), "cuda", None)
        dependencies.append({"name": "torch", "available": True, "version": torch_version,
                             "cuda_available": cuda_available, "cuda_runtime": cuda_runtime})
        if not cuda_available:
            blockers.append("QLoRA 需要 CUDA 可用的 PyTorch；当前环境未检测到可用 CUDA GPU。")
    except (importlib.metadata.PackageNotFoundError, ImportError, OSError, RuntimeError):
        dependencies.append({"name": "torch", "available": False, "version": None,
                             "cuda_available": False, "cuda_runtime": None})
        blockers.append("QLoRA 需要可导入且支持 CUDA 的 PyTorch；当前环境无法验证。")

    return {"ok": not blockers, "mode_id": mode_id, "dependencies": dependencies,
            "blockers": blockers}
