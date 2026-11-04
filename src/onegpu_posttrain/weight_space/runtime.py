"""Probe the exact, isolated Docker runtime used for Qwen3.5 inference."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .manifest import DOCKER_IMAGE, MODEL_PROCESSOR_SHA256
from .registry import ResolvedInputs

PROBE_CODE = (
    "import json,torch,transformers; "
    "from transformers import AutoProcessor,AutoModelForMultimodalLM; "
    "print(json.dumps({'torch':torch.__version__,"
    "'transformers':transformers.__version__,"
    "'cuda':torch.cuda.is_available(),"
    "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
)
MOUNT_PROBE_CODE = """
import hashlib, json, os
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

model = Path('/model')
practice = Path('/suite/practice.jsonl')
final = Path('/suite/final.jsonl')
paths = (model, practice, final)
print(json.dumps({
    'processor_sha256': sha(model / 'chat_template.jinja'),
    'model_config': (model / 'config.json').is_file(),
    'model_weights': any(model.glob('*.safetensors')),
    'practice_sha256': sha(practice),
    'final_sha256': sha(final),
    'practice_count': sum(1 for _ in practice.open(encoding='utf-8')),
    'final_count': sum(1 for _ in final.open(encoding='utf-8')),
    'read_only': all(bool(os.statvfs(path).f_flag & os.ST_RDONLY) for path in paths),
}))
"""


@dataclass(frozen=True)
class RuntimeProbe:
    image: str
    torch_version: str
    transformers_version: str
    gpu: str


def docker_prefix() -> list[str]:
    if os.name == "nt":
        return ["wsl.exe", "-d", "Ubuntu-22.04", "--exec", "/usr/bin/docker"]
    return ["docker"]


def docker_host_path(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if os.name != "nt":
        return str(resolved)
    result = subprocess.run(
        ["wsl.exe", "-d", "Ubuntu-22.04", "--exec", "/usr/bin/wslpath", "-a", str(resolved)],
        capture_output=True, timeout=15, check=False,
    )
    if result.returncode:
        raise RuntimeError("cannot translate registered artifact path for WSL Docker")
    translated = result.stdout.decode("utf-8", errors="replace").strip()
    if not translated.startswith("/mnt/"):
        raise RuntimeError("registered artifact is not a WSL mount path")
    return translated


def probe_runtime(*, timeout_seconds: int = 90) -> RuntimeProbe:
    command = [
        *docker_prefix(), "run", "--rm", "--network", "none", "--gpus", "all",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "128", DOCKER_IMAGE, "python", "-c", PROBE_CODE,
    ]
    result = subprocess.run(command, capture_output=True, timeout=timeout_seconds, check=False)
    if result.returncode:
        raise RuntimeError(f"pinned Docker runtime failed with exit code {result.returncode}")
    try:
        output = result.stdout.decode("utf-8", errors="replace")
        payload = next(json.loads(line) for line in reversed(output.splitlines()) if line.startswith("{"))
    except (StopIteration, ValueError) as exc:
        raise RuntimeError("pinned Docker runtime did not return probe data") from exc
    if (payload.get("torch") != "2.6.0+cu124"
            or payload.get("transformers") != "5.8.0"
            or payload.get("cuda") is not True
            or not isinstance(payload.get("gpu"), str) or not payload["gpu"]):
        raise RuntimeError("pinned Docker runtime versions or GPU capability changed")
    return RuntimeProbe(DOCKER_IMAGE, payload["torch"], payload["transformers"], payload["gpu"])


def probe_registered_mounts(inputs: ResolvedInputs, *, timeout_seconds: int = 90) -> dict[str, object]:
    """Check exact read-only artifact mounts from inside the pinned container."""
    mounts = [
        (inputs.model_path, "/model"),
        (inputs.practice_path, "/suite/practice.jsonl"),
        (inputs.final_path, "/suite/final.jsonl"),
    ]
    command = [
        *docker_prefix(), "run", "--rm", "--network", "none", "--gpus", "all",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "128",
    ]
    for source, target in mounts:
        command.extend(["--mount", f"type=bind,source={docker_host_path(source)},target={target},readonly"])
    command.extend([DOCKER_IMAGE, "python", "-c", MOUNT_PROBE_CODE])
    result = subprocess.run(command, capture_output=True, timeout=timeout_seconds, check=False)
    if result.returncode:
        raise RuntimeError(f"registered Docker mount probe failed with exit code {result.returncode}")
    try:
        output = result.stdout.decode("utf-8", errors="replace")
        payload = next(json.loads(line) for line in reversed(output.splitlines()) if line.startswith("{"))
    except (StopIteration, ValueError) as exc:
        raise RuntimeError("registered Docker mount probe returned no result") from exc
    if (payload.get("processor_sha256") != MODEL_PROCESSOR_SHA256
            or payload.get("practice_sha256") != inputs.manifest.practice.canonical_sha256
            or payload.get("final_sha256") != inputs.manifest.final_holdout.canonical_sha256
            or payload.get("practice_count") != inputs.manifest.practice.record_count
            or payload.get("final_count") != inputs.manifest.final_holdout.record_count
            or payload.get("model_config") is not True
            or payload.get("model_weights") is not True
            or payload.get("read_only") is not True):
        raise RuntimeError("registered Docker mounts differ from pinned artifacts")
    return payload
