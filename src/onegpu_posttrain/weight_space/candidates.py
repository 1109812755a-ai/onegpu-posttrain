"""Create immutable candidate checkpoints from validated bounded edits."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from safetensors.torch import load_file, save_file

from ..lineage import sha256_directory
from .tool_provider import ToolCall
from .weight_tools import _tensor_file, dispatch_weight_tool


@dataclass(frozen=True)
class CandidateArtifact:
    candidate_id: str
    model_path: Path
    metadata_path: Path
    parent_sha256: str
    candidate_sha256: str


def _candidate_record(workspace: str | Path, experiment_id: str, candidate_id: str) -> tuple[Path, dict]:
    if (not re.fullmatch(r"exp-weight-space-[a-z0-9-]{8,64}", experiment_id)
            or not re.fullmatch(r"candidate-[0-9a-f]{16}", candidate_id)):
        raise ValueError("invalid candidate identity")
    root = Path(workspace).resolve(strict=True)
    directory = (root / "runs" / "weight-space" / experiment_id / "candidates" / candidate_id).resolve(strict=True)
    if not directory.is_relative_to(root) or directory.is_symlink():
        raise ValueError("candidate escapes workspace")
    record_path = directory / "edit.json"
    if record_path.is_symlink() or (directory / "model").is_symlink():
        raise ValueError("candidate record cannot use symbolic links")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("candidate_id") != candidate_id or record.get("experiment_id") != experiment_id:
        raise ValueError("candidate record identity mismatch")
    if sha256_directory(directory / "model") != record.get("candidate_sha256"):
        raise ValueError("candidate checkpoint digest changed")
    return directory, record


def candidate_state(workspace: str | Path, experiment_id: str, candidate_id: str) -> dict:
    directory, _ = _candidate_record(workspace, experiment_id, candidate_id)
    state_path = directory / "state.json"
    if state_path.is_symlink():
        raise ValueError("candidate state cannot use a symbolic link")
    if not state_path.exists():
        return {"schema_version": "1.0", "state": "retained", "history": []}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("state") not in {"retained", "discarded"} or not isinstance(state.get("history"), list):
        raise ValueError("candidate state record is invalid")
    return state


def transition_candidate(workspace: str | Path, experiment_id: str, candidate_id: str,
                         action: str) -> dict:
    """Persist retain/discard/restore events without altering checkpoint bytes."""
    if action not in {"retain", "discard", "restore"}:
        raise ValueError("unknown candidate transition")
    directory, _ = _candidate_record(workspace, experiment_id, candidate_id)
    current = candidate_state(workspace, experiment_id, candidate_id)
    if action == "restore" and current["state"] != "discarded":
        raise ValueError("only a discarded candidate can be restored")
    if action == "discard" and current["state"] == "discarded":
        raise ValueError("candidate is already discarded")
    if action == "retain" and current["state"] == "discarded":
        raise ValueError("discarded candidate requires explicit restore")
    new_state = "discarded" if action == "discard" else "retained"
    updated = {"schema_version": "1.0", "state": new_state,
               "history": [*current["history"], {"action": action,
                                                "at": datetime.now(timezone.utc).isoformat()}]}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".state-", suffix=".tmp",
                                     dir=directory, delete=False) as stream:
        staged = Path(stream.name)
        json.dump(updated, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    staged.replace(directory / "state.json")
    return updated


def require_finalizable_candidate(workspace: str | Path, experiment_id: str,
                                  candidate_id: str) -> CandidateArtifact:
    directory, record = _candidate_record(workspace, experiment_id, candidate_id)
    if candidate_state(workspace, experiment_id, candidate_id)["state"] != "retained":
        raise ValueError("discarded candidate cannot enter final evaluation")
    return CandidateArtifact(candidate_id, directory / "model", directory / "edit.json",
                             record["parent_sha256"], record["candidate_sha256"])


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("candidate source cannot contain symbolic links")


def apply_candidate_edit(workspace: str | Path, experiment_id: str,
                         parent_path: str | Path, parent_sha256: str,
                         call: ToolCall) -> CandidateArtifact:
    """Validate, copy, edit, and atomically publish one candidate branch."""
    if call.name != "propose_edit" or not re.fullmatch(r"exp-weight-space-[a-z0-9-]{8,64}", experiment_id):
        raise ValueError("invalid experiment or edit tool")
    root = Path(workspace).resolve(strict=True)
    parent = Path(parent_path).resolve(strict=True)
    _reject_symlinks(parent)
    if sha256_directory(parent) != parent_sha256:
        raise ValueError("parent checkpoint digest changed")
    edit = dispatch_weight_tool(call, parent, parent_sha256)
    tensor_name = edit["tensor_name"]
    source_weight = _tensor_file(parent, tensor_name)
    candidates_root = root / "runs" / "weight-space" / experiment_id / "candidates"
    candidates_root.mkdir(parents=True, exist_ok=True)
    resolved_candidates_root = candidates_root.resolve(strict=True)
    if not resolved_candidates_root.is_relative_to(root):
        raise ValueError("candidate root escapes workspace")
    candidate_id = f"candidate-{uuid4().hex[:16]}"
    destination = resolved_candidates_root / candidate_id
    temporary = Path(tempfile.mkdtemp(prefix=".candidate-", dir=resolved_candidates_root))
    try:
        model_dir = temporary / "model"
        shutil.copytree(parent, model_dir, symlinks=False)
        copied_weight = model_dir / source_weight.name
        tensors = load_file(str(copied_weight), device="cpu")
        weight = tensors[tensor_name]
        start = edit["offset"]
        end = start + edit["element_count"]
        before = weight.reshape(-1)[start:end].to(torch.float32).tolist()
        if any(abs(observed - expected) > 1e-5
               for observed, expected in zip(before, edit["before"], strict=True)):
            raise ValueError("copied candidate does not match validated parent values")
        weight.reshape(-1)[start:end] = torch.tensor(edit["after"], dtype=weight.dtype)
        staged_weight = copied_weight.with_suffix(".safetensors.tmp")
        save_file(tensors, str(staged_weight))
        os.replace(staged_weight, copied_weight)
        del tensors
        candidate_sha256 = sha256_directory(model_dir)
        if sha256_directory(parent) != parent_sha256:
            raise ValueError("parent checkpoint changed during candidate edit")
        metadata = {"schema_version": "1.0", "candidate_id": candidate_id,
                    "experiment_id": experiment_id, "parent_sha256": parent_sha256,
                    "candidate_sha256": candidate_sha256, "tensor_name": tensor_name,
                    "offset": start, "changed_elements": edit["element_count"],
                    "before": edit["before"], "after": edit["after"],
                    "tool_call_id": call.id}
        (temporary / "edit.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
        temporary.replace(destination)
        return CandidateArtifact(candidate_id, destination / "model", destination / "edit.json",
                                 parent_sha256, candidate_sha256)
    finally:
        if temporary.exists() and temporary.resolve().is_relative_to(resolved_candidates_root):
            shutil.rmtree(temporary)
