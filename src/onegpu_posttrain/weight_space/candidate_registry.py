"""Register a verified edited candidate as a discoverable derived model."""

from __future__ import annotations

import json
from pathlib import Path

from ..lineage import Digest, ModelLineage, sha256_directory
from .candidates import CandidateArtifact


def register_candidate(workspace: str | Path, artifact: CandidateArtifact,
                       parent_model_id: str) -> ModelLineage:
    root = Path(workspace).resolve(strict=True)
    model_path = artifact.model_path.resolve(strict=True)
    edit_path = artifact.metadata_path.resolve(strict=True)
    if (not model_path.is_relative_to(root) or not edit_path.is_relative_to(root)
            or edit_path.parent != model_path.parent or not parent_model_id.startswith("model-")):
        raise ValueError("candidate lineage paths or parent identity are invalid")
    edit = json.loads(edit_path.read_text(encoding="utf-8"))
    if (edit.get("candidate_id") != artifact.candidate_id
            or edit.get("parent_sha256") != artifact.parent_sha256
            or edit.get("candidate_sha256") != artifact.candidate_sha256
            or sha256_directory(model_path) != artifact.candidate_sha256):
        raise ValueError("candidate edit evidence does not match model bytes")
    parent_registries = list((root / "runs").glob(f"**/model_artifacts/{parent_model_id}.json"))[:2]
    if len(parent_registries) != 1:
        raise ValueError("candidate parent model registration is missing or ambiguous")
    parent = ModelLineage.model_validate_json(parent_registries[0].read_text(encoding="utf-8"))
    if parent.model_id != parent_model_id or parent.digest.value != artifact.parent_sha256:
        raise ValueError("candidate parent model digest does not match registration")
    model_id = f"model-{artifact.candidate_id.removeprefix('candidate-')}"
    registry_dir = edit_path.parent / "model_artifacts"
    registry_dir.mkdir(exist_ok=True)
    registry_path = registry_dir / f"{model_id}.json"
    if registry_path.exists():
        raise ValueError("candidate model is already registered")
    record = ModelLineage(
        model_id=model_id, kind="weight_space_candidate", path=str(model_path),
        digest=Digest(status="available", value=artifact.candidate_sha256),
        parent_model_ids=[parent_model_id], stage="weight_space", edit_artifact_path=str(edit_path),
    )
    registry_path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return record
