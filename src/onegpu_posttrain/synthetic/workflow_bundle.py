"""Immutable, workspace-local input bundles for the SFT generation workflow."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..lineage import sha256_file
from ..data.targeted_plan import load_targeted_plan
from ..data.policy import load_policy
from .generation import prepare_generation


_BUNDLE_ID = re.compile(r"^plan-targeted-[0-9a-f]{20}$")
_INPUT_NAMES = {
    "plan", "profile", "template", "protocol", "definition", "report",
    "registry", "train_seeds", "policy", "diagnostic", "holdout",
}


class WorkflowBundle(BaseModel):
    """A content-pinned reference set; browser requests only use bundle_id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    bundle_id: str = Field(pattern=r"^plan-targeted-[0-9a-f]{20}$")
    plan_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inputs: dict[str, dict[str, str]]
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _inside_workspace_file(workspace: Path, path: Path) -> Path:
    root = workspace.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    current = candidate
    while current != root:
        if current.is_symlink():
            raise ValueError("generation bundle inputs cannot contain symbolic links")
        if current.parent == current:
            raise ValueError("generation bundle input escapes workspace")
        current = current.parent
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("generation bundle input must be a file inside workspace")
    return resolved


def _bundle_digest(bundle_id: str, plan_hash: str, inputs: dict[str, dict[str, str]]) -> str:
    raw = json.dumps({"bundle_id": bundle_id, "plan_config_sha256": plan_hash, "inputs": inputs},
                     sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def register_sft_workflow_bundle(
    workspace: str | Path,
    *,
    inputs: dict[str, str | Path],
) -> WorkflowBundle:
    """Validate every prerequisite and persist a new content-pinned bundle."""
    root = Path(workspace).resolve(strict=True)
    if set(inputs) != _INPUT_NAMES:
        raise ValueError("workflow bundle must provide exactly the registered SFT input set")
    resolved = {name: _inside_workspace_file(root, Path(value)) for name, value in inputs.items()}
    plan = load_targeted_plan(resolved["plan"])
    if plan.stage != "sft":
        raise ValueError("only SFT generation plans can be registered in this workflow")
    policy = load_policy(resolved["policy"])
    if policy.stage != "sft":
        raise ValueError("SFT workflow bundle requires an SFT data policy")
    # Full semantic and contamination validation is deliberately provider-free.
    prepare_generation(
        plan_path=resolved["plan"], profile_path=resolved["profile"],
        template_path=resolved["template"], protocol_path=resolved["protocol"],
        definition_path=resolved["definition"], report_path=resolved["report"],
        registry_path=resolved["registry"], train_seed_path=resolved["train_seeds"],
        protected_dataset_paths=[resolved["diagnostic"], resolved["holdout"]],
    )
    entries = {
        name: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for name, path in sorted(resolved.items())
    }
    bundle = WorkflowBundle(
        bundle_id=plan.plan_id,
        plan_config_sha256=plan.config_sha256,
        inputs=entries,
        bundle_sha256=_bundle_digest(plan.plan_id, plan.config_sha256, entries),
    )
    output = root / "artifacts" / "synthetic" / plan.plan_id / "workflow_bundle.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(bundle.model_dump(mode="json"), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return bundle


def load_sft_workflow_bundle(workspace: str | Path, bundle_id: str) -> tuple[WorkflowBundle, dict[str, Path]]:
    root = Path(workspace).resolve(strict=True)
    if not _BUNDLE_ID.fullmatch(bundle_id):
        raise ValueError("invalid generation bundle ID")
    path = root / "artifacts" / "synthetic" / bundle_id / "workflow_bundle.json"
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise FileNotFoundError("registered SFT workflow bundle not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        bundle = WorkflowBundle.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError):
        raise ValueError("registered SFT workflow bundle is invalid") from None
    if bundle.bundle_id != bundle_id or set(bundle.inputs) != _INPUT_NAMES:
        raise ValueError("registered SFT workflow bundle identity is invalid")
    if bundle.bundle_sha256 != _bundle_digest(bundle.bundle_id, bundle.plan_config_sha256, bundle.inputs):
        raise ValueError("registered SFT workflow bundle digest is invalid")
    resolved: dict[str, Path] = {}
    for name, entry in bundle.inputs.items():
        rel = Path(entry.get("path", ""))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("registered SFT workflow bundle path is invalid")
        file_path = _inside_workspace_file(root, root / rel)
        if sha256_file(file_path) != entry.get("sha256"):
            raise ValueError("registered SFT workflow bundle input changed after registration")
        resolved[name] = file_path
    plan = load_targeted_plan(resolved["plan"])
    if plan.plan_id != bundle_id or plan.config_sha256 != bundle.plan_config_sha256:
        raise ValueError("registered SFT plan identity changed after registration")
    return bundle, resolved
