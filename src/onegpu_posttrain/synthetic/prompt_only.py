"""Immutable, source-free SFT generation plans and bundles."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..data.manifest import sha256_file
from ..data.policy import DataPolicy, digest_json
from .providers import (
    GenerationRequest,
    OutputProtocol,
    ProviderProfile,
    validate_request_support,
)

_BUNDLE_ID = re.compile(r"^plan-sft-prompt-[0-9a-f]{20}$")
PROTOCOL = OutputProtocol(protocol_id="prompt-only-sft-candidate", version="1",
                          required_fields=["question", "solution", "final_answer"])
TEMPLATE_ID = "prompt-only-sft-math-v1"
TEMPLATE_TEXT = (
    "Create one original math training example. Topic: {topic}. Skill: {skill}. "
    "Difficulty: {difficulty}. Constraints: {constraints}. Return exactly one JSON object "
    "with string fields question, solution, final_answer. The question must be solvable; "
    "solution must show a clear derivation; final_answer must be concise. Do not copy or "
    "refer to any seed example."
)


class PromptOnlyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0"] = "1.0"
    mode: Literal["prompt_only_sft"] = "prompt_only_sft"
    topic: str = Field(min_length=2, max_length=120)
    skill: str = Field(min_length=2, max_length=120)
    difficulty: Literal["introductory", "intermediate", "advanced"]
    constraints: str = Field(default="", max_length=500)
    target_count: int = Field(ge=1, le=1000)
    request_limit: int = Field(ge=1, le=3000)
    max_output_tokens: int = Field(default=1024, ge=64, le=4096)
    temperature: float = Field(default=0.4, ge=0, le=2, allow_inf_nan=False)
    provider_profile_id: str
    provider_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_id: str
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_template_id: str = TEMPLATE_ID
    prompt_template_sha256: str
    output_protocol_id: str = PROTOCOL.protocol_id
    output_protocol_sha256: str

    @field_validator("topic", "skill", "constraints")
    @classmethod
    def reject_controls(cls, value: str) -> str:
        if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
            raise ValueError("control characters are not allowed")
        return value.strip()

    @property
    def config_sha256(self) -> str:
        return digest_json(self.model_dump(mode="json"))

    @property
    def plan_id(self) -> str:
        return f"plan-sft-prompt-{self.config_sha256[:20]}"


def _inside_file(root: Path, path: Path) -> Path:
    current = path
    while current != root:
        if current.is_symlink() or current.parent == current:
            raise ValueError("prompt-only bundle inputs must be regular workspace files")
        current = current.parent
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("prompt-only bundle input is outside workspace")
    return resolved


def prepare_prompt_only_bundle(workspace: str | Path, *, topic: str, skill: str,
                               difficulty: str, constraints: str = "", target_count: int,
                               profile_id: str | None = None) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    profile_root = root / "configs" / "synthetic"
    profiles: list[tuple[ProviderProfile, Path]] = []
    for path in profile_root.rglob("provider*.yaml") if profile_root.is_dir() else ():
        if path.is_symlink() or "example" in path.name or "fake" in path.name:
            continue
        try:
            path = _inside_file(root, path)
            profile = ProviderProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, yaml.YAMLError):
            continue
        if not profile.api_key_env:
            continue
        env_ready = bool(__import__("os").environ.get(profile.api_key_env))
        env_file = root / ".env.local"
        if not env_ready and env_file.is_file():
            env_ready = any(line.strip().partition("=")[0].removeprefix("export ").strip() == profile.api_key_env
                            and line.strip().partition("=")[2].strip()
                            for line in env_file.read_text(encoding="utf-8").splitlines()
                            if line.strip() and not line.lstrip().startswith("#") and "=" in line)
        if env_ready:
            profiles.append((profile, path))
    selected = next((pair for pair in profiles if pair[0].profile_id == profile_id), None) if profile_id else None
    if profile_id and selected is None:
        raise ValueError("selected Provider profile is not configured")
    if not profile_id:
        if len(profiles) != 1:
            raise ValueError("select one configured Provider profile")
        selected = profiles[0]
    assert selected is not None
    profile, profile_path = selected

    policy_artifact = next((item for item in __import__("onegpu_posttrain.data.policy", fromlist=["policy_templates"])
                            .policy_templates() if item["config"]["stage"] == "sft"), None)
    if policy_artifact is None:
        raise ValueError("default SFT policy is unavailable")
    policy = DataPolicy.model_validate(policy_artifact["config"])
    validate_request_support(profile, GenerationRequest(prompt="validation-only", max_output_tokens=1024,
        temperature=0.4, output_protocol=PROTOCOL))
    template_sha = hashlib.sha256(TEMPLATE_TEXT.encode("utf-8")).hexdigest()
    protocol_sha = hashlib.sha256(PROTOCOL.model_dump_json().encode("utf-8")).hexdigest()
    plan = PromptOnlyPlan(topic=topic, skill=skill, difficulty=difficulty, constraints=constraints,
        target_count=target_count, request_limit=min(max(target_count * 2, target_count), 3000),
        provider_profile_id=profile.profile_id, provider_profile_sha256=sha256_file(profile_path),
        policy_id=policy.policy_id, policy_sha256=policy.config_sha256,
        prompt_template_sha256=template_sha, output_protocol_sha256=protocol_sha)
    bundle_id = plan.plan_id
    base = root / "artifacts" / "synthetic" / bundle_id
    base.mkdir(parents=True, exist_ok=True)
    for name, payload in (("prompt_only_plan.json", plan.model_dump(mode="json") | {"plan_id": plan.plan_id}),
                          ("prompt_only_policy.json", policy.model_dump(mode="json"))):
        path = base / name
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError("immutable prompt-only plan input differs")
        if not path.exists():
            path.write_text(encoded, encoding="utf-8")
    profile_path = _inside_file(root, profile_path)
    inputs = {"plan": base / "prompt_only_plan.json", "policy": base / "prompt_only_policy.json",
              "profile": profile_path}
    entries = {name: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
               for name, path in sorted(inputs.items())}
    digest = hashlib.sha256(json.dumps({"bundle_id": bundle_id, "plan": plan.config_sha256, "inputs": entries},
                                      sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    bundle = {"schema_version": "1.0", "mode": "prompt_only_sft", "bundle_id": bundle_id,
              "plan_config_sha256": plan.config_sha256, "inputs": entries, "bundle_sha256": digest}
    bundle_path = base / "prompt_only_workflow_bundle.json"
    encoded = json.dumps(bundle, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if bundle_path.exists() and bundle_path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError("immutable prompt-only bundle differs")
    if not bundle_path.exists():
        bundle_path.write_text(encoded, encoding="utf-8")
    return {"mode": "prompt_only_sft", "bundle_id": bundle_id, "provider_profile_id": profile.profile_id,
            "model": profile.model, "target_count": target_count, "provider_calls": 0,
            "network_used": False, "status": "prepared_not_executed"}


def load_prompt_only_bundle(workspace: str | Path, bundle_id: str) -> tuple[PromptOnlyPlan, dict[str, Path]]:
    root = Path(workspace).resolve(strict=True)
    if not _BUNDLE_ID.fullmatch(bundle_id):
        raise ValueError("invalid prompt-only bundle id")
    base = _inside_file(root, root / "artifacts" / "synthetic" / bundle_id / "prompt_only_workflow_bundle.json")
    bundle = json.loads(base.read_text(encoding="utf-8"))
    entries = bundle.get("inputs", {})
    if bundle.get("bundle_id") != bundle_id or set(entries) != {"plan", "policy", "profile"}:
        raise ValueError("invalid prompt-only bundle identity")
    paths: dict[str, Path] = {}
    for name, item in entries.items():
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("invalid prompt-only input path")
        path = _inside_file(root, root / rel)
        if sha256_file(path) != item["sha256"]:
            raise ValueError("prompt-only input digest changed")
        paths[name] = path
    payload = json.loads(paths["plan"].read_text(encoding="utf-8"))
    plan_id = payload.pop("plan_id", None)
    plan = PromptOnlyPlan.model_validate(payload)
    if plan.plan_id != bundle_id or plan_id != bundle_id or plan.config_sha256 != bundle["plan_config_sha256"]:
        raise ValueError("prompt-only plan digest mismatch")
    computed = hashlib.sha256(json.dumps({"bundle_id": bundle_id, "plan": plan.config_sha256, "inputs": entries},
                                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if computed != bundle.get("bundle_sha256"):
        raise ValueError("prompt-only bundle digest mismatch")
    return plan, paths
