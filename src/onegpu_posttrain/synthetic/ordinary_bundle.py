"""Immutable workspace bundle registration for ordinary SFT generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..data.manifest import load_manifest, sha256_file
from ..data.policy import DataPolicy, load_policy
from ..data.targeted import GenerationParameters
from .ordinary_plan import OrdinarySFTPlan
from .providers import OutputProtocol, ProviderProfile, PromptTemplate, GenerationRequest, validate_request_support


_BUNDLE_ID = r"^plan-sft-ordinary-[0-9a-f]{20}$"
_INPUTS = {"source_manifest", "source_data", "profile", "template", "protocol", "policy", "plan"}


class OrdinarySFTBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0"
    mode: str = "ordinary_sft"
    bundle_id: str = Field(pattern=_BUNDLE_ID)
    plan_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    inputs: dict[str, dict[str, str]]
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _inside_file(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    candidate = candidate if candidate.is_absolute() else root / candidate
    current = candidate
    while current != root:
        if current.is_symlink() or current.parent == current:
            raise ValueError("ordinary bundle input must be a regular workspace file")
        current = current.parent
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("ordinary bundle input must be a file inside workspace")
    return resolved


def _digest(bundle_id: str, plan_sha: str, inputs: dict[str, dict[str, str]]) -> str:
    raw = json.dumps({"bundle_id": bundle_id, "plan_config_sha256": plan_sha, "inputs": inputs},
                     sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def register_ordinary_sft_bundle(
    workspace: str | Path,
    *,
    source_dataset_id: str,
    source_manifest_path: str | Path,
    source_data_path: str | Path,
    profile_path: str | Path,
    template_path: str | Path,
    protocol_path: str | Path,
    policy_payload: dict[str, Any],
    target_count: int,
    request_limit: int | None = None,
) -> OrdinarySFTBundle:
    """Validate allowlisted local identities and create a content-pinned bundle."""
    root = Path(workspace).resolve(strict=True)
    manifest_path = _inside_file(root, source_manifest_path)
    data_path = _inside_file(root, source_data_path)
    provider_path = _inside_file(root, profile_path)
    prompt_path = _inside_file(root, template_path)
    output_path = _inside_file(root, protocol_path)

    manifest = load_manifest(manifest_path)
    if manifest.stage != "sft" or manifest.dataset_id != source_dataset_id:
        raise ValueError("普通 SFT 方案要求所选 SFT dataset ID 与 manifest 一致")
    source_hash = sha256_file(data_path)
    if manifest.hashes.get("canonical_records_sha256") != source_hash:
        raise ValueError("source dataset 摘要与 manifest 不一致")

    try:
        profile = ProviderProfile.model_validate(yaml.safe_load(provider_path.read_text(encoding="utf-8")))
        template = PromptTemplate.model_validate(yaml.safe_load(prompt_path.read_text(encoding="utf-8")))
        protocol = OutputProtocol.model_validate(yaml.safe_load(output_path.read_text(encoding="utf-8")))
        policy = DataPolicy.model_validate(policy_payload)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError, TypeError, ValueError):
        raise ValueError("普通 SFT 方案中的本地 profile/template/protocol/policy 无效") from None
    if policy.stage != "sft":
        raise ValueError("普通生成只接受 SFT policy")
    target_count = int(target_count)
    if not 1 <= target_count <= 10000:
        raise ValueError("target_count 必须在 1 到 10000 之间")
    request_limit = request_limit if request_limit is not None else min(max(target_count * 2, target_count), 30000)
    if not target_count <= request_limit <= 30000:
        raise ValueError("request_limit 必须大于等于目标数量且不超过 30000")
    parameters = GenerationParameters(temperature=0.2, provider_seed=42)
    validate_request_support(profile, GenerationRequest(
        prompt="validation-only", max_output_tokens=1536, temperature=parameters.temperature,
        seed=None, output_protocol=protocol,
    ))

    policy_path = root / "artifacts" / "synthetic" / "ordinary-policy-snapshots" / f"{policy.policy_id}.yaml"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    if not policy_path.exists():
        with policy_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(policy.model_dump(mode="json"), stream, allow_unicode=True, sort_keys=False)
    policy_path = _inside_file(root, policy_path)
    if load_policy(policy_path).config_sha256 != policy.config_sha256:
        raise ValueError("普通 SFT policy 快照已变化；拒绝覆盖历史配置")

    plan = OrdinarySFTPlan(
        source_dataset_id=manifest.dataset_id,
        source_content_sha256=source_hash,
        source_manifest_sha256=sha256_file(manifest_path),
        provider_profile_id=profile.profile_id,
        provider_profile_sha256=sha256_file(provider_path),
        prompt_template_id=template.template_id,
        prompt_template_sha256=sha256_file(prompt_path),
        output_protocol_id=protocol.protocol_id,
        output_protocol_sha256=sha256_file(output_path),
        policy_id=policy.policy_id,
        policy_config_sha256=policy.config_sha256,
        target_count=target_count,
        request_limit=request_limit,
    )
    bundle_id = plan.plan_id
    plan_path = root / "artifacts" / "synthetic" / bundle_id / "ordinary_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    if not plan_path.exists():
        with plan_path.open("x", encoding="utf-8") as stream:
            json.dump(plan.model_dump(mode="json") | {"plan_id": plan.plan_id,
                      "config_sha256": plan.config_sha256}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    plan_path = _inside_file(root, plan_path)
    try:
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        existing_plan_id = existing_plan.pop("plan_id", None)
        existing_plan_sha = existing_plan.pop("config_sha256", None)
        parsed_existing_plan = OrdinarySFTPlan.model_validate(existing_plan)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, TypeError, ValueError):
        raise ValueError("普通 SFT 方案文件无效；不会覆盖历史方案") from None
    if (parsed_existing_plan != plan or existing_plan_id != plan.plan_id
            or existing_plan_sha != plan.config_sha256):
        raise ValueError("已有普通 SFT 方案与当前解析配置不一致；不会覆盖历史方案")

    paths = {"source_manifest": manifest_path, "source_data": data_path, "profile": provider_path,
             "template": prompt_path, "protocol": output_path, "policy": policy_path, "plan": plan_path}
    entries = {name: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
               for name, path in sorted(paths.items())}
    bundle = OrdinarySFTBundle(
        bundle_id=bundle_id, plan_config_sha256=plan.config_sha256,
        inputs=entries, bundle_sha256=_digest(bundle_id, plan.config_sha256, entries),
    )
    bundle_path = plan_path.parent / "ordinary_workflow_bundle.json"
    if bundle_path.exists():
        existing = OrdinarySFTBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
        if existing != bundle:
            raise FileExistsError("已有普通 SFT bundle 不同；不会覆盖历史方案")
        return existing
    with bundle_path.open("x", encoding="utf-8") as stream:
        json.dump(bundle.model_dump(mode="json"), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return bundle


def load_ordinary_sft_bundle(workspace: str | Path, bundle_id: str) -> tuple[OrdinarySFTBundle, dict[str, Path]]:
    root = Path(workspace).resolve(strict=True)
    if not __import__("re").fullmatch(_BUNDLE_ID, bundle_id):
        raise ValueError("invalid ordinary SFT bundle ID")
    bundle_path = root / "artifacts" / "synthetic" / bundle_id / "ordinary_workflow_bundle.json"
    bundle_path = _inside_file(root, bundle_path)
    try:
        bundle = OrdinarySFTBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, ValidationError):
        raise ValueError("ordinary SFT bundle is invalid") from None
    if bundle.bundle_id != bundle_id or set(bundle.inputs) != _INPUTS:
        raise ValueError("ordinary SFT bundle identity is invalid")
    if bundle.bundle_sha256 != _digest(bundle.bundle_id, bundle.plan_config_sha256, bundle.inputs):
        raise ValueError("ordinary SFT bundle digest is invalid")
    resolved: dict[str, Path] = {}
    for name, entry in bundle.inputs.items():
        rel = Path(entry.get("path", ""))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("ordinary SFT bundle path is invalid")
        path = _inside_file(root, root / rel)
        if sha256_file(path) != entry.get("sha256"):
            raise ValueError("ordinary SFT bundle input changed after registration")
        resolved[name] = path
    plan_payload = json.loads(resolved["plan"].read_text(encoding="utf-8"))
    plan_id = plan_payload.pop("plan_id", None)
    plan_sha = plan_payload.pop("config_sha256", None)
    plan = OrdinarySFTPlan.model_validate(plan_payload)
    if plan.plan_id != bundle_id or plan_id != plan.plan_id or plan_sha != plan.config_sha256:
        raise ValueError("ordinary SFT plan identity changed")
    if plan.config_sha256 != bundle.plan_config_sha256:
        raise ValueError("ordinary SFT plan digest changed")
    return bundle, resolved
