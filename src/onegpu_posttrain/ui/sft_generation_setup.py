"""Read-only discovery of local inputs for the guided SFT generation workflow."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from ..data.manifest import load_manifest, sha256_file
from ..data.policy import DataPolicy, policy_templates
from ..data.roles import load_role_registry
from ..data.targeted import ScenarioDefinition
from ..synthetic.ordinary_bundle import register_ordinary_sft_bundle
from ..synthetic.prompt_only import prepare_prompt_only_bundle
from ..synthetic.providers import OutputProtocol, PromptTemplate, ProviderProfile


def _credential_available(root: Path, env_name: str | None) -> bool:
    """Report only presence; never return or load a local secret during discovery."""
    if not env_name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
        return False
    if os.environ.get(env_name):
        return True
    path = root / ".env.local"
    if not _inside(root, path) or not path.is_file():
        return False
    try:
        found = False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, separator, value = line.partition("=")
            if not separator or key.strip() != env_name or found:
                return False
            value = value.strip()
            if value.startswith(("'", '"')):
                if len(value) < 2 or value[-1] != value[0]:
                    return False
                value = value[1:-1]
            if "\x00" in value:
                return False
            found = bool(value)
        return found
    except (OSError, UnicodeError):
        return False
    return False


def _inside(root: Path, path: Path) -> bool:
    try:
        return not path.is_symlink() and path.resolve(strict=True).is_relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False


def _sft_datasets(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in (root / "data" / "processed", root / "artifacts" / "ui-demo"):
        if not _inside(root, base) or not base.is_dir():
            continue
        for manifest_path in base.rglob("manifests/*.yaml"):
            if len(rows) >= 500 or not _inside(root, manifest_path):
                continue
            try:
                manifest = load_manifest(manifest_path)
                if manifest.stage != "sft":
                    continue
                artifact_root = manifest_path.parent.parent
                canonical = artifact_root / "canonical" / "sft.jsonl"
                digest = sha256_file(canonical) if _inside(root, canonical) and canonical.is_file() else None
                expected = manifest.hashes.get("canonical_records_sha256")
                hash_matches = digest is not None and expected is not None and digest == expected
                rows.append({
                    "dataset_id": manifest.dataset_id,
                    "stage": manifest.stage,
                    "sample_count": manifest.sample_count,
                    "content_sha256": digest,
                    "manifest_sha256": sha256_file(manifest_path),
                    "origin": manifest.origin.kind if manifest.origin else "unknown",
                    "eligible": bool(hash_matches),
                    "_manifest_path": manifest_path,
                    "_canonical_path": canonical,
                    "missing_reasons": [] if hash_matches else [
                        "canonical SFT 文件缺失、越界或摘要与 manifest 不一致"
                    ],
                })
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                continue
    return sorted(rows, key=lambda row: (not row["eligible"], row["dataset_id"]))


def _profiles(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = root / "configs" / "synthetic"
    if not _inside(root, base) or not base.is_dir():
        return rows
    for path in base.rglob("provider*.yaml"):
        if not _inside(root, path):
            continue
        try:
            profile = ProviderProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
            rows.append({
                "profile_id": profile.profile_id,
                "adapter": profile.adapter,
                "model": profile.model,
                "credential_env_configured": _credential_available(root, profile.api_key_env),
                "credential_env_name": profile.api_key_env,
                "is_example": "example" in path.name or profile.adapter == "fake",
            })
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            continue
    return sorted(rows, key=lambda row: (row["is_example"], row["profile_id"]))


def _validated_local_configs(root: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {"templates": [], "protocols": [], "policies": []}
    base = root / "configs" / "synthetic" / "examples"
    if _inside(root, base) and base.is_dir():
        for path in base.glob("*.yaml"):
            if not _inside(root, path):
                continue
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    continue
                for key, model in (("templates", PromptTemplate), ("protocols", OutputProtocol)):
                    try:
                        parsed = model.model_validate(raw)
                    except ValueError:
                        continue
                    result[key].append({"id": parsed.template_id if key == "templates" else parsed.protocol_id,
                                        "version": parsed.version})
            except (OSError, yaml.YAMLError):
                continue
    for item in policy_templates():
        result["policies"].append({"id": item["policy_id"], "version": item["config_sha256"][:12]})
    for path in (root / "configs" / "data" / "policies").glob("*.yaml") if (root / "configs" / "data" / "policies").is_dir() else ():
        if _inside(root, path):
            result["policies"].append({"id": path.stem, "version": "local"})
    for key, rows in result.items():
        result[key] = sorted({(row["id"], row["version"]) for row in rows})
        result[key] = [{"id": item[0], "version": item[1]} for item in result[key]]
    return result


def _scenario_reports(root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for base in (root / "artifacts" / "synthetic", root / "artifacts" / "targeted"):
        if not _inside(root, base) or not base.is_dir():
            continue
        for path in base.rglob("*scenario*report*.json"):
            if len(reports) >= 200 or not _inside(root, path):
                continue
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict) or not isinstance(raw.get("scenarios"), list):
                    continue
                reports.append({
                    "report_id": raw.get("report_id"),
                    "eval_id": raw.get("eval_id"),
                    "dataset_id": raw.get("dataset_id"),
                    "dataset_sha256": raw.get("dataset_sha256"),
                    "model_id": raw.get("model_id"),
                    "report_sha256": sha256_file(path),
                    "scenarios": [{"scenario_id": item.get("scenario_id"), "status": item.get("status"),
                                   "denominator": item.get("denominator")}
                                  for item in raw["scenarios"] if isinstance(item, dict)],
                })
            except (OSError, TypeError, ValueError):
                continue
    return reports


def _targeting_evidence(root: Path) -> dict[str, list[dict[str, Any]]]:
    definitions: list[dict[str, Any]] = []
    registries: list[dict[str, Any]] = []
    for base in (root / "artifacts", root / "configs"):
        if not _inside(root, base) or not base.is_dir():
            continue
        for path in base.rglob("*.yaml"):
            if len(definitions) >= 200 or not _inside(root, path) or "scenario" not in path.name.lower():
                continue
            try:
                definition = ScenarioDefinition.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
                definitions.append({"definition_id": definition.definition_id,
                                    "dataset_id": definition.dataset_id,
                                    "dataset_sha256": definition.dataset_sha256,
                                    "scenario_ids": [item.scenario_id for item in definition.scenarios]})
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                continue
        for path in base.rglob("*.json"):
            if len(registries) >= 200 or not _inside(root, path) or "registry" not in path.name.lower():
                continue
            try:
                registry = load_role_registry(path)
                datasets: dict[str, dict[str, Any]] = {}
                for record in registry.records:
                    entry = datasets.setdefault(record.dataset_id, {"dataset_id": record.dataset_id,
                        "dataset_sha256": record.dataset_sha256, "roles": {}})
                    entry["roles"][record.role] = entry["roles"].get(record.role, 0) + 1
                registries.append({"registry_id": registry.registry_id,
                                   "registry_sha256": registry.registry_sha256,
                                   "datasets": list(datasets.values())})
            except (OSError, TypeError, ValueError, KeyError):
                continue
    return {"definitions": definitions, "role_registries": registries}


def _registered_dataset_hashes(root: Path) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    visited = 0
    for base in (root / "data" / "processed", root / "artifacts"):
        if not _inside(root, base) or not base.is_dir():
            continue
        for path in base.rglob("manifests/*.yaml"):
            if visited >= 2000:
                return identities
            visited += 1
            if not _inside(root, path):
                continue
            try:
                manifest = load_manifest(path)
                metadata_path = manifest.metadata.get("canonical_path")
                canonical = root / metadata_path if isinstance(metadata_path, str) else path.parent.parent / "canonical" / f"{manifest.stage}.jsonl"
                if not _inside(root, canonical) or not canonical.is_file():
                    continue
                actual = sha256_file(canonical)
                expected = manifest.hashes.get("canonical_records_sha256")
                if actual == expected:
                    identities.add((manifest.dataset_id, actual))
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                continue
    return identities


def discover_sft_generation_setup(workspace: str | Path) -> dict[str, Any]:
    """Return safe identities and explicit gaps; never reads or returns credential values."""
    root = Path(workspace).resolve(strict=True)
    datasets = _sft_datasets(root)
    profiles = _profiles(root)
    configs = _validated_local_configs(root)
    reports = _scenario_reports(root)
    targeting = _targeting_evidence(root)
    registered_hashes = _registered_dataset_hashes(root)
    ready = []
    configured_profiles = [
        profile for profile in profiles
        if not profile["is_example"]
        and profile["adapter"] != "fake"
        and profile["credential_env_configured"]
    ]
    ordinary_global_reasons: list[str] = []
    if not configs["templates"]:
        ordinary_global_reasons.append("缺少有效的项目提示词模板")
    if not configs["protocols"]:
        ordinary_global_reasons.append("缺少有效的候选输出协议")
    if not configured_profiles:
        ordinary_global_reasons.append("没有凭据已配置的真实 Provider profile")
    for dataset in datasets:
        ordinary_reasons = list(dataset["missing_reasons"])
        ordinary_reasons.extend(ordinary_global_reasons)
        reasons = list(ordinary_reasons)
        role_registry = next((registry for registry in targeting["role_registries"]
                              if any(item["dataset_id"] == dataset["dataset_id"]
                                     and item["dataset_sha256"] == dataset["content_sha256"]
                                     and item["roles"].get("train_seed", 0) > 0
                                     for item in registry["datasets"])), None)
        role_data = {item["dataset_id"]: item for item in (role_registry or {}).get("datasets", [])}
        diagnostic = next((item for item in role_data.values() if item["roles"].get("diagnostic_dev", 0)), None)
        holdout = next((item for item in role_data.values() if item["roles"].get("final_holdout", 0)), None)
        diagnostic = diagnostic if diagnostic and (diagnostic["dataset_id"], diagnostic["dataset_sha256"]) in registered_hashes else None
        holdout = holdout if holdout and (holdout["dataset_id"], holdout["dataset_sha256"]) in registered_hashes else None
        if role_registry is None or diagnostic is None or holdout is None:
            reasons.append("缺少固定的角色登记：训练 seed、独立 diagnostic_dev 与 final_holdout 必须分离")
        definition = next((item for item in targeting["definitions"] if diagnostic
                           and item["dataset_id"] == diagnostic["dataset_id"]
                           and item["dataset_sha256"] == diagnostic["dataset_sha256"]), None)
        if definition is None:
            reasons.append("缺少与 diagnostic_dev 数据 ID/摘要匹配的场景定义")
        report = next((item for item in reports if diagnostic
                       and item.get("dataset_id") == diagnostic["dataset_id"]
                       and item.get("dataset_sha256") == diagnostic["dataset_sha256"]
                       and definition and any(scenario["scenario_id"] in definition["scenario_ids"]
                                              for scenario in item["scenarios"])), None)
        if report is None:
            reasons.append("缺少与 diagnostic_dev、场景定义及题目级预测匹配的诊断报告")
        ready.append({"dataset_id": dataset["dataset_id"],
                      # Keep the legacy top-level eligibility targeted-only for current UI clients.
                      "eligible": not reasons, "missing_reasons": reasons,
                      "ordinary_sft": {"eligible": not ordinary_reasons,
                                        "missing_reasons": ordinary_reasons,
                                        "sample_count": dataset["sample_count"]},
                      "targeted": {"eligible": not reasons, "missing_reasons": reasons},
                      "matching_report_id": report.get("report_id") if report else None,
                      "scenario_ids": definition["scenario_ids"] if definition else [],
                      "role_registry_id": role_registry["registry_id"] if role_registry else None,
                      "diagnostic_dataset_id": diagnostic["dataset_id"] if diagnostic else None,
                      "holdout_dataset_id": holdout["dataset_id"] if holdout else None})
    return {
        "read_only": True,
        "provider_calls": 0,
        "datasets": [{key: value for key, value in dataset.items() if not key.startswith("_")}
                     for dataset in datasets],
        "profiles": profiles,
        "configs": configs,
        "scenario_reports": reports,
        "scenario_definitions": targeting["definitions"],
        "role_registries": targeting["role_registries"],
        "readiness": ready,
        "summary": {"dataset_count": len(datasets), "ready_count": sum(item["eligible"] for item in ready),
                    "ordinary_ready_count": sum(item["ordinary_sft"]["eligible"] for item in ready),
                    "targeted_ready_count": sum(item["targeted"]["eligible"] for item in ready),
                    "scenario_report_count": len(reports)},
    }


def prepare_ordinary_sft_generation(
    workspace: str | Path,
    *,
    dataset_id: str,
    target_count: int,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Prepare an immutable ordinary SFT bundle using discovered IDs only."""
    root = Path(workspace).resolve(strict=True)
    dataset = next((item for item in _sft_datasets(root) if item["dataset_id"] == dataset_id), None)
    if dataset is None or not dataset["eligible"]:
        raise ValueError("所选 SFT dataset 不存在或 canonical 摘要校验失败")

    profiles_root = root / "configs" / "synthetic"
    valid_profiles: list[tuple[ProviderProfile, Path]] = []
    for path in profiles_root.rglob("provider*.yaml") if _inside(root, profiles_root) and profiles_root.is_dir() else ():
        if not _inside(root, path):
            continue
        try:
            profile = ProviderProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            continue
        if profile.adapter == "fake" or "example" in path.name:
            continue
        if _credential_available(root, profile.api_key_env):
            valid_profiles.append((profile, path))
    if profile_id:
        selected = next((pair for pair in valid_profiles if pair[0].profile_id == profile_id), None)
        if selected is None:
            raise ValueError("所选 Provider profile 不存在或本地凭据未配置")
    elif len(valid_profiles) == 1:
        selected = valid_profiles[0]
    elif not valid_profiles:
        raise ValueError("没有凭据已配置的真实 Provider profile")
    else:
        raise ValueError("发现多个可用 Provider profile；请明确选择 profile_id")
    profile, profile_path = selected

    examples = root / "configs" / "synthetic" / "examples"
    template_path = examples / "prompt.sft-math.yaml"
    protocol_path = examples / "output.sft-candidate.yaml"
    if not all(_inside(root, path) and path.is_file() for path in (template_path, protocol_path)):
        raise ValueError("项目默认 SFT 模板或输出协议缺失")
    policy_artifact = next((item for item in policy_templates()
                            if item["config"]["stage"] == "sft"
                            and item["config"]["rules"]["correctness"] == "filter"
                            and item["config"]["rules"]["completeness"] == "filter"), None)
    if policy_artifact is None:
        raise ValueError("项目默认 SFT 数据策略缺失")
    policy = DataPolicy.model_validate(policy_artifact["config"])
    bundle = register_ordinary_sft_bundle(
        root,
        source_dataset_id=dataset_id,
        source_manifest_path=dataset["_manifest_path"],
        source_data_path=dataset["_canonical_path"],
        profile_path=profile_path,
        template_path=template_path,
        protocol_path=protocol_path,
        policy_payload=policy.model_dump(mode="json"),
        target_count=target_count,
    )
    return {
        "mode": "ordinary_sft",
        "bundle_id": bundle.bundle_id,
        "source_dataset_id": dataset_id,
        "source_content_sha256": dataset["content_sha256"],
        "provider_profile_id": profile.profile_id,
        "model": profile.model,
        "target_count": target_count,
        "provider_calls": 0,
        "network_used": False,
        "status": "prepared_not_executed",
    }


def prepare_prompt_only_sft_generation(
    workspace: str | Path,
    *,
    topic: str,
    skill: str,
    difficulty: str,
    constraints: str = "",
    target_count: int,
    profile_id: str | None = None,
) -> dict[str, Any]:
    """Prepare a bounded source-free plan; this operation never calls the provider."""
    return prepare_prompt_only_bundle(workspace, topic=topic, skill=skill, difficulty=difficulty,
                                      constraints=constraints, target_count=target_count,
                                      profile_id=profile_id)
