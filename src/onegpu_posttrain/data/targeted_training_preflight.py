"""Strict preflight for training runs that consume targeted datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..lineage import sha256_directory, sha256_file
from .manifest import load_manifest, validate_manifest_evidence
from .targeted_stage_plan import load_registered_targeted_stage_plan


def _dataset_source_and_digest(dataset: str | Path) -> tuple[Path, str | None]:
    path = Path(dataset).resolve()
    sidecar = path.with_suffix(path.suffix + ".adapter.json")
    if sidecar.is_file() and not sidecar.is_symlink():
        try:
            report = json.loads(sidecar.read_text(encoding="utf-8"))
            source = Path(str(report.get("input", "")))
            if not source.is_absolute():
                source = (Path.cwd() / source).resolve()
            if source.is_file() and report.get("input_sha256") == sha256_file(source):
                if report.get("output_sha256") != sha256_file(path):
                    raise ValueError("registered dataset adapter output digest does not match the consumed file")
                return source, sha256_file(source)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid registered dataset adapter evidence: {exc}") from exc
    return path, sha256_file(path) if path.is_file() else None


def _matching_manifest(dataset: str | Path) -> tuple[Path, Path] | None:
    canonical, content_sha = _dataset_source_and_digest(dataset)
    if not content_sha:
        return None
    for parent in [canonical.parent, *list(canonical.parents)[:6]]:
        manifest_dir = parent / "manifests"
        if not manifest_dir.is_dir() or manifest_dir.is_symlink():
            continue
        for manifest_path in manifest_dir.glob("*.yaml"):
            if manifest_path.is_symlink() or not manifest_path.is_file():
                continue
            try:
                manifest = load_manifest(manifest_path)
            except (OSError, ValueError, TypeError):
                continue
            if manifest.hashes.get("canonical_records_sha256") == content_sha:
                return manifest_path, canonical
    return None


def _validate_records(canonical: Path, stage: str, plan_id: str, plan_version: int,
                      reward_contract: Any | None = None) -> int:
    count = 0
    try:
        with canonical.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"canonical dataset JSONL line {line_number} is invalid") from exc
                if not isinstance(row, dict):
                    raise TypeError(f"canonical dataset line {line_number} is not an object")
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    raise TypeError(f"targeted dataset line {line_number} has no metadata evidence")
                expected_role = "train_pairs" if stage == "preference" else "train_seed"
                ancestry_role_ok = (metadata.get("source_data_role") == "train_seed"
                                    if stage == "preference" else True)
                if (metadata.get("targeted_plan_id") != plan_id
                        or metadata.get("targeted_plan_version") != plan_version
                        or metadata.get("data_role") != expected_role or not ancestry_role_ok):
                    raise ValueError(f"targeted dataset line {line_number} has mismatched plan or non-train ancestry")
                if stage == "preference":
                    if (metadata.get("preference_pairing_protocol") != "independent-correctness-v1"
                            or metadata.get("chosen_correct") is not True
                            or metadata.get("rejected_correct") is not False
                            or not metadata.get("parent_record_ids")):
                        raise ValueError(f"preference pair on line {line_number} lacks independent ranking evidence")
                    if not isinstance(row.get("chosen"), str) or not row["chosen"].strip():
                        raise ValueError(f"preference pair on line {line_number} has an empty chosen answer")
                    if not isinstance(row.get("rejected"), str) or not row["rejected"].strip():
                        raise ValueError(f"preference pair on line {line_number} has an empty rejected answer")
                elif stage == "grpo":
                    if reward_contract is None:
                        raise ValueError("GRPO target is missing an executable reward contract")
                    expected = {
                        "reward_contract_id": reward_contract.contract_id,
                        "reward_contract_version": reward_contract.version,
                        "reward_contract_protocol_sha256": reward_contract.protocol_sha256,
                        "reward_contract_implementation_sha256": reward_contract.implementation_sha256,
                        "reference_source": "registered_train_seed",
                    }
                    if any(metadata.get(key) != value for key, value in expected.items()):
                        raise ValueError(f"GRPO prompt on line {line_number} has mismatched reward/reference evidence")
                    if not isinstance(row.get("answer"), str) or not row["answer"].strip():
                        raise ValueError(f"GRPO prompt on line {line_number} has no trusted reference answer")
                    if not metadata.get("parent_record_ids"):
                        raise ValueError(f"GRPO prompt on line {line_number} has no seed ancestry")
                count += 1
    except OSError as exc:
        raise ValueError(f"cannot read canonical targeted dataset: {exc}") from exc
    if count == 0:
        raise ValueError("targeted training dataset has no usable records")
    return count


def preflight_targeted_training(
    *,
    dataset: str | Path,
    model: str | Path,
    stage: str,
    workspace: str | Path,
    training: dict[str, Any] | None = None,
    external_plugin: str | Path | None = None,
) -> dict[str, Any] | None:
    """Validate targeted plan, registered dataset, parent model and GRPO reward.

    Non-targeted/legacy datasets return ``None`` and retain their existing CLI
    behavior. A dataset that carries a plan identity is always validated strictly.
    """
    found = _matching_manifest(dataset)
    if found is None:
        return None
    manifest_path, canonical = found
    manifest = load_manifest(manifest_path)
    plan_id = manifest.metadata.get("targeted_plan_id")
    plan_version = manifest.metadata.get("targeted_plan_version")
    if plan_id is None and plan_version is None:
        return None
    if (not isinstance(plan_id, str) or not re.fullmatch(r"plan-targeted-[a-f0-9]{20}", plan_id)
            or not isinstance(plan_version, int) or plan_version < 1):
        raise ValueError("registered targeted dataset has invalid plan identity")
    expected_data_stage = "preference" if stage in {"dpo", "simpo"} else stage
    if manifest.stage != expected_data_stage:
        raise ValueError(f"targeted dataset stage {manifest.stage!r} cannot be used for {stage} training")
    errors = validate_manifest_evidence(manifest_path)
    if errors:
        raise ValueError("targeted dataset evidence validation failed: " + "; ".join(errors))
    if manifest.lineage is None:
        raise ValueError("targeted dataset is missing dataset lineage")
    plan = load_registered_targeted_stage_plan(workspace, plan_id, plan_version)
    if plan.target.stage != expected_data_stage:
        raise ValueError(f"targeted plan stage {plan.target.stage!r} does not match {stage} training")
    if plan.training_seed.dataset_id not in manifest.lineage.parents:
        raise ValueError("targeted dataset lineage does not include the plan's registered train-seed dataset")
    model_path = Path(model)
    if model_path.is_dir():
        model_sha = sha256_directory(model_path)
    elif model_path.is_file():
        model_sha = sha256_file(model_path)
    else:
        raise ValueError("targeted training parent model must exist so its content identity can be verified")
    if model_sha != plan.starting_model.content_sha256:
        raise ValueError("training parent model content does not match the immutable targeted plan")
    contract = None
    if stage == "grpo":
        from .targeted_grpo import ExecutableRewardContract

        contract = ExecutableRewardContract.create()
    count = _validate_records(canonical, expected_data_stage, plan_id, plan_version, contract)
    if manifest.sample_count != count:
        raise ValueError("targeted dataset sample_count does not match its canonical records")
    reward_context = None
    if stage == "grpo":
        from .targeted_grpo import ExecutableRewardContract, validate_reward_contract

        assert isinstance(contract, ExecutableRewardContract)
        validate_reward_contract(contract)
        evidence = getattr(plan.target, "reward_contract", None)
        if not evidence or evidence.evidence_id != contract.contract_id or evidence.content_sha256 != contract.protocol_sha256:
            raise ValueError("GRPO targeted plan does not bind the current executable reward contract")
        metadata = manifest.metadata
        expected_reward_metadata = {
            "reward_contract_verified": True,
            "reward_contract_id": contract.contract_id,
            "reward_contract_protocol_sha256": contract.protocol_sha256,
            "reward_contract_implementation_sha256": contract.implementation_sha256,
        }
        if any(metadata.get(key) != value for key, value in expected_reward_metadata.items()):
            raise ValueError("registered GRPO dataset is missing matching executable reward-contract evidence")
        settings = training or {}
        reward_funcs = list(settings.get("reward_funcs", ["onegpu_accuracy", "onegpu_boxed_format"]))
        if reward_funcs != ["onegpu_accuracy", "onegpu_boxed_format"]:
            raise ValueError("GRPO training reward functions differ from the verified targeted reward contract")
        plugin_path = Path(external_plugin) if external_plugin is not None else (
            Path(workspace) / "plugins/grpo_rewards.py"
        )
        shipped_plugin = Path(__file__).resolve().parents[3] / "plugins/grpo_rewards.py"
        if (plugin_path.is_symlink() or not plugin_path.is_file() or not shipped_plugin.is_file()
                or sha256_file(plugin_path) != sha256_file(shipped_plugin)):
            raise ValueError("GRPO executable reward plugin is missing or differs from the verified project adapter")
        reward_context = {
            "contract_id": contract.contract_id,
            "contract_version": contract.version,
            "protocol_sha256": contract.protocol_sha256,
            "implementation_sha256": contract.implementation_sha256,
        }
    return {
        "plan_id": plan.plan_id,
        "plan_version": plan.version,
        "dataset_id": manifest.dataset_id,
        "dataset_stage": manifest.stage,
        "record_count": count,
        "starting_model_id": plan.starting_model.model_id,
        "starting_model_sha256": model_sha,
        "training_seed_dataset_id": plan.training_seed.dataset_id,
        "reward_contract": reward_context,
        "preflight": "passed",
    }
