"""Versioned executable reward contract and safe targeted GRPO prompt records."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..lineage import sha256_json
from ..reporting.data_report import length_summary
from ..rewards.math import accuracy_reward, boxed_format_reward
from ..schemas import GRPORecord, Message
from .dataset_import import register_jsonl
from .grpo_prompt_core import build_grpo_prompt_record
from .manifest import load_manifest
from .registry import DataSource
from .targeted_stage_plan import GRPOTarget, TargetedStagePlan

RewardName = Literal["accuracy", "boxed_format"]


def _targeted_prompt_quality(rows: list[dict[str, Any]], *, candidate_count: int) -> dict[str, Any]:
    prompt_lengths = [len(row["raw_problem"]) for row in rows]
    token_lengths = [row["metadata"]["prompt_tokens"] for row in rows
                     if isinstance(row.get("metadata", {}).get("prompt_tokens"), int)
                     and not isinstance(row["metadata"].get("prompt_tokens"), bool)
                     and row["metadata"]["prompt_tokens"] >= 0]
    truncation_values = [row["metadata"]["truncated"] for row in rows
                         if isinstance(row.get("metadata", {}).get("truncated"), bool)]
    return {
        "status": "report_only",
        "candidate_count": candidate_count,
        "usable_prompt_count": len(rows),
        "trusted_reference_coverage": {
            "status": "measured" if candidate_count else "not_measured",
            "numerator": len(rows), "denominator": candidate_count,
            "value": len(rows) / candidate_count if candidate_count else None,
            "unit": "ratio",
        },
        "prompt_length_chars": {"status": "measured" if prompt_lengths else "not_measured",
                                "denominator": len(prompt_lengths),
                                **length_summary(prompt_lengths)},
        "prompt_length_tokens": {"status": "measured" if token_lengths else "not_measured",
                                  "denominator": len(token_lengths),
                                  **length_summary(token_lengths),
                                  "reason": None if token_lengths else "No pinned tokenizer/template rendering was run during targeted prompt construction."},
        "truncation": {
            "status": "measured" if truncation_values else "not_measured",
            "truncated_count": sum(truncation_values) if truncation_values else None,
            "known_count": len(truncation_values), "denominator": len(rows),
            "value": sum(truncation_values) / len(truncation_values) if truncation_values else None,
            "unit": "ratio",
            "reason": None if truncation_values else "No tokenizer or training-time truncation check was run.",
        },
        "reward_variance": {"status": "not_measured", "denominator": 0,
                            "reason": "No model rollouts exist before GRPO training."},
    }


def prepare_imported_targeted_grpo_prompts(
    candidates: list[dict[str, Any]], *, plan: TargetedStagePlan,
    contract: ExecutableRewardContract,
) -> dict[str, Any]:
    """Validate already canonical GRPO prompts without inferring their ancestry."""
    if not isinstance(plan.target, GRPOTarget):
        raise TypeError("GRPO import requires a GRPO targeted plan")
    validate_reward_contract(contract)
    evidence = plan.target.reward_contract
    if not evidence or evidence.evidence_id != contract.contract_id or evidence.content_sha256 != contract.protocol_sha256:
        raise ValueError("GRPO plan must reference this exact executable reward contract ID and protocol hash")
    if len(candidates) > plan.budget.max_candidates:
        raise ValueError("imported rows exceed the immutable plan max_candidates")
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    unselected: Counter[str] = Counter()
    seen: set[str] = set()
    expected = {
        "targeted_plan_id": plan.plan_id,
        "targeted_plan_version": plan.version,
        "data_role": "train_seed",
        "reward_contract_id": contract.contract_id,
        "reward_contract_version": contract.version,
        "reward_contract_protocol_sha256": contract.protocol_sha256,
        "reward_contract_implementation_sha256": contract.implementation_sha256,
        "reference_source": "registered_train_seed",
    }
    for candidate in candidates:
        try:
            record = GRPORecord.model_validate(candidate)
        except (TypeError, ValueError):
            rejected["invalid_canonical_record"] += 1
            continue
        if len(rows) >= plan.budget.expected_usable_count:
            unselected["expected_usable_count_reached"] += 1
            continue
        metadata = record.metadata
        if any(metadata.get(key) != value for key, value in expected.items()):
            rejected["mismatched_plan_reward_or_training_role"] += 1
            continue
        parents = metadata.get("parent_record_ids")
        if not isinstance(parents, list) or not parents or any(not isinstance(item, str) or not item for item in parents):
            rejected["missing_seed_ancestry"] += 1
            continue
        if record.id in seen:
            rejected["duplicate_record_id"] += 1
            continue
        seen.add(record.id)
        rows.append(record.model_dump(mode="json"))
    return {
        "schema_version": "1.0",
        "status": "completed" if len(rows) >= plan.budget.expected_usable_count else "partial",
        "source_kind": "canonical_grpo_import",
        "plan_id": plan.plan_id, "plan_version": plan.version,
        "reward_contract_id": contract.contract_id,
        "reward_contract_protocol_sha256": contract.protocol_sha256,
        "requested_prompts": plan.budget.expected_usable_count,
        "candidate_count": len(candidates), "usable_prompts": len(rows),
        "shortfall": max(0, plan.budget.expected_usable_count - len(rows)),
        "rejections": dict(rejected), "not_selected": dict(unselected),
        "quality": _targeted_prompt_quality(rows, candidate_count=len(candidates)),
        "records": rows,
    }


class ExecutableRewardContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    contract_id: str = Field(pattern=r"^reward-contract-[0-9a-f]{20}$")
    version: str = Field(pattern=r"^[0-9]+(?:\.[0-9]+){0,2}$")
    rewards: tuple[RewardName, ...] = Field(min_length=1)
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, rewards: tuple[RewardName, ...] = ("accuracy", "boxed_format"), *, version: str = "1.0") -> ExecutableRewardContract:
        if len(rewards) != len(set(rewards)):
            raise ValueError("reward functions must be unique")
        code_path = Path(__file__).resolve().parents[1] / "rewards" / "math.py"
        implementation_sha = _sha256_file(code_path)
        protocol_sha = sha256_json({"version": version, "rewards": rewards,
                                    "accuracy": "extract_answer+answers_equal:v1",
                                    "boxed_format": "last_boxed_answer:v1"})
        identity = sha256_json({"implementation_sha256": implementation_sha,
                                "protocol_sha256": protocol_sha})
        return cls(contract_id=f"reward-contract-{identity[:20]}", version=version,
                   rewards=rewards, implementation_sha256=implementation_sha,
                   protocol_sha256=protocol_sha)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reward_contract(contract: ExecutableRewardContract) -> None:
    current = ExecutableRewardContract.create(contract.rewards, version=contract.version)
    if current != contract:
        raise ValueError("reward contract implementation/protocol hash is stale or mismatched")


def evaluate_reward_contract(
    contract: ExecutableRewardContract,
    completion: str,
    reference_answer: str,
) -> dict[str, float]:
    """Execute allowlisted deterministic rewards; do not trust provider labels."""
    validate_reward_contract(contract)
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("completion must be non-empty text")
    if not isinstance(reference_answer, str) or not reference_answer.strip():
        raise ValueError("a trusted reference answer is required")
    values = {}
    if "accuracy" in contract.rewards:
        values["accuracy"] = accuracy_reward(completion, reference_answer)
    if "boxed_format" in contract.rewards:
        values["boxed_format"] = boxed_format_reward(completion)
    return values


def build_targeted_grpo_prompts(
    seeds: list[dict[str, Any]],
    *,
    plan: TargetedStagePlan,
    contract: ExecutableRewardContract,
) -> dict[str, Any]:
    """Prepare target-count prompts only from train seeds with trusted answers."""
    if not isinstance(plan.target, GRPOTarget):
        raise TypeError("GRPO prompt creation requires a GRPO targeted plan")
    validate_reward_contract(contract)
    evidence = plan.target.reward_contract
    if not evidence or evidence.evidence_id != contract.contract_id or evidence.content_sha256 != contract.protocol_sha256:
        raise ValueError("GRPO plan must reference this exact executable reward contract ID and protocol hash")
    if len(seeds) > plan.budget.max_candidates:
        raise ValueError("seed rows exceed the immutable plan max_candidates")
    rows, rejected, unselected = [], Counter(), Counter()
    seen = set()
    for seed in seeds:
        if not isinstance(seed, dict):
            rejected["invalid_seed"] += 1
            continue
        if len(rows) >= plan.budget.expected_usable_count:
            unselected["expected_usable_count_reached"] += 1
            continue
        if seed.get("role", seed.get("data_role", "train_seed")) != "train_seed":
            rejected["non_training_ancestry"] += 1
            continue
        record_id = seed.get("record_id", seed.get("id"))
        problem_id = seed.get("problem_id", record_id)
        prompt = seed.get("problem", seed.get("raw_problem", seed.get("question")))
        reference = seed.get("reference_answer", seed.get("answer"))
        if not all(isinstance(value, str) and value.strip() for value in (record_id, problem_id, prompt, reference)):
            rejected["missing_prompt_or_trusted_reference"] += 1
            continue
        if record_id in seen:
            rejected["duplicate_seed_id"] += 1
            continue
        seen.add(record_id)
        identity = sha256_json({"plan_id": plan.plan_id, "source_record_id": record_id,
                                "problem_id": problem_id, "prompt": prompt})[:24]
        rows.append(build_grpo_prompt_record(
            record_id=f"grpo-targeted-{identity}", prompt=prompt, answer=reference,
            problem_id=problem_id, parent_record_ids=[record_id], contract=contract,
            reference_source="registered_train_seed",
            context={"targeted_plan_id": plan.plan_id, "targeted_plan_version": plan.version},
            prompt_template_id="targeted-grpo-prompt-v1",
        ))
    quality = _targeted_prompt_quality(rows, candidate_count=len(seeds))
    return {
        "schema_version": "1.0", "status": "completed" if len(rows) >= plan.budget.expected_usable_count else "partial",
        "plan_id": plan.plan_id, "plan_version": plan.version,
        "reward_contract_id": contract.contract_id, "reward_contract_protocol_sha256": contract.protocol_sha256,
        "requested_prompts": plan.budget.expected_usable_count,
        "candidate_count": len(seeds), "usable_prompts": len(rows),
        "shortfall": max(0, plan.budget.expected_usable_count - len(rows)),
        "rejections": dict(rejected), "not_selected": dict(unselected),
        "quality": quality, "records": rows,
    }


def summarize_rollout_rewards(
    contract: ExecutableRewardContract,
    rollouts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Report observed reward coverage/variance using true rollout denominators."""
    validate_reward_contract(contract)
    values: dict[str, list[float]] = defaultdict(list)
    by_prompt: dict[str, list[float]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    completion_lengths: list[int] = []
    completion_token_lengths: list[int] = []
    truncation_known = 0
    truncation_count = 0
    identified_count = 0
    for item in rollouts:
        if not isinstance(item, dict) or not isinstance(item.get("prompt_id"), str):
            rejected["invalid_rollout_identity"] += 1
            continue
        identified_count += 1
        completion = item.get("completion")
        if isinstance(completion, str) and completion.strip():
            completion_lengths.append(len(completion))
        token_length = item.get("completion_tokens")
        if isinstance(token_length, int) and not isinstance(token_length, bool) and token_length >= 0:
            completion_token_lengths.append(token_length)
        if isinstance(item.get("truncated"), bool):
            truncation_known += 1
            truncation_count += int(item["truncated"])
        try:
            rewards = evaluate_reward_contract(contract, completion, item.get("reference_answer"))
        except (TypeError, ValueError):
            rejected["unverifiable_rollout"] += 1
            continue
        aggregate = sum(rewards.values()) / len(rewards)
        by_prompt[item["prompt_id"]].append(aggregate)
        for name, value in rewards.items():
            values[name].append(value)
    prompt_stds = [statistics.pstdev(scores) for scores in by_prompt.values() if len(scores) >= 2]
    zero_std = sum(value == 0 for value in prompt_stds)
    verified_count = len(next(iter(values.values()), [])) if values else 0
    return {
        "contract_id": contract.contract_id,
        "rollout_count": len(rollouts),
        "identified_rollout_count": identified_count,
        "verified_rollout_count": verified_count,
        "verifier_coverage": {
            "status": "measured" if rollouts else "not_measured",
            "numerator": verified_count, "denominator": len(rollouts),
            "value": verified_count / len(rollouts) if rollouts else None,
            "unit": "ratio",
        },
        "reward_means": {key: (sum(items) / len(items) if items else None) for key, items in values.items()},
        "reward_variances": {key: (statistics.pvariance(items) if items else None) for key, items in values.items()},
        "prompt_count": len(by_prompt), "prompt_count_with_repeated_rollouts": len(prompt_stds),
        "zero_std_prompt_count": zero_std,
        "zero_std_ratio": zero_std / len(prompt_stds) if prompt_stds else None,
        "completion_length_chars": {"status": "measured" if completion_lengths else "not_measured",
                                    "denominator": len(completion_lengths), **length_summary(completion_lengths)},
        "completion_length_tokens": {"status": "measured" if completion_token_lengths else "not_measured",
                                     "denominator": len(completion_token_lengths),
                                     **length_summary(completion_token_lengths)},
        "truncation": {
            "status": "measured" if truncation_known else "not_measured",
            "truncated_count": truncation_count if truncation_known else None,
            "known_count": truncation_known, "denominator": len(rollouts),
            "value": truncation_count / truncation_known if truncation_known else None,
            "unit": "ratio",
        },
        "rejections": dict(rejected), "status": "measured" if values else "not_measured",
    }


def register_targeted_grpo_prompts(
    report: dict[str, Any], *, plan: TargetedStagePlan, contract: ExecutableRewardContract,
    workspace: str | Path, output_root: str | Path, provider_profile_id: str,
) -> dict[str, Any]:
    if not isinstance(plan.target, GRPOTarget):
        raise TypeError("GRPO registration requires a GRPO targeted plan")
    evidence = plan.target.reward_contract
    if not evidence or evidence.evidence_id != contract.contract_id or evidence.content_sha256 != contract.protocol_sha256:
        raise ValueError("GRPO plan does not reference the exact executable reward contract")
    if report.get("plan_id") != plan.plan_id or report.get("reward_contract_id") != contract.contract_id:
        raise ValueError("GRPO report does not match plan/reward contract")
    rows = report.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("zero usable GRPO prompts are not registered")
    if report.get("plan_version") != plan.version:
        raise ValueError("GRPO report does not match the targeted plan version")
    for index, row in enumerate(rows, start=1):
        record = GRPORecord.model_validate(row)
        metadata = record.metadata
        expected = {
            "targeted_plan_id": plan.plan_id,
            "targeted_plan_version": plan.version,
            "data_role": "train_seed",
            "reward_contract_id": contract.contract_id,
            "reward_contract_version": contract.version,
            "reward_contract_protocol_sha256": contract.protocol_sha256,
            "reward_contract_implementation_sha256": contract.implementation_sha256,
            "reference_source": "registered_train_seed",
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"GRPO candidate {index} has mismatched plan or reward evidence")
        if not metadata.get("parent_record_ids"):
            raise ValueError(f"GRPO candidate {index} is missing train-seed ancestry")
    candidate_count = report.get("candidate_count")
    if (not isinstance(candidate_count, int) or isinstance(candidate_count, bool)
            or candidate_count < len(rows) or candidate_count > plan.budget.max_candidates):
        raise ValueError("GRPO report has an invalid candidate denominator")
    quality = report.get("quality")
    expected_quality = _targeted_prompt_quality(rows, candidate_count=candidate_count)
    if quality != expected_quality:
        raise ValueError("GRPO report quality evidence does not match its actual candidates")
    validate_reward_contract(contract)
    source = DataSource(source_id=f"source-targeted-{plan.plan_id}", kind="local_upload",
                        name="Targeted GRPO prompt view derived from registered train seeds",
                        uri=f"dataset://{plan.training_seed.dataset_id}", revision=None,
                        revision_status="unknown", license_status="not_provided", declared_by="adapter")
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
                       for row in rows)
    registered = register_jsonl(BytesIO(payload), workspace=workspace, output_root=output_root,
                                stage="grpo", origin=source,
                                registration_metadata={
                                    "quality_status": "report_only", "usage": "exploratory",
                                    "targeted_plan_id": plan.plan_id, "targeted_plan_version": plan.version,
                                    "parent_dataset_ids": [plan.training_seed.dataset_id],
                                    "provider_profile_id": provider_profile_id,
                                    "reward_contract_verified": True,
                                    "reward_contract_id": contract.contract_id,
                                    "reward_contract_protocol_sha256": contract.protocol_sha256,
                                    "reward_contract_implementation_sha256": contract.implementation_sha256,
                                    "targeted_grpo_quality": quality,
                                })
    manifest = load_manifest(registered.path / "manifests" / "grpo-user-upload-v1.yaml")
    return {"plan_id": plan.plan_id, "plan_version": plan.version,
            "dataset_id": registered.dataset_id, "record_count": registered.record_count,
            "canonical_sha256": registered.canonical_sha256,
            "quality_report_id": manifest.quality_report.report_id if manifest.quality_report else None,
            "registration_path": registered.path.relative_to(Path(workspace).resolve()).as_posix(),
            "generation_status": report["status"], "shortfall": report["shortfall"]}
