"""Offline-testable orchestration for the registered SFT generation path."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from ..data.benchmarks import benchmark_index_from_jsonl
from ..data.io import read_jsonl
from ..data.policy import load_policy
from ..data.roles import load_role_registry
from ..data.tokenization import HuggingFaceTokenizer, WhitespaceTokenizer
from ..lineage import sha256_file
from .artifacts import write_targeted_artifacts
from .candidates import validate_candidate_response
from .generation import execute_targeted_generation, prepare_generation
from .policy_export import CandidatePolicyInput, export_candidates_with_policy
from .providers import FakeProvider, OpenAICompatibleProvider
from .secrets import load_local_provider_credential
from .workflow_bundle import load_sft_workflow_bundle


def _contained_path(workspace: Path, relative: str, *, must_exist: bool = False) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("workflow path must be workspace-relative")
    candidate = workspace / rel
    current = candidate
    while current != workspace:
        if current.is_symlink():
            raise ValueError("workflow paths cannot contain symbolic links")
        current = current.parent
    resolved = candidate.resolve(strict=must_exist)
    if not resolved.is_relative_to(workspace):
        raise ValueError("workflow path escapes workspace")
    return resolved


def _tokenizer(workspace: Path, policy):
    config = policy.tokenizer
    if config.test_only:
        return WhitespaceTokenizer()
    if config.local_path:
        local = _contained_path(workspace, config.local_path, must_exist=True)
        if not local.is_dir():
            raise ValueError("configured local tokenizer path is not a directory")
        return HuggingFaceTokenizer(str(local), revision=None, local_files_only=True)
    return HuggingFaceTokenizer(config.repo, revision=config.revision, local_files_only=True)


def preflight_sft_generation(
    workspace: str | Path,
    bundle_id: str,
    *,
    usable_target: int,
    max_requests: int,
    target_candidates: int,
) -> dict[str, Any]:
    """Validate a request without credential lookup, provider calls, or tokenizer download."""
    root = Path(workspace).resolve(strict=True)
    bundle, files = load_sft_workflow_bundle(root, bundle_id)
    from ..data.targeted_plan import load_targeted_plan
    plan = load_targeted_plan(files["plan"])
    if min(usable_target, max_requests, target_candidates) < 1:
        raise ValueError("generation targets and budgets must be positive")
    if max_requests > plan.budget.max_requests or target_candidates > plan.budget.target_candidates:
        raise ValueError("per-run request/candidate budgets cannot exceed registered plan limits")
    profile = yaml.safe_load(files["profile"].read_text(encoding="utf-8"))
    policy = load_policy(files["policy"])
    if policy.tokenizer.test_only and profile.get("adapter") != "fake":
        raise ValueError("test-only tokenizer is permitted only with the local fake provider")
    return {
        "status": "preflight_ready",
        "bundle_id": bundle.bundle_id,
        "plan_config_sha256": bundle.plan_config_sha256,
        "source_dataset_id": plan.training_seed.dataset_id,
        "stage": "sft",
        "provider_profile_id": profile.get("profile_id"),
        "provider_adapter": profile.get("adapter"),
        "model": profile.get("model"),
        "prompt_template_sha256": plan.prompt_template_sha256,
        "policy_id": policy.policy_id,
        "desired_usable_records": usable_target,
        "candidate_budget": target_candidates,
        "max_requests": max_requests,
        "max_output_tokens_per_request": plan.budget.max_output_tokens,
        "input_cost_per_million_tokens": profile.get("input_cost_per_million_tokens"),
        "output_cost_per_million_tokens": profile.get("output_cost_per_million_tokens"),
        "estimated_output_cost_upper_bound_usd": (
            max_requests * plan.budget.max_output_tokens * profile["output_cost_per_million_tokens"] / 1_000_000
            if isinstance(profile.get("output_cost_per_million_tokens"), (int, float)) else None
        ),
        "cost_status": "output_only_upper_bound" if isinstance(profile.get("output_cost_per_million_tokens"), (int, float)) else "unknown",
        "provider_calls": 0,
        "network_used": False,
        "plan_status": "registered_not_executed",
    }


def execute_sft_generation(
    workspace: str | Path,
    bundle_id: str,
    *,
    execution_id: str | None = None,
    usable_target: int,
    max_requests: int,
    target_candidates: int,
    provider: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run a bounded registered SFT plan and persist candidate/policy artifacts."""
    root = Path(workspace).resolve(strict=True)
    preflight_sft_generation(root, bundle_id, usable_target=usable_target,
                             max_requests=max_requests, target_candidates=target_candidates)
    _, files = load_sft_workflow_bundle(root, bundle_id)
    execution_id = execution_id or f"sftgen-{uuid4().hex[:20]}"
    if not execution_id.startswith("sftgen-") or len(execution_id) > 64:
        raise ValueError("invalid generation execution ID")
    run_root = root / "runs" / "synthetic" / execution_id
    run_root.mkdir(parents=True, exist_ok=False)
    from ..data.targeted_plan import load_targeted_plan
    original_plan = load_targeted_plan(files["plan"])
    bounded_budget = original_plan.budget.model_copy(update={
        "max_requests": max_requests, "target_candidates": target_candidates,
    })
    bounded_plan = original_plan.model_copy(update={"budget": bounded_budget})
    resolved_plan_path = run_root / "resolved_plan.yaml"
    resolved_plan_path.write_text(yaml.safe_dump(bounded_plan.model_dump(mode="json"), allow_unicode=True), encoding="utf-8")
    setup = prepare_generation(
        plan_path=resolved_plan_path, profile_path=files["profile"], template_path=files["template"],
        protocol_path=files["protocol"], definition_path=files["definition"], report_path=files["report"],
        registry_path=files["registry"], train_seed_path=files["train_seeds"],
        protected_dataset_paths=[files["diagnostic"], files["holdout"]],
    )
    profile = setup.profile
    if profile.adapter == "openai_compatible" and provider is None:
        load_local_provider_credential(profile.api_key_env or "", root / ".env.local")
    if provider is None:
        provider = FakeProvider() if profile.adapter == "fake" else OpenAICompatibleProvider()

    plan_dir = root / "artifacts" / "synthetic" / setup.plan.plan_id
    raw_path = plan_dir / "private" / f"responses-{execution_id}.jsonl"
    generation = execute_targeted_generation(
        setup, output_path=raw_path, provider=provider,
        progress_callback=progress_callback, stop_requested=stop_requested,
    )
    if not raw_path.is_file() or generation.get("successful_candidates", 0) == 0:
        result = {
            "execution_id": execution_id, "status": "partial", "bundle_id": bundle_id,
            "desired_usable_records": usable_target, "usable_records": 0,
            "candidate_count": generation.get("successful_candidates", 0),
            "request_count": generation.get("request_count", 0),
            "shortfall": usable_target,
            "stop_reason": generation.get("stop_reason", "no_candidates"),
            "provider_calls": generation.get("provider_calls", 0),
            "network_used": generation.get("network_used", False),
            "raw_output_path": raw_path.relative_to(root).as_posix(),
        }
        (run_root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    policy = load_policy(files["policy"])
    tokenizer = _tokenizer(root, policy)
    seeds = {row.get("id"): row for row in read_jsonl(files["train_seeds"])}
    protocol = setup.protocol
    candidates, policy_inputs = [], []
    for raw in read_jsonl(raw_path):
        seed = seeds.get(raw.get("record_id"))
        if not seed:
            continue
        metadata = seed.get("metadata") if isinstance(seed.get("metadata"), dict) else {}
        question = next((seed.get(key) for key in ("raw_problem", "question", "problem", "text")
                         if isinstance(seed.get(key), str) and seed.get(key).strip()), "")
        if not question:
            continue
        candidate = validate_candidate_response(
            raw.get("response", ""), source_record_id=raw["record_id"],
            expected_problem_id=raw["problem_id"], candidate_slot=raw["candidate_slot"],
            request_sha256=raw["request_sha256"], finish_reason=raw.get("finish_reason"),
            question=question, protocol=protocol, tokenizer=tokenizer,
            max_training_tokens=policy.resources.max_length,
        )
        candidates.append(candidate)
        policy_inputs.append(CandidatePolicyInput(
            candidate=candidate, question=question,
            reference_answer=metadata.get("reference", seed.get("answer", seed.get("reference_answer"))),
        ))
    registry = load_role_registry(files["registry"])
    protected_paths = [files["diagnostic"], files["holdout"]]
    protected_hash = hashlib.sha256("".join(sha256_file(path) for path in protected_paths).encode()).hexdigest()
    protected_index_id = f"protected-index-{protected_hash[:16]}"
    policy_export = export_candidates_with_policy(
        policy_inputs, policy=policy, tokenizer=tokenizer, registry=registry,
        protected_index=benchmark_index_from_jsonl(protected_paths),
        protected_index_id=protected_index_id, protected_index_sha256=protected_hash,
    )
    artifact_root = plan_dir / "builds" / execution_id
    artifacts = write_targeted_artifacts(
        output_dir=artifact_root, plan=setup.plan, profile=profile, tokenizer=tokenizer,
        candidates=candidates, policy_export=policy_export, generation_summary=generation,
        validation_summary={
            "candidate_count": len(candidates),
            "valid_format": sum(item.format_status == "valid" for item in candidates),
            "invalid_format": sum(item.format_status == "invalid" for item in candidates),
            "truncated": sum(item.completion_status == "truncated" for item in candidates),
            "over_length": sum(item.length_status == "over_limit" for item in candidates),
        },
    )
    usable = int(artifacts["training_rows"])
    result = {
        "execution_id": execution_id,
        "status": ("cancelled" if generation.get("stop_reason") == "cancelled_before_next_request"
                   else "completed" if usable >= usable_target else "partial"),
        "bundle_id": bundle_id,
        "source_dataset_id": original_plan.training_seed.dataset_id,
        "desired_usable_records": usable_target,
        "candidate_budget": target_candidates,
        "candidate_count": len(candidates),
        "usable_records": usable,
        "request_count": generation.get("request_count", 0),
        "max_requests": max_requests,
        "shortfall": max(0, usable_target - usable),
        "stop_reason": generation.get("stop_reason"),
        "provider_calls": generation.get("provider_calls", 0),
        "network_used": generation.get("network_used", False),
        "raw_output_path": raw_path.relative_to(root).as_posix(),
        "valid_format_count": sum(item.format_status == "valid" for item in candidates),
        "invalid_format_count": sum(item.format_status == "invalid" for item in candidates),
        "truncated_count": sum(item.completion_status == "truncated" for item in candidates),
        "over_length_count": sum(item.length_status == "over_limit" for item in candidates),
        "generation_artifacts": {
            **{key: Path(value).relative_to(root).as_posix()
               for key, value in artifacts.items()
               if isinstance(value, str) and Path(value).is_absolute()},
            "training_path": (Path(artifacts["private_root"]) / "canonical" / "sft.jsonl").relative_to(root).as_posix(),
        },
        "training_dataset_id": _manifest_dataset_id(artifact_root / artifacts["training_manifest"]),
        "quality_status": "report_only",
    }
    (run_root / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _manifest_dataset_id(path: Path) -> str:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(value["dataset_id"])
