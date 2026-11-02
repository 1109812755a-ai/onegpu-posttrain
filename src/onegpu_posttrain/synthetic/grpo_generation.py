"""GRPO prompt candidate generation without manufacturing reward evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from ..data.targeted_grpo import ExecutableRewardContract, validate_reward_contract
from ..data.targeted_stage_plan import GRPOTarget, TargetedStagePlan
from .provider_identity import provider_identity_evidence
from .providers import GenerationRequest, OutputProtocol, ProviderError, ProviderProfile

_PROMPT_PROTOCOL = OutputProtocol(
    protocol_id="grpo-prompt-candidate", version="1", required_fields=["raw_problem"],
)


def generate_grpo_prompt_candidates(
    seeds: list[dict[str, Any]],
    *,
    mode: str,
    profile: ProviderProfile,
    provider: Any,
    contract: ExecutableRewardContract,
    request_budget: int,
    candidate_budget: int,
    max_output_tokens: int,
    timeout_seconds: float = 120,
    targeted_plan: TargetedStagePlan | None = None,
    progress_callback=None,
    stop_requested=None,
) -> dict[str, Any]:
    """Generate bounded prompt-only candidates; never accept model answers/reward labels.

    This is intentionally not a trainable GRPO dataset builder. Ordinary
    generation has no trusted reference; targeted generation may reference
    only exact train-seed ancestors. A separate evidence/registration workflow
    must attach trusted answers before the rows can pass the GRPO training gate.
    """
    if mode not in {"ordinary", "targeted"}:
        raise ValueError("mode must be ordinary or targeted")
    for name, value in (("request_budget", request_budget), ("candidate_budget", candidate_budget),
                        ("max_output_tokens", max_output_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if request_budget > 10_000 or candidate_budget > 10_000 or max_output_tokens > 8192:
        raise ValueError("GRPO generation budget exceeds the safety limit")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 1 <= timeout_seconds <= 600:
        raise ValueError("Provider request timeout must be between 1 and 600 seconds")
    validate_reward_contract(contract)
    if mode == "targeted":
        if targeted_plan is None or not isinstance(targeted_plan.target, GRPOTarget):
            raise ValueError("targeted GRPO generation requires a registered GRPO plan")
        evidence = targeted_plan.target.reward_contract
        if (evidence is None or evidence.evidence_id != contract.contract_id
                or evidence.content_sha256 != contract.protocol_sha256
                or request_budget > targeted_plan.budget.max_requests
                or candidate_budget > targeted_plan.budget.max_candidates
                or max_output_tokens > targeted_plan.budget.max_output_tokens):
            raise ValueError("reward contract or generation budget does not match the immutable GRPO plan")
    elif targeted_plan is not None:
        raise ValueError("ordinary GRPO generation cannot carry a targeted plan")

    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    requests = 0
    for index, seed in enumerate(seeds):
        if stop_requested and stop_requested():
            break
        if len(accepted) >= candidate_budget or requests >= request_budget:
            break
        if not isinstance(seed, dict):
            rejected["invalid_seed"] += 1
            continue
        metadata = seed.get("metadata") if isinstance(seed.get("metadata"), dict) else {}
        role = seed.get("data_role", seed.get("role", metadata.get("data_role", metadata.get("role"))))
        if mode == "targeted" and role != "train_seed":
            rejected["non_training_ancestry"] += 1
            continue
        seed_id = seed.get("id", seed.get("record_id"))
        prompt = next((seed.get(key) for key in ("raw_problem", "problem", "question", "prompt")
                       if isinstance(seed.get(key), str) and seed[key].strip()), None)
        if not isinstance(seed_id, str) or not seed_id.strip() or not prompt:
            rejected["missing_seed_identity_or_prompt"] += 1
            continue
        problem_id = metadata.get("problem_id", seed_id)
        if not isinstance(problem_id, str) or not problem_id.strip():
            rejected["missing_problem_identity"] += 1
            continue
        generation_prompt = (
            "Generate one self-contained math problem prompt for GRPO training. "
            "Return only JSON with exactly the field raw_problem. Do not include an answer, "
            "reference, reward, correctness label, or scoring instructions.\n"
            f"Source problem/topic: {prompt.strip()}\n"
            f"Requested variation index: {index + 1}"
        )
        requests += 1
        try:
            response = provider.generate(profile, GenerationRequest(
                prompt=generation_prompt, max_output_tokens=max_output_tokens,
                timeout_seconds=float(timeout_seconds), output_protocol=_PROMPT_PROTOCOL,
            ))
            payload = json.loads(response.text)
            # Reject additional answer-like fields rather than silently trusting them.
            if (not isinstance(payload, dict) or set(payload) != {"raw_problem"}
                    or not isinstance(payload.get("raw_problem"), str)
                    or not payload["raw_problem"].strip() or len(payload["raw_problem"]) > 16_000):
                rejected["invalid_or_answer_bearing_provider_response"] += 1
                continue
            identity = hashlib.sha256(
                f"{mode}\0{seed_id}\0{problem_id}\0{requests}\0{payload['raw_problem']}".encode("utf-8")
            ).hexdigest()[:24]
            accepted.append({
                "candidate_id": f"grpo-prompt-candidate-{identity}",
                "raw_problem": payload["raw_problem"].strip(),
                "messages": [{"role": "user", "content": payload["raw_problem"].strip()}],
                "problem_id": problem_id,
                "source_record_id": seed_id,
                "source_role": role or "ordinary_topic",
                "targeted_plan_id": targeted_plan.plan_id if targeted_plan else None,
                "targeted_plan_version": targeted_plan.version if targeted_plan else None,
                "reward_contract_id": contract.contract_id,
                "reward_contract_protocol_sha256": contract.protocol_sha256,
                "reference_status": "missing",
                "reward_evidence_status": "not_evaluated",
                "trainable": False,
            })
        except ProviderError as exc:
            errors[exc.category] += 1
        except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
            errors["provider_or_response_error"] += 1
        if progress_callback:
            progress_callback({"attempted_requests": requests, "accepted_candidates": len(accepted),
                               "candidate_budget": candidate_budget})

    cancelled = bool(stop_requested and stop_requested())
    return {
        "schema_version": "1.0", "stage": "grpo", "mode": mode,
        "status": "cancelled" if cancelled else "completed" if len(accepted) == candidate_budget else "partial",
        "cancelled": cancelled,
        "candidate_count": len(accepted), "requested_candidate_count": candidate_budget,
        "request_count": requests, "request_budget": request_budget, "timeout_seconds": float(timeout_seconds),
        "rejections": dict(rejected), "request_errors": dict(errors), "candidates": accepted,
        **provider_identity_evidence(profile_id=profile.profile_id, model=profile.model,
                                     revision=profile.revision, adapter=profile.adapter),
        "network_used": profile.adapter == "openai_compatible",
        "targeted_plan_id": targeted_plan.plan_id if targeted_plan else None,
        "targeted_plan_version": targeted_plan.version if targeted_plan else None,
        "reward_contract_id": contract.contract_id,
        "reward_contract_protocol_sha256": contract.protocol_sha256,
        "trainable": False,
        "limitations": [
            "Provider 输出仅作为 prompt 候选，不会被当成可信答案或 reward 标签。",
            "普通生成没有可信 reference；定向生成仍须独立绑定 train_seed reference 后才能进入训练数据登记。",
            "本结果不是 canonical GRPO 数据集，也不会绕过 reward-contract 训练预检。",
        ],
    }
