"""Evidence-based same-problem preference pair construction for targeted data."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

from ..schemas import Message, PreferenceRecord
from ..synthetic.verification import assess_candidate_answers_batch
from .dataset_import import register_jsonl
from .manifest import load_manifest
from .registry import DataSource
from .targeted_stage_plan import PreferenceTarget, TargetedStagePlan
from ..synthetic.providers import GenerationRequest, OutputProtocol, ProviderError, ProviderProfile


_PREFERENCE_CANDIDATE_PROTOCOL = OutputProtocol(
    protocol_id="targeted-preference-candidate", version="1", required_fields=["completion"],
)


def generate_targeted_preference_candidates(
    seed_records: list[dict[str, Any]],
    *,
    plan: TargetedStagePlan,
    profile: ProviderProfile,
    provider: Any,
    max_requests: int | None = None,
    max_candidates: int | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Generate bounded same-seed answer attempts; correctness is decided later by a verifier.

    Only explicitly marked training-seed rows are sent to a provider. The second
    attempt is prompted as a plausible error, but that prompt is never treated
    as ranking evidence: :func:`build_targeted_preference_pairs` independently
    checks both outputs against the seed's trusted scalar reference.
    """
    if not isinstance(plan.target, PreferenceTarget):
        raise TypeError("Preference candidate generation requires a Preference targeted plan")
    request_budget = plan.budget.max_requests if max_requests is None else max_requests
    candidate_budget = plan.budget.max_candidates if max_candidates is None else max_candidates
    output_budget = plan.budget.max_output_tokens if max_output_tokens is None else max_output_tokens
    if not 0 <= request_budget <= plan.budget.max_requests:
        raise ValueError("max_requests cannot exceed the immutable plan budget")
    if not 0 <= candidate_budget <= plan.budget.max_candidates:
        raise ValueError("max_candidates cannot exceed the immutable plan budget")
    if not 1 <= output_budget <= plan.budget.max_output_tokens:
        raise ValueError("max_output_tokens cannot exceed the immutable plan budget")

    selected: list[tuple[dict[str, Any], str, str, Any]] = []
    rejected: Counter[str] = Counter()
    for seed in seed_records:
        if not isinstance(seed, dict):
            rejected["invalid_seed_record"] += 1
            continue
        metadata = seed.get("metadata") if isinstance(seed.get("metadata"), dict) else {}
        role = seed.get("data_role", seed.get("role", metadata.get("data_role", metadata.get("role"))))
        record_id = seed.get("id")
        problem = next((seed.get(key) for key in ("raw_problem", "problem", "question", "prompt")
                        if isinstance(seed.get(key), str) and seed.get(key).strip()), None)
        reference = seed.get("reference_answer", seed.get("answer", metadata.get("reference")))
        problem_id = metadata.get("problem_id", record_id)
        if role != "train_seed":
            rejected["non_training_ancestry"] += 1
            continue
        if not isinstance(record_id, str) or not record_id.strip() or not isinstance(problem_id, str) or not problem_id.strip() or not problem:
            rejected["missing_seed_identity_or_problem"] += 1
            continue
        if reference is None or isinstance(reference, bool) or not isinstance(reference, (str, int, float)):
            rejected["missing_or_unsupported_reference"] += 1
            continue
        if metadata.get("scenario_id") not in (None, plan.target.scenario_id):
            rejected["scenario_mismatch"] += 1
            continue
        selected.append((seed, record_id.strip(), problem, reference))

    candidates: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    requests = 0
    stop_reason = "candidate_budget_reached"
    # Alternate requested answer types per problem. A provider's stated role or
    # correctness is not returned to the pair builder as evidence.
    attempts = [(item, slot) for item in selected for slot in ("solution_attempt", "plausible_error")]
    for (seed, source_record_id, problem, reference), slot in attempts:
        if requests >= request_budget:
            stop_reason = "request_budget_exhausted"
            break
        if len(candidates) >= candidate_budget:
            stop_reason = "candidate_budget_reached"
            break
        prompt = (
            "Create one candidate completion for a training preference example.\n"
            f"Target scenario: {plan.target.scenario_id}\nQuestion: {problem}\n"
            + (f"Trusted final answer for the solution attempt: {reference}\n"
               "Produce a concise, valid solution whose final answer agrees with it.\n"
               if slot == "solution_attempt" else
               f"Do not use the trusted final answer {reference!r}. Produce a plausible but incorrect solution attempt.\n")
            + 'Return only a JSON object with exactly one field: {"completion":"..."}. '
            "Do not return labels, correctness claims, or explanations outside the completion."
        )
        requests += 1
        try:
            response = provider.generate(profile, GenerationRequest(
                prompt=prompt, max_output_tokens=output_budget,
                timeout_seconds=plan.budget.timeout_seconds, output_protocol=_PREFERENCE_CANDIDATE_PROTOCOL,
            ))
            payload = json.loads(response.text)
            if (not isinstance(payload, dict) or set(payload) != {"completion"}
                    or not isinstance(payload.get("completion"), str) or not payload["completion"].strip()
                    or len(payload["completion"]) > 16_000):
                errors["invalid_candidate_response"] += 1
                continue
            candidate_hash = hashlib.sha256(
                f"{plan.plan_id}\0{source_record_id}\0{slot}\0{requests}".encode("utf-8")
            ).hexdigest()[:24]
            candidates.append({
                "candidate_id": f"pref-candidate-{candidate_hash}",
                "problem_id": str((seed.get("metadata") or {}).get("problem_id") or seed.get("id")),
                "source_record_id": source_record_id, "source_role": "train_seed",
                "problem": problem, "completion": payload["completion"].strip(),
                "reference_answer": reference,
                "candidate_slot": slot,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "provider_request_id": response.provider_request_id,
            })
        except ProviderError as exc:
            errors[exc.category] += 1
        except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
            errors["provider_or_response_error"] += 1
    else:
        stop_reason = "all_seed_attempts_consumed"

    return {
        "schema_version": "1.0", "status": "completed" if len(candidates) == candidate_budget else "partial",
        "plan_id": plan.plan_id, "plan_version": plan.version,
        "training_seed_dataset_id": plan.training_seed.dataset_id,
        "requested_candidate_count": candidate_budget, "candidate_count": len(candidates),
        "request_count": requests, "max_requests": request_budget,
        "provider_profile_id": profile.profile_id, "provider_model": profile.model,
        "provider_revision": profile.revision, "max_output_tokens": output_budget,
        "network_used": profile.adapter == "openai_compatible", "stop_reason": stop_reason,
        "rejected_seeds": dict(rejected), "request_errors": dict(errors), "candidates": candidates,
        "correctness_status": "not_independently_verified",
    }


def build_targeted_preference_pairs(
    candidates: list[dict[str, Any]],
    *,
    plan: TargetedStagePlan,
) -> dict[str, Any]:
    """Pair only one independently verified-correct and one verified-wrong answer.

    Provider self-reported correctness is ignored. Candidates without a trusted
    scalar reference, unsupported verifier outputs, same-quality pairs, reused
    IDs, and non-training ancestry are reported as rejected, never guessed.
    One pair at most is emitted per seed problem to avoid accidental weighting.
    """
    if not isinstance(plan.target, PreferenceTarget):
        raise TypeError("Preference pair construction requires a Preference targeted plan")
    if len(candidates) > plan.budget.max_candidates:
        raise ValueError("candidate list exceeds the immutable plan max_candidates")

    groups: dict[tuple[str, str], list[tuple[dict[str, Any], str, str, dict[str, Any]]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_content: set[tuple[str, str]] = set()
    answers: list[str] = []
    references: list[Any] = []
    valid_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            rejected["invalid_candidate"] += 1
            continue
        cid = candidate.get("candidate_id")
        problem_id = candidate.get("problem_id")
        source_id = candidate.get("source_record_id")
        role = candidate.get("source_role", "train_seed")
        problem = candidate.get("problem")
        answer = candidate.get("completion")
        if role != "train_seed":
            rejected["non_training_ancestry"] += 1
            continue
        if not all(isinstance(v, str) and v.strip() for v in (cid, problem_id, source_id, problem, answer)):
            rejected["missing_identity_or_text"] += 1
            continue
        if cid in seen_ids:
            rejected["duplicate_candidate_id"] += 1
            continue
        seen_ids.add(cid)
        digest = hashlib.sha256(answer.strip().encode("utf-8")).hexdigest()
        content_key = (source_id, digest)
        if content_key in seen_content:
            rejected["duplicate_candidate_content"] += 1
            continue
        seen_content.add(content_key)
        valid_rows.append(candidate)
        answers.append(answer)
        references.append(candidate.get("reference_answer"))

    assessments = assess_candidate_answers_batch(answers, references)
    for candidate, assessment in zip(valid_rows, assessments, strict=True):
        if assessment.status not in {"correct", "incorrect"}:
            rejected[f"verification_{assessment.status}"] += 1
            continue
        groups[(candidate["source_record_id"], candidate["problem_id"])].append((
            candidate, assessment.status, assessment.method, assessment.model_dump(mode="json"),
        ))

    records: list[dict[str, Any]] = []
    examined = 0
    for (source_id, problem_id), rows in sorted(groups.items()):
        if len(records) >= plan.budget.expected_usable_count:
            rejected["target_budget_reached"] += len(rows)
            continue
        if len({row[0]["problem"] for row in rows}) != 1:
            rejected["problem_text_conflict"] += len(rows)
            continue
        correct = [row for row in rows if row[1] == "correct"]
        incorrect = [row for row in rows if row[1] == "incorrect"]
        examined += 1
        if not correct or not incorrect:
            reason = "both_correct_or_no_wrong_candidate" if correct else "both_incorrect_or_no_correct_candidate"
            rejected[reason] += len(rows)
            continue
        chosen = min(correct, key=lambda item: item[0]["candidate_id"])
        rejected_candidate = min(incorrect, key=lambda item: item[0]["candidate_id"])
        chosen_text = chosen[0]["completion"].strip()
        rejected_text = rejected_candidate[0]["completion"].strip()
        if chosen_text == rejected_text:
            rejected["identical_pair_sides"] += 1
            continue
        identity = hashlib.sha256(
            f"{plan.plan_id}\0{source_id}\0{chosen[0]['candidate_id']}\0{rejected_candidate[0]['candidate_id']}".encode()
        ).hexdigest()[:24]
        record = PreferenceRecord(
            id=f"pref-targeted-{identity}", raw_problem=chosen[0]["problem"],
            prompt=[Message(role="user", content=chosen[0]["problem"])],
            chosen=chosen_text, rejected=rejected_text, prompt_template_id="targeted-preference-v1",
            metadata={
                "problem_id": problem_id, "data_role": "train_pairs", "source_data_role": "train_seed",
                "parent_record_ids": [source_id],
                "candidate_ids": [chosen[0]["candidate_id"], rejected_candidate[0]["candidate_id"]],
                "chosen_correct": True, "rejected_correct": False,
                "chosen_verification": chosen[3], "rejected_verification": rejected_candidate[3],
                "targeted_plan_id": plan.plan_id, "targeted_plan_version": plan.version,
                "preference_pairing_protocol": "independent-correctness-v1",
                "ordering_evidence_status": "verified",
                "ordering_evidence_method": "independent_final_answer_reference_check",
            },
        )
        records.append(record.model_dump(mode="json"))

    partial = len(records) < plan.budget.expected_usable_count
    return {
        "schema_version": "1.0", "status": "partial" if partial else "completed",
        "plan_id": plan.plan_id, "plan_version": plan.version,
        "target_pairs": plan.budget.expected_usable_count,
        "candidate_count": len(candidates), "verified_problem_count": examined,
        "usable_pairs": len(records), "shortfall": max(0, plan.budget.expected_usable_count - len(records)),
        "rejections": dict(rejected), "records": records,
        "evidence_limitations": [
            "答案正确性使用独立 parser/reference 检查；不代表推理过程质量已验证。",
            "只允许每个训练种子问题最多生成一条 pair；不会用模型自报标签填补排序。",
        ],
    }


def register_targeted_preference_pairs(
    report: dict[str, Any],
    *,
    plan: TargetedStagePlan,
    workspace: str | Path,
    output_root: str | Path,
    provider_profile_id: str,
    provider_model: str | None = None,
    provider_revision: str | None = None,
    provider_adapter: str | None = None,
    generation_bundle_id: str | None = None,
    generation_execution_id: str | None = None,
) -> dict[str, Any]:
    """Register only the actual non-empty pair slice; preserve partial status."""
    if not isinstance(plan.target, PreferenceTarget):
        raise TypeError("Preference registration requires a Preference targeted plan")
    if report.get("plan_id") != plan.plan_id or report.get("plan_version") != plan.version:
        raise ValueError("preference report does not match the immutable plan")
    rows = report.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("zero usable preference pairs are not registered")
    if len(rows) != report.get("usable_pairs"):
        raise ValueError("preference report count does not match records")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"preference row {index} is not an object")
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if (metadata.get("targeted_plan_id") != plan.plan_id
                or metadata.get("targeted_plan_version") != plan.version):
            raise ValueError(f"preference row {index} is not bound to the registered plan")
        if (metadata.get("data_role") != "train_pairs" or metadata.get("source_data_role") != "train_seed"
                or not isinstance(metadata.get("parent_record_ids"), list)
                or len(metadata["parent_record_ids"]) != 1):
            raise ValueError(f"preference row {index} lacks one training-seed ancestor")
        if (metadata.get("chosen_correct") is not True or metadata.get("rejected_correct") is not False
                or (metadata.get("chosen_verification") or {}).get("status") != "correct"
                or (metadata.get("rejected_verification") or {}).get("status") != "incorrect"):
            raise ValueError(f"preference row {index} lacks independent chosen/rejected evidence")
        candidate_ids = metadata.get("candidate_ids")
        if not isinstance(candidate_ids, list) or len(candidate_ids) != 2 or candidate_ids[0] == candidate_ids[1]:
            raise ValueError(f"preference row {index} must identify two distinct candidates")
        if (not isinstance(row.get("chosen"), str) or not row["chosen"].strip()
                or not isinstance(row.get("rejected"), str) or not row["rejected"].strip()
                or row["chosen"].strip() == row["rejected"].strip()):
            raise ValueError(f"preference row {index} has empty or identical pair sides")
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    source = DataSource(
        source_id=f"source-targeted-{plan.plan_id}", kind="model_generated",
        name=f"Targeted Preference generation via {provider_profile_id}",
        revision=provider_revision, revision_status="known" if provider_revision else "unknown",
        license_status="not_provided", declared_by="provider",
    )
    parent_ref = {
        "dataset_id": plan.training_seed.dataset_id,
        "content_sha256": plan.training_seed.content_sha256,
        "role": "train_seed",
    }
    registered = register_jsonl(
        BytesIO(payload), workspace=workspace, output_root=output_root, stage="preference", origin=source,
        registration_metadata={
            "quality_status": "report_only", "usage": "exploratory",
            "targeted_plan_id": plan.plan_id, "targeted_plan_version": plan.version,
            "parent_dataset_ids": [plan.training_seed.dataset_id],
            # DPO and SimPO consume the registered Preference artifact, while
            # this immutable ref proves exactly which seed dataset produced it.
            "parent_dataset_refs": [parent_ref],
            "generation_protocol": "targeted-preference-independent-correctness-v1",
            "provider_profile_id": provider_profile_id,
            "provider_model": provider_model,
            "provider_revision": provider_revision,
            "provider_adapter": provider_adapter,
            "provider_revision_status": "available" if provider_revision else "unavailable",
            "provider_identity_status": "profile_declared_not_independently_verified",
            "registered_model_association_status": "not_proven",
            "registered_model_id": None, "registered_model_sha256": None,
            "generation_bundle_id": generation_bundle_id,
            "generation_execution_id": generation_execution_id,
            "verification_scope": "final-answer-only",
            "ordering_evidence_protocol": "independent-correctness-v1",
            "ordering_evidence_status": "verified",
            "preference_quality_summary": {
                "status": report.get("status"),
                "target_pairs": report.get("target_pairs"),
                "candidate_count": report.get("candidate_count"),
                "usable_pairs": report.get("usable_pairs"),
                "shortfall": report.get("shortfall"),
                "rejections": report.get("rejections", {}),
                "independently_verified_pairs": len(rows),
                "evidence_scope": "final-answer-only",
                "parent_dataset_refs": [parent_ref],
            },
        },
    )
    manifest = load_manifest(registered.path / "manifests" / "preference-model-generated-v1.yaml")
    return {
        "plan_id": plan.plan_id, "plan_version": plan.version,
        "dataset_id": registered.dataset_id, "record_count": registered.record_count,
        "canonical_sha256": registered.canonical_sha256,
        "quality_report_id": manifest.quality_report.report_id if manifest.quality_report else None,
        "parent_dataset_refs": [parent_ref],
        "registration_path": registered.path.relative_to(Path(workspace).resolve()).as_posix(),
        "generation_status": report["status"], "shortfall": report["shortfall"],
    }
