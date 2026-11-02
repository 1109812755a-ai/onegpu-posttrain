"""Ordinary Preference candidate generation and explicit human ranking."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from ..schemas import Message, PreferenceRecord
from ..synthetic.providers import (
    GenerationRequest,
    OutputProtocol,
    ProviderError,
    ProviderProfile,
)
from .dataset_import import register_jsonl
from .manifest import load_manifest
from .registry import DataSource

_PAIR_SIDE_PROTOCOL = OutputProtocol(
    protocol_id="ordinary-preference-candidate", version="1", required_fields=["completion"],
)


def generate_ordinary_preference_candidates(
    prompts: list[str], *, profile: ProviderProfile, provider: Any,
    request_budget: int, candidate_budget: int, max_output_tokens: int, timeout_seconds: float = 120,
) -> dict[str, Any]:
    """Generate unranked same-prompt candidates; no provider label becomes evidence."""
    if not prompts or len(prompts) > 500:
        raise ValueError("普通 Preference 生成需要 1..500 个非空题目")
    if any(not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16_000 for prompt in prompts):
        raise ValueError("题目必须是 1..16000 字符的非空文本")
    for name, value in (("request_budget", request_budget), ("candidate_budget", candidate_budget),
                        ("max_output_tokens", max_output_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须为正整数")
    if request_budget > 10_000 or candidate_budget > 1_000 or max_output_tokens > 8192:
        raise ValueError("Preference 生成预算超过安全上限")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 1 <= timeout_seconds <= 600:
        raise ValueError("Provider 单次超时必须在 1 到 600 秒之间")
    candidates: list[dict[str, Any]] = []
    errors: Counter[str] = Counter()
    requests = 0
    stop_reason = "candidate_budget_reached"
    for problem_index, raw_problem in enumerate(prompts):
        problem_id = "problem-" + hashlib.sha256(raw_problem.strip().encode("utf-8")).hexdigest()[:20]
        for slot in range(2):
            if requests >= request_budget or len(candidates) >= candidate_budget:
                stop_reason = "request_budget_exhausted" if requests >= request_budget else "candidate_budget_reached"
                break
            generation_prompt = (
                "Provide one candidate response to the user question below. Do not state a preference rank, "
                "correctness label, reward, or evaluation result. Return only JSON with exactly one field: "
                '{"completion":"..."}.\nQuestion: ' + raw_problem.strip()
            )
            requests += 1
            try:
                response = provider.generate(profile, GenerationRequest(
                    prompt=generation_prompt, max_output_tokens=max_output_tokens,
                    timeout_seconds=float(timeout_seconds), output_protocol=_PAIR_SIDE_PROTOCOL,
                ))
                value = json.loads(response.text)
                if (not isinstance(value, dict) or set(value) != {"completion"}
                        or not isinstance(value.get("completion"), str)
                        or not value["completion"].strip() or len(value["completion"]) > 16_000):
                    errors["invalid_or_rank_bearing_response"] += 1
                    continue
                candidate_id = "pref-candidate-" + hashlib.sha256(
                    f"{problem_id}\0{slot}\0{requests}\0{value['completion']}".encode("utf-8")
                ).hexdigest()[:24]
                candidates.append({
                    "candidate_id": candidate_id, "problem_id": problem_id,
                    "problem_index": problem_index, "candidate_slot": slot,
                    "raw_problem": raw_problem.strip(), "completion": value["completion"].strip(),
                    "provider_request_id": response.provider_request_id,
                    "ordering_status": "unreviewed",
                })
            except ProviderError as exc:
                errors[exc.category] += 1
            except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
                errors["provider_or_response_error"] += 1
        else:
            continue
        break
    else:
        stop_reason = "all_prompts_consumed"
    candidate_set_sha256 = hashlib.sha256(json.dumps(
        candidates, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return {
        "schema_version": "1.0", "stage": "preference", "mode": "ordinary",
        "status": "completed" if len(candidates) == candidate_budget else "partial",
        "requested_candidate_count": candidate_budget, "candidate_count": len(candidates),
        "request_count": requests, "request_budget": request_budget, "timeout_seconds": float(timeout_seconds),
        "provider_profile_id": profile.profile_id, "provider_model": profile.model,
        "provider_revision": profile.revision, "request_errors": dict(errors),
        "stop_reason": stop_reason, "candidates": candidates,
        "candidate_set_sha256": candidate_set_sha256,
        "ordering_evidence_status": "unreviewed", "trainable": False,
    }


def build_human_reviewed_preference_pairs(
    candidates: list[dict[str, Any]], reviews: list[dict[str, Any]], *, reviewer_id: str,
) -> dict[str, Any]:
    """Create pairs only from explicit same-prompt user-selected ranking evidence."""
    if not isinstance(reviewer_id, str) or not reviewer_id.strip() or len(reviewer_id) > 120:
        raise ValueError("必须提供审阅者标识（最多 120 字符）")
    by_id: dict[str, dict[str, Any]] = {}
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    candidate_set_sha256 = hashlib.sha256(json.dumps(
        candidates, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            rejected["invalid_candidate"] += 1
            continue
        cid = candidate.get("candidate_id")
        pid = candidate.get("problem_id")
        prompt = candidate.get("raw_problem")
        completion = candidate.get("completion")
        if not all(isinstance(value, str) and value.strip() for value in (cid, pid, prompt, completion)):
            rejected["missing_candidate_fields"] += 1
            continue
        if cid in by_id:
            rejected["duplicate_candidate_id"] += 1
            continue
        by_id[cid] = candidate
        by_problem[pid].append(candidate)

    records: list[dict[str, Any]] = []
    reviewed_problem_ids: set[str] = set()
    for review in reviews:
        if not isinstance(review, dict):
            rejected["invalid_review"] += 1
            continue
        problem_id = review.get("problem_id")
        chosen_id, rejected_id = review.get("chosen_candidate_id"), review.get("rejected_candidate_id")
        rationale = review.get("rationale")
        if (not isinstance(problem_id, str) or not problem_id.strip()
                or not isinstance(chosen_id, str) or not isinstance(rejected_id, str)
                or chosen_id == rejected_id or not isinstance(rationale, str) or len(rationale.strip()) < 8):
            rejected["review_missing_identity_or_rationale"] += 1
            continue
        if problem_id in reviewed_problem_ids:
            rejected["duplicate_problem_review"] += 1
            continue
        chosen, rejected_candidate = by_id.get(chosen_id), by_id.get(rejected_id)
        if (chosen is None or rejected_candidate is None
                or chosen.get("problem_id") != problem_id or rejected_candidate.get("problem_id") != problem_id
                or chosen.get("raw_problem") != rejected_candidate.get("raw_problem")):
            rejected["review_candidates_do_not_match_same_problem"] += 1
            continue
        if chosen["completion"].strip() == rejected_candidate["completion"].strip():
            rejected["identical_pair_sides"] += 1
            continue
        reviewed_problem_ids.add(problem_id)
        evidence = {
            "kind": "human_review", "reviewer_id": reviewer_id.strip(),
            "rationale": rationale.strip(), "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "chosen_candidate_id": chosen_id, "rejected_candidate_id": rejected_id,
            "candidate_content_sha256": hashlib.sha256(
                (chosen["completion"].strip() + "\0" + rejected_candidate["completion"].strip()).encode("utf-8")
            ).hexdigest(),
            "candidate_set_sha256": candidate_set_sha256,
            "scope": "user_declared_ranking; reviewer_identity_not_independently_authenticated",
        }
        row = PreferenceRecord(
            id="pref-human-" + hashlib.sha256(
                f"{problem_id}\0{chosen_id}\0{rejected_id}".encode("utf-8")
            ).hexdigest()[:24],
            raw_problem=chosen["raw_problem"],
            prompt=[Message(role="user", content=chosen["raw_problem"])],
            chosen=chosen["completion"].strip(), rejected=rejected_candidate["completion"].strip(),
            prompt_template_id="ordinary-preference-human-review-v1",
            metadata={
                "problem_id": problem_id, "data_role": "train_pairs",
                "candidate_ids": [chosen_id, rejected_id],
                "ordering_evidence_status": "human_verified",
                "ordering_evidence": evidence,
            },
        )
        records.append(row.model_dump(mode="json"))
    unreviewed_problem_count = max(0, len(by_problem) - len(reviewed_problem_ids))
    if unreviewed_problem_count:
        rejected["unreviewed_problem"] += unreviewed_problem_count
    return {
        "schema_version": "1.0", "stage": "preference", "status": "partial" if rejected else "completed",
        "candidate_count": len(candidates), "review_count": len(reviews),
        "candidate_set_sha256": candidate_set_sha256,
        "usable_pairs": len(records), "rejections": dict(rejected), "records": records,
        "ordering_evidence_status": "human_verified" if records else "unreviewed",
        "reviewer_scope": "user_declared; reviewer identity is not independently authenticated",
        "trainable": bool(records),
    }


def register_human_reviewed_preference_pairs(
    report: dict[str, Any], *, workspace: str | Path, output_root: str | Path,
    provider_profile_id: str, provider_model: str, provider_revision: str | None,
    generation_bundle_id: str | None = None, generation_execution_id: str | None = None,
    provider_adapter: str | None = None,
) -> dict[str, Any]:
    """Register only the reviewed non-empty slice, retaining uncertainty and provenance."""
    rows = report.get("records")
    if (report.get("stage") != "preference" or not isinstance(rows, list) or not rows
            or len(rows) != report.get("usable_pairs")):
        raise ValueError("没有可登记的人工审核 Preference pair")
    for index, row in enumerate(rows):
        metadata = row.get("metadata") if isinstance(row, dict) else None
        evidence = metadata.get("ordering_evidence") if isinstance(metadata, dict) else None
        if (metadata.get("ordering_evidence_status") != "human_verified"
                or not isinstance(evidence, dict) or evidence.get("kind") != "human_review"
                or not evidence.get("chosen_candidate_id") or not evidence.get("rejected_candidate_id")
                or not evidence.get("rationale")
                or evidence.get("candidate_set_sha256") != report.get("candidate_set_sha256")):
            raise ValueError(f"Preference pair {index} lacks auditable human ranking evidence")
        PreferenceRecord.model_validate(row)
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    source = DataSource(
        source_id="source-pref-human-" + hashlib.sha256(payload).hexdigest()[:20],
        kind="model_generated", name=f"Ordinary Preference candidates via {provider_profile_id}",
        revision=provider_revision, revision_status="known" if provider_revision else "unknown",
        license_status="not_provided", declared_by="provider",
    )
    registration = register_jsonl(
        BytesIO(payload), workspace=workspace, output_root=output_root,
        stage="preference", origin=source,
        registration_metadata={
            "quality_status": "report_only", "usage": "exploratory",
            "generation_protocol": "ordinary-preference-candidate-human-review-v1",
            "provider_profile_id": provider_profile_id, "provider_model": provider_model,
            "provider_revision": provider_revision,
            "provider_adapter": provider_adapter,
            "provider_revision_status": "available" if provider_revision else "unavailable",
            "generation_bundle_id": generation_bundle_id,
            "generation_execution_id": generation_execution_id,
            "candidate_set_sha256": report.get("candidate_set_sha256"),
            "provider_identity_status": "profile_declared_not_independently_verified",
            "registered_model_association_status": "not_proven",
            "registered_model_id": None, "registered_model_sha256": None,
            "ordering_evidence_protocol": "human-review-v1",
            "ordering_evidence_status": "human_verified",
            "reviewer_identity_status": "not_independently_authenticated",
            "preference_quality_summary": {
                "candidate_count": report.get("candidate_count"), "review_count": report.get("review_count"),
                "usable_pairs": len(rows), "rejections": report.get("rejections", {}),
                "ordering_evidence_status": "human_verified",
                "reviewer_scope": report.get("reviewer_scope"),
            },
        },
    )
    manifest = load_manifest(registration.path / "manifests" / "preference-model-generated-v1.yaml")
    return {
        "dataset_id": registration.dataset_id, "record_count": registration.record_count,
        "canonical_sha256": registration.canonical_sha256,
        "quality_report_id": manifest.quality_report.report_id if manifest.quality_report else None,
        "registration_path": registration.path.relative_to(Path(workspace).resolve()).as_posix(),
        "parent_dataset_refs": [], "ordering_evidence_status": "human_verified",
        "reviewer_identity_status": "not_independently_authenticated",
    }
