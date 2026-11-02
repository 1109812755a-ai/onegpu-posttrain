"""Bounded, evidence-bound CPT corpus generation for one targeted plan."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from io import BytesIO
from pathlib import Path

from ..synthetic.cpt_generation import validate_cpt_candidate_payload
from ..synthetic.providers import GenerationRequest, OutputProtocol, ProviderProfile
from .dataset_import import RegistrationResult, register_jsonl
from .decontaminate import BenchmarkIndex
from .manifest import load_manifest
from .registry import DataSource
from .targeted_stage_plan import CPTTarget, TargetedStagePlan


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def generate_targeted_cpt_dataset(
    plan: TargetedStagePlan,
    *,
    provider: object,
    profile: ProviderProfile,
    benchmark_index: BenchmarkIndex,
    benchmark_index_id: str,
    benchmark_index_sha256: str,
    tokenizer_id: str,
    count_tokens: Callable[[str], int],
    output_root: str | Path,
    workspace: str | Path,
    fuzzy_threshold: float = 0.85,
) -> dict:
    """Generate up to the immutable plan's budgets; register genuine partial data.

    This function intentionally reports factuality as unverified. The token
    counter is injected so token targets always identify the tokenizer used.
    """
    target = plan.target
    if not isinstance(target, CPTTarget):
        raise TypeError("CPT data generation requires a CPT targeted plan")
    if profile.profile_id != plan.provider_profile_id:
        raise ValueError("provider profile does not match the immutable plan")
    if not benchmark_index_id or len(benchmark_index_sha256) != 64:
        raise ValueError("CPT generation requires identified benchmark contamination evidence")
    if not tokenizer_id:
        raise ValueError("tokenizer_id is required for auditable token accounting")
    token_budget = plan.budget.expected_usable_count
    topics = list(target.subtopics)
    desired_shares = {item.subtopic: item.minimum_token_share for item in target.coverage_targets}
    records: list[dict] = []
    rejections: Counter[str] = Counter()
    seen_text: set[str] = set()
    topic_tokens: Counter[str] = Counter()
    request_count = 0
    measured_tokens = 0

    while (measured_tokens < token_budget
           and request_count < min(plan.budget.max_requests, plan.budget.max_candidates)
           and len(records) < plan.budget.max_candidates):
        request_count += 1
        if desired_shares and measured_tokens:
            topic = min(topics, key=lambda name: (
                topic_tokens[name] / measured_tokens - desired_shares.get(name, 0), name,
            ))
        else:
            topic = topics[(request_count - 1) % len(topics)]
        prompt = (
            "Write one self-contained educational source passage for continued-pretraining (CPT). "
            "Do not write a question, answer key, instruction, or fabricated citation. Return exactly "
            "a JSON object with fields text and subtopic. Set subtopic to the exact requested topic string, "
            "without adding a subtitle. Keep the text focused on the requested topic.\n"
            f"Domain: {target.domain}\nSubtopic: {topic}"
        )
        request = GenerationRequest(
            prompt=prompt, max_output_tokens=plan.budget.max_output_tokens,
            temperature=0.4 if "temperature" in profile.supported_parameters else None,
            seed=plan.seed + request_count if "seed" in profile.supported_parameters else None,
            timeout_seconds=plan.budget.timeout_seconds,
            output_protocol=OutputProtocol(protocol_id="targeted-cpt-text", version="1",
                                           required_fields=["text", "subtopic"]),
        )
        try:
            response = provider.generate(profile, request)
            payload = json.loads(response.text)
        except (ValueError, TypeError, json.JSONDecodeError, AttributeError):
            rejections["invalid_candidate"] += 1
            continue
        candidate, count, rejection = validate_cpt_candidate_payload(
            payload, expected_subtopic=topic, benchmark_index=benchmark_index,
            fuzzy_threshold=fuzzy_threshold, count_tokens=count_tokens,
            max_output_tokens=plan.budget.max_output_tokens,
        )
        if rejection:
            rejections[rejection] += 1
            continue
        assert candidate is not None and count is not None
        normalized = _normalise_text(candidate.text)
        if normalized in seen_text:
            rejections["exact_duplicate"] += 1
            continue
        record_id = "cpt-gen-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
        seen_text.add(normalized)
        records.append({
            "id": record_id,
            "text": candidate.text,
            "source": f"provider:{profile.profile_id}",
            "metadata": {
                "data_role": "train_seed",
                "domain": target.domain,
                "subtopic": candidate.subtopic,
                "tokens": count,
                "plan_id": plan.plan_id,
                "plan_version": plan.version,
                "provider_profile_id": profile.profile_id,
                "provider_model": profile.model,
                "parent_dataset_id": plan.training_seed.dataset_id,
            },
        })
        measured_tokens += count
        topic_tokens[candidate.subtopic] += count

    partial = measured_tokens < token_budget
    if not records:
        return {
            "status": "failed", "registered": False, "partial": True,
            "plan_id": plan.plan_id, "request_count": request_count,
            "usable_tokens": 0, "target_tokens": token_budget,
            "rejections": dict(rejections), "reason": "no_candidate_passed_validation",
            "factuality": "not_verified",
        }

    payload = b"".join((json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                       for row in records)
    source = DataSource(
        source_id=f"source-targeted-{plan.plan_id}", kind="model_generated",
        name=f"Targeted CPT generation via {profile.profile_id}",
        revision=profile.revision, revision_status="known" if profile.revision else "unknown",
        license_status="not_provided", declared_by="provider",
    )
    registration: RegistrationResult = register_jsonl(
        BytesIO(payload), workspace=workspace, output_root=output_root,
        stage="cpt", origin=source,
        registration_metadata={
            "targeted_plan_id": plan.plan_id,
            "targeted_plan_version": plan.version,
            "parent_dataset_ids": [plan.training_seed.dataset_id],
            "generation_protocol": "targeted-cpt-generation-v1",
            "tokenizer_id": tokenizer_id,
            "benchmark_index_id": benchmark_index_id,
            "benchmark_index_sha256": benchmark_index_sha256,
        },
    )
    manifest = load_manifest(registration.path / "manifests" / "cpt-model-generated-v1.yaml")
    report = {
        "schema_version": "1.0", "protocol": "targeted-cpt-generation-v1",
        "status": "partial" if partial else "completed", "partial": partial,
        "plan_id": plan.plan_id, "plan_version": plan.version,
        "dataset_id": registration.dataset_id,
        "record_count": len(records), "usable_tokens": measured_tokens,
        "target_tokens": token_budget, "shortfall_tokens": max(0, token_budget - measured_tokens),
        "request_count": request_count, "rejections": dict(rejections),
        "topic_tokens": dict(topic_tokens), "topic_token_shares": {
            topic: topic_tokens[topic] / measured_tokens for topic in topics
        },
        "tokenizer_id": tokenizer_id,
        "benchmark_index_id": benchmark_index_id,
        "benchmark_index_sha256": benchmark_index_sha256,
        "fuzzy_threshold": fuzzy_threshold,
        "provider_profile_id": profile.profile_id, "provider_model": profile.model,
        "canonical_sha256": registration.canonical_sha256,
        "quality_report_id": manifest.quality_report.report_id if manifest.quality_report else None,
        "factuality": "not_verified",
        "limitations": [
            "Provider output is not independent evidence of factuality.",
            "Contamination screening is limited to the identified exact+shingle index.",
            "The registered token count uses the caller-declared tokenizer; no model training is implied.",
        ],
    }
    report_path = registration.path / "targeted_generation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**report, "registered": True, "artifact_path": str(registration.path), "report_path": str(report_path)}
