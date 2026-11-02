"""Provider-free preflight and explicit theme-driven SFT generation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..data.evidence import ProcessingReport, ReportReference, assess_dataset, step
from ..data.manifest import (
    DatasetManifest,
    SourceSpec,
    TokenizerSpec,
    save_manifest,
    sha256_file,
)
from ..data.policy import DataPolicy
from ..lineage import ArtifactRef, DatasetLineage, RunRecord, new_run_id
from .prompt_only import (
    PROTOCOL,
    TEMPLATE_ID,
    TEMPLATE_TEXT,
    load_prompt_only_bundle,
)
from .providers import (
    FakeProvider,
    GenerationRequest,
    OpenAICompatibleProvider,
    ProviderError,
    ProviderProfile,
)
from .secrets import load_local_provider_credential
from .request_guard import RequestGuard, UncertainProviderRequest
from .sft_workflow import _tokenizer


def preflight_prompt_only_sft_generation(workspace: str | Path, bundle_id: str) -> dict[str, Any]:
    plan, files = load_prompt_only_bundle(workspace, bundle_id)
    profile = ProviderProfile.model_validate(yaml.safe_load(files["profile"].read_text(encoding="utf-8")))
    policy = DataPolicy.model_validate(yaml.safe_load(files["policy"].read_text(encoding="utf-8")))
    if profile.profile_id != plan.provider_profile_id or policy.config_sha256 != plan.policy_sha256:
        raise ValueError("prompt-only bundle config identity mismatch")
    return {"status": "preflight_ready", "mode": "prompt_only_sft", "bundle_id": bundle_id,
            "plan_config_sha256": plan.config_sha256, "provider_profile_id": profile.profile_id,
            "model": profile.model, "topic": plan.topic, "skill": plan.skill,
            "difficulty": plan.difficulty, "target_count": plan.target_count,
            "request_limit": plan.request_limit, "desired_usable_records": plan.target_count,
            "max_output_tokens_per_request": plan.max_output_tokens,
            "semantic_correctness": "not_independently_verified", "provider_calls": 0, "network_used": False}


def execute_prompt_only_sft_generation(
    workspace: str | Path, bundle_id: str, *, execution_id: str, provider: Any | None = None,
    max_requests: int | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    preflight_prompt_only_sft_generation(root, bundle_id)
    plan, files = load_prompt_only_bundle(root, bundle_id)
    request_limit = plan.request_limit if max_requests is None else max_requests
    if isinstance(request_limit, bool) or not isinstance(request_limit, int) or not 1 <= request_limit <= plan.request_limit:
        raise ValueError("本次请求预算必须在已登记方案上限内")
    profile = ProviderProfile.model_validate(yaml.safe_load(files["profile"].read_text(encoding="utf-8")))
    policy = DataPolicy.model_validate(yaml.safe_load(files["policy"].read_text(encoding="utf-8")))
    if profile.adapter == "openai_compatible" and provider is None:
        load_local_provider_credential(profile.api_key_env or "", root / ".env.local")
    provider = provider or (FakeProvider() if profile.adapter == "fake" else OpenAICompatibleProvider())
    provider = RequestGuard(root / "artifacts" / "synthetic" / bundle_id, execution_id, provider)
    tokenizer = _tokenizer(root, policy)
    build_id = f"build-sft-prompt-{hashlib.sha256((bundle_id + execution_id).encode()).hexdigest()[:20]}"
    build_root = root / "artifacts" / "synthetic" / bundle_id / "builds" / execution_id
    private_root = build_root / "private"
    training_path = private_root / "canonical" / "sft.jsonl"
    training_path.parent.mkdir(parents=True, exist_ok=False)
    raw_path = private_root / "provider-responses.jsonl"
    canonical: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    seen_questions: set[str] = set()
    started_at = datetime.now(timezone.utc).isoformat()
    request_count = 0
    uncertain_request = False
    prompt = TEMPLATE_TEXT.format(topic=plan.topic, skill=plan.skill, difficulty=plan.difficulty,
                                  constraints=plan.constraints or "none")
    rendered_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    for index in range(request_limit):
        if len(canonical) >= plan.target_count or (stop_requested and stop_requested()):
            break
        request = GenerationRequest(prompt=prompt, max_output_tokens=plan.max_output_tokens,
                                    temperature=plan.temperature, output_protocol=PROTOCOL)
        request_sha = hashlib.sha256(json.dumps(request.model_dump(mode="json"), sort_keys=True,
                                                  ensure_ascii=False).encode()).hexdigest()
        request_count += 1
        try:
            response = provider.generate(profile, request)
            payload = json.loads(response.text)
        except UncertainProviderRequest:
            raise
        except ProviderError as exc:
            error_key = f"provider_{exc.category}"
            if exc.status_code is not None:
                error_key += f"_http_{exc.status_code}"
            rejected[error_key] = rejected.get(error_key, 0) + 1
            response = None
            payload = None
            uncertain_request = exc.uncertain
        except Exception as exc:  # noqa: BLE001 - provider text may contain secrets; retain only error class
            rejected[f"provider_or_json_{type(exc).__name__}"] = rejected.get(f"provider_or_json_{type(exc).__name__}", 0) + 1
            response = None
            payload = None
        if response is not None:
            with raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"request_sha256": request_sha, "response_sha256": hashlib.sha256(
                    response.text.encode("utf-8")).hexdigest(), "finish_reason": response.finish_reason},
                    ensure_ascii=False) + "\n")
            if response.finish_reason in {"length", "max_tokens"}:
                rejected["truncated"] = rejected.get("truncated", 0) + 1
                payload = None
        if not isinstance(payload, dict) or any(not isinstance(payload.get(key), str) or not payload[key].strip()
                                                 for key in ("question", "solution", "final_answer")):
            if response is not None:
                rejected["invalid_candidate_schema"] = rejected.get("invalid_candidate_schema", 0) + 1
            payload = None
        if payload is not None:
            question, solution, answer = (payload[key].strip() for key in ("question", "solution", "final_answer"))
            normalized = " ".join(question.casefold().split())
            if normalized in seen_questions:
                rejected["duplicate_question"] = rejected.get("duplicate_question", 0) + 1
            else:
                seen_questions.add(normalized)
                record_id = "sft-prompt-" + hashlib.sha256(f"{execution_id}:{index}:{request_sha}".encode()).hexdigest()[:24]
                messages = [{"role": "user", "content": question},
                            {"role": "assistant", "content": f"{solution}\n\nFinal answer: {answer}"}]
                token_count = tokenizer.count_messages(messages)
                candidate = {"id": record_id, "raw_problem": question, "messages": messages,
                             "prompt_template_id": TEMPLATE_ID,
                             "metadata": {"topic": plan.topic, "skill": plan.skill,
                                          "difficulty": plan.difficulty, "semantic_correctness": "unverified"}}
                candidate_rows.append({"id": record_id, "question": question, "solution": solution,
                                       "final_answer": answer, "token_count": token_count})
                if token_count > policy.resources.max_length:
                    rejected["over_token_limit"] = rejected.get("over_token_limit", 0) + 1
                else:
                    canonical.append(candidate)
        if progress_callback:
            progress_callback({"completed_candidates": index + 1, "attempted_requests": request_count,
                               "candidate_target": plan.target_count, "request_budget": request_limit})
        if uncertain_request:
            break
    training_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in canonical),
                             encoding="utf-8")
    content_sha = sha256_file(training_path)
    profile_sha = sha256_file(files["profile"])
    generated_dataset_id = "ds-sft-prompt-build-" + hashlib.sha256(
        f"{bundle_id}:{execution_id}:{content_sha}".encode()).hexdigest()[:24]
    build_run_id = new_run_id("data_build", label="prompt-only-sft-generation")
    quality_run_id = new_run_id("quality_assess", label="prompt-only-sft-generation")
    created_at = datetime.now(timezone.utc).isoformat()
    quality = assess_dataset(canonical, stage="sft", dataset_id=generated_dataset_id,
                             content_sha256=content_sha, build_id=build_id,
                             tokenizer_name=tokenizer.name, overlap=None)
    quality.run_id = quality_run_id
    reports = build_root / "reports"
    quality_path = reports / "quality" / "sft.json"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(quality.model_dump(mode="json", by_alias=True), ensure_ascii=False,
                                       indent=2) + "\n", encoding="utf-8")
    config = {"mode": "prompt_only_sft", "bundle_id": bundle_id, "execution_id": execution_id,
              "topic_sha256": hashlib.sha256(plan.topic.encode()).hexdigest(),
              "skill_sha256": hashlib.sha256(plan.skill.encode()).hexdigest(), "difficulty": plan.difficulty,
              "provider_profile_id": profile.profile_id, "provider_profile_sha256": profile_sha,
              "prompt_template_id": TEMPLATE_ID, "prompt_template_sha256": plan.prompt_template_sha256,
              "output_protocol_sha256": plan.output_protocol_sha256,
              "rendered_prompt_sha256": rendered_prompt_sha256,
              "constraints_sha256": hashlib.sha256(plan.constraints.encode("utf-8")).hexdigest(),
              "target_count": plan.target_count}
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    processing = ProcessingReport(
        build_id=build_id, created_at=created_at, processing_protocol="prompt-only-sft-generation-v1",
        source={"repo": f"{profile.adapter}:{profile.model}", "config": profile.profile_id, "revision": profile.revision},
        tokenizer={"repo": tokenizer.name, "revision": "local-or-profile-pinned", "backend": tokenizer.name},
        benchmark_screening={"enabled": False, "status": "not_assessed",
                             "reason": "newly generated examples require later benchmark screening"},
        config=config, config_sha256=config_sha,
        source_acquisition={"mode": "llm_generate", "generation_mode": "prompt_only_sft",
                            "provider_profile_id": profile.profile_id, "model": profile.model},
        stop_reason=("uncertain_request_requires_explicit_review" if uncertain_request else
                     "targets_met" if len(canonical) >= plan.target_count else "request_budget_exhausted"),
        processing_steps=[step(1, "prompt_only_sft_generation_v1", "transform", "prompt:topic_skill_difficulty",
                               f"dataset:{generated_dataset_id}", request_count, len(canonical), "requests",
                               output_unit="examples", rejected=rejected,
                               details={"candidate_count": len(candidate_rows), "target_count": plan.target_count},
                               limitations=["模型自生成答案未独立验证正确性"] )],
        outputs={"sft": {"dataset_id": generated_dataset_id, "path": "private/canonical/sft.jsonl",
                         "content_sha256": content_sha, "record_count": len(canonical),
                         "budget_count": len(canonical), "budget_target": plan.target_count,
                         "budget_unit": "examples"}},
        limitations=["无种子数据集；prompt-only 生成数据，semantic correctness 未独立验证，登记前需人工审核"])
    processing_path = reports / "processing_report.json"
    processing_path.write_text(json.dumps(processing.model_dump(mode="json", by_alias=True), ensure_ascii=False,
                                          indent=2) + "\n", encoding="utf-8")
    quality_ref = ReportReference(path="../reports/quality/sft.json", sha256=sha256_file(quality_path),
                                  report_id=quality.quality_report_id)
    processing_ref = ReportReference(path="../reports/processing_report.json", sha256=sha256_file(processing_path),
                                     report_id=build_id)
    manifest = DatasetManifest(
        dataset_id=generated_dataset_id, stage="sft",
        source=SourceSpec(repo=f"{profile.adapter}:{profile.model}", config=profile.profile_id, revision=profile.revision or ""),
        tokenizer=TokenizerSpec(repo=tokenizer.name, revision="local-or-profile-pinned"),
        prompt_template_id=TEMPLATE_ID,
        protocol_versions={"canonical_schema": "stage-canonical-v1", "generation": "prompt-only-sft-v1"},
        sample_count=len(canonical), hashes={"canonical_records_sha256": content_sha},
        processing_report=processing_ref, quality_report=quality_ref,
        lineage=DatasetLineage(parents=[], produced_by_run_id=build_run_id,
            transform={"name": "prompt_only_sft_generation", "version": "1", "config_sha256": config_sha},
            source_evidence={"source_kind": "model_generated", "generation_mode": "prompt_only_sft",
                "bundle_id": bundle_id, "execution_id": execution_id, "provider_profile_id": profile.profile_id,
                "provider_profile_sha256": profile_sha, "prompt_template_id": TEMPLATE_ID,
                "prompt_template_sha256": plan.prompt_template_sha256,
                "output_protocol_sha256": plan.output_protocol_sha256,
                    "rendered_prompt_sha256": rendered_prompt_sha256,
                "topic_sha256": hashlib.sha256(plan.topic.encode()).hexdigest(),
                "skill_sha256": hashlib.sha256(plan.skill.encode()).hexdigest(),
                    "constraints_sha256": hashlib.sha256(plan.constraints.encode()).hexdigest(),
                "difficulty": plan.difficulty, "requested_target_count": plan.target_count,
                "semantic_correctness": "not_independently_verified"},
            record_provenance="record", count_unit="examples", content_sha256=content_sha,
            processing_report_id=build_id, quality_report_ids=[quality.quality_report_id]),
        metadata={"canonical_path": "private/canonical/sft.jsonl", "generation_mode": "prompt_only_sft",
                  "generation_execution_id": execution_id, "bundle_id": bundle_id,
                  "quality_status": "report_only", "usage": "exploratory", "visibility": "private"})
    manifest_path = build_root / "manifests" / "sft-generated-v1.yaml"
    save_manifest(manifest_path, manifest)
    result_ref = ArtifactRef(id=generated_dataset_id, kind="dataset", path=training_path.relative_to(root).as_posix(),
                             digest={"value": content_sha, "status": "available"}, role="generated")
    build_run = RunRecord(run_id=build_run_id, run_type="data_build", status="completed",
        run_label="prompt-only-sft-generation", started_at=started_at, completed_at=created_at,
        inputs=[ArtifactRef(id=execution_id, kind="file", role="prompt_generation"),
                ArtifactRef(id=bundle_id, kind="file", digest={"value": plan.config_sha256,
                "status": "available"}, role="plan")], outputs=[result_ref], resolved_config_sha256=config_sha)
    quality_run = RunRecord(run_id=quality_run_id, run_type="quality_assess", status="completed",
        run_label="prompt-only-sft-quality", started_at=created_at, completed_at=created_at,
        inputs=[result_ref.model_copy(update={"role": "assessed"})],
        outputs=[ArtifactRef(id=quality.quality_report_id, kind="quality_report",
                 digest={"value": sha256_file(quality_path), "status": "available"}, role="quality")])
    for run in (build_run, quality_run):
        run_dir = build_root / "runs" / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "lineage.json").write_text(json.dumps(run.model_dump(mode="json"), ensure_ascii=False,
                                                           indent=2) + "\n", encoding="utf-8")
    cost_path = reports / "generation_validation_cost.json"
    cost_path.write_text(json.dumps({"mode": "prompt_only_sft", "bundle_id": bundle_id,
        "execution_id": execution_id, "provider_profile_id": profile.profile_id,
        "provider_adapter": profile.adapter, "model": profile.model, "request_count": request_count,
        "candidate_count": len(candidate_rows), "shortfall": max(0, plan.target_count - len(canonical)),
        "candidate_rejections": rejected, "semantic_correctness": "not_independently_verified",
        "prompt_sha256": rendered_prompt_sha256},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shortfall = max(0, plan.target_count - len(canonical))
    return {"execution_id": execution_id, "mode": "prompt_only_sft", "bundle_id": bundle_id,
        "status": "completed" if shortfall == 0 else ("partial" if canonical else "failed"),
        "desired_usable_records": plan.target_count, "usable_records": len(canonical),
        "candidate_count": len(candidate_rows) + sum(rejected.values()), "request_count": request_count,
        "max_requests": request_limit, "shortfall": shortfall,
        "stop_reason": ("uncertain_request_requires_explicit_review" if uncertain_request else
                        "targets_met" if shortfall == 0 else "request_budget_exhausted"),
        "provider_calls": request_count, "network_used": profile.adapter != "fake",
        "candidate_rejections": rejected, "quality_status": "report_only",
        "semantic_correctness": "not_independently_verified", "training_dataset_id": generated_dataset_id,
        "generation_artifacts": {"training_path": training_path.relative_to(root).as_posix(),
            "training_manifest": manifest_path.relative_to(root).as_posix(),
            "generation_validation_cost_report": cost_path.relative_to(root).as_posix()}}
