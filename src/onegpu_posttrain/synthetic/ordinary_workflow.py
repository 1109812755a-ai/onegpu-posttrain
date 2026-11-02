"""Provider-free preflight and explicit execution for ordinary seeded SFT."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from ..data.evidence import ProcessingReport, ReportReference, assess_dataset, step
from ..data.io import read_jsonl
from ..data.manifest import DatasetManifest, SourceSpec, TokenizerSpec, save_manifest, sha256_file
from ..data.policy import DataPolicy, digest_json
from ..data.tokenization import WhitespaceTokenizer
from ..lineage import ArtifactRef, DatasetLineage, RunRecord, new_run_id
from .candidates import validate_candidate_response
from .ordinary_bundle import load_ordinary_sft_bundle
from .providers import (FakeProvider, GenerationRequest, OpenAICompatibleProvider,
                        ProviderError, ProviderProfile, OutputProtocol, PromptTemplate)
from .secrets import load_local_provider_credential
from .request_guard import RequestGuard, UncertainProviderRequest
from .sft_workflow import _tokenizer


def _model_inputs(files: dict[str, Path]):
    profile = ProviderProfile.model_validate(yaml.safe_load(files["profile"].read_text(encoding="utf-8")))
    template = PromptTemplate.model_validate(yaml.safe_load(files["template"].read_text(encoding="utf-8")))
    protocol = OutputProtocol.model_validate(yaml.safe_load(files["protocol"].read_text(encoding="utf-8")))
    policy = DataPolicy.model_validate(yaml.safe_load(files["policy"].read_text(encoding="utf-8")))
    plan = json.loads(files["plan"].read_text(encoding="utf-8"))
    return profile, template, protocol, policy, plan


def preflight_ordinary_sft_generation(workspace: str | Path, bundle_id: str) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    bundle, files = load_ordinary_sft_bundle(root, bundle_id)
    profile, template, protocol, policy, plan = _model_inputs(files)
    if policy.tokenizer.test_only and profile.adapter != "fake":
        raise ValueError("test-only tokenizer is permitted only with the local fake provider")
    if profile.profile_id != plan["provider_profile_id"] or template.template_id != plan["prompt_template_id"]:
        raise ValueError("ordinary SFT bundle config identity mismatch")
    if sha256_file(files["source_data"]) != plan["source_content_sha256"]:
        raise ValueError("ordinary SFT source data changed")
    rows = list(read_jsonl(files["source_data"]))
    eligible = sum(1 for row in rows if _question(row))
    return {
        "status": "preflight_ready",
        "mode": "ordinary_sft",
        "bundle_id": bundle.bundle_id,
        "plan_config_sha256": bundle.plan_config_sha256,
        "source_dataset_id": plan["source_dataset_id"],
        "source_record_count": len(rows),
        "eligible_source_count": eligible,
        "model": profile.model,
        "provider_profile_id": profile.profile_id,
        "prompt_template_id": template.template_id,
        "target_count": plan["target_count"],
        "request_limit": plan["request_limit"],
        "max_output_tokens_per_request": plan["max_output_tokens"],
        "policy_id": policy.policy_id,
        "provider_calls": 0,
        "network_used": False,
    }


def _question(row: dict[str, Any]) -> str:
    raw_problem = row.get("raw_problem")
    if isinstance(raw_problem, str) and raw_problem.strip():
        return raw_problem.strip()
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "user" \
                and isinstance(message.get("content"), str) and message["content"].strip():
            return message["content"].strip()
    return ""


def _answer(row: dict[str, Any]) -> str | None:
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "assistant" \
                and isinstance(message.get("content"), str):
            return message["content"]
    return None


def execute_ordinary_sft_generation(
    workspace: str | Path,
    bundle_id: str,
    *,
    execution_id: str,
    max_requests: int | None = None,
    provider: Any | None = None,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Generate bounded candidates from an SFT source, retaining source-to-dataset lineage."""
    root = Path(workspace).resolve(strict=True)
    preflight = preflight_ordinary_sft_generation(root, bundle_id)
    bundle, files = load_ordinary_sft_bundle(root, bundle_id)
    profile, template, protocol, policy, plan = _model_inputs(files)
    request_limit = plan["request_limit"] if max_requests is None else max_requests
    if isinstance(request_limit, bool) or not isinstance(request_limit, int) or not 1 <= request_limit <= plan["request_limit"]:
        raise ValueError("本次请求预算必须在已登记方案上限内")
    rows = list(read_jsonl(files["source_data"]))
    selected = [row for row in rows if _question(row)][:min(plan["target_count"], request_limit)]
    if profile.adapter == "openai_compatible" and provider is None:
        load_local_provider_credential(profile.api_key_env or "", root / ".env.local")
    provider = provider or (FakeProvider() if profile.adapter == "fake" else OpenAICompatibleProvider())
    provider = RequestGuard(root / "artifacts" / "synthetic" / bundle_id, execution_id, provider)
    tokenizer = _tokenizer(root, policy)

    build_id = f"build-sft-ordinary-{hashlib.sha256((bundle_id + execution_id).encode()).hexdigest()[:20]}"
    build_root = root / "artifacts" / "synthetic" / bundle_id / "builds" / execution_id
    private_root = build_root / "private"
    candidate_path = private_root / "candidates.jsonl"
    training_path = private_root / "canonical" / "sft.jsonl"
    candidate_path.parent.mkdir(parents=True, exist_ok=False)
    training_path.parent.mkdir(parents=True, exist_ok=False)
    candidates: list[dict[str, Any]] = []
    canonical: list[dict[str, Any]] = []
    request_count = 0
    uncertain_request = False
    rejected: dict[str, int] = {}
    raw_path = private_root / "provider-responses.jsonl"
    started_at = datetime.now(timezone.utc).isoformat()
    max_items = min(len(selected), request_limit)
    try:
        with raw_path.open("x", encoding="utf-8") as raw_stream:
            for index, row in enumerate(selected):
                if stop_requested and stop_requested():
                    break
                question = _question(row)
                record_id = row.get("id")
                if not isinstance(record_id, str) or not record_id:
                    rejected["missing_source_record_id"] = rejected.get("missing_source_record_id", 0) + 1
                    continue
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                problem_id = metadata.get("problem_id") or record_id
                if not isinstance(problem_id, str) or not problem_id:
                    problem_id = record_id
                topic = metadata.get("topic") or metadata.get("subject") or "math"
                template_values = {"question": question, "topic": str(topic)}
                prompt = template.render({key: value for key, value in template_values.items()
                                          if key in template.allowed_variables})
                required_fields = ", ".join(protocol.required_fields)
                prompt += (f"\n\nOutput exactly one JSON object with fields: {required_fields}. "
                           f"Set problem_id exactly to {json.dumps(problem_id, ensure_ascii=False)}.")
                request = GenerationRequest(prompt=prompt, max_output_tokens=plan["max_output_tokens"],
                                            temperature=plan["temperature"], output_protocol=protocol)
                request_sha = hashlib.sha256(json.dumps(request.model_dump(mode="json"), sort_keys=True,
                                                         ensure_ascii=False).encode()).hexdigest()
                try:
                    response = provider.generate(profile, request)
                except UncertainProviderRequest:
                    raise
                except Exception as exc:  # sanitized; don't persist provider exception text
                    rejected[f"provider_{type(exc).__name__}"] = rejected.get(f"provider_{type(exc).__name__}", 0) + 1
                    request_count += 1
                    if progress_callback:
                        progress_callback({"completed_candidates": index + 1, "attempted_requests": request_count,
                                           "candidate_target": plan["target_count"], "request_budget": request_limit})
                    if isinstance(exc, ProviderError) and exc.uncertain:
                        uncertain_request = True
                        break
                    continue
                request_count += 1
                raw_stream.write(json.dumps({"source_record_id": record_id, "problem_id": problem_id,
                                             "request_sha256": request_sha, "response": response.text,
                                             "finish_reason": response.finish_reason}, ensure_ascii=False) + "\n")
                candidate = validate_candidate_response(
                    response.text, source_record_id=record_id, expected_problem_id=problem_id,
                    candidate_slot=0, request_sha256=request_sha, finish_reason=response.finish_reason,
                    question=question, protocol=protocol, tokenizer=tokenizer,
                    max_training_tokens=policy.resources.max_length,
                )
                candidates.append(candidate.model_dump(mode="json"))
                if candidate.validation_status != "candidate_ready_for_quality_review":
                    rejected[candidate.validation_status] = rejected.get(candidate.validation_status, 0) + 1
                elif candidate.answer_consistency in {"mismatch", "unparseable"}:
                    rejected[f"answer_{candidate.answer_consistency}"] = rejected.get(f"answer_{candidate.answer_consistency}", 0) + 1
                else:
                    generated_id = "sft-gen-" + hashlib.sha256(
                        f"{bundle_id}:{record_id}:{candidate.response_summary.response_sha256}".encode()
                    ).hexdigest()[:24]
                    canonical.append({
                        "id": generated_id,
                        "raw_problem": question,
                        "messages": [{"role": "user", "content": question},
                                     {"role": "assistant", "content": candidate.solution or ""}],
                        "prompt_template_id": template.template_id,
                        "rendered_template_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "metadata": {
                            "problem_id": problem_id,
                            "parent_record_ids": [record_id],
                            "source_dataset_id": plan["source_dataset_id"],
                            "candidate_id": candidate.candidate_id,
                            "generation_mode": "ordinary_sft",
                            "generation_execution_id": execution_id,
                            "answer_consistency": candidate.answer_consistency,
                            "semantic_correctness": "unverified",
                            "source_answer_available": _answer(row) is not None,
                        },
                    })
                if progress_callback:
                    progress_callback({"completed_candidates": index + 1, "attempted_requests": request_count,
                                       "candidate_target": plan["target_count"], "request_budget": request_limit})
    except OSError:
        raise
    finally:
        pass

    for path, records in ((candidate_path, candidates), (training_path, canonical)):
        with path.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

    content_sha = sha256_file(training_path)
    profile_sha = sha256_file(files["profile"])
    config = {"mode": "ordinary_sft", "bundle_id": bundle_id, "execution_id": execution_id,
              "provider_profile_id": profile.profile_id, "provider_profile_sha256": profile_sha,
              "model": profile.model, "source_dataset_id": plan["source_dataset_id"],
              "source_content_sha256": plan["source_content_sha256"],
              "target_count": plan["target_count"], "prompt_template_id": template.template_id,
              "prompt_template_sha256": plan["prompt_template_sha256"], "policy_id": policy.policy_id}
    config_sha = digest_json(config)
    generated_dataset_id = "ds-sft-ordinary-build-" + hashlib.sha256(
        f"{bundle_id}:{execution_id}:{content_sha}".encode()
    ).hexdigest()[:24]
    build_run_id = new_run_id("data_build", label="ordinary-sft-generation")
    quality_run_id = new_run_id("quality_assess", label="ordinary-sft-generation")
    created_at = datetime.now(timezone.utc).isoformat()
    quality = assess_dataset(canonical, stage="sft", dataset_id=generated_dataset_id,
                             content_sha256=content_sha, build_id=build_id,
                             tokenizer_name=tokenizer.name, overlap=None)
    quality.run_id = quality_run_id
    reports = build_root / "reports"
    quality_path = reports / "quality" / "sft.json"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(quality.model_dump(mode="json", by_alias=True),
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    processing = ProcessingReport(
        build_id=build_id, created_at=created_at, processing_protocol="ordinary-sft-generation-v1",
        source={"repo": f"{profile.adapter}:{profile.model}", "config": profile.profile_id,
                "revision": profile.revision},
        tokenizer={"repo": tokenizer.name, "revision": "local-or-profile-pinned", "backend": tokenizer.name},
        benchmark_screening={"enabled": False, "status": "not_reassessed",
                             "reason": "源 SFT 数据已登记；本生成步骤未执行独立 benchmark contamination screening"},
        config=config, config_sha256=config_sha,
        source_acquisition={"mode": "llm_generate", "provider_profile_id": profile.profile_id,
                            "model": profile.model, "generation_mode": "ordinary_sft"},
        stop_reason=("uncertain_request_requires_explicit_review" if uncertain_request else
                     "targets_met" if len(canonical) >= plan["target_count"] else "source_exhausted"),
        processing_steps=[step(1, "ordinary_sft_generation_v1", "transform",
                               f"dataset:{plan['source_dataset_id']}", f"dataset:{generated_dataset_id}",
                               request_count, len(canonical), "requests", output_unit="examples",
                               rejected=rejected,
                               details={"candidate_count": len(candidates), "target_count": plan["target_count"],
                                        "answer_consistency": "checked when parseable; otherwise reported"},
                               limitations=["生成模型自报答案不等同独立正确性验证"] )],
        outputs={"sft": {"dataset_id": generated_dataset_id,
                          "path": "private/canonical/sft.jsonl", "content_sha256": content_sha,
                          "record_count": len(canonical), "budget_count": len(canonical),
                          "budget_target": plan["target_count"], "budget_unit": "examples"}},
        limitations=["未重新执行 benchmark 去污染；继承源数据声明但不宣称新增样本通过 benchmark screen",
                     "语义正确性未独立验证，登记前须人工审核"],
    )
    processing_path = reports / "processing_report.json"
    processing_path.write_text(json.dumps(processing.model_dump(mode="json", by_alias=True),
                                          ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    quality_ref = ReportReference(path="../reports/quality/sft.json", sha256=sha256_file(quality_path),
                                  report_id=quality.quality_report_id)
    processing_ref = ReportReference(path="../reports/processing_report.json",
                                     sha256=sha256_file(processing_path), report_id=build_id)
    source_manifest = DatasetManifest(
        dataset_id=generated_dataset_id, stage="sft",
        source=SourceSpec(repo=f"{profile.adapter}:{profile.model}", config=profile.profile_id,
                          revision=profile.revision or ""),
        tokenizer=TokenizerSpec(repo=tokenizer.name, revision="local-or-profile-pinned"),
        prompt_template_id=template.template_id,
        protocol_versions={"canonical_schema": "stage-canonical-v1", "generation": "ordinary-sft-v1"},
        sample_count=len(canonical), hashes={"canonical_records_sha256": content_sha},
        processing_report=processing_ref, quality_report=quality_ref,
        lineage=DatasetLineage(
            parents=[plan["source_dataset_id"]], produced_by_run_id=build_run_id,
            transform={"name": "ordinary_sft_generation", "version": "1", "config_sha256": config_sha},
            source_evidence={"generation_mode": "ordinary_sft", "bundle_id": bundle_id,
                             "execution_id": execution_id, "provider_profile_id": profile.profile_id,
                             "provider_profile_sha256": profile_sha,
                             "source_dataset_sha256": plan["source_content_sha256"]},
            record_provenance="record", count_unit="examples", content_sha256=content_sha,
            processing_report_id=build_id, quality_report_ids=[quality.quality_report_id],
        ),
        metadata={"canonical_path": "private/canonical/sft.jsonl", "generation_mode": "ordinary_sft",
                  "generation_execution_id": execution_id, "bundle_id": bundle_id,
                  "quality_status": "report_only", "usage": "exploratory", "visibility": "private"},
    )
    manifest_path = build_root / "manifests" / "sft-generated-v1.yaml"
    save_manifest(manifest_path, source_manifest)
    # save_manifest creates parent directories and writes once for this new execution.
    source_ref = ArtifactRef(id=plan["source_dataset_id"], kind="dataset",
                             digest={"value": plan["source_content_sha256"], "status": "available"}, role="source")
    result_ref = ArtifactRef(id=generated_dataset_id, kind="dataset",
                             path=training_path.relative_to(root).as_posix(),
                             digest={"value": content_sha, "status": "available"}, role="generated")
    build_run = RunRecord(
        run_id=build_run_id, run_type="data_build", status="completed", run_label="ordinary-sft-generation",
        started_at=started_at, completed_at=created_at, inputs=[source_ref], outputs=[result_ref],
        resolved_config_sha256=config_sha,
    )
    quality_run = RunRecord(
        run_id=quality_run_id, run_type="quality_assess", status="completed", run_label="ordinary-sft-quality",
        started_at=created_at, completed_at=created_at,
        inputs=[result_ref.model_copy(update={"role": "assessed"})],
        outputs=[ArtifactRef(id=quality.quality_report_id, kind="quality_report",
                             digest={"value": sha256_file(quality_path), "status": "available"}, role="quality")],
    )
    for run in (build_run, quality_run):
        run_dir = build_root / "runs" / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "lineage.json").write_text(json.dumps(run.model_dump(mode="json"), ensure_ascii=False,
                                                             indent=2) + "\n", encoding="utf-8")
    shortfall = max(0, plan["target_count"] - len(canonical))
    result = {
        "execution_id": execution_id, "mode": "ordinary_sft", "bundle_id": bundle_id,
        "source_dataset_id": plan["source_dataset_id"], "status": "completed" if shortfall == 0 else "partial",
        "desired_usable_records": plan["target_count"], "usable_records": len(canonical),
        "candidate_count": len(candidates), "request_count": request_count,
        "max_requests": request_limit, "shortfall": shortfall,
        "stop_reason": ("uncertain_request_requires_explicit_review" if uncertain_request else
                        "targets_met" if shortfall == 0 else "source_exhausted"),
        "provider_calls": request_count, "network_used": profile.adapter != "fake",
        "candidate_rejections": rejected,
        "quality_status": "report_only",
        "training_dataset_id": generated_dataset_id,
        "generation_artifacts": {
            "training_path": training_path.relative_to(root).as_posix(),
            "training_manifest": manifest_path.relative_to(root).as_posix(),
            "generation_validation_cost_report": "",
        },
    }
    cost_path = reports / "generation_validation_cost.json"
    cost_path.write_text(json.dumps({"mode": "ordinary_sft", "bundle_id": bundle_id,
                                     "execution_id": execution_id, "provider_profile_id": profile.profile_id,
                                     "provider_adapter": profile.adapter, "model": profile.model,
                                     "generation": {key: result[key] for key in (
                                         "request_count", "candidate_count", "shortfall", "stop_reason", "network_used")},
                                     "candidate_rejections": rejected, "quality_status": "report_only",
                                     "semantic_correctness": "not_independently_verified"},
                                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["generation_artifacts"]["generation_validation_cost_report"] = cost_path.relative_to(root).as_posix()
    return result
