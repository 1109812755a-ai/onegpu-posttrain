"""Private candidate/data artifacts with standard OneGPU manifests and lineage."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..data.evidence import (
    ProcessingReport,
    ReportReference,
    assess_dataset,
    step,
)
from ..data.manifest import (
    DatasetManifest,
    SourceSpec,
    TokenizerSpec,
    save_manifest,
    sha256_file,
)
from ..data.policy import digest_json
from ..data.tokenization import TokenizerBackend
from ..lineage import ArtifactRef, DatasetLineage, RunRecord, new_run_id
from ..reporting.data_report import write_report
from .candidates import CandidateRecord
from .providers import ProviderProfile
from .provider_identity import provider_identity_evidence


def _write_jsonl_new(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _ref(path: Path, root: Path, artifact_id: str, kind: str, role: str) -> ArtifactRef:
    return ArtifactRef(
        id=artifact_id,
        kind=kind,
        path=path.relative_to(root).as_posix(),
        digest={"algorithm": "sha256", "value": sha256_file(path), "status": "available"},
        role=role,
    )


def _dataset_id(stage: str, content_sha256: str, config_sha256: str) -> str:
    digest = hashlib.sha256(f"{stage}:{content_sha256}:{config_sha256}".encode()).hexdigest()
    return f"ds-targeted-{stage}-{digest[:24]}"


def write_targeted_artifacts(
    *,
    output_dir: str | Path,
    plan: Any,
    profile: ProviderProfile,
    tokenizer: TokenizerBackend,
    candidates: list[CandidateRecord],
    policy_export: dict[str, Any],
    generation_summary: dict[str, Any],
    validation_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist only structured private data and sanitized public evidence."""
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("定向数据产物需要一个新的空 build 目录；不覆盖历史记录")
    root.mkdir(parents=True, exist_ok=True)
    stage = policy_export["stage"]
    if stage not in {"sft", "preference"}:
        raise ValueError("targeted data export supports only sft or preference")
    if policy_export.get("policy_id") is None:
        raise ValueError("policy export is missing its policy identity")
    build_id = f"build-targeted-{digest_json({'plan': plan.plan_id, 'profile': profile.profile_id, 'stage': stage, 'content': [c.response_summary.response_sha256 for c in candidates]})[:24]}"
    build_run_id = new_run_id("data_build", label="targeted-synthetic")
    quality_run_id = new_run_id("quality_assess", label=f"targeted-{stage}")
    created_at = datetime.now(timezone.utc).isoformat()

    private = root / "private"
    candidate_path = private / "candidates.jsonl"
    training_path = private / "canonical" / f"{stage}.jsonl"
    candidate_rows = [candidate.model_dump(mode="json") for candidate in candidates]
    training_rows = policy_export["records"]
    _write_jsonl_new(candidate_path, candidate_rows)
    _write_jsonl_new(training_path, training_rows)
    candidate_hash = sha256_file(candidate_path)
    training_hash = sha256_file(training_path)
    config = {
        "plan_id": plan.plan_id,
        "provider_profile_id": profile.profile_id,
        "provider_adapter": profile.adapter,
        "model": profile.model,
        "model_revision": profile.revision,
        "model_revision_status": "available" if profile.revision else "unavailable",
        **provider_identity_evidence(profile_id=profile.profile_id, model=profile.model,
                                     revision=profile.revision, adapter=profile.adapter),
        "prompt_template_sha256": plan.prompt_template_sha256,
        "policy_id": policy_export["policy_id"],
        "policy_config_sha256": policy_export["policy_config_sha256"],
        "stage": stage,
        "generation_parameters": plan.generation_parameters.model_dump(mode="json"),
    }
    config_sha256 = digest_json(config)
    candidate_dataset_id = _dataset_id("synthetic_candidates", candidate_hash, config_sha256)
    training_dataset_id = _dataset_id(stage, training_hash, config_sha256)
    source = SourceSpec(
        repo=f"{profile.adapter}:{profile.model}",
        config=profile.profile_id,
        revision=profile.revision or "",
    )
    tokenizer_spec = TokenizerSpec(repo=tokenizer.name, revision="local-or-profile-pinned")
    process_steps = [
        step(
            1, "llm_generate_v1", "transform", f"dataset:{plan.training_seed.dataset_id}",
            f"dataset:{candidate_dataset_id}", len({c.source_record_id for c in candidates}),
            len(candidates), "training_seed_problems", output_unit="candidates",
            details={"provider_adapter": profile.adapter, "model": profile.model,
                     "model_revision_status": "available" if profile.revision else "unavailable"},
            limitations=["候选正文保存在 private/；模型自报答案不构成独立正确性证据"],
        ),
        step(
            2, "candidate_policy_export_v1", "export", f"dataset:{candidate_dataset_id}",
            f"dataset:{training_dataset_id}", len(candidates), len(training_rows), "candidates",
            output_unit="preference_pairs" if stage == "preference" else "examples",
            details={"policy_id": policy_export["policy_id"],
                     "assessment_counts": policy_export.get("assessment_counts", {}),
                     "rejected": policy_export.get("rejected", {})},
            limitations=["只导出通过当前 policy 的记录；不足配额时不复制或伪造样本"],
        ),
    ]
    stop_reason = "targets_met" if len(candidates) >= plan.budget.target_candidates else "source_exhausted"
    report_outputs = {
        "synthetic_candidates": {"dataset_id": candidate_dataset_id, "path": "private/candidates.jsonl",
                                 "content_sha256": candidate_hash, "record_count": len(candidates),
                                 "budget_count": len(candidates), "budget_target": plan.budget.target_candidates,
                                 "budget_unit": "candidates"},
        stage: {"dataset_id": training_dataset_id,
                "path": f"private/canonical/{stage}.jsonl", "content_sha256": training_hash,
                "record_count": len(training_rows), "budget_count": len(training_rows),
                "budget_target": plan.training_budget,
                "budget_unit": "preference_pairs" if stage == "preference" else "examples"},
    }
    processing = ProcessingReport(
        build_id=build_id,
        created_at=created_at,
        processing_protocol="targeted-synthetic-data-build-v1",
        source={"repo": source.repo, "config": source.config, "revision": source.revision or None},
        source_acquisition={"mode": "llm_generate", "provider_adapter": profile.adapter,
                            "model": profile.model,
                            "revision_status": "available" if profile.revision else "unavailable"},
        tokenizer={"repo": tokenizer_spec.repo, "revision": tokenizer_spec.revision, "backend": tokenizer.name},
        benchmark_screening={"enabled": True, "index_id": policy_export["contamination_index_id"],
                             "index_sha256": policy_export["contamination_index_sha256"]},
        config=config,
        config_sha256=config_sha256,
        stop_reason=stop_reason,
        processing_steps=process_steps,
        outputs=report_outputs,
        limitations=["候选与canonical训练文本属于 private 数据，不进入 UI artifact roots",
                     "未配置不可变 provider revision 时不能精确复现远端模型行为",
                     "费用为供应方 usage 与用户费率形成的估算；未知费用不按零处理"],
    )
    reports_dir = root / "reports"
    write_report(reports_dir / "processing_report.json", processing.model_dump(mode="json", by_alias=True))
    processing_path = reports_dir / "processing_report.json"
    processing_ref = ReportReference(path="../reports/processing_report.json",
                                     sha256=sha256_file(processing_path), report_id=build_id)

    validation_summary = validation_summary or {}
    validation_keys = {
        "candidate_count", "valid_format", "invalid_format", "identity_mismatch",
        "truncated", "over_length", "answer_consistency_mismatch", "answer_unparseable",
    }
    sanitized_validation = {
        key: value for key, value in validation_summary.items()
        if key in validation_keys and isinstance(value, (int, float, bool, type(None)))
    }
    allowed_generation_keys = (
        "request_count", "successful_candidates", "target_candidate_count", "shortfall", "stop_reason",
        "cache_reused_count", "cache_recovered_count", "network_used", "execution_status", "ledger",
    )
    sanitized_generation = {key: generation_summary[key] for key in allowed_generation_keys
                            if key in generation_summary}
    cost_report = {
        "schema_version": "1.0",
        "plan_id": plan.plan_id,
        "provider_profile_id": profile.profile_id,
        "provider_adapter": profile.adapter,
        "model": profile.model,
        "model_revision": profile.revision,
        "model_revision_status": "available" if profile.revision else "unavailable",
        "generation": sanitized_generation,
        "candidate_validation": sanitized_validation,
        "policy_export": {
            "stage": stage,
            "policy_id": policy_export["policy_id"],
            "selected_count": policy_export["selected_count"],
            "assessment_counts": policy_export.get("assessment_counts", {}),
            "rejected": policy_export.get("rejected", {}),
        },
        "private_response_body_included": False,
        "limitations": ["估算费用不是账单或供应商硬上限", "没有 usage/费率时 cost_status 保持 unknown"],
    }
    write_report(reports_dir / "generation_validation_cost.json", cost_report)

    quality = assess_dataset(
        training_rows,
        stage=stage,
        dataset_id=training_dataset_id,
        content_sha256=training_hash,
        build_id=build_id,
        tokenizer_name=tokenizer.name,
        overlap=0,
    )
    quality.run_id = quality_run_id
    quality_path = reports_dir / "quality" / f"{stage}.json"
    write_report(quality_path, quality.model_dump(mode="json", by_alias=True))
    quality_ref = ReportReference(path=f"../reports/quality/{stage}.json",
                                 sha256=sha256_file(quality_path), report_id=quality.quality_report_id)

    candidate_manifest = DatasetManifest(
        dataset_id=candidate_dataset_id,
        stage="synthetic_candidates",
        source=source,
        tokenizer=tokenizer_spec,
        prompt_template_id=f"sha256:{plan.prompt_template_sha256}",
        protocol_versions={"candidate_schema": "1.0", "generation": "1.0"},
        sample_count=len(candidate_rows),
        hashes={"canonical_records_sha256": candidate_hash},
        processing_report=processing_ref,
        lineage=DatasetLineage(
            parents=[plan.training_seed.dataset_id], produced_by_run_id=build_run_id,
            transform={"name": "llm_generate_v1", "version": "1", "config_sha256": config_sha256},
            source_evidence={"repo": source.repo, "config": source.config,
                             "revision": source.revision or None,
                             "revision_status": "available" if profile.revision else "unavailable",
                             **provider_identity_evidence(profile_id=profile.profile_id, model=profile.model,
                                                          revision=profile.revision, adapter=profile.adapter)},
            record_provenance="record", count_unit="candidates", content_sha256=candidate_hash,
            processing_report_id=build_id,
        ),
        metadata={"canonical_path": "private/candidates.jsonl", "visibility": "private", "plan_id": plan.plan_id},
    )
    training_manifest = DatasetManifest(
        dataset_id=training_dataset_id,
        stage=stage,
        source=source,
        tokenizer=tokenizer_spec,
        prompt_template_id=f"sha256:{plan.prompt_template_sha256}",
        protocol_versions={"policy_export": "1.0", "candidate_schema": "1.0"},
        sample_count=len(training_rows),
        hashes={"canonical_records_sha256": training_hash},
        processing_report=processing_ref,
        quality_report=quality_ref,
        lineage=DatasetLineage(
            parents=[candidate_dataset_id], produced_by_run_id=build_run_id,
            transform={"name": "candidate_policy_export_v1", "version": "1",
                       "config_sha256": policy_export["policy_config_sha256"]},
            source_evidence={"repo": source.repo, "config": source.config,
                             "revision": source.revision or None,
                             "revision_status": "available" if profile.revision else "unavailable",
                             **provider_identity_evidence(profile_id=profile.profile_id, model=profile.model,
                                                          revision=profile.revision, adapter=profile.adapter)},
            record_provenance="record", count_unit="preference_pairs" if stage == "preference" else "examples",
            content_sha256=training_hash, processing_report_id=build_id,
            quality_report_ids=[quality.quality_report_id],
        ),
        metadata={"canonical_path": f"private/canonical/{stage}.jsonl",
                  "usage": "exploratory", "visibility": "private", "plan_id": plan.plan_id,
                  "data_policy": policy_export["policy_id"]},
    )
    manifests_dir = root / "manifests"
    save_manifest(manifests_dir / "synthetic-candidates-v1.yaml", candidate_manifest)
    save_manifest(manifests_dir / f"{stage}-targeted-v1.yaml", training_manifest)

    candidate_ref = _ref(candidate_path, root, candidate_dataset_id, "dataset", "candidate")
    training_ref = _ref(training_path, root, training_dataset_id, "dataset", "train")
    processing_artifact = _ref(processing_path, root, build_id, "processing_report", "processing")
    quality_ref_artifact = _ref(quality_path, root, quality.quality_report_id, "quality_report", "assessment")
    build_run = RunRecord(
        run_id=build_run_id, run_type="data_build", status="completed", run_label="targeted-synthetic",
        started_at=created_at, completed_at=datetime.now(timezone.utc).isoformat(),
        inputs=[ArtifactRef(id=plan.training_seed.dataset_id, kind="dataset", digest={
            "algorithm": "sha256", "value": plan.training_seed.content_sha256, "status": "available",
        }, role="train_seed")],
        outputs=[candidate_ref, training_ref, processing_artifact],
    )
    quality_run = RunRecord(
        run_id=quality_run_id, run_type="quality_assess", status="completed", run_label=f"targeted-{stage}",
        started_at=quality.created_at, completed_at=quality.created_at,
        inputs=[training_ref.model_copy(update={"role": "assessed"})],
        outputs=[quality_ref_artifact],
    )
    write_report(root / "runs" / build_run_id / "lineage.json", build_run.model_dump(mode="json"))
    write_report(root / "runs" / quality_run_id / "lineage.json", quality_run.model_dump(mode="json"))
    return {
        "build_id": build_id,
        "data_build_run_id": build_run_id,
        "quality_assess_run_id": quality_run_id,
        "candidate_dataset_id": candidate_dataset_id,
        "training_dataset_id": training_dataset_id,
        "candidate_manifest": str(manifests_dir / "synthetic-candidates-v1.yaml"),
        "training_manifest": str(manifests_dir / f"{stage}-targeted-v1.yaml"),
        "processing_report": str(processing_path),
        "generation_validation_cost_report": str(reports_dir / "generation_validation_cost.json"),
        "quality_report": str(quality_path),
        "candidate_rows": len(candidate_rows),
        "training_rows": len(training_rows),
        "private_root": str(private),
        "public_response_bodies_included": False,
        "model_revision_status": "available" if profile.revision else "unavailable",
        **provider_identity_evidence(profile_id=profile.profile_id, model=profile.model,
                                     revision=profile.revision, adapter=profile.adapter),
    }
