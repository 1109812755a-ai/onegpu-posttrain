"""Register reviewed synthetic SFT output as an immutable local dataset."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..lineage import ArtifactRef, DatasetLineage, RunRecord, new_run_id
from .evidence import ProcessingReport, ReportReference, assess_dataset, step
from .importer import MAX_IMPORT_BYTES, MAX_IMPORT_RECORDS, validate_canonical_jsonl
from .io import read_jsonl
from .manifest import DatasetManifest, sha256_file
from .registry import DataSource


def _inside(root: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("generation artifact path must be workspace-relative")
    candidate = root / rel
    current = candidate
    while current != root:
        if current.is_symlink():
            raise ValueError("generation artifact path cannot contain symlinks")
        current = current.parent
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
    if not resolved.is_file():
        raise ValueError("generation artifact is not a file")
    return resolved


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def preview_generated_sft_dataset(workspace: str | Path, state: dict[str, Any], *, limit: int = 3) -> dict[str, Any]:
    """Return a small allowlisted preview for explicit human review before registration."""
    if not 1 <= limit <= 5:
        raise ValueError("preview limit must be between 1 and 5")
    root = Path(workspace).resolve(strict=True)
    result = state.get("result")
    if state.get("status") not in {"completed", "partial", "cancelled"} or not isinstance(result, dict):
        raise ValueError("generation has no completed reviewable output")
    if result.get("execution_id") != state.get("execution_id") or result.get("bundle_id") != state.get("bundle_id"):
        raise ValueError("generation execution identity mismatch")
    refs = result.get("generation_artifacts")
    if not isinstance(refs, dict) or not isinstance(refs.get("training_path"), str):
        raise TypeError("generation output is empty or unavailable")
    canonical = _inside(root, refs["training_path"])
    if canonical.stat().st_size > MAX_IMPORT_BYTES:
        raise ValueError("generation output exceeds review size limit")
    with canonical.open("rb") as stream:
        validation = validate_canonical_jsonl(stream, "sft", max_bytes=MAX_IMPORT_BYTES,
                                               max_records=MAX_IMPORT_RECORDS)
    if not validation.valid:
        raise ValueError("generation output is empty or schema-invalid")
    records = []
    for row in read_jsonl(canonical):
        records.append({
            "id": str(row["id"])[:200],
            "raw_problem": str(row["raw_problem"])[:2000],
            "messages": [{"role": message["role"], "content": str(message["content"])[:4000]}
                         for message in row["messages"][:8]],
        })
        if len(records) >= limit:
            break
    return {"execution_id": state["execution_id"], "dataset_id": result.get("training_dataset_id"),
            "count": validation.record_count, "canonical_sha256": sha256_file(canonical),
            "records": records, "truncated": validation.record_count > len(records),
            "semantic_correctness": "not_independently_verified"}


def register_generated_sft_dataset(workspace: str | Path, state: dict[str, Any]) -> dict[str, Any]:
    """Copy an explicitly reviewed generation output into the registered dataset catalog.

    Inputs are taken only from the persisted generation result, never from browser paths.
    Re-registering identical bytes reuses the existing immutable artifact.
    """
    root = Path(workspace).resolve(strict=True)
    result = state.get("result")
    if state.get("status") not in {"completed", "partial", "cancelled"} or not isinstance(result, dict):
        raise ValueError("only a completed or partial generation with a reviewable result can be registered")
    if state.get("registered_dataset_id"):
        return {"dataset_id": state["registered_dataset_id"], "reused": True}
    if result.get("bundle_id") != state.get("bundle_id") or result.get("execution_id") != state.get("execution_id"):
        raise ValueError("generation execution identity mismatch")
    generation_artifacts = result.get("generation_artifacts")
    if not isinstance(generation_artifacts, dict):
        raise TypeError("generation did not produce a reviewable training manifest")
    manifest_rel = generation_artifacts.get("training_manifest")
    canonical_rel = generation_artifacts.get("training_path")
    cost_rel = generation_artifacts.get("generation_validation_cost_report")
    if not all(isinstance(item, str) for item in (manifest_rel, canonical_rel, cost_rel)):
        raise ValueError("generation output references are incomplete")
    source_manifest_path = _inside(root, manifest_rel)
    source_manifest = yaml.safe_load(source_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source_manifest, dict) or source_manifest.get("stage") != "sft":
        raise ValueError("generation output is not an SFT dataset")
    source_manifest_root = source_manifest_path.parent.parent.resolve(strict=True)
    source_canonical = _inside(root, canonical_rel)
    source_canonical.relative_to(source_manifest_root)
    canonical_digest = sha256_file(source_canonical)
    if source_manifest.get("hashes", {}).get("canonical_records_sha256") != canonical_digest:
        raise ValueError("generation output digest does not match its manifest")

    with source_canonical.open("rb") as stream:
        validation = validate_canonical_jsonl(
            stream, "sft", max_bytes=MAX_IMPORT_BYTES, max_records=MAX_IMPORT_RECORDS,
        )
    if not validation.valid or validation.record_count < 1:
        raise ValueError("empty or schema-invalid generated output cannot be registered")
    if validation.record_count != result.get("usable_records"):
        raise ValueError("generation result count does not match canonical SFT output")
    rows = list(read_jsonl(source_canonical))
    cost = json.loads(_inside(root, cost_rel).read_text(encoding="utf-8"))
    source_name = str(cost.get("model") or "未注明模型")
    profile_id = str(cost.get("provider_profile_id") or "unknown")
    source_evidence = (source_manifest.get("lineage") or {}).get("source_evidence") or {}
    provider_model = str(cost.get("model") or source_evidence.get("provider_model") or "") or None
    provider_revision = cost.get("model_revision", source_evidence.get("provider_revision"))
    provider_adapter = cost.get("provider_adapter", source_evidence.get("provider_adapter"))
    identity_evidence = {
        "provider_profile_id": profile_id,
        "provider_adapter": provider_adapter,
        "provider_model": provider_model,
        "provider_revision": provider_revision,
        "provider_revision_status": "available" if provider_revision else "unavailable",
        "provider_identity_status": "profile_declared_not_independently_verified",
        "registered_model_association_status": "not_proven",
        "registered_model_id": None,
        "registered_model_sha256": None,
    }
    generation_mode = str(source_evidence.get("generation_mode") or "targeted")
    source_dataset_id = result.get("source_dataset_id")
    targeted_stage_plan = result.get("targeted_stage_plan") if isinstance(result.get("targeted_stage_plan"), dict) else {}
    targeted_plan_id = targeted_stage_plan.get("plan_id")
    targeted_plan_version = targeted_stage_plan.get("plan_version")
    prompt_only = generation_mode == "prompt_only_sft"
    content_id = hashlib.sha256(f"sft:{canonical_digest}".encode()).hexdigest()[:24]
    dataset_id = f"ds-sft-generated-{content_id}"
    target_root = root / "data" / "processed" / "model-generated" / dataset_id
    if target_root.is_symlink():
        raise ValueError("registered dataset target cannot be a symbolic link")
    if target_root.exists():
        existing_manifest = target_root / "manifests" / "sft-generated-v1.yaml"
        try:
            existing = yaml.safe_load(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise FileExistsError("dataset ID already exists but cannot be verified") from exc
        if (existing.get("dataset_id") != dataset_id
                or existing.get("hashes", {}).get("canonical_records_sha256") != canonical_digest):
            raise FileExistsError("dataset ID collision; existing immutable artifact was not modified")
        return {"dataset_id": dataset_id, "count": existing.get("sample_count"),
                "canonical_sha256": canonical_digest, "manifest_path": str(existing_manifest.relative_to(root).as_posix()),
                "reused": True}

    parent = target_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id}-", dir=parent))
    try:
        canonical = staging / "canonical" / "sft.jsonl"
        canonical.parent.mkdir(parents=True)
        shutil.copyfile(source_canonical, canonical)
        copied_digest = sha256_file(canonical)
        if copied_digest != canonical_digest:
            raise OSError("generated dataset copy digest mismatch")

        now = datetime.now(timezone.utc).isoformat()
        build_run_id = new_run_id("data_build", label="generated-sft-register")
        quality_run_id = new_run_id("quality_assess", label="generated-sft")
        processing_id = f"processing-generated-{content_id}"
        quality = assess_dataset(rows, stage="sft", dataset_id=dataset_id,
                                 content_sha256=canonical_digest, build_id=processing_id,
                                 tokenizer_name="not_recomputed_during_registration", overlap=None)
        quality.run_id = quality_run_id

        processing_config = {"bundle_id": state["bundle_id"], "execution_id": state["execution_id"],
                             "source_dataset_id": source_dataset_id,
                             "generation_mode": generation_mode,
                             "intermediate_dataset_id": source_manifest.get("dataset_id"),
                             "registered_dataset_id": dataset_id, **identity_evidence,
                             "model": source_name,
                             "targeted_stage_plan_id": targeted_stage_plan.get("plan_id"),
                             "targeted_stage_plan_version": targeted_stage_plan.get("plan_version")}
        processing_config_sha = hashlib.sha256(json.dumps(
            processing_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()).hexdigest()
        processing_report = ProcessingReport(
            build_id=processing_id, created_at=now,
            processing_protocol="reviewed-synthetic-sft-registration-v1",
            source={"repo": f"model_generated:{source_name}", "config": profile_id,
                    "revision": provider_revision},
            tokenizer={"repo": "not_recomputed", "revision": "not_recomputed"},
            benchmark_screening={"enabled": False, "reason": "Generation plan screening evidence is retained in its source build."},
            config=processing_config, config_sha256=processing_config_sha,
            source_acquisition={"mode": "llm_generate", **identity_evidence,
                                "model": source_name, "license_status": "not_provided",
                                "generation_execution_id": state["execution_id"]},
            stop_reason="targets_met" if result.get("shortfall", 0) == 0 else "request_budget_exhausted",
            processing_steps=[step(
                1, "reviewed_synthetic_sft_registration_v1", "export",
                ("prompt:topic_skill_difficulty" if prompt_only else
                 f"dataset:{source_manifest.get('dataset_id')}"), f"dataset:{dataset_id}",
                validation.record_count, validation.record_count, "examples", output_unit="examples",
                details={"generation_execution_id": state["execution_id"], "bundle_id": state["bundle_id"]},
                limitations=["登记时再次验证 canonical schema；不代表语义正确性"],
            )],
            outputs={"sft": {"dataset_id": dataset_id, "path": "canonical/sft.jsonl",
                             "content_sha256": canonical_digest, "record_count": validation.record_count,
                             "budget_count": validation.record_count, "budget_target": result.get("desired_usable_records"),
                             "budget_unit": "examples"}},
            limitations=["生成模型与许可由 provider 配置声明，未独立验证"],
        ).model_dump(mode="json", by_alias=True)
        processing_path = staging / "reports" / "processing_report.json"
        _write_json(processing_path, processing_report)
        quality_path = staging / "reports" / "quality" / "sft.json"
        _write_json(quality_path, quality.model_dump(mode="json", by_alias=True))
        processing_ref = ReportReference(path="../reports/processing_report.json",
                                         sha256=sha256_file(processing_path), report_id=processing_id)
        quality_ref = ReportReference(path="../reports/quality/sft.json",
                                      sha256=sha256_file(quality_path), report_id=quality.quality_report_id)

        source = DataSource(
            source_id=f"src-generated-{content_id}", kind="model_generated", name=source_name,
            revision=provider_revision, revision_status="known" if provider_revision else "unknown",
            license_status="not_provided", declared_by="provider",
        )
        manifest = DatasetManifest(
            dataset_id=dataset_id, stage="sft", origin=source,
            sample_count=validation.record_count,
            protocol_versions={"canonical_schema": "stage-canonical-v1", "registration": "generated-sft-v1"},
            hashes={"canonical_records_sha256": canonical_digest},
            processing_report=processing_ref, quality_report=quality_ref,
            lineage=DatasetLineage(
                parents=[] if prompt_only else [str(source_dataset_id)], produced_by_run_id=build_run_id,
                transform={"name": "reviewed-synthetic-sft-registration", "version": "1",
                           "config_sha256": hashlib.sha256(
                               f"{state['bundle_id']}:{state['execution_id']}:{canonical_digest}".encode()).hexdigest()},
                source_evidence={"kind": "model_generated", **identity_evidence,
                                 "model": source_name, "license_status": "not_provided",
                                 "generation_mode": generation_mode,
                                 "provider_profile_sha256": (source_manifest.get("lineage", {})
                                                             .get("source_evidence", {})
                                                             .get("provider_profile_sha256")),
                                 "requested_target_count": result.get("desired_usable_records"),
                                 "generated_candidate_count": result.get("candidate_count"),
                                 "prompt_template_id": source_evidence.get("prompt_template_id"),
                                 "prompt_template_sha256": source_evidence.get("prompt_template_sha256"),
                                 "output_protocol_sha256": source_evidence.get("output_protocol_sha256"),
                                 "rendered_prompt_sha256": source_evidence.get("rendered_prompt_sha256"),
                                 "topic_sha256": source_evidence.get("topic_sha256"),
                                 "skill_sha256": source_evidence.get("skill_sha256"),
                                 "constraints_sha256": source_evidence.get("constraints_sha256"),
                                 "difficulty": source_evidence.get("difficulty"),
                                 "semantic_correctness": source_evidence.get("semantic_correctness",
                                     "not_independently_verified"),
                                 "generation_execution_id": state["execution_id"], "bundle_id": state["bundle_id"],
                                 "targeted_stage_plan_id": targeted_stage_plan.get("plan_id"),
                                 "targeted_stage_plan_version": targeted_stage_plan.get("plan_version")},
                record_provenance="record", count_unit="examples", content_sha256=canonical_digest,
                processing_report_id=processing_id, quality_report_ids=[quality.quality_report_id],
            ),
            metadata={"canonical_path": "canonical/sft.jsonl", "source_kind": "model_generated",
                      "quality_status": "report_only", "usage": "exploratory", "visibility": "registered",
                      "generation_mode": generation_mode,
                      "provider_profile_sha256": (source_manifest.get("lineage", {})
                                                  .get("source_evidence", {}).get("provider_profile_sha256")),
                      "generation_execution_id": state["execution_id"],
                      "generation_plan_id": state["bundle_id"],
                      "targeted_stage_plan_id": targeted_stage_plan.get("plan_id"),
                      "targeted_stage_plan_version": targeted_stage_plan.get("plan_version")},
        )
        manifests = staging / "manifests"
        from .manifest import save_manifest
        save_manifest(manifests / "sft-generated-v1.yaml", manifest)

        dataset_ref = ArtifactRef(id=dataset_id, kind="dataset", path="canonical/sft.jsonl",
                                  digest={"value": canonical_digest, "status": "available"}, role="train")
        build_run = RunRecord(
            run_id=build_run_id, run_type="data_build", status="completed", run_label="generated-sft-register",
            targeted_plan_id=targeted_plan_id, targeted_plan_version=targeted_plan_version,
            started_at=now, completed_at=now,
            inputs=([ArtifactRef(id=str(source_manifest.get("dataset_id")),
                                 kind="file" if prompt_only else "dataset",
                                 path=canonical.relative_to(staging).as_posix() if prompt_only else None,
                                 role="reviewed_generation_output" if prompt_only else "generated_intermediate")]
                    + ([] if prompt_only else [ArtifactRef(id=str(source_dataset_id), kind="dataset", role="generation_seed")])
                    + [ArtifactRef(id=state["execution_id"], kind="file",
                                   path=f"runs/synthetic-workflow/{state['execution_id']}/state.json", role="generation_execution")]),
            outputs=[dataset_ref,
                     ArtifactRef(id=processing_id, kind="processing_report", path="reports/processing_report.json",
                                 digest={"value": processing_ref.sha256, "status": "available"})],
        )
        quality_run = RunRecord(
            run_id=quality_run_id, run_type="quality_assess", status="completed", run_label="generated-sft",
            targeted_plan_id=targeted_plan_id, targeted_plan_version=targeted_plan_version,
            started_at=quality.created_at, completed_at=quality.created_at,
            inputs=[dataset_ref.model_copy(update={"role": "assessed"})],
            outputs=[ArtifactRef(id=quality.quality_report_id, kind="quality_report", path="reports/quality/sft.json",
                                 digest={"value": quality_ref.sha256, "status": "available"})],
        )
        for run in (build_run, quality_run):
            run_dir = staging / "runs" / run.run_id
            run_dir.mkdir(parents=True)
            _write_json(run_dir / "lineage.json", run.model_dump(mode="json"))

        try:
            os.replace(staging, target_root)
        except OSError:
            if target_root.exists():
                raise FileExistsError("dataset registration raced with another immutable write")
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    manifest_rel = (target_root / "manifests" / "sft-generated-v1.yaml").relative_to(root).as_posix()
    return {"dataset_id": dataset_id, "count": validation.record_count,
            "canonical_sha256": canonical_digest, "manifest_path": manifest_rel, "reused": False}
