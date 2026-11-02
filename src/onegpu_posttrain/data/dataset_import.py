"""Atomic local registration of validated canonical JSONL uploads."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from ..lineage import ArtifactRef, DatasetLineage, RunRecord, new_run_id, sha256_json
from .evidence import ProcessingReport, ReportReference, assess_dataset, step
from .importer import ImportValidation, validate_canonical_jsonl
from .manifest import DatasetComposition, DatasetManifest, save_manifest, sha256_file
from .registry import DataSource, TrainingStage


class _HashingReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.digest = hashlib.sha256()

    def readline(self, size: int = -1) -> bytes:
        chunk = self.stream.readline(size)
        self.digest.update(chunk)
        return chunk


@dataclass(frozen=True)
class RegistrationResult:
    dataset_id: str
    stage: str
    record_count: int
    original_sha256: str
    canonical_sha256: str
    path: Path
    reused: bool
    validation: ImportValidation


def _safe_output_root(workspace: Path, output_root: Path) -> Path:
    workspace = workspace.resolve(strict=True)
    # Inspect the lexical path before resolving it: resolve() would hide a
    # symlink by returning its destination, which could be outside the workspace.
    candidate = Path(os.path.abspath(output_root))
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("dataset output root escapes workspace") from exc
    current = candidate
    while current != workspace:
        if current.is_symlink():
            raise ValueError("dataset output root cannot contain symbolic links")
        current = current.parent
    return candidate.resolve(strict=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_jsonl(
    stream: BinaryIO,
    *,
    workspace: str | Path,
    output_root: str | Path,
    stage: TrainingStage,
    origin: DataSource,
    registration_metadata: dict | None = None,
    identity_material_sha256: str | None = None,
    composition: DatasetComposition | None = None,
    provenance_file: str | Path | None = None,
    raw_source_bytes: bytes | None = None,
    source_format: str | None = None,
    converter_config: dict | None = None,
) -> RegistrationResult:
    """Validate, canonicalize and atomically persist a dataset under workspace."""
    generated_grpo_supported = (
        origin.kind == "model_generated" and stage == "grpo"
        and registration_metadata is not None
        and registration_metadata.get("reward_contract_verified") is True
        and isinstance(registration_metadata.get("reward_contract_id"), str)
        and isinstance(registration_metadata.get("reward_contract_protocol_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", registration_metadata["reward_contract_protocol_sha256"]) is not None
    )
    generated_supported = (
        origin.kind == "model_generated" and stage in {"cpt", "preference"}
    ) or generated_grpo_supported
    generated_parent_refs_supported = (
        origin.kind == "model_generated" and stage == "cpt" and generated_supported
        and isinstance(registration_metadata, dict)
        and isinstance(registration_metadata.get("parent_dataset_refs"), list)
    )
    mixture_supported = (origin.kind == "dataset_mixture" and stage in {"cpt", "sft", "preference", "grpo"}
                         and isinstance(registration_metadata, dict)
                         and registration_metadata.get("composition_protocol") == "dataset-mixture-v1"
                         and isinstance(registration_metadata.get("parent_dataset_ids"), list)
                         and len(registration_metadata["parent_dataset_ids"]) >= 2
                         and composition is not None and provenance_file is not None)
    derived_grpo_supported = (
        origin.kind == "dataset_derived" and stage == "grpo"
        and isinstance(registration_metadata, dict)
        and registration_metadata.get("derivation_protocol") == "grpo-openr1-seed-v1"
        and registration_metadata.get("reward_contract_verified") is True
        and isinstance(registration_metadata.get("reward_contract_id"), str)
        and isinstance(registration_metadata.get("reward_contract_protocol_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", registration_metadata["reward_contract_protocol_sha256"]) is not None
        and isinstance(registration_metadata.get("parent_dataset_refs"), list)
        and len(registration_metadata["parent_dataset_refs"]) > 0
    )
    cpt_role_repair_supported = (
        origin.kind == "dataset_derived" and stage == "cpt"
        and isinstance(registration_metadata, dict)
        and registration_metadata.get("derivation_protocol") == "cpt-role-repair-v1"
        and registration_metadata.get("role_source_protocol") == "finemath-build-v2"
        and registration_metadata.get("data_role_protocol") == "cpt-data-role-v1"
        and isinstance(registration_metadata.get("parent_dataset_refs"), list)
        and len(registration_metadata["parent_dataset_refs"]) == 1
    )
    if (origin.kind != "local_upload" and not generated_supported and not mixture_supported
            and not derived_grpo_supported and not cpt_role_repair_supported):
        raise ValueError("JSONL registration requires local_upload, a supported generated-data protocol, or a verified dataset mixture")
    if identity_material_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", identity_material_sha256):
        raise ValueError("identity_material_sha256 must be a lowercase SHA-256 digest")
    if composition is not None and (origin.kind != "dataset_mixture" or composition.stage != stage):
        raise ValueError("composition metadata must match a dataset_mixture artifact and stage")
    workspace_path = Path(workspace).expanduser().resolve(strict=True)
    registration_metadata = registration_metadata or {}
    parent_dataset_ids = registration_metadata.get("parent_dataset_ids", [])
    if (not isinstance(parent_dataset_ids, list)
            or any(not isinstance(value, str) or not value.strip() for value in parent_dataset_ids)
            or len(parent_dataset_ids) != len(set(parent_dataset_ids))):
        raise ValueError("parent_dataset_ids must be a unique list of non-empty dataset IDs")
    if derived_grpo_supported:
        refs = registration_metadata["parent_dataset_refs"]
        if (any(not isinstance(ref, dict) or set(ref) != {"dataset_id", "content_sha256"}
                or not isinstance(ref.get("dataset_id"), str) or not ref["dataset_id"].strip()
                or not isinstance(ref.get("content_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", ref["content_sha256"]) is None for ref in refs)
                or [ref["dataset_id"] for ref in refs] != parent_dataset_ids):
            raise ValueError("derived GRPO parent refs must match parent_dataset_ids and contain canonical SHA-256 digests")
    if generated_parent_refs_supported:
        refs = registration_metadata["parent_dataset_refs"]
        if (any(not isinstance(ref, dict) or set(ref) != {"dataset_id", "content_sha256"}
                or not isinstance(ref.get("dataset_id"), str) or not ref["dataset_id"].strip()
                or not isinstance(ref.get("content_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", ref["content_sha256"]) is None for ref in refs)
                or [ref["dataset_id"] for ref in refs] != parent_dataset_ids):
            raise ValueError("generated CPT parent refs must match parent_dataset_ids and contain canonical SHA-256 digests")
    if cpt_role_repair_supported:
        refs = registration_metadata["parent_dataset_refs"]
        if (any(not isinstance(ref, dict) or set(ref) != {"dataset_id", "content_sha256"}
                or not isinstance(ref.get("dataset_id"), str) or not ref["dataset_id"].strip()
                or not isinstance(ref.get("content_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", ref["content_sha256"]) is None for ref in refs)
                or [ref["dataset_id"] for ref in refs] != parent_dataset_ids):
            raise ValueError("CPT role repair parent refs must match parent_dataset_ids and contain canonical SHA-256 digests")
        role_counts = registration_metadata.get("data_role_counts")
        if (not isinstance(role_counts, dict) or set(role_counts) != {"train_seed"}
                or not isinstance(role_counts["train_seed"], int) or role_counts["train_seed"] < 1):
            raise ValueError("CPT role repair requires a positive train_seed count")
    lineage_metadata = {
        key: registration_metadata[key]
        for key in (
            "targeted_plan_id", "targeted_plan_version", "parent_dataset_ids", "provider_profile_id",
            "generation_protocol", "verification_scope", "reward_contract_verified", "reward_contract_id",
            "reward_contract_protocol_sha256", "reward_contract_implementation_sha256",
            "benchmark_index_id", "benchmark_index_sha256", "tokenizer_id", "composition_protocol",
            "composition_config_sha256", "composition_strategy", "source_record_counts",
            "cross_source_duplicate_ids",
            "derivation_protocol", "derivation_config_sha256", "derivation_seed", "requested_count",
            "selected_source_record_ids_sha256", "parent_dataset_refs", "reward_contract_version",
            "reference_verification_scope",
            "role_source_protocol", "role_assignment", "data_role_protocol", "data_role_counts",
            "factuality",
            "generation_execution_id", "generation_candidate_sha256", "generation_request_count",
            "generation_bundle_id", "candidate_set_sha256", "provider_model", "provider_revision",
            "provider_adapter", "provider_revision_status", "provider_identity_status",
            "registered_model_association_status", "registered_model_id", "registered_model_sha256",
            "ordering_evidence_protocol", "ordering_evidence_status",
        )
        if key in registration_metadata
    }
    if "targeted_plan_id" not in lineage_metadata and "targeted_stage_plan_id" in registration_metadata:
        lineage_metadata["targeted_plan_id"] = registration_metadata["targeted_stage_plan_id"]
    if "targeted_plan_version" not in lineage_metadata and "targeted_stage_plan_version" in registration_metadata:
        lineage_metadata["targeted_plan_version"] = registration_metadata["targeted_stage_plan_version"]
    root = _safe_output_root(workspace_path, Path(output_root).expanduser())
    root.mkdir(parents=True, exist_ok=True)
    # Re-resolve after creation to defend against a symlinked parent being swapped in.
    root = _safe_output_root(workspace_path, root)

    with tempfile.TemporaryDirectory(prefix=".dataset-import-", dir=root) as staging_name:
        staging = Path(staging_name)
        canonical_path = staging / "canonical" / f"{stage}.jsonl"
        canonical_path.parent.mkdir(parents=True)
        with canonical_path.open("wb") as canonical:
            reader = _HashingReader(stream)

            def write_record(record: dict) -> None:
                encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                canonical.write(encoded + b"\n")

            validation = validate_canonical_jsonl(reader, stage, on_record=write_record)
            canonical.flush()
            os.fsync(canonical.fileno())
            original_sha256 = hashlib.sha256(raw_source_bytes).hexdigest() if raw_source_bytes is not None else reader.digest.hexdigest()

        if not validation.valid:
            raise ValueError(json.dumps(validation.report(), ensure_ascii=False))

        provenance_digest = None
        converter_config = converter_config or {}
        converter_config_sha256 = sha256_json(converter_config) if source_format else None
        if source_format:
            if stage not in {"sft", "eval"} or raw_source_bytes is None:
                raise ValueError("source format conversion is only supported for SFT/evaluation uploads with retained source bytes")
            if len(raw_source_bytes) + canonical_path.stat().st_size > 512 * 1024 * 1024:
                raise ValueError("raw and canonical artifacts exceed the 512 MiB combined storage limit")
            raw_path = staging / "raw" / "source.jsonl"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(raw_source_bytes)
            if _sha256_file(raw_path) != original_sha256:
                raise ValueError("retained source digest mismatch")
        if composition is not None and provenance_file is not None:
            provenance_source = Path(provenance_file)
            if provenance_source.is_symlink() or not provenance_source.is_file():
                raise ValueError("composition provenance sidecar is unavailable")
            provenance_digest = _sha256_file(provenance_source)
            if provenance_digest != composition.provenance_sha256:
                raise ValueError("composition provenance sidecar digest mismatch")
            provenance_target = staging / composition.provenance_path
            provenance_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(provenance_source, provenance_target)
            if _sha256_file(provenance_target) != provenance_digest:
                raise ValueError("composition provenance sidecar changed during registration")

        canonical_digest = _sha256_file(canonical_path)
        source_label = {"local_upload": "upload", "model_generated": "generated",
                        "dataset_mixture": "mixture", "dataset_derived": "derived"}.get(origin.kind, "source")
        identity_material = converter_config_sha256 if source_format else identity_material_sha256
        identity_digest = (canonical_digest if identity_material is None else hashlib.sha256(
            f"{identity_material}:{canonical_digest}:{original_sha256 if source_format else ''}".encode("ascii")
        ).hexdigest())
        dataset_id = f"ds-{stage}-{source_label}-{identity_digest[:20]}"
        source = origin.model_copy(update={
            "source_id": f"source-{source_label}-{original_sha256[:20]}",
            "content_sha256": original_sha256,
        })
        build_id = f"build-{source_label}-{uuid4().hex}"
        build_run_id = new_run_id("data_build", label=f"{source_label}-{stage}")
        quality_run_id = new_run_id("quality_assess", label=f"{source_label}-{stage}")
        now = datetime.now(timezone.utc).isoformat()
        config = {
            "import_protocol": "source-format-import-v1" if source_format else "canonical-jsonl-import-v1",
            "stage": stage,
            "schema": "stage-canonical-v1",
            "normalization": "pydantic-json-mode-sort-keys-v1",
            **({"source_format": source_format, "converter_version": converter_config.get("converter_version"),
                "converter_config": converter_config, "converter_config_sha256": converter_config_sha256,
                "source_record_count": converter_config.get("source_record_count", validation.record_count),
                "output_record_count": validation.record_count} if source_format else {}),
            "lineage_context": lineage_metadata,
            "composition": composition.model_dump(mode="json") if composition is not None else None,
        }
        config_sha256 = sha256_json(config)

        # Read the bounded canonical artifact for stage-specific static quality
        # checks. This reports unavailable semantic checks honestly as not_measured.
        records = []
        with canonical_path.open("r", encoding="utf-8") as canonical:
            for line in canonical:
                records.append(json.loads(line))

        processing = ProcessingReport(
            build_id=build_id,
            created_at=now,
            processing_protocol="source-format-import-v1" if source_format else "canonical-jsonl-import-v1",
            source=source.model_dump(mode="json"),
            source_acquisition={
                "mode": origin.kind,
                "original_sha256": original_sha256,
                "bytes_read": len(raw_source_bytes) if raw_source_bytes is not None else validation.bytes_read,
                **({"source_format": source_format, "source_sha256": original_sha256,
                    "canonical_sha256": canonical_digest, "converter_version": converter_config.get("converter_version"),
                    "converter_config_sha256": converter_config_sha256} if source_format else {}),
                "revision_status": source.revision_status,
                "license_status": source.license_status,
            },
            tokenizer={"status": "not_measured", "reason": "Import does not run a tokenizer"},
            benchmark_screening={"status": "not_measured", "reason": "Import does not run benchmark contamination screening"},
            config=config,
            config_sha256=config_sha256,
            stop_reason="source_exhausted",
            processing_steps=[step(
                1, "validate-and-canonicalize-jsonl", "transform",
                f"upload:sha256:{original_sha256}", f"canonical/{stage}.jsonl",
                validation.record_count, validation.record_count, "records",
                details={"schema": "stage-canonical-v1", "normalization": config["normalization"],
                         **({"source_format": source_format, "converter_version": converter_config.get("converter_version"),
                             "converter_config_sha256": converter_config_sha256,
                             "source_sha256": original_sha256, "canonical_sha256": canonical_digest} if source_format else {})},
                limitations=["Canonicalization validates schema and stable serialization; it does not clean semantic content"],
            )],
            outputs={stage: {
                "dataset_id": dataset_id,
                "path": f"canonical/{stage}.jsonl",
                "content_sha256": canonical_digest,
                "record_count": validation.record_count,
                "budget_count": validation.record_count,
                "budget_target": None,
                "budget_unit": {"cpt": "records", "sft": "examples", "preference": "preference_pairs",
                                 "grpo": "prompts", "eval": "evaluation_questions"}[stage],
                **({"data_role_counts": registration_metadata["data_role_counts"]}
                   if stage == "cpt" and "data_role_counts" in registration_metadata else {}),
            }},
            limitations=[
                "Import validation is structural; semantic correctness and benchmark contamination were not assessed",
                ("Source and license declarations are user-provided and are not independently verified"
                 if origin.kind == "local_upload" else
                 "Generated corpus factuality is not independently verified; license status is not applicable to provider output"
                 if origin.kind == "model_generated" else
                 "Derived dataset inherits parent source/license limitations; upstream reference labels are not independently verified"),
            ],
        )
        reports_dir = staging / "reports"
        quality_dir = reports_dir / "quality"
        manifests_dir = staging / "manifests"
        runs_dir = staging / "runs"
        quality_dir.mkdir(parents=True)
        manifests_dir.mkdir()
        processing_path = reports_dir / "processing_report.json"
        processing_path.write_text(
            json.dumps(processing.model_dump(mode="json", by_alias=True), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        processing_ref = ReportReference(
            path="../reports/processing_report.json", sha256=sha256_file(processing_path), report_id=build_id,
        )

        quality = assess_dataset(
            records, stage=stage, dataset_id=dataset_id, content_sha256=canonical_digest,
            build_id=build_id, tokenizer_name="not_measured", overlap=None,
        )
        if composition is not None:
            quality.summaries["mixture"] = {
                "strategy": composition.strategy,
                "parents": [parent.model_dump(mode="json") for parent in composition.parents],
                "source_record_counts": registration_metadata.get("source_record_counts", {}),
                "cross_source_duplicate_ids": registration_metadata.get("cross_source_duplicate_ids", 0),
                "provenance_sha256": provenance_digest,
            }
        quality.run_id = quality_run_id
        quality_path = quality_dir / f"{stage}.json"
        quality_path.write_text(
            json.dumps(quality.model_dump(mode="json", by_alias=True), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        quality_ref = ReportReference(
            path=f"../reports/quality/{stage}.json", sha256=sha256_file(quality_path),
            report_id=quality.quality_report_id,
        )
        quality_run = RunRecord(
            run_id=quality_run_id, run_type="quality_assess", status="completed", run_label=f"{source_label}-{stage}",
            targeted_plan_id=lineage_metadata.get("targeted_plan_id"),
            targeted_plan_version=lineage_metadata.get("targeted_plan_version"),
            started_at=quality.created_at, completed_at=quality.created_at,
            inputs=[ArtifactRef(id=dataset_id, kind="dataset", path=f"canonical/{stage}.jsonl", role="assessed",
                                digest={"value": canonical_digest, "status": "available"})],
            outputs=[ArtifactRef(id=quality.quality_report_id, kind="quality_report",
                                 path=f"reports/quality/{stage}.json", role="assessment",
                                 digest={"value": quality_ref.sha256, "status": "available"})],
        )
        (runs_dir / quality_run_id).mkdir(parents=True)
        (runs_dir / quality_run_id / "lineage.json").write_text(
            json.dumps(quality_run.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        build_run = RunRecord(
            run_id=build_run_id, run_type="data_build", status="completed", run_label=f"{source_label}-{stage}",
            targeted_plan_id=lineage_metadata.get("targeted_plan_id"),
            targeted_plan_version=lineage_metadata.get("targeted_plan_version"),
            started_at=now, completed_at=datetime.now(timezone.utc).isoformat(),
            inputs=[ArtifactRef(id=source.source_id, kind="file", role="derived_records",
                                digest={"value": original_sha256, "status": "available"})]
                   + [ArtifactRef(id=parent.dataset_id, kind="dataset", role="parent_dataset",
                                  digest={"value": parent.content_sha256, "status": "available"})
                      for parent in composition.parents] if composition is not None else
                   ([ArtifactRef(id=source.source_id, kind="file", role="derived_records",
                                digest={"value": original_sha256, "status": "available"})]
                   + [ArtifactRef(id=parent["dataset_id"], kind="dataset", role="parent_dataset",
                                  digest={"value": parent["content_sha256"], "status": "available"})
                      for parent in registration_metadata.get("parent_dataset_refs", [])]
                   if derived_grpo_supported or cpt_role_repair_supported or generated_parent_refs_supported else
                   [ArtifactRef(id=source.source_id, kind="file", path="raw/source.jsonl" if source_format else None,
                                role="uploaded_source",
                                digest={"value": original_sha256, "status": "available"})]),
            outputs=[
                ArtifactRef(id=dataset_id, kind="dataset", path=f"canonical/{stage}.jsonl", role=stage,
                            digest={"value": canonical_digest, "status": "available"}),
                ArtifactRef(id=build_id, kind="processing_report", path="reports/processing_report.json",
                            digest={"value": processing_ref.sha256, "status": "available"}),
            ],
        )
        (runs_dir / build_run_id).mkdir(parents=True)
        (runs_dir / build_run_id / "lineage.json").write_text(
            json.dumps(build_run.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        lineage = DatasetLineage(
            parents=parent_dataset_ids, produced_by_run_id=build_run_id,
            transform={"name": "dataset-mixture" if composition is not None else
                       "source-format-import" if source_format else
                       registration_metadata.get("derivation_protocol", "canonical-jsonl-import"),
                       "version": composition.protocol if composition is not None else
                       converter_config.get("converter_version", "1") if source_format else
                       registration_metadata.get("derivation_protocol", "1"),
                       "config_sha256": composition.config_sha256 if composition is not None else
                       converter_config_sha256 if source_format else
                       registration_metadata.get("derivation_config_sha256", config_sha256)},
            source_evidence={"source": source.model_dump(mode="json"), "original_sha256": original_sha256,
                             **({"source_format": source_format, "source_path": "raw/source.jsonl",
                                 "converter_version": converter_config.get("converter_version", "unknown"),
                                 "converter_config_sha256": converter_config_sha256,
                                 "canonical_sha256": canonical_digest} if source_format else {}),
                             "registration_context": lineage_metadata},
            record_provenance="record" if composition is not None else "dataset_only", count_unit={"cpt": "records", "sft": "examples",
                                                            "preference": "preference_pairs", "grpo": "prompts",
                                                            "eval": "evaluation_questions"}[stage],
            content_sha256=canonical_digest,
            processing_report_id=build_id, quality_report_ids=[quality.quality_report_id],
        )
        manifest = DatasetManifest(
            dataset_id=dataset_id, stage=stage, origin=source, sample_count=validation.record_count,
            protocol_versions={"canonical_schema": "stage-canonical-v1", "import": "source-format-import-v1" if source_format else "canonical-jsonl-import-v1",
                               **({"source_converter": converter_config.get("converter_version", "unknown")} if source_format else {}),
                               "quality": quality.quality_protocol,
                               **({"data_role": registration_metadata["data_role_protocol"]}
                                  if "data_role_protocol" in registration_metadata else {})},
            hashes={"original_file_sha256": original_sha256, "canonical_records_sha256": canonical_digest,
                    **({"source_file_sha256": original_sha256, "converter_config_sha256": converter_config_sha256} if source_format else {})},
            processing_report=processing_ref, quality_report=quality_ref, lineage=lineage, composition=composition,
            metadata={"canonical_path": f"canonical/{stage}.jsonl", "source_kind": origin.kind,
                      **({"source_path": "raw/source.jsonl", "source_format": source_format,
                          "converter_version": converter_config.get("converter_version", "unknown"),
                          "converter_config": converter_config, "converter_config_sha256": converter_config_sha256,
                          "source_record_count": converter_config.get("source_record_count", validation.record_count),
                          "canonical_record_count": validation.record_count} if source_format else {}),
                      "quality_status": quality.gate.summary_status, **(registration_metadata or {})},
        )
        manifest_name = (f"{stage}-user-upload-v1.yaml" if source_label == "upload" else
                         f"{stage}-model-generated-v1.yaml" if source_label == "generated" else
                         f"{stage}-dataset-mixture-v1.yaml" if source_label == "mixture" else
                         f"{stage}-dataset-derived-v1.yaml")
        save_manifest(manifests_dir / manifest_name, manifest)
        metadata = {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "stage": stage,
            "source": source.model_dump(mode="json"),
            "record_count": validation.record_count,
            "original_sha256": original_sha256,
            "canonical_sha256": canonical_digest,
            **({"source_format": source_format, "converter_version": converter_config.get("converter_version", "unknown"),
                "converter_config_sha256": converter_config_sha256, "source_path": "raw/source.jsonl"} if source_format else {}),
            "manifest_path": f"manifests/{manifest_name}",
        }
        (staging / "registration.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8",
        )

        target = root / dataset_id
        if target.is_symlink():
            raise ValueError("existing dataset target cannot be a symbolic link")
        if target.exists():
            existing_path = target / "registration.json"
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise FileExistsError("dataset ID already exists but cannot be safely verified") from exc
            if (existing.get("dataset_id") != dataset_id
                    or existing.get("canonical_sha256") != canonical_digest
                    or existing.get("stage") != stage):
                raise FileExistsError("dataset ID collision; existing artifact was not modified")
            return RegistrationResult(dataset_id, stage, validation.record_count, original_sha256,
                                      canonical_digest, target, True, validation)

        # Directory rename is atomic because staging and target share the same parent.
        try:
            os.replace(staging, target)
        except OSError:
            # A concurrent registration may have committed the same stable ID.
            # Never overwrite it; verify and reuse only an exact content match.
            if not target.is_dir() or target.is_symlink():
                raise
            try:
                existing = json.loads((target / "registration.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise FileExistsError("dataset ID already exists but cannot be safely verified") from exc
            if (existing.get("dataset_id") != dataset_id
                    or existing.get("canonical_sha256") != canonical_digest
                    or existing.get("stage") != stage):
                raise FileExistsError("dataset ID collision; existing artifact was not modified")
            return RegistrationResult(dataset_id, stage, validation.record_count, original_sha256,
                                      canonical_digest, target, True, validation)
        return RegistrationResult(dataset_id, stage, validation.record_count, original_sha256,
                                  canonical_digest, target, False, validation)
