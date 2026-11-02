"""Derive a smaller SFT length cohort from an existing immutable dataset."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..adapters.ms_swift import adapt_sft, validate_native_record
from ..lineage import ArtifactRef, DatasetLineage, Digest, RunRecord, new_run_id, sha256_file, sha256_json
from ..reporting.data_report import write_report
from .evidence import ProcessingReport, ReportReference, step
from .io import read_jsonl, write_jsonl
from .manifest import DatasetManifest, load_manifest, save_manifest, validate_manifest_evidence


def derive_sft_length_subset(manifest: str | Path, *, max_length: int, output: str | Path) -> dict[str, Any]:
    """Filter by recorded rendered length, emitting immutable child data and full lineage."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    manifest_path = Path(manifest).expanduser().resolve(strict=True)
    parent = load_manifest(manifest_path)
    if parent.stage != "sft" or not parent.lineage:
        raise ValueError("parent must be a lineage-aware SFT dataset manifest")
    parent_revision = (parent.source.revision if parent.source else None)
    if parent_revision is None and parent.origin:
        parent_revision = parent.origin.revision
    parent_issues = validate_manifest_evidence(manifest_path)
    if parent_issues:
        raise ValueError("parent dataset evidence is invalid: " + "; ".join(parent_issues))

    parent_root = manifest_path.parent.parent.resolve(strict=True)
    canonical_path = (parent_root / str(parent.metadata.get("canonical_path", ""))).resolve(strict=True)
    try:
        canonical_path.relative_to(parent_root)
    except ValueError as exc:
        raise ValueError("parent canonical dataset path escapes its artifact root") from exc
    parent_digest = sha256_file(canonical_path)
    if parent.hashes.get("canonical_records_sha256") != parent_digest:
        raise ValueError("parent canonical dataset digest mismatch")
    parent_rows = list(read_jsonl(canonical_path))
    if len(parent_rows) != parent.sample_count:
        raise ValueError("parent record count does not match its manifest")

    selected: list[dict[str, Any]] = []
    rejected = 0
    for row in parent_rows:
        length = row.get("metadata", {}).get("rendered_tokens")
        if not isinstance(length, int) or length < 0:
            raise ValueError(f"record {row.get('id')} has no valid measured rendered_tokens")
        if length <= max_length:
            selected.append(row)
        else:
            rejected += 1
    if not selected:
        raise ValueError("no SFT records fit the requested max_length")

    config = {
        "protocol": "sft-rendered-length-subset-v1",
        "parent_dataset_id": parent.dataset_id,
        "parent_content_sha256": parent_digest,
        "max_rendered_tokens": max_length,
        "selection_rule": "record metadata.rendered_tokens <= max_rendered_tokens; no truncation",
    }
    config_sha = sha256_json(config)
    identity = sha256_json({**config, "selected_records_identity": [row["id"] for row in selected]})
    dataset_id = f"ds-sft-derived-{max_length}-{identity[:20]}"
    output_path = Path(output).expanduser().resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"derived dataset output already exists; choose a new immutable path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_base = Path.cwd()
    if temp_base.drive.lower() != output_path.drive.lower():
        temp_base = output_path.parent
    with tempfile.TemporaryDirectory(prefix=".sft-derive-", dir=temp_base) as temp_name:
        temp_root = Path(temp_name)
        canonical = temp_root / "canonical" / "sft.jsonl"
        native = temp_root / "native" / "sft.jsonl"
        write_jsonl(canonical, selected)
        native_rows = [adapt_sft(row) for row in selected]
        for row in native_rows:
            validate_native_record("sft", row)
        write_jsonl(native, native_rows)
        canonical_sha = sha256_file(canonical)
        native_sha = sha256_file(native)
        build_run_id = new_run_id("data_build", label=f"derive-sft-{max_length}")
        created = datetime.now(timezone.utc).isoformat()
        processing_config = {**config, "output_dataset_id": dataset_id}
        processing = ProcessingReport(
            build_id=build_run_id,
            created_at=created,
            processing_protocol="sft-rendered-length-subset-v1",
            source={
                "dataset_id": parent.dataset_id,
                "manifest_path": str(manifest_path),
                "revision": parent_revision,
                "content_sha256": parent_digest,
            },
            tokenizer=parent.tokenizer.model_dump(mode="json") if parent.tokenizer else {},
            benchmark_screening={"status": "inherited_not_reassessed", "parent_dataset_id": parent.dataset_id},
            config=processing_config,
            config_sha256=config_sha,
            source_acquisition={"mode": "local_parent_dataset", "network_used": False},
            stop_reason="source_exhausted",
            processing_steps=[step(
                1, "rendered_length_filter", "filter", parent.dataset_id, dataset_id,
                len(parent_rows), len(selected), "examples",
                rejected={"rendered_tokens_exceeds_limit": rejected},
                details={"max_rendered_tokens": max_length, "length_field": "metadata.rendered_tokens",
                         "truncation": "none"},
                limitations=["This subset operation changes eligibility by length only; semantic quality was not reassessed."],
            )],
            outputs={"sft": {
                "dataset_id": dataset_id,
                "canonical_path": "canonical/sft.jsonl",
                "native_path": "native/sft.jsonl",
                "sample_count": len(selected),
                "count_unit": "examples",
                "canonical_sha256": canonical_sha,
                "native_sha256": native_sha,
            }},
            limitations=[
                "Exploratory child cohort; its actual count is not an official Mini qualification gate.",
                "The parent exact-answer selection does not establish semantic reasoning correctness.",
                "Quality evidence was inherited as context, not reassessed for this child cohort.",
            ],
        )
        processing_path = temp_root / "reports" / "processing_report.json"
        write_report(processing_path, processing.model_dump(mode="json", by_alias=True))
        processing_sha = sha256_file(processing_path)
        processing_ref = ReportReference(
            path="../reports/processing_report.json", sha256=processing_sha, report_id=build_run_id,
        )
        metadata = {
            "usage": "exploratory",
            "provenance_type": "sft_length_subset",
            "max_rendered_tokens": max_length,
            "canonical_path": "canonical/sft.jsonl",
            "native_path": "native/sft.jsonl",
            "native_sha256": native_sha,
            "parent_manifest_path": str(manifest_path),
            "parent_dataset_id": parent.dataset_id,
            "quality_evidence": {
                "status": "not_reassessed",
                "parent_dataset_id": parent.dataset_id,
                "limitations": ["Length-only child selection does not independently verify reasoning quality."],
            },
        }
        child = DatasetManifest(
            dataset_id=dataset_id,
            stage="sft",
            source=parent.source,
            tokenizer=parent.tokenizer,
            prompt_template_id=parent.prompt_template_id,
            protocol_versions={**parent.protocol_versions, "length_subset": "sft-rendered-length-subset-v1"},
            split_seed=parent.split_seed,
            sample_count=len(selected),
            hashes={
                "canonical_records_sha256": canonical_sha,
                "native_records_sha256": native_sha,
                "parent_content_sha256": parent_digest,
                "selected_record_ids_sha256": sha256_json([row["id"] for row in selected]),
            },
            processing_report=processing_ref,
            lineage=DatasetLineage(
                parents=[parent.dataset_id],
                produced_by_run_id=build_run_id,
                transform={"name": config["protocol"], "version": "1", "config_sha256": config_sha,
                           "parameters": {"max_rendered_tokens": max_length, "truncation": "none"}},
                source_evidence={"parent_dataset_id": parent.dataset_id,
                                 "parent_manifest_sha256": sha256_file(manifest_path),
                                 "source_revision": parent_revision,
                                 "source_snapshot_sha256": parent.lineage.source_evidence.get("source_snapshot_sha256")},
                record_provenance="record",
                count_unit="examples",
                content_sha256=canonical_sha,
                processing_report_id=build_run_id,
            ),
            metadata=metadata,
        )
        registered_path = temp_root / "manifests" / "sft-derived-v1.yaml"
        save_manifest(registered_path, child)
        run = RunRecord(
            run_id=build_run_id,
            run_type="data_build",
            status="completed",
            lineage_status="complete",
            run_label=f"derive-sft-{max_length}",
            experiment_id=parent.dataset_id,
            started_at=created,
            completed_at=datetime.now(timezone.utc).isoformat(),
            inputs=[
                ArtifactRef(id=parent.dataset_id, kind="dataset", path=str(canonical_path),
                            digest=Digest(value=parent_digest, status="available"), role="parent_dataset"),
                ArtifactRef(id=f"file-{sha256_file(manifest_path)[:24]}", kind="file", path=str(manifest_path),
                            digest=Digest(value=sha256_file(manifest_path), status="available"), role="parent_manifest"),
            ],
            outputs=[
                ArtifactRef(id=dataset_id, kind="dataset", path=str(output_path / "canonical" / "sft.jsonl"),
                            digest=Digest(value=canonical_sha, status="available"), role="derived_dataset"),
                ArtifactRef(id=build_run_id, kind="processing_report",
                            path=str(output_path / "reports" / "processing_report.json"),
                            digest=Digest(value=processing_sha, status="available"), role="processing_report"),
            ],
            resolved_config_sha256=config_sha,
        )
        run_path = temp_root / "runs" / build_run_id / "lineage.json"
        run_path.parent.mkdir(parents=True)
        run_path.write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
        issues = validate_manifest_evidence(registered_path)
        if issues:
            raise ValueError("derived dataset manifest failed validation: " + "; ".join(issues))
        os.replace(temp_root, output_path)

    return {
        "status": "built",
        "dataset_id": dataset_id,
        "parent_dataset_id": parent.dataset_id,
        "manifest": str(output_path / "manifests" / "sft-derived-v1.yaml"),
        "canonical": str(output_path / "canonical" / "sft.jsonl"),
        "native": str(output_path / "native" / "sft.jsonl"),
        "processing_report": str(output_path / "reports" / "processing_report.json"),
        "run_id": build_run_id,
        "parent_count": len(parent_rows),
        "sample_count": len(selected),
        "length_rejected": rejected,
        "max_rendered_tokens": max_length,
        "canonical_sha256": canonical_sha,
        "native_sha256": native_sha,
    }
