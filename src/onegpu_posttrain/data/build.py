from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..lineage import ArtifactRef, DatasetLineage, RunRecord, new_run_id
from ..reporting.data_report import (
    write_markdown_report,
    write_processing_markdown_report,
    write_report,
)
from .decontaminate import BenchmarkIndex
from .evidence import ProcessingReport, ReportReference, assess_dataset
from .finemath import FineMathBuildConfig, FineMathBuilder
from .io import write_jsonl
from .manifest import (
    DatasetManifest,
    SourceSpec,
    TokenizerSpec,
    save_manifest,
    sha256_file,
)
from .openr1 import OpenR1BuildConfig, OpenR1Builder
from .tokenization import TokenizerBackend


def _hash_ids(ids: Iterable[str]) -> str:
    h = hashlib.sha256()
    for value in sorted(ids):
        h.update(value.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _budget_evidence(stage: str, report: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the exact stage budget count from the builder's authoritative report."""
    if stage in {"cpt", "cpt_holdout"}:
        key = "train" if stage == "cpt" else "holdout"
        section = report.get(key, {})
        count, target, unit = section.get("tokens"), section.get("token_budget"), "tokens"
    else:
        selected_key = "holdout" if stage == "eval" else stage
        count = report.get("selected", {}).get(selected_key)
        target = report.get("targets", {}).get(selected_key)
        unit = {
            "sft": "examples",
            "preference": "preference_pairs",
            "preference_eval": "preference_pairs",
            "grpo": "prompts",
            "eval": "examples",
        }.get(stage)
    if not isinstance(count, int) or count < 0 or not unit:
        raise ValueError(f"Builder report has no valid budget count for stage {stage!r}")
    if stage not in {"cpt", "cpt_holdout"} and count != len(rows):
        raise ValueError(f"Builder count for {stage!r} does not match emitted record count")
    if target is not None and (not isinstance(target, int) or target < 0):
        raise ValueError(f"Builder report has an invalid budget target for stage {stage!r}")
    return {"budget_count": count, "budget_target": target, "budget_unit": unit}


def _write_evidence(*, output_dir: Path, name: str, report: dict[str, Any],
                    config: dict[str, Any], datasets: dict[str, tuple[Path, list[dict[str, Any]]]],
                    source: SourceSpec, tokenizer: TokenizerSpec, tokenizer_name: str,
                    benchmark_index: BenchmarkIndex | None, benchmark_metadata: list[dict[str, Any]] | None,
                    overlap: int, source_acquisition: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Write report artifacts once; manifest paths stay compatible, identities follow content/config."""
    config_sha256 = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    build_id = f"build-{name}-{uuid4().hex}"
    build_run_id = new_run_id("data_build", label=name)
    started_at = datetime.now(timezone.utc).isoformat()
    benchmark_screening = {"enabled": benchmark_index is not None, "sources": benchmark_metadata or []}
    if benchmark_index is not None:
        benchmark_payload = sorted((text, sorted(ids)) for text, ids in benchmark_index.normalized_to_ids.items())
        benchmark_screening.update({"normalized_index_sha256": hashlib.sha256(json.dumps(benchmark_payload).encode()).hexdigest(),
                                    "shingle_n": benchmark_index.shingle_n, "indexed_items": len(benchmark_index.shingles_by_id)})
    outputs = {}
    for stage, (path, rows) in datasets.items():
        content_sha256 = sha256_file(path)
        identity = {"stage": stage, "source": source.model_dump(), "tokenizer": tokenizer.model_dump(),
                    "tokenizer_backend": tokenizer_name, "config_sha256": config_sha256, "content_sha256": content_sha256,
                    "processing_protocol": report["protocol"],
                    "benchmark_screening": {k: v for k, v in benchmark_screening.items() if k != "sources"}}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        output = {"dataset_id": f"ds-{name}-{stage}-{digest}", "path": path.relative_to(output_dir).as_posix(),
                  "content_sha256": content_sha256, "record_count": len(rows),
                  **_budget_evidence(stage, report, rows)}
        if stage in {"cpt", "cpt_holdout"}:
            role_counts: dict[str, int] = {}
            for row in rows:
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                role = metadata.get("data_role")
                role_counts[role] = role_counts.get(role, 0) + 1
            output["data_role_counts"] = role_counts
        outputs[stage] = output
    processing = ProcessingReport(
        build_id=build_id, created_at=datetime.now(timezone.utc).isoformat(), processing_protocol=report["protocol"],
        source=source.model_dump(), source_acquisition=source_acquisition or {"mode": "caller_supplied"},
        tokenizer={**tokenizer.model_dump(), "backend": tokenizer_name},
        benchmark_screening=benchmark_screening,
        config=config, config_sha256=config_sha256, stop_reason=report["stop_reason"],
        processing_steps=report["processing_steps"], outputs=outputs,
        limitations=["Step summaries cover the processed source prefix, not the full upstream dataset",
                     "Aggregate branch order describes a DAG, not a serial CPT/SFT/alignment training chain",
                     "Code revision was not captured; source/config/output identities are recorded explicitly"],
    )
    processing_path = output_dir / "reports" / "processing_report.json"
    payload = processing.model_dump(mode="json", by_alias=True)
    write_report(processing_path, payload)
    write_processing_markdown_report(processing_path.with_suffix(".md"), payload)
    processing_ref = ReportReference(path="../reports/processing_report.json", sha256=sha256_file(processing_path), report_id=build_id)
    evidence = {}
    failures = []
    build_outputs: list[ArtifactRef] = [
        ArtifactRef(id=item["dataset_id"], kind="dataset", path=item["path"],
                    digest={"algorithm": "sha256", "value": item["content_sha256"], "status": "available"}, role=stage)
        for stage, item in outputs.items()
    ]
    build_outputs.append(ArtifactRef(id=build_id, kind="processing_report", path="reports/processing_report.json",
                                    digest={"algorithm": "sha256", "value": processing_ref.sha256, "status": "available"}))
    for stage, (_, rows) in datasets.items():
        quality_run_id = new_run_id("quality_assess", label=stage)
        quality = assess_dataset(rows, stage=stage, dataset_id=outputs[stage]["dataset_id"],
                                 content_sha256=outputs[stage]["content_sha256"], build_id=build_id,
                                 tokenizer_name=tokenizer_name, overlap=overlap)
        quality.run_id = quality_run_id
        quality_path = output_dir / "reports" / "quality" / f"{stage}.json"
        if quality.gate.summary_status == "FAIL":
            failures.append(stage)
        quality_payload = quality.model_dump(mode="json", by_alias=True)
        write_report(quality_path, quality_payload)
        write_markdown_report(quality_path.with_suffix(".md"), f"{stage} Static Data Quality", quality_payload)
        quality_ref = ReportReference(path=f"../reports/quality/{stage}.json",
                                      sha256=sha256_file(quality_path), report_id=quality.quality_report_id)
        quality_run = RunRecord(
            run_id=quality_run_id, run_type="quality_assess", status="completed", run_label=stage,
            started_at=quality.created_at, completed_at=quality.created_at,
            inputs=[ArtifactRef(id=outputs[stage]["dataset_id"], kind="dataset", path=outputs[stage]["path"],
                                digest={"algorithm": "sha256", "value": outputs[stage]["content_sha256"],
                                        "status": "available"}, role="assessed")],
            outputs=[ArtifactRef(id=quality.quality_report_id, kind="quality_report",
                                 path=f"reports/quality/{stage}.json", role="assessment")],
        )
        write_report(output_dir / "runs" / quality_run_id / "lineage.json", quality_run.model_dump(mode="json"))
        manifest_count_unit = {
            "cpt": "chunks", "cpt_holdout": "chunks", "sft": "examples",
            "preference": "preference_pairs", "preference_eval": "preference_pairs",
            "grpo": "prompts", "eval": "examples",
        }[stage]
        evidence[stage] = {
            "dataset_id": outputs[stage]["dataset_id"], "processing_report": processing_ref,
            "quality_report": quality_ref,
            "lineage": DatasetLineage(
                parents=[], produced_by_run_id=build_run_id,
                transform={"name": f"build_{name}_artifacts", "version": report["protocol"], "config_sha256": config_sha256},
                source_evidence={"repo": source.repo, "config": source.config, "revision": source.revision or None,
                                 "revision_status": "available" if source.revision else "unavailable"},
                record_provenance="record" if any(
                    r.get("metadata", {}).get("source_uuid")
                    or r.get("metadata", {}).get("source_record_id")
                    or r.get("metadata", {}).get("source_row_id")
                    for r in rows
                ) else "dataset_only",
                count_unit=manifest_count_unit, content_sha256=outputs[stage]["content_sha256"],
                processing_report_id=build_id, quality_report_ids=[quality.quality_report_id],
            ),
        }
    build_run = RunRecord(
        run_id=build_run_id, run_type="data_build", status="completed", run_label=name,
        started_at=started_at, completed_at=datetime.now(timezone.utc).isoformat(),
        inputs=[ArtifactRef(id=source.repo, kind="file", role="source")], outputs=build_outputs,
    )
    write_report(output_dir / "runs" / build_run_id / "lineage.json", build_run.model_dump(mode="json"))
    report["build_id"] = build_id
    report["data_build_run_id"] = build_run_id
    report["quality_gate_failures"] = failures
    report["dataset_ids"] = {stage: output["dataset_id"] for stage, output in outputs.items()}
    report["processing_report"] = {"path": "reports/processing_report.json", "sha256": processing_ref.sha256}
    return evidence


def build_openr1_artifacts(
    rows: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    tokenizer: TokenizerBackend,
    config: OpenR1BuildConfig,
    source_repo: str,
    source_config: str | None,
    source_revision: str,
    tokenizer_repo: str,
    tokenizer_revision: str,
    benchmark_index: BenchmarkIndex | None = None,
    benchmark_metadata: list[dict[str, Any]] | None = None,
    source_acquisition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    canonical = output_dir / "canonical"
    reports = output_dir / "reports"
    manifests = output_dir / "manifests"
    canonical.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    builder = OpenR1Builder(tokenizer, config, benchmark_index)
    result = builder.build(rows)
    result.report["benchmark_decontamination"] = benchmark_metadata or []
    raw_rows = int(result.report.get("counters", {}).get("raw_rows", 0))
    result.report["retention"] = {
        "sft_per_raw": len(result.sft) / raw_rows if raw_rows else None,
        "preference_per_raw": len(result.preference) / raw_rows if raw_rows else None,
        "preference_eval_per_raw": len(result.preference_eval) / raw_rows if raw_rows else None,
        "grpo_per_raw": len(result.grpo) / raw_rows if raw_rows else None,
        "holdout_per_raw": len(result.holdout) / raw_rows if raw_rows else None,
    }
    result.report["retention_semantics"] = "Legacy build yields per processed raw row; units differ by stage, so these are not quality scores"

    files = {
        "sft": canonical / "sft.jsonl",
        "preference": canonical / "preference.jsonl",
        "preference_eval": canonical / "preference_eval.jsonl",
        "grpo": canonical / "grpo.jsonl",
        "eval": canonical / "internal_holdout.jsonl",
        "contamination": reports / "contamination.jsonl",
    }
    write_jsonl(files["sft"], result.sft)
    write_jsonl(files["preference"], result.preference)
    write_jsonl(files["preference_eval"], result.preference_eval)
    write_jsonl(files["grpo"], result.grpo)
    write_jsonl(files["eval"], result.holdout)
    write_jsonl(files["contamination"], result.contaminated)

    stage_rows = {
        "sft": result.sft,
        "preference": result.preference,
        "preference_eval": result.preference_eval,
        "grpo": result.grpo,
        "eval": result.holdout,
    }
    source = SourceSpec(repo=source_repo, config=source_config, revision=source_revision)
    tokenizer_spec = TokenizerSpec(repo=tokenizer_repo, revision=tokenizer_revision)
    evidence = _write_evidence(
        output_dir=output_dir, name="openr1", report=result.report, config=asdict(config),
        datasets={stage: (files[stage], rows) for stage, rows in stage_rows.items()},
        source=source, tokenizer=tokenizer_spec, tokenizer_name=tokenizer.name,
        benchmark_index=benchmark_index, benchmark_metadata=benchmark_metadata,
        overlap=result.report["overlap"]["train_holdout"],
        source_acquisition=source_acquisition,
    )
    write_report(reports / "openr1_data_report.json", result.report)
    write_markdown_report(reports / "openr1_data_report.md", "OpenR1 Smoke Data Report", result.report)
    for stage, stage_data in stage_rows.items():
        stage_path = files[stage]
        ids = [r.get("metadata", {}).get("source_uuid") or r.get("id") for r in stage_data]
        manifest = DatasetManifest(
            **evidence[stage],
            stage=stage,
            source=source,
            tokenizer=tokenizer_spec,
            prompt_template_id=config.prompt_template_id if stage != "eval" else None,
            protocol_versions={
                "normalization": "v1",
                "decontamination": "v1",
                "verifiability_gate": "v1",
                "length_gate": "v1",
                "split": "v1",
            },
            split_seed=config.seed,
            sample_count=len(stage_data),
            hashes={
                "source_row_ids_sha256": _hash_ids(str(x) for x in ids),
                "canonical_records_sha256": sha256_file(stage_path),
            },
            metadata={"tokenizer_backend": tokenizer.name, "build_protocol": "openr1-build-v1",
                      "canonical_path": stage_path.relative_to(output_dir).as_posix(),
                      "benchmark_decontamination": benchmark_metadata or [],
                      "source_acquisition": source_acquisition or {"mode": "caller_supplied"}},
        )
        save_manifest(manifests / f"{stage}-smoke-v1.yaml", manifest)
    return result.report


def build_finemath_artifacts(
    rows: Iterable[dict[str, Any]],
    *,
    output_dir: str | Path,
    tokenizer: TokenizerBackend,
    config: FineMathBuildConfig,
    source_repo: str,
    source_config: str | None,
    source_revision: str,
    tokenizer_repo: str,
    tokenizer_revision: str,
    benchmark_index: BenchmarkIndex | None = None,
    benchmark_metadata: list[dict[str, Any]] | None = None,
    source_acquisition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    canonical = output_dir / "canonical"
    reports = output_dir / "reports"
    manifests = output_dir / "manifests"
    canonical.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    builder = FineMathBuilder(tokenizer, config, benchmark_index)
    result = builder.build(rows)
    result.report["benchmark_contamination_verification"] = benchmark_metadata or []
    train_path = canonical / "cpt_train.jsonl"
    holdout_path = canonical / "cpt_holdout.jsonl"
    write_jsonl(train_path, result.train)
    write_jsonl(holdout_path, result.holdout)
    write_jsonl(reports / "finemath_contamination.jsonl", result.contamination)

    source = SourceSpec(repo=source_repo, config=source_config, revision=source_revision)
    tokenizer_spec = TokenizerSpec(repo=tokenizer_repo, revision=tokenizer_revision)
    evidence = _write_evidence(
        output_dir=output_dir, name="finemath", report=result.report, config=asdict(config),
        datasets={"cpt": (train_path, result.train), "cpt_holdout": (holdout_path, result.holdout)},
        source=source, tokenizer=tokenizer_spec, tokenizer_name=tokenizer.name,
        benchmark_index=benchmark_index, benchmark_metadata=benchmark_metadata,
        overlap=result.report["document_overlap"],
        source_acquisition=source_acquisition,
    )
    write_report(reports / "finemath_data_report.json", result.report)
    write_markdown_report(reports / "finemath_data_report.md", "FineMath Smoke Data Report", result.report)

    for stage, path, rows_ in (("cpt", train_path, result.train), ("cpt_holdout", holdout_path, result.holdout)):
        doc_ids = [str(r.get("metadata", {}).get("document_id") or r["id"]) for r in rows_]
        data_role_counts = {"train_seed" if stage == "cpt" else "final_holdout": len(rows_)}
        manifest = DatasetManifest(
            **evidence[stage],
            stage=stage,
            source=source,
            tokenizer=tokenizer_spec,
            protocol_versions={"document_split": "v1", "contamination_verification": "v1", "chunking": "v1",
                               "record_identity": "v2", "data_role": "cpt-data-role-v1"},
            split_seed=42,
            sample_count=len(rows_),
            hashes={
                "source_document_ids_sha256": _hash_ids(doc_ids),
                "canonical_records_sha256": sha256_file(path),
            },
            metadata={"tokenizer_backend": tokenizer.name, "build_protocol": "finemath-build-v3",
                      "data_role_protocol": "cpt-data-role-v1",
                      "data_role_counts": data_role_counts,
                      "canonical_path": path.relative_to(output_dir).as_posix(),
                      "benchmark_contamination_verification": benchmark_metadata or [],
                      "source_acquisition": source_acquisition or {"mode": "caller_supplied"}},
        )
        save_manifest(manifests / f"{stage}-smoke-v1.yaml", manifest)
    return result.report
