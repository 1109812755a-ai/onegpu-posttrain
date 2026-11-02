"""Register an existing SFT length-calibration candidate in dataset lineage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..lineage import ArtifactRef, DatasetLineage, Digest, RunRecord, new_run_id, sha256_json
from .io import read_jsonl
from .manifest import DatasetManifest, SourceSpec, TokenizerSpec, load_manifest, save_manifest, sha256_file, validate_manifest_evidence
from .sft_calibration import _local_snapshot_revision


def _read_calibration_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid SFT calibration manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.1":
        raise ValueError("SFT calibration manifest must use schema_version 1.1")
    if payload.get("stage") != "sft" or payload.get("status") != "exploratory":
        raise ValueError("only exploratory SFT calibration candidates may be registered")
    for key in ("dataset_id", "source", "tokenizer", "sampling", "eligibility", "counts", "hashes", "paths"):
        if not isinstance(payload.get(key), dict) and key != "dataset_id":
            raise ValueError(f"SFT calibration manifest is missing object field: {key}")
    if not isinstance(payload.get("dataset_id"), str) or not payload["dataset_id"].strip():
        raise ValueError("SFT calibration manifest is missing dataset_id")
    return payload


def _candidate_files(manifest_path: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    paths = payload["paths"]
    root = manifest_path.parent
    canonical = (root / str(paths.get("canonical", ""))).resolve(strict=True)
    native = (root / str(paths.get("native", ""))).resolve(strict=True)
    for candidate in (canonical, native):
        try:
            candidate.relative_to(root.resolve(strict=True))
        except ValueError as exc:
            raise ValueError("calibration candidate paths must remain inside their artifact directory") from exc
        if not candidate.is_file():
            raise ValueError(f"calibration candidate is not a file: {candidate}")
    expected = payload["hashes"]
    for label, candidate in (("canonical_sha256", canonical), ("native_sha256", native)):
        actual = sha256_file(candidate)
        if expected.get(label) != actual:
            raise ValueError(f"calibration {label} mismatch for {candidate}")
    return canonical, native


def _verify_source_revision(payload: dict[str, Any]) -> Path:
    source = payload["source"]
    source_path = Path(str(source.get("local_path", ""))).expanduser()
    if not source_path.is_dir():
        raise ValueError("pinned local source snapshot is unavailable; refusing to register unverifiable provenance")
    expected_revision = source.get("revision")
    actual_revision = _local_snapshot_revision(source_path)
    if not expected_revision or actual_revision != expected_revision:
        raise ValueError(
            f"source revision mismatch: manifest={expected_revision!r}, local snapshot={actual_revision!r}"
        )
    if not source.get("snapshot_sha256"):
        raise ValueError("calibration manifest has no source snapshot digest")
    return source_path.resolve(strict=True)


def _existing_registration(manifest_path: Path, payload: dict[str, Any], canonical: Path,
                           native: Path) -> dict[str, Any] | None:
    registered_path = manifest_path.parent / "manifests" / "sft-calibration-v1.yaml"
    if not registered_path.exists():
        return None
    registered = load_manifest(registered_path)
    expected_canonical = sha256_file(canonical)
    expected_native = sha256_file(native)
    if (registered.dataset_id != payload["dataset_id"]
            or registered.hashes.get("canonical_records_sha256") != expected_canonical
            or registered.metadata.get("native_sha256") != expected_native):
        raise ValueError("existing calibration registration conflicts with current candidate identity or content")
    expected_metadata = {
        "provenance_type": "sft_length_calibration",
        "max_rendered_tokens": int(payload["eligibility"]["max_rendered_tokens"]),
        "length_eligibility": payload["eligibility"],
        "sampling": payload["sampling"],
        "quality_evidence": payload.get("quality_evidence", {}),
    }
    for key, value in expected_metadata.items():
        existing = registered.metadata.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"existing calibration registration metadata conflicts: {key}")
        registered.metadata[key] = value
    artifact_root = registered_path.parent.parent.resolve(strict=True)
    try:
        registered.metadata["canonical_path"] = canonical.resolve(strict=True).relative_to(artifact_root).as_posix()
    except ValueError as exc:
        raise ValueError("calibration canonical file must remain within its registration artifact root") from exc
    # This only fills metadata derived from the already-verified immutable calibration
    # manifest; record content, identity, and historical JSONL files are not rewritten.
    save_manifest(registered_path, registered)
    issues = validate_manifest_evidence(registered_path)
    if issues:
        raise ValueError("existing calibration registration is invalid: " + "; ".join(issues))
    run_id = registered.lineage.produced_by_run_id if registered.lineage else None
    run_path = manifest_path.parent / "runs" / str(run_id) / "lineage.json"
    if not run_id or not run_path.is_file():
        raise ValueError("existing calibration registration has no producing data_build Run")
    run = RunRecord.model_validate_json(run_path.read_text(encoding="utf-8"))
    if run.run_type != "data_build" or run.run_id != run_id or not any(
        item.id == payload["dataset_id"] for item in run.outputs
    ):
        raise ValueError("existing calibration registration Run does not produce the registered dataset")
    return {"status": "reused", "dataset_id": payload["dataset_id"], "manifest": str(registered_path),
            "run_id": run_id, "canonical_sha256": expected_canonical, "native_sha256": expected_native,
            "sample_count": registered.sample_count}


def register_sft_calibration_candidate(manifest: str | Path) -> dict[str, Any]:
    """Add canonical lineage records for an existing calibration candidate without rewriting its data."""
    manifest_path = Path(manifest).expanduser().resolve(strict=True)
    payload = _read_calibration_manifest(manifest_path)
    canonical, native = _candidate_files(manifest_path, payload)
    source_path = _verify_source_revision(payload)
    record_count = sum(1 for _ in read_jsonl(canonical))
    native_count = sum(1 for _ in read_jsonl(native))
    expected_count = payload["counts"].get("verification_selected_training_examples")
    if not isinstance(expected_count, int) or record_count != expected_count or native_count != expected_count:
        raise ValueError(
            f"candidate record count mismatch: manifest={expected_count}, canonical={record_count}, native={native_count}"
        )

    reused = _existing_registration(manifest_path, payload, canonical, native)
    if reused is not None:
        return reused

    artifact_root = manifest_path.parent
    manifests_dir = artifact_root / "manifests"
    if (manifests_dir / "sft-calibration-v1.yaml").exists():
        raise ValueError("calibration manifest path is already occupied")

    now = datetime.now(timezone.utc).isoformat()
    calibration_sha = sha256_file(manifest_path)
    canonical_sha = sha256_file(canonical)
    native_sha = sha256_file(native)
    transform = {
        "name": "register_sft_length_calibration_candidate",
        "version": "1",
        "config_sha256": sha256_json({
            "eligibility": payload["eligibility"],
            "sampling": payload["sampling"],
            "tokenizer": payload["tokenizer"],
        }),
        "calibration_id": payload.get("calibration_id"),
        "calibration_manifest_sha256": calibration_sha,
    }
    run_id = new_run_id("data_build", label=f"register-{payload['dataset_id']}")
    source = payload["source"]
    tokenizer = payload["tokenizer"]
    run = RunRecord(
        run_id=run_id,
        run_type="data_build",
        status="completed",
        lineage_status="complete",
        run_label=f"register-{payload['dataset_id']}",
        experiment_id=payload.get("calibration_id"),
        started_at=now,
        completed_at=now,
        inputs=[
            ArtifactRef(id=f"file-{calibration_sha[:24]}", kind="file", path=str(manifest_path),
                        digest=Digest(value=calibration_sha, status="available"), role="calibration_manifest"),
            ArtifactRef(id=f"file-{canonical_sha[:24]}", kind="file", path=str(canonical),
                        digest=Digest(value=canonical_sha, status="available"), role="canonical_records"),
            ArtifactRef(id=f"file-{native_sha[:24]}", kind="file", path=str(native),
                        digest=Digest(value=native_sha, status="available"), role="native_records"),
            ArtifactRef(id=f"source-{source['revision']}", kind="file", path=str(source_path),
                        revision=source["revision"], role="pinned_source_snapshot"),
        ],
        outputs=[ArtifactRef(id=payload["dataset_id"], kind="dataset", path=str(canonical),
                             digest=Digest(value=canonical_sha, status="available"), role="registered_dataset")],
        recipe_path="calibration manifest registration",
        resolved_config_sha256=sha256_json(transform),
    )
    dataset_manifest = DatasetManifest(
        dataset_id=payload["dataset_id"],
        stage="sft",
        source=SourceSpec(repo=source["name"], revision=source["revision"]),
        tokenizer=TokenizerSpec(repo=tokenizer["repo"], revision=tokenizer["revision"]),
        protocol_versions={
            "sft_length_calibration": str(payload["schema_version"]),
            "answer_verification": str(payload["eligibility"].get("answer_verification_mode", "unknown")),
        },
        split_seed=int(payload["sampling"]["seed"]),
        sample_count=record_count,
        hashes={
            "canonical_records_sha256": canonical_sha,
            "source_row_ids_sha256": str(payload["sampling"]["source_row_ids_sha256"]),
            "source_snapshot_sha256": str(source["snapshot_sha256"]),
        },
        lineage=DatasetLineage(
            parents=[],
            produced_by_run_id=run_id,
            transform=transform,
            source_evidence={
                "source_repo": source["name"],
                "source_revision": source["revision"],
                "source_snapshot_sha256": source["snapshot_sha256"],
                "calibration_manifest_sha256": calibration_sha,
            },
            record_provenance="dataset_only",
            count_unit="examples",
            content_sha256=canonical_sha,
        ),
        metadata={
            "usage": "exploratory",
            "provenance_type": "sft_length_calibration",
            "max_rendered_tokens": int(payload["eligibility"]["max_rendered_tokens"]),
            "native_path": "native/sft.jsonl",
            "canonical_path": "canonical/sft.jsonl",
            "native_sha256": native_sha,
            "length_eligibility": payload["eligibility"],
            "sampling": payload["sampling"],
            "quality_evidence": payload.get("quality_evidence", {}),
            "calibration_manifest_path": str(manifest_path),
            "source_local_path": str(source_path),
        },
    )

    run_dir = artifact_root / "runs" / run_id
    manifests_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    registered_path = manifests_dir / "sft-calibration-v1.yaml"
    try:
        save_manifest(registered_path, dataset_manifest)
        (run_dir / "lineage.json").write_text(run.model_dump_json(indent=2) + "\n", encoding="utf-8")
        issues = validate_manifest_evidence(registered_path)
        if issues:
            raise ValueError("new calibration registration failed validation: " + "; ".join(issues))
    except Exception:
        registered_path.unlink(missing_ok=True)
        (run_dir / "lineage.json").unlink(missing_ok=True)
        run_dir.rmdir()
        raise

    return {"status": "registered", "dataset_id": payload["dataset_id"], "manifest": str(registered_path),
            "run_id": run_id, "canonical_sha256": canonical_sha, "native_sha256": native_sha,
            "sample_count": record_count}
