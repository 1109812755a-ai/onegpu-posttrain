"""Fail-closed preflight and preparation for a registered targeted CPT dataset."""

from __future__ import annotations

import json
from pathlib import Path

from ..adapters.registered_dataset import adapt_registered_dataset
from ..lineage import sha256_file
from .cpt_corpus import validate_cpt_corpus_jsonl
from .manifest import load_manifest, resolve_canonical_path, validate_manifest_evidence


def _local_base(model: str | Path, revision: str | None) -> tuple[Path, str]:
    supplied = Path(model).expanduser()
    if supplied.is_dir():
        resolved = supplied.resolve(strict=True)
    else:
        if not revision:
            raise ValueError("Base model 不是本地目录，且没有 pinned revision；不会联网下载")
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError as exc:
            raise ValueError("huggingface_hub 不可用，无法只读检查 Base 模型缓存") from exc
        config_file = try_to_load_from_cache(str(model), "config.json", revision=revision)
        if not isinstance(config_file, str):
            raise ValueError(f"pinned Base 配置未在本地缓存：{model}@{revision}；不会联网下载")
        resolved = Path(config_file).parent.resolve(strict=True)
    config_path = resolved / "config.json"
    if not config_path.is_file():
        raise ValueError("Base 模型缺少本地 config.json")
    if not any((resolved / name).is_file() for name in (
        "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json",
    )):
        raise ValueError("Base 模型缺少本地权重或权重索引；不会联网下载")
    return resolved, sha256_file(config_path)


def inspect_registered_cpt_dataset(manifest_path: str | Path) -> dict:
    """Validate registered CPT identity, evidence, content digest, and train role only."""
    supplied = Path(manifest_path).expanduser()
    if supplied.is_symlink():
        raise ValueError("CPT manifest path cannot be a symbolic link")
    manifest_file = supplied.resolve(strict=True)
    manifest = load_manifest(manifest_file)
    if manifest.stage != "cpt" or manifest.lineage is None or manifest.processing_report is None:
        raise ValueError("registered dataset must be a complete CPT artifact with lineage")
    evidence_errors = validate_manifest_evidence(manifest_file)
    if evidence_errors:
        raise ValueError("CPT manifest evidence invalid: " + "; ".join(evidence_errors))
    canonical = resolve_canonical_path(manifest_file, manifest)
    digest = sha256_file(canonical)
    if digest != manifest.hashes.get("canonical_records_sha256"):
        raise ValueError("CPT canonical content hash does not match manifest")
    with canonical.open("rb") as stream:
        corpus_check = validate_cpt_corpus_jsonl(stream, expected_role="train_seed")
    if not corpus_check.valid:
        raise ValueError("CPT training corpus role/schema invalid: " + json.dumps(corpus_check.report(), ensure_ascii=False))
    return {
        "status": "pass", "dataset_id": manifest.dataset_id,
        "dataset_sha256": digest, "record_count": manifest.sample_count,
        "manifest_path": str(manifest_file),
        "quality_report_id": manifest.quality_report.report_id if manifest.quality_report else None,
        "holdout_role_isolated": True,
    }


def inspect_local_base_model(model: str | Path, revision: str | None) -> dict:
    """Inspect pinned base-model files without network access or model loading."""
    model_path, config_sha = _local_base(model, revision)
    return {"status": "pass", "model_path": str(model_path), "config_sha256": config_sha,
            "model_revision": revision, "downloads_performed": False}


def preflight_registered_cpt(
    manifest_path: str | Path,
    *,
    model: str | Path,
    revision: str | None = None,
) -> dict:
    """Verify CPT manifest, reports, hashes, train role, and local Base weights."""
    dataset = inspect_registered_cpt_dataset(manifest_path)
    base = inspect_local_base_model(model, revision)
    return {
        "status": "pass", **dataset, "model_path": base["model_path"],
        "model_config_sha256": base["config_sha256"], "model_revision": revision,
        "downloads_performed": False,
    }


def prepare_registered_cpt_training(
    manifest_path: str | Path,
    *,
    model: str | Path,
    revision: str | None,
    adapter_output: str | Path,
) -> dict:
    """Preflight first, then create an immutable trainer-native CPT data view."""
    evidence = preflight_registered_cpt(manifest_path, model=model, revision=revision)
    adapter = adapt_registered_dataset(manifest_path, adapter_output)
    if adapter["dataset_id"] != evidence["dataset_id"] or adapter["canonical_sha256"] != evidence["dataset_sha256"]:
        raise ValueError("registered CPT adapter input identity changed after preflight")
    native_path = Path(adapter["output_dir"]) / adapter["native_path"]
    return {"preflight": evidence, "adapter": adapter, "dataset": str(native_path)}
