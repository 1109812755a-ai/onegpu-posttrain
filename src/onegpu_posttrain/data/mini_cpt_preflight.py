"""Fail-closed local preflight for the versioned Mini CPT recipe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..lineage import sha256_directory, sha256_file
from .io import read_jsonl


class MiniCPTPreflightError(ValueError):
    """Raised when a Mini CPT input differs from its pinned local evidence."""


def _local_model_evidence(recipe: dict[str, Any], model: str | Path | None, root: Path) -> dict[str, Any]:
    model_config_path = Path(recipe["model_config"])
    if not model_config_path.is_absolute():
        model_config_path = Path.cwd() / model_config_path
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    configured = model_config["model"]
    repo, revision = str(configured["repo"]), str(configured["revision"])
    local: Path | None = None
    if model is not None:
        requested = Path(model)
        if requested.is_dir():
            local = requested.resolve()
        elif str(model) != repo:
            raise MiniCPTPreflightError(f"Mini CPT 只接受本地 pinned Base 快照或 recipe 中的 repo：{model}")
    if local is None:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError as exc:  # pragma: no cover - installed with Transformers
            raise MiniCPTPreflightError("huggingface_hub 不可用，无法只读检查 pinned Base 本地缓存。") from exc
        for filename in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"):
            cached = try_to_load_from_cache(repo, filename, revision=revision)
            if isinstance(cached, str) and Path(cached).is_file():
                local = Path(cached).parent.resolve()
                break
    if local is None:
        raise MiniCPTPreflightError(f"pinned Base 权重未在本机缓存（{repo}@{revision}）；不会联网下载或替换模型。")
    if not (local / "config.json").is_file():
        raise MiniCPTPreflightError(f"本地 Base 快照缺少 config.json：{local}")
    has_weights = any((local / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
    has_index = any((local / name).is_file() for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"))
    if not has_weights and not has_index:
        raise MiniCPTPreflightError(f"本地 pinned Base 快照缺少模型权重文件：{local}")
    if model is not None:
        parts = [part.lower() for part in local.parts]
        expected_cache_name = f"models--{repo.replace('/', '--')}".lower()
        if expected_cache_name not in parts or revision.lower() not in parts:
            raise MiniCPTPreflightError("本地模型目录无法证明属于 recipe 固定的 repo/revision。")
    model_json = json.loads((local / "config.json").read_text(encoding="utf-8"))
    if model_json.get("model_type") != "qwen3":
        raise MiniCPTPreflightError("本地 Base config.json 的 model_type 与 pinned Qwen3 不匹配。")
    return {"repo": repo, "revision": revision, "local_path": str(local),
            "directory_sha256": sha256_directory(local), "weights_present": True}


def preflight_mini_cpt(recipe: dict[str, Any], dataset: str | Path, *, root: str | Path = ".",
                       model: str | Path | None = None) -> dict[str, Any]:
    """Validate a Mini CPT dataset and return usage/review evidence without network access."""
    root_path = Path(root).resolve()
    model_evidence = _local_model_evidence(recipe, model, root_path)
    spec = recipe.get("dataset")
    if not isinstance(spec, dict):
        return {"status": "not_applicable", "reason": "recipe does not declare a Mini dataset"}

    def local_path(key: str) -> Path:
        raw = Path(str(spec.get(key, "")))
        return raw if raw.is_absolute() else root_path / raw

    native = Path(dataset)
    if not native.is_absolute():
        native = root_path / native
    expected_native = local_path("native")
    if native.resolve() != expected_native.resolve():
        raise MiniCPTPreflightError(f"指定的数据文件与 Mini CPT 配方不一致：{native}")

    manifest_path = local_path("manifest")
    canonical_path = local_path("canonical")
    adapter_path = native.with_suffix(native.suffix + ".adapter.json")
    review_path = local_path("quality_review")
    for label, path in (("manifest", manifest_path), ("canonical dataset", canonical_path),
                        ("native dataset", native), ("adapter evidence", adapter_path),
                        ("quality review", review_path)):
        if not path.is_file():
            raise MiniCPTPreflightError(f"本地 {label} 不存在：{path}；不会联网下载或替换来源。")

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    canonical_digest = sha256_file(canonical_path)
    native_digest = sha256_file(native)
    expected_id = str(spec.get("dataset_id", ""))
    expected_budget = int(spec.get("expected_budget_count", -1))
    outputs = json.loads((manifest_path.parent.parent / "reports" / "processing_report.json").read_text(encoding="utf-8"))["outputs"]
    cpt_output = outputs.get("cpt", {})

    checks = {
        "dataset_id_matches": manifest.get("dataset_id") == expected_id,
        "stage_is_cpt": manifest.get("stage") == "cpt",
        "canonical_digest_matches_manifest": manifest.get("hashes", {}).get("canonical_records_sha256") == canonical_digest,
        "adapter_source_matches_canonical": Path(adapter.get("input", "")).resolve() == canonical_path.resolve(),
        "adapter_input_digest_matches": adapter.get("input_sha256") == canonical_digest,
        "adapter_output_digest_matches": adapter.get("output_sha256") == native_digest,
        "adapter_count_matches_manifest": adapter.get("records") == manifest.get("sample_count"),
        "budget_matches_recipe": cpt_output.get("budget_count") == expected_budget,
        "budget_matches_manifest": cpt_output.get("budget_count") == expected_budget and cpt_output.get("dataset_id") == expected_id,
        "review_matches_dataset": review.get("dataset_id") == expected_id and review.get("content_sha256") == canonical_digest,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise MiniCPTPreflightError("Mini CPT 数据/血缘证据不匹配：" + ", ".join(failures))

    native_rows = list(read_jsonl(native))
    token_count = sum(int(row.get("metadata", {}).get("tokens", 0)) for row in read_jsonl(canonical_path))
    if len(native_rows) != manifest.get("sample_count") or token_count != expected_budget:
        raise MiniCPTPreflightError(
            f"记录数/token 预算不匹配：records={len(native_rows)}, tokens={token_count}, expected={expected_budget}"
        )

    pending = review.get("outcome") != "approved" or review.get("usage") in {"legacy_unknown", "exploratory"}
    if pending and spec.get("usage") != "exploratory":
        raise MiniCPTPreflightError("质量 review 未批准；只能以 usage=exploratory 执行。")
    return {
        "status": "pass",
        "dataset_id": expected_id,
        "canonical_sha256": canonical_digest,
        "native_sha256": native_digest,
        "record_count": len(native_rows),
        "budget_count": token_count,
        "budget_unit": spec.get("expected_budget_unit"),
        "quality_review_outcome": review.get("outcome"),
        "quality_review_usage": review.get("usage"),
        "usage": "exploratory" if pending else "official",
        "downloads_performed": False,
        "base_model": model_evidence,
        "checks": checks,
    }
