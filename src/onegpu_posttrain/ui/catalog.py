"""Bounded, read-only adapter from local data artifacts to the Data page."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..data.evidence import ProcessingReport, QualityReport
from ..data.manifest import DatasetManifest
from ..data.source_adapters import adapt_open_source_manifest, source_id_for_manifest
from ..lineage import RunRecord, sha256_directory, sha256_file

MAX_BYTES = 2 * 1024 * 1024
MAX_FILES = 2000
STAGE_ORDER = ("cpt", "cpt_holdout", "sft", "preference", "preference_eval", "grpo", "eval")
STAGES = set(STAGE_ORDER)
EXCLUDED_DIRS = {"canonical", "private", "responses", "train", "merged", "checkpoint", "checkpoints",
                 "logs", "evalscope", "node_modules", ".git", ".venv"}
PRIVATE_DIRS = {"private", "responses"}
PRIVATE_KEYS = re.compile(r"(^|_)(secret|password|credential|authorization|api_key|access_token|auth_token)$", re.IGNORECASE)
RAW_KEYS = {"text", "messages", "chosen", "rejected", "prompt", "answer", "problem", "raw_problem",
            "generations", "environment", "stdout", "stderr", "command", "command_string"}


class ArtifactError(ValueError):
    pass


def _key(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:20]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_view(value: Any, depth: int = 0, budget: list[int] | None = None) -> Any:
    """Reports may contain arbitrary metadata. Never render raw records or execute their content."""
    budget = budget if budget is not None else [5000]
    budget[0] -= 1
    if depth > 12 or budget[0] < 0:
        return "[bounded preview]"
    if isinstance(value, dict):
        return {str(k): "[not exposed]" if PRIVATE_KEYS.search(str(k)) or str(k).lower() in RAW_KEYS
                else _safe_view(v, depth + 1, budget) for k, v in list(value.items())[:200]}
    if isinstance(value, list):
        return [_safe_view(v, depth + 1, budget) for v in value[:200]]
    if isinstance(value, str):
        return value[:4000] + ("… [truncated]" if len(value) > 4000 else "")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value if value is None or isinstance(value, (int, float, bool)) else str(value)


def _count(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


class ArtifactCatalog:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.roots = [self.workspace / "data/processed", self.workspace / "artifacts/ui-demo",
                      self.workspace / "artifacts/synthetic"]
        self.run_root = self.workspace / "runs"
        self.lock = threading.RLock()
        self.artifacts: dict[str, Path] = {}
        self.builds: dict[str, dict[str, Any]] = {}
        self.scanned_at: str | None = None
        self.issues: list[dict[str, str]] = []

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.workspace).as_posix()

    def _resolve(self, path: Path) -> Path:
        if PRIVATE_DIRS.intersection(path.parts):
            raise ArtifactError("Private or raw dataset paths are not exposed by the artifact reader")
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != self.workspace.parent):
            raise ArtifactError("Symbolic links are not exposed by the artifact reader")
        resolved = path.resolve()
        allowed = [root for root in [*self.roots, self.run_root]
                   if root.resolve().is_relative_to(self.workspace) and resolved.is_relative_to(root.resolve())]
        if not resolved.is_relative_to(self.workspace) or not allowed:
            raise ArtifactError("Artifact path escapes the configured roots")
        return resolved

    def _walk(self, root: Path):
        if not root.exists():
            return
        self._resolve(root)
        visited = 0
        for directory, dirs, files in os.walk(root, followlinks=False):
            current = Path(directory)
            depth = len(current.relative_to(root).parts)
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")
                             and not (current / d).is_symlink()) if depth < 5 else []
            for name in sorted(files):
                visited += 1
                if visited > MAX_FILES:
                    self.issues.append({"path": self._relative(root), "message": "扫描达到文件数量上限，结果不完整"})
                    return
                yield current / name

    def _read(self, path: Path) -> tuple[dict[str, Any], str]:
        resolved = self._resolve(path)
        if resolved.stat().st_size > MAX_BYTES:
            raise ArtifactError("Artifact exceeds the 2 MiB read limit")
        with resolved.open("rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ArtifactError("Artifact exceeds the 2 MiB read limit")
        try:
            text = raw.decode("utf-8-sig")
            data = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
        except (ValueError, UnicodeError, yaml.YAMLError, RecursionError) as exc:
            raise ArtifactError("Artifact could not be parsed as a report object") from exc
        if not isinstance(data, dict):
            raise ArtifactError("Expected a report object")
        return data, hashlib.sha256(raw).hexdigest()

    def _register(self, path: Path) -> dict[str, str]:
        self._resolve(path)
        relative = self._relative(path)
        token = _key(relative)
        self.artifacts[token] = path
        return {"id": token, "path": relative, "label": path.name}

    def artifact(self, token: str) -> dict[str, Any]:
        with self.lock:
            if token not in self.artifacts:
                raise KeyError(token)
            path = self.artifacts[token]
            data, digest = self._read(path)  # Recheck containment, size and parseability on every read.
            return {"path": self._relative(path), "sha256": digest, "data": _safe_view(data),
                    "notice": "只读汇总预览；敏感字段和原始样本字段不予展示，深度/行数已限制。"}

    def scan(self) -> dict[str, Any]:
        with self.lock:
            self.artifacts = {}
            self.builds = {}
            self.issues = []
            candidates: dict[Path, Path] = {}
            for root in self.roots:
                try:
                    for path in self._walk(root):
                        if path.parent.name != "reports":
                            continue
                        if path.name == "processing_report.json":
                            candidates[path.parent.parent] = path
                        elif path.name.endswith("_data_report.json"):
                            candidates.setdefault(path.parent.parent, path)
                except (OSError, ArtifactError) as exc:
                    self.issues.append({"path": self._relative(root), "message": str(exc) if isinstance(exc, ArtifactError) else "数据根目录不可读取"})
            for base, path in sorted(candidates.items()):
                key = _key(self._relative(base))
                try:
                    detail = self._build(base, path, key)
                    self.builds[key] = detail
                except (OSError, ArtifactError, ValidationError, TypeError, KeyError, AttributeError, RecursionError):
                    self.issues.append({"path": self._relative(path), "message": "构建报告损坏、版本不支持或关联文件不可读取"})
            self.scanned_at = _utc()
            self.issues = list({(issue["path"], issue["message"]): issue for issue in self.issues}.values())
            ordered = sorted(self.builds.values(), key=lambda b: (-len(b["datasets"]), b["path"]))
            return {"builds": [self._summary(b) for b in ordered], "issues": self.issues,
                    "scanned_at": self.scanned_at, "read_only": True, "roots": [self._relative(p) for p in self.roots]}

    @staticmethod
    def _summary(build: dict[str, Any]) -> dict[str, Any]:
        keys = ("id", "name", "path", "source", "revision", "mode", "is_demo", "raw_count", "raw_unit", "created_at", "warnings")
        dataset_keys = ("stage", "dataset_id", "display_name", "count", "target", "unit", "tokens", "token_target",
                        "content_sha256", "mixture_eligible", "mixture_block_reason")
        datasets = []
        for dataset in build["datasets"]:
            item = {key: dataset.get(key) for key in dataset_keys}
            source = dataset.get("source") or {}
            quality = dataset.get("quality") or {}
            item["source"] = {key: source.get(key) for key in ("source_id", "kind", "name", "revision_status", "license_status")}
            item["quality_status"] = (quality.get("gate") or {}).get("summary_status")
            datasets.append(item)
        return {**{k: build[k] for k in keys}, "datasets": datasets}

    def detail(self, key: str) -> dict[str, Any]:
        with self.lock:
            if not self.scanned_at:
                self.scan()
            if key not in self.builds:
                raise KeyError(key)
            return _safe_view(self.builds[key])

    def _build(self, base: Path, path: Path, key: str) -> dict[str, Any]:
        raw, digest = self._read(path)
        modern = path.name == "processing_report.json"
        processing = ProcessingReport.model_validate(raw).model_dump(mode="json", by_alias=True) if modern else None
        warnings: list[str] = []
        source_artifacts = [self._register(path)]
        if not modern:
            warnings.append("旧版聚合报告：处理流程由已有计数推导，未回填逐步质量快照。")
        datasets = []
        manifests = sorted((base / "manifests").glob("*.yaml")) + sorted((base / "manifests").glob("*.json"))
        source, revision = (processing["source"].get("repo"), processing["source"].get("revision")) if processing else (None, None)
        for manifest_path in manifests[:30]:
            try:
                manifest_raw, manifest_digest = self._read(manifest_path)
                manifest = DatasetManifest.model_validate(manifest_raw)
                if manifest.stage not in STAGES:
                    continue
                if manifest.source:
                    source = source or manifest.source.repo
                    revision = revision or manifest.source.revision
                if manifest.origin:
                    source = source or manifest.origin.name
                    revision = revision or manifest.origin.revision
                artifact = self._register(manifest_path)
                artifact["sha256"] = manifest_digest
                source_artifacts.append(artifact)
                quality = None
                quality_artifact = None
                if manifest.processing_report:
                    reference = (manifest_path.parent / manifest.processing_report.path)
                    if reference.resolve() != path.resolve() or manifest.processing_report.sha256 != digest:
                        warnings.append(f"{manifest.stage} 的 processing report 引用或摘要不匹配。")
                if manifest.quality_report:
                    reference = manifest_path.parent / manifest.quality_report.path
                    # A manifest is not allowed to turn the viewer into an arbitrary JSON browser.
                    expected = base / "reports/quality" / f"{manifest.stage}.json"
                    if reference.resolve() != expected.resolve():
                        raise ArtifactError("Quality report is outside the allowed stage location")
                    quality_raw, quality_digest = self._read(reference)
                    if quality_digest != manifest.quality_report.sha256:
                        raise ArtifactError("Quality report digest mismatch")
                    parsed = QualityReport.model_validate(quality_raw)
                    if parsed.dataset_id != manifest.dataset_id or parsed.content_sha256 != manifest.hashes.get("canonical_records_sha256"):
                        raise ArtifactError("Quality report does not describe this dataset")
                    quality = parsed.model_dump(mode="json", by_alias=True)
                    quality_artifact = self._register(expected)
                    source_artifacts.append(quality_artifact)
                dataset = self._dataset(manifest, raw, modern, quality, artifact, quality_artifact)
                source_id = source_id_for_manifest(manifest)
                source_repo = (manifest.source.repo if manifest.source else "") or ""
                generated = source_repo.partition(":")[0] in {"fake", "openai_compatible"}
                dataset["source"] = {
                    "source_id": source_id or (manifest.origin.source_id if manifest.origin else "llm-generation" if generated else None),
                    "kind": manifest.origin.kind if manifest.origin else "model_generated" if generated else "open_source" if source_id else "unknown",
                    "name": manifest.origin.name if manifest.origin else source_repo or "未记录",
                    "revision": manifest.origin.revision if manifest.origin else manifest.source.revision if manifest.source else None,
                    "revision_status": manifest.origin.revision_status if manifest.origin else "known" if manifest.source and manifest.source.revision else "not_provided",
                    "license_status": manifest.origin.license_status if manifest.origin else "not_provided",
                }
                if source_id and manifest.stage in {"cpt", "sft", "preference", "grpo"}:
                    try:
                        dataset["registry_artifact"] = adapt_open_source_manifest(manifest_path).model_dump(mode="json")
                    except (OSError, ValueError, ValidationError, TypeError, KeyError):
                        # Legacy reports may predate complete registry evidence. Keep
                        # them visible in the read-only catalog without claiming that
                        # they are consumable registered datasets.
                        dataset["registry_artifact"] = None
                        dataset["registry_status"] = "evidence_incomplete"
                        warnings.append(
                            f"{manifest.stage} 的开源来源登记证据不完整；旧报告仍可查看，"
                            "但不能作为已登记数据集消费。"
                        )
                datasets.append(dataset)
            except (OSError, ArtifactError, ValidationError, TypeError, KeyError, RecursionError):
                warnings.append(f"{manifest_path.name} 或关联质量报告无效，已隐藏该数据集的质量结果。")
                # Keep a visible diagnostic rather than silently showing unverified quality.
        if not datasets:
            warnings.append("未找到可读取的 dataset manifest；不能确认最终数据集身份。")
        counter = raw.get("counters", {}) if not modern else {}
        steps = processing["processing_steps"] if processing else self._legacy_steps(raw)
        if modern:
            raw_count = steps[0]["input_count"]["value"] if steps else None
        else:
            raw_count = _count(counter.get("raw_rows"))
        build = {"id": key, "name": "OpenR1" if "openr1" in str(source or base.name).lower() else "FineMath" if "finemath" in str(source or base.name).lower() else base.name,
                 "path": self._relative(base), "source": source or "未记录", "revision": revision,
                 "mode": "versioned" if modern else "legacy", "is_demo": base.is_relative_to(self.workspace / "artifacts/ui-demo"),
                 "raw_count": raw_count, "raw_unit": "raw_rows", "created_at": processing["created_at"] if modern else None,
                 "datasets": sorted(datasets, key=lambda d: STAGE_ORDER.index(d["stage"])),
                 "steps": steps, "warnings": warnings, "artifacts": source_artifacts,
                 "overlap": raw.get("overlap", {}) if not modern else {},
                 "source_distributions": raw.get("distribution_before", {}) if not modern else {},
                 "model_effect": {"status": "not_evaluated", "reason": "尚无登记的清洗策略受控训练对照，不能由质量变化推断模型提升。"}}
        build["related_runs"] = self._related_runs({
            d["dataset_id"]: {"manifest_sha256": d["manifest"]["sha256"], "content_sha256": d["content_sha256"]}
            for d in datasets
        })
        return build

    @staticmethod
    def _dataset(manifest: DatasetManifest, raw: dict[str, Any], modern: bool, quality: dict[str, Any] | None,
                 artifact: dict[str, str], quality_artifact: dict[str, str] | None) -> dict[str, Any]:
        stage = manifest.stage
        count = manifest.sample_count
        unit = "preference_pairs" if stage.startswith("preference") else "prompts" if stage == "grpo" else "examples" if not stage.startswith("cpt") else "chunks"
        target = None
        tokens, token_target = None, None
        lengths, distribution = {}, {}
        if modern:
            config = raw.get("config", {})
            target = _count(config.get(f"{'holdout' if stage == 'eval' else stage}_target"))
            if stage.startswith("cpt"):
                token_target = _count(config.get("holdout_token_budget" if stage == "cpt_holdout" else "train_token_budget"))
            if quality:
                lengths = quality["summaries"].get("lengths", {})
                distribution = quality["summaries"].get("source_distribution", {})
                token_length = lengths.get("tokens", {})
                if token_length.get("count") and token_length.get("mean") is not None:
                    tokens = round(token_length["count"] * token_length["mean"])
        elif stage.startswith("cpt"):
            info = raw.get("holdout" if stage == "cpt_holdout" else "train", {})
            tokens, token_target = _count(info.get("tokens")), _count(info.get("token_budget"))
        else:
            target = _count(raw.get("targets", {}).get("holdout" if stage == "eval" else stage))
            lengths = {k: v for k, v in raw.get("lengths", {}).items() if k.startswith(f"{stage}_") and "candidate" not in k}
            distribution = raw.get("distribution_after", {}).get(f"{stage}.source", {})
        mixture_eligible = bool(manifest.lineage and manifest.processing_report
                                and manifest.metadata.get("canonical_path"))
        return {"stage": stage, "dataset_id": manifest.dataset_id,
                "display_name": manifest.metadata.get("display_name") if isinstance(manifest.metadata, dict) else None,
                "count": count, "unit": unit,
                "target": target, "tokens": tokens, "token_target": token_target, "quality": quality,
                "lengths": lengths, "source_distribution": distribution,
                "content_sha256": manifest.hashes.get("canonical_records_sha256"), "manifest": artifact,
                "quality_artifact": quality_artifact, "mixture_eligible": mixture_eligible,
                "mixture_block_reason": None if mixture_eligible else
                "缺少登记 lineage、processing report 或 canonical_path"}

    @staticmethod
    def _legacy_steps(raw: dict[str, Any]) -> list[dict[str, Any]]:
        c = raw.get("counters", {})
        if not isinstance(c, dict) or not isinstance(raw.get("protocol", ""), str):
            raise ArtifactError("Malformed legacy processing counters")
        def item(order, name, n_in, n_out, unit, out_unit=None, reasons=None):
            return {"step_id": f"legacy-{order}", "order": order, "operation": name, "operation_version": "legacy",
                    "kind": "transform", "status": "derived", "input_ref": "legacy:aggregate", "output_ref": "legacy:aggregate",
                    "input_count": {"value": n_in, "unit": unit}, "output_count": {"value": n_out, "unit": out_unit or unit},
                    "output_is_subset": False, "rejected_count": sum(reasons.values()) if reasons else None,
                    "first_rejection_reasons": reasons or {}, "unselected_count": None, "unselected_reasons": {},
                    "metrics_before": {}, "metrics_after": {}, "details": {},
                    "limitations": ["从旧版聚合计数推导；缺失步骤和前后质量快照不能还原。"]}
        raw_count = _count(c.get("raw_rows"))
        if raw.get("protocol", "").startswith("openr1"):
            unique = _count(c.get("normalized_unique"))
            removed = _count(c.get("benchmark_contamination_removed"))
            screened = unique - removed if unique is not None and removed is not None else None
            return [item(1, "normalize / exact dedup", raw_count, unique, "raw_rows", "unique_problems",
                         {key: c[key] for key in ("empty_problem", "exact_duplicate_problem") if _count(c.get(key)) is not None}),
                    item(2, "benchmark screening", unique, screened, "unique_problems",
                         reasons={"benchmark_contamination_removed": removed} if removed is not None else None),
                    item(3, "verifiable math core", screened, _count(c.get("verifiable_core")), "unique_problems",
                         reasons={"no_verifier_correct_trace": c["no_verifier_correct_trace"]} if "no_verifier_correct_trace" in c else None)]
        return [item(1, "source text validation", raw_count, None, "raw_rows",
                     reasons={"empty_text": c["empty_text"]} if "empty_text" in c else None),
                item(2, "document split / screening", None, None, "documents",
                     reasons={"exact_contamination_removed": c["exact_contamination_removed"]} if "exact_contamination_removed" in c else None),
                item(3, "chunk / token budget", None, sum(raw.get(p, {}).get("tokens", 0) for p in ("train", "holdout")), "chunks", "tokens")]

    def _related_runs(self, dataset_ids: dict[str, dict[str, str | None]]) -> list[dict[str, Any]]:
        linked = []
        try:
            for path in self._walk(self.run_root):
                if path.name not in {"lineage.json", "evaluation_run.json"}:
                    continue
                try:
                    lineage, _ = self._read(path)
                    versioned = lineage.get("lineage_schema_version") == "1.0"
                    if versioned:
                        record = RunRecord.model_validate(lineage)
                        inputs = [item.model_dump(mode="json") for item in record.inputs if item.kind == "dataset"]
                        matched = [d for d in inputs if d.get("id") in dataset_ids]
                        run_id, stage, status = record.run_id, path.parent.name, record.status
                    else:
                        inputs = lineage.get("inputs", {}).get("datasets", [])
                        matched = [d for d in inputs if isinstance(d, dict) and d.get("dataset_id") in dataset_ids]
                        run_id, stage = lineage.get("run_id"), lineage.get("stage")
                        run_path = path.parent / "run.json"
                        run, _ = self._read(run_path) if run_path.exists() else ({}, "")
                        status = run.get("train", {}).get("status", "unknown")
                    if not matched:
                        continue
                    hash_verified = True
                    dataset_keys = []
                    for item in matched:
                        dataset_id = item.get("id") if versioned else item.get("dataset_id")
                        dataset_keys.append(dataset_id)
                        expected = dataset_ids[dataset_id].get("content_sha256") if versioned else dataset_ids[dataset_id].get("manifest_sha256")
                        digest = item.get("digest") or {}
                        observed = digest.get("value") if versioned else item.get("manifest_sha256")
                        if observed and expected and observed != expected:
                            raise ArtifactError("Dataset digest mismatch in run lineage")
                        if versioned and item.get("path") and digest.get("status") == "available":
                            resolved = self._resolve(Path(item["path"]))
                            actual = sha256_directory(resolved) if resolved.is_dir() else sha256_file(resolved)
                            if actual != digest.get("value"):
                                raise ArtifactError("Dataset path digest mismatch in run lineage")
                        if not observed:
                            hash_verified = False
                    evaluation_path = path.parent / "eval.json"
                    if not evaluation_path.exists():
                        evaluation_path = path.parent / "evaluation/eval.json"
                    metrics = []
                    if evaluation_path.exists():
                        evaluation, _ = self._read(evaluation_path)
                        for benchmark, info in evaluation.get("evalscope_reports", {}).items():
                            metrics.append({"name": benchmark, "value": _count(info.get("score")), "unit": "ratio", "samples": info.get("num_samples")})
                        for label, block, metric in (("Internal accuracy", "internal_math", "accuracy"), ("Domain PPL", "domain_perplexity", "perplexity")):
                            if block in evaluation:
                                info = evaluation[block]
                                metrics.append({"name": label, "value": _count(info.get(metric)), "unit": "ratio" if metric == "accuracy" else "ppl", "samples": info.get("samples")})
                    verification = ("content_sha256" if versioned else "manifest_hash") if hash_verified else "id_only"
                    verified_note = "已核对数据内容摘要。" if versioned else "已核对 manifest hash。"
                    linked.append({"run_id": run_id, "stage": stage, "dataset_ids": dataset_keys,
                                   "status": status, "metrics": metrics,
                                   "source": self._relative(path), "verification": verification,
                                   "note": "显式 dataset ID 关联；" + (verified_note if hash_verified else "未提供摘要，关联仅依据 ID。") + "不构成清洗策略因果对照。"})
                except (OSError, ArtifactError, ValidationError, TypeError, KeyError, AttributeError, RecursionError):
                    self.issues.append({"path": self._relative(path), "message": "训练血缘或评测文件无效、摘要不匹配"})
        except (OSError, ArtifactError, TypeError, KeyError, AttributeError, RecursionError):
            self.issues.append({"path": "runs", "message": "部分训练血缘或评测文件不可读取"})
        return linked
