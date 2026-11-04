"""Read-only, bounded projection of local run workspace records for the assistant."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..ui.catalog import ArtifactCatalog
from ..ui.run_workspace import RunWorkspace
from .contracts import EvidenceFact, EvidenceQuery, EvidenceRef, EvidenceStatus

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_SAFE_LINEAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_SAFE_REVISION = re.compile(r"^[A-Za-z0-9._+-]{1,80}$")
_FAILED_STATUSES = {"failed", "oom", "interrupted"}
_CONFIG_FACTS = (
    "stage", "model_revision", "seed", "max_length", "max_steps",
    "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps",
)


def _scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value[:240] if isinstance(value, str) else value
    return None


def _failure_reason(detail: dict[str, Any], lineage: dict[str, Any]) -> str | None:
    """Return a short, sanitized structured error for failed evaluation evidence."""
    records = [{"failure_reason": detail.get("failure_reason")}, detail.get("normalized_record"),
               detail.get("launcher_record"), lineage]
    for record in records:
        if not isinstance(record, dict):
            continue
        errors = record.get("errors")
        candidates = errors if isinstance(errors, list) else [record.get("failure_reason") or record.get("error")]
        for value in candidates:
            if not isinstance(value, str) or not value.strip():
                continue
            message = " ".join(value.split())
            message = re.sub(r"(?i)\b(api[_-]?key|access[_-]?token|authorization)\s*[:=]\s*[^\s,;]+", r"\1=[已隐藏]", message)
            message = re.sub(r"https?://\S+", "[远程地址]", message)
            message = re.sub(r"\b[A-Za-z]:\\[^\s\"']+", "[本地路径]", message)
            return message[:1200]
    return None


def _record_status(run: dict[str, Any], lineage: dict[str, Any] | None) -> EvidenceStatus:
    if not run:
        return "missing"
    schema_version = (lineage or {}).get("lineage_schema_version") or run.get("lineage_schema_version")
    if not isinstance(schema_version, str):
        return "legacy"
    if (lineage or {}).get("lineage_status") in {"incomplete", "partial"}:
        return "partial"
    return "available"


class WorkspaceEvidenceReader:
    """Expose only small, typed facts and local UI links, never artifact contents."""

    def __init__(self, workspace: Path, *, catalog: ArtifactCatalog | None = None):
        self._workspace = workspace.resolve()
        self._runs = RunWorkspace(self._workspace)
        self._catalog = catalog or ArtifactCatalog(self._workspace)
        self._catalog_scanned = bool(catalog and catalog.scanned_at)

    def search(self, request: EvidenceQuery) -> list[EvidenceRef]:
        selected = set(request.selected_entity_ids)
        dataset_ids = {value for value in selected if value.startswith(("ds-", "dataset-"))}
        model_ids = {value for value in selected if value.startswith("model-")}
        if request.kind == "trace_data_lineage" and dataset_ids:
            datasets = self._search_datasets(dataset_ids, request.limit)
            if datasets:
                return datasets
        if request.kind == "trace_data_lineage" and model_ids:
            return self._search_model_lineage(model_ids, request.limit)
        if selected:
            rows = []
            for entity_id in sorted(selected):
                if not _SAFE_ID.fullmatch(entity_id):
                    continue
                found = self._runs.list_runs(q=entity_id, limit=5).get("runs", [])
                rows.extend(row for row in found if row.get("run_id") == entity_id)
        else:
            rows = self._runs.list_runs(limit=min(request.limit, 50)).get("runs", [])
        if request.kind == "recent_failed_runs":
            rows = [row for row in rows if row.get("status") in _FAILED_STATUSES]
        elif request.kind == "compare_evaluations":
            rows = [row for row in rows if row.get("run_kind") == "evaluation"]
        elif request.query and not selected:
            needle = request.query.casefold()[:120]
            rows = [
                row for row in rows
                if needle in " ".join(
                    str(row.get(key) or "")
                    for key in ("run_id", "run_label", "experiment_id", "stage", "status")
                ).casefold()
            ]
        output: list[EvidenceRef] = []
        for row in rows[:request.limit]:
            run_id = row.get("run_id")
            if not isinstance(run_id, str) or not _SAFE_ID.fullmatch(run_id):
                continue
            detail = self._runs.run_detail(run_id, _summary=row)
            if not isinstance(detail, dict):
                output.append(self._missing_run(run_id))
                continue
            lineage = detail.get("lineage") if isinstance(detail.get("lineage"), dict) else None
            output.extend(self._project_run(row, detail, lineage))
            if len(output) >= 50:
                break
        return output[:50]

    def _search_model_lineage(self, model_ids: set[str], limit: int) -> list[EvidenceRef]:
        """Resolve registered model IDs through bounded training Run lineage records."""
        rows = self._runs.list_runs(limit=200).get("runs", [])
        found: dict[str, tuple[str, dict[str, Any]]] = {}
        merge_sources: dict[str, str] = {}
        for row in rows:
            if row.get("run_kind") != "training":
                continue
            run_id = row.get("run_id")
            if not isinstance(run_id, str) or not _SAFE_ID.fullmatch(run_id):
                continue
            lineage = self._runs.run_lineage_from_row(run_id, row)
            if not isinstance(lineage, dict):
                continue
            outputs = lineage.get("outputs")
            if not isinstance(outputs, list):
                continue
            for output in outputs[:40]:
                if not isinstance(output, dict) or output.get("kind") != "model":
                    continue
                model_id = output.get("id")
                if isinstance(model_id, str) and model_id in model_ids:
                    found.setdefault(model_id, (run_id, lineage))
            if len(found) == len(model_ids):
                break

        # A merged model is produced by a merge Run rather than directly by
        # training. Reuse the existing digest-verified handoff index.
        if model_ids - found.keys():
            from ..ui.model_playground_catalog import ModelPlaygroundCatalog
            verified = ModelPlaygroundCatalog(self._workspace)._verified_sources()
            rows_by_id = {row.get("run_id"): row for row in rows}
            for model_id in model_ids - found.keys():
                source = verified.get(model_id) or {}
                run_id = source.get("source_run_id")
                row = rows_by_id.get(run_id)
                if not isinstance(run_id, str) or not isinstance(row, dict):
                    continue
                lineage = self._runs.run_lineage_from_row(run_id, row)
                if isinstance(lineage, dict):
                    found[model_id] = (run_id, lineage)
                    merge_id = source.get("produced_by_run_id")
                    if isinstance(merge_id, str) and _SAFE_ID.fullmatch(merge_id):
                        merge_sources[model_id] = merge_id

        refs: list[EvidenceRef] = []
        for model_id in sorted(model_ids)[:limit]:
            source = found.get(model_id)
            if source is None:
                refs.append(EvidenceRef(
                    evidence_id=f"model:{model_id}", kind="model", entity_id=model_id,
                    title=f"模型 {model_id}", status="missing",
                    summary="没有在受限训练运行血缘中找到该模型身份。",
                    limitations=["未读取模型权重或任意模型路径；无法据此确认来源。"],
                    source_href="/runs",
                ))
                continue
            run_id, lineage = source
            inputs = lineage.get("inputs")
            dataset_ids = sorted({
                item.get("id") for item in inputs[:40]
                if isinstance(item, dict) and item.get("kind") == "dataset"
                and isinstance(item.get("id"), str) and _SAFE_LINEAGE_ID.fullmatch(item["id"])
            }) if isinstance(inputs, list) else []
            facts = [
                EvidenceFact(key="model_id", label="模型 ID", value=model_id),
                EvidenceFact(key="produced_by_run_id", label="来源训练 Run", value=run_id),
            ]
            if model_id in merge_sources:
                facts.append(EvidenceFact(key="merge_run_id", label="合并 Run", value=merge_sources[model_id]))
            if dataset_ids:
                facts.append(EvidenceFact(key="dataset_ids", label="登记数据集", value=", ".join(dataset_ids)))
            refs.append(EvidenceRef(
                evidence_id=f"model:{model_id}", kind="model", entity_id=model_id,
                title=f"模型 {model_id}", status="available",
                version=_scalar(lineage.get("lineage_schema_version")) or "legacy",
                summary=f"模型来源记录指向训练运行 {run_id}。",
                facts=facts,
                limitations=[] if dataset_ids else ["训练运行没有登记可验证的数据集输入。"],
                source_href=f"/runs/{run_id}",
            ))
            run_row = next((item for item in rows if item.get("run_id") == run_id), None)
            detail = self._runs.run_detail(run_id, _summary=run_row) if run_row else None
            if run_row and isinstance(detail, dict):
                refs.extend(self._project_run(run_row, detail, lineage))
        return refs[:50]

    def _search_datasets(self, selected: set[str], limit: int) -> list[EvidenceRef]:
        if not self._catalog_scanned:
            self._catalog.scan()
            self._catalog_scanned = True
        result: list[EvidenceRef] = []
        found_ids: set[str] = set()
        for build in list(self._catalog.builds.values()):
            datasets = build.get("datasets")
            if not isinstance(datasets, list):
                continue
            safe_datasets = [item for item in datasets if isinstance(item, dict)
                             and item.get("dataset_id") in selected]
            if not safe_datasets:
                continue
            detail = self._catalog.detail(str(build.get("id", "")))
            detail_by_id = {
                item.get("dataset_id"): item for item in detail.get("datasets", [])
                if isinstance(item, dict) and isinstance(item.get("dataset_id"), str)
            }
            for summary in safe_datasets:
                dataset_id = summary.get("dataset_id")
                if not isinstance(dataset_id, str) or not _SAFE_LINEAGE_ID.fullmatch(dataset_id):
                    continue
                found_ids.add(dataset_id)
                item = detail_by_id.get(dataset_id, summary)
                result.extend(self._project_dataset(dataset_id, item, build.get("mode")))
                if len(result) >= limit * 3:
                    return result[:limit * 3]
        for dataset_id in sorted(selected - found_ids)[:limit]:
            result.extend(self._missing_dataset(dataset_id))
        return result

    @staticmethod
    def _missing_dataset(dataset_id: str) -> list[EvidenceRef]:
        return [
            EvidenceRef(
                evidence_id=f"dataset:{dataset_id}", kind="dataset", entity_id=dataset_id,
                title=f"数据集 {dataset_id}", status="missing",
                summary="工作区限定目录中没有找到该数据集身份。",
                limitations=["没有读取任意文件路径；不能依据标识推断数据集内容。"],
                source_href="/data",
            ),
            EvidenceRef(
                evidence_id=f"quality:{dataset_id}", kind="quality_report", entity_id=dataset_id,
                title=f"{dataset_id} 质量报告", status="missing",
                summary="没有可验证的质量报告。", limitations=["质量状态未知。"], source_href="/data",
            ),
        ]

    @staticmethod
    def _project_dataset(dataset_id: str, item: dict[str, Any], mode: Any) -> list[EvidenceRef]:
        stage = _scalar(item.get("stage")) or "unknown"
        count = _scalar(item.get("count"))
        unit = _scalar(item.get("unit")) or "records"
        dataset_status: EvidenceStatus = "legacy" if mode == "legacy" else "available"
        facts = [
            EvidenceFact(key="dataset_id", label="数据集 ID", value=dataset_id),
            EvidenceFact(key="stage", label="训练阶段", value=stage),
            EvidenceFact(key="record_count", label="记录数", value=count, unit=str(unit)),
        ]
        digest = item.get("content_sha256")
        if isinstance(digest, str) and re.fullmatch(r"[a-fA-F0-9]{64}", digest):
            facts.append(EvidenceFact(key="content_sha256", label="数据摘要", value=digest))
        lineage = item.get("manifest", {}).get("lineage") if isinstance(item.get("manifest"), dict) else None
        lineage_facts: list[EvidenceFact] = []
        if isinstance(lineage, dict):
            for key in ("produced_by_run_id", "processing_report_id"):
                value = lineage.get(key)
                if isinstance(value, str) and _SAFE_LINEAGE_ID.fullmatch(value):
                    lineage_facts.append(EvidenceFact(key=key, label=key, value=value))
            parents = lineage.get("parents")
            if isinstance(parents, list):
                ids = [str(value) for value in parents[:10]
                       if isinstance(value, str) and _SAFE_LINEAGE_ID.fullmatch(value)]
                if ids:
                    lineage_facts.append(EvidenceFact(key="parents", label="上游数据集", value=", ".join(ids)))
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        safe_source_kind = source.get("kind") if source.get("kind") in {
            "open_source", "local_upload", "model_generated", "declared_source", "unknown"
        } else "unknown"
        facts.append(EvidenceFact(key="source_kind", label="来源类型", value=safe_source_kind))
        dataset_ref = EvidenceRef(
            evidence_id=f"dataset:{dataset_id}", kind="dataset", entity_id=dataset_id,
            title=f"{stage} 数据集 {dataset_id}", status=dataset_status,
            version=digest if isinstance(digest, str) and re.fullmatch(r"[a-fA-F0-9]{64}", digest) else "legacy",
            content_sha256=digest if isinstance(digest, str) and re.fullmatch(r"[a-fA-F0-9]{64}", digest) else None,
            summary=f"{stage} 数据集；{count if count is not None else '记录数未知'} {unit}；来源类型 {safe_source_kind}。",
            facts=facts,
            limitations=["仅提供数据集身份、计数和来源摘要，不包含样本正文。"],
            source_href=f"/data#dataset-{dataset_id}",
        )
        quality = item.get("quality") if isinstance(item.get("quality"), dict) else None
        if quality is None:
            quality_ref = EvidenceRef(
                evidence_id=f"quality:{dataset_id}", kind="quality_report", entity_id=dataset_id,
                title=f"{dataset_id} 质量报告", status="missing",
                summary="没有可验证的质量报告。", limitations=["质量状态未知，不能据此判断数据质量。"],
                source_href=f"/data#dataset-{dataset_id}",
            )
        else:
            gate = quality.get("gate") if isinstance(quality.get("gate"), dict) else {}
            report_status = gate.get("summary_status") if gate.get("summary_status") in {"FAIL", "NOT_EVALUATED"} else "unknown"
            metrics = quality.get("metrics") if isinstance(quality.get("metrics"), dict) else {}
            quality_facts = [
                EvidenceFact(key="quality_status", label="质量门禁", value=report_status),
                EvidenceFact(key="quality_protocol", label="质量协议", value=_scalar(quality.get("quality_protocol"))),
            ]
            for metric_id, metric in list(metrics.items())[:30]:
                if not isinstance(metric_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{1,120}", metric_id):
                    continue
                if not isinstance(metric, dict):
                    continue
                value = _scalar(metric.get("value"))
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                quality_facts.append(EvidenceFact(
                    key=f"metric.{metric_id}", label=metric_id, value=value,
                    unit=_scalar(metric.get("unit")) if isinstance(metric.get("unit"), str) else None,
                    denominator=_scalar(metric.get("denominator")) if isinstance(metric.get("denominator"), (str, int)) else None,
                ))
            quality_ref = EvidenceRef(
                evidence_id=f"quality:{dataset_id}", kind="quality_report", entity_id=dataset_id,
                title=f"{dataset_id} 质量报告", status="available",
                version=_scalar(quality.get("schema_version")) or "unknown",
                summary=f"质量门禁状态 {report_status}；指标 {len(quality_facts) - 2} 项。",
                facts=quality_facts,
                limitations=["质量报告仅描述预训练数据结构与检查项，不代表模型效果。"],
                source_href=f"/data#dataset-{dataset_id}",
            )
        lineage_ref = EvidenceRef(
            evidence_id=f"dataset-lineage:{dataset_id}", kind="lineage", entity_id=dataset_id,
            title=f"{dataset_id} 数据血缘", status="available" if lineage_facts else "missing",
            version="dataset-lineage-v1" if lineage_facts else "legacy",
            summary="；".join(f"{fact.label}：{fact.value}" for fact in lineage_facts)
            if lineage_facts else "未登记可验证的数据血缘。",
            facts=lineage_facts,
            limitations=[] if lineage_facts else ["不能从数据集名称推断来源或生成过程。"],
            source_href=f"/data#dataset-{dataset_id}",
        )
        return [dataset_ref, quality_ref, lineage_ref]

    def _project_run(
        self,
        row: dict[str, Any],
        detail: dict[str, Any],
        lineage: dict[str, Any] | None,
    ) -> list[EvidenceRef]:
        run_id = str(row["run_id"])
        run_status = _scalar(row.get("status")) or "unknown"
        title = _scalar(row.get("run_label")) or run_id
        run_facts = [
            EvidenceFact(key="run_id", label="运行 ID", value=run_id),
            EvidenceFact(key="run_kind", label="运行类型", value=_scalar(row.get("run_kind"))),
            EvidenceFact(key="stage", label="训练阶段", value=_scalar(row.get("stage"))),
            EvidenceFact(key="status", label="运行状态", value=run_status),
            EvidenceFact(key="started_at", label="开始时间", value=_scalar(row.get("started_at"))),
            EvidenceFact(key="completed_at", label="完成时间", value=_scalar(row.get("completed_at"))),
        ]
        status = _record_status(detail, lineage)
        refs = [
            EvidenceRef(
                evidence_id=f"run:{run_id}", kind="run", entity_id=run_id,
                title=str(title)[:240], status=status,
                version=(lineage or {}).get("lineage_schema_version")
                if isinstance((lineage or {}).get("lineage_schema_version"), str) else "legacy",
                summary=f"{title}；阶段 {row.get('stage') or '未知'}；运行状态 {run_status}。",
                facts=run_facts, limitations=["仅返回结构化运行摘要；完整日志与配置不发送到模型。"],
                source_href=f"/runs/{run_id}",
            )
        ]

        if row.get("run_kind") == "evaluation":
            refs.append(self._project_evaluation(row, detail, lineage))

        config = detail.get("config")
        if isinstance(config, dict):
            training = config.get("training") if isinstance(config.get("training"), dict) else {}
            config_facts = []
            for key in _CONFIG_FACTS:
                value = config.get(key, training.get(key))
                scalar = _scalar(value)
                if isinstance(scalar, str) and ("/" in scalar or "\\" in scalar or ":" in scalar):
                    scalar = None
                if key in {"seed", "max_length", "max_steps", "per_device_train_batch_size", "gradient_accumulation_steps"}:
                    if isinstance(scalar, bool) or not isinstance(scalar, int):
                        scalar = None
                if key == "learning_rate" and (isinstance(scalar, bool) or not isinstance(scalar, (int, float))):
                    scalar = None
                if key == "model_revision" and isinstance(scalar, str) and not _SAFE_REVISION.fullmatch(scalar):
                    scalar = None
                if key == "stage" and isinstance(scalar, str) and not _SAFE_REVISION.fullmatch(scalar):
                    scalar = None
                if scalar is not None:
                    config_facts.append(EvidenceFact(key=key, label=key, value=scalar))
            refs.append(EvidenceRef(
                evidence_id=f"configuration:{run_id}", kind="configuration", entity_id=run_id,
                title=f"{title} 配置摘要", status=status, version="resolved-config-summary-v1",
                summary="配置摘要仅包含阶段、模型 revision 和白名单训练参数。",
                facts=config_facts,
                limitations=["已省略本地路径、环境信息、命令参数和未列入白名单的配置字段。"],
                source_href=f"/runs/{run_id}",
            ))
        else:
            refs.append(EvidenceRef(
                evidence_id=f"configuration:{run_id}", kind="configuration", entity_id=run_id,
                title=f"{title} 配置", status="missing", summary="没有可读取的结构化配置记录。",
                limitations=["配置证据缺失。"], source_href=f"/runs/{run_id}",
            ))

        logs = detail.get("logs")
        logs_available = isinstance(logs, dict) and isinstance(logs.get("path"), str)
        refs.append(EvidenceRef(
            evidence_id=f"log:{run_id}", kind="log", entity_id=run_id,
            title=f"{title} 日志", status="available" if logs_available else "missing",
            version="bounded-log-presence-v1" if logs_available else None,
            summary="存在运行日志，可在运行详情页查看。" if logs_available else "没有可读取的运行日志。",
            facts=[EvidenceFact(key="available", label="日志可用", value=logs_available)],
            limitations=["未读取或发送日志正文；需要查看具体日志时请打开运行详情。"],
            source_href=f"/runs/{run_id}",
        ))

        if lineage is None:
            refs.append(EvidenceRef(
                evidence_id=f"lineage:{run_id}", kind="lineage", entity_id=run_id,
                title=f"{title} 模型血缘", status="missing",
                summary="没有可读取的模型血缘记录。", limitations=["无法据此确认数据或模型来源。"],
                source_href=f"/runs/{run_id}",
            ))
        else:
            facts: list[EvidenceFact] = []
            safe_links: list[str] = []
            for direction in ("inputs", "outputs"):
                entries = lineage.get(direction)
                if not isinstance(entries, list):
                    continue
                for entry in entries[:20]:
                    if not isinstance(entry, dict):
                        continue
                    item_id = _scalar(entry.get("id"))
                    kind = _scalar(entry.get("kind"))
                    role = _scalar(entry.get("role"))
                    availability = _scalar(entry.get("availability"))
                    if (isinstance(item_id, str) and _SAFE_LINEAGE_ID.fullmatch(item_id)
                            and isinstance(kind, str) and _SAFE_REVISION.fullmatch(kind)):
                        safe_links.append(f"{direction[:-1]} {kind} {item_id} ({role or 'role unknown'}, {availability or 'availability unknown'})")
            for key, label in (("run_type", "运行类型"), ("lineage_status", "血缘状态"),
                               ("lineage_schema_version", "血缘版本"), ("code_revision", "代码版本")):
                value = lineage.get(key)
                if isinstance(value, dict):
                    value = value.get("value") or value.get("status") or value.get("reason")
                scalar = _scalar(value)
                if scalar is not None:
                    facts.append(EvidenceFact(key=key, label=label, value=scalar))
            summaries = safe_links[:12]
            warnings = lineage.get("lineage_warnings")
            limitations = ["来源路径和原始产物内容未发送。"]
            if (lineage.get("lineage_status") or "unknown") != "complete":
                limitations.append("血缘未完整，无法确认全部来源。")
            if isinstance(warnings, list) and warnings:
                limitations.append(f"血缘记录包含 {min(len(warnings), 20)} 条警告；请打开来源查看。")
            refs.append(EvidenceRef(
                evidence_id=f"lineage:{run_id}", kind="lineage", entity_id=run_id,
                title=f"{title} 模型血缘", status=status,
                version=_scalar(lineage.get("lineage_schema_version")) or "legacy",
                summary="；".join(summaries) if summaries else "血缘记录存在，但没有可安全摘要的来源/产物身份。",
                facts=facts, limitations=limitations[:20], source_href=f"/runs/{run_id}",
            ))
        return refs

    @staticmethod
    def _project_evaluation(
        row: dict[str, Any], detail: dict[str, Any], lineage: dict[str, Any] | None
    ) -> EvidenceRef:
        run_id = str(row["run_id"])
        lineage = lineage or {}
        model_id = lineage.get("model_id")
        protocol = lineage.get("protocol_version")
        dataset_ids = lineage.get("dataset_ids")
        evaluation_status = lineage.get("status") or row.get("status")
        status: EvidenceStatus = {
            "completed": "available", "partial": "partial", "failed": "failed",
            "incomplete": "partial",
        }.get(str(evaluation_status).lower(), "unknown")
        facts: list[EvidenceFact] = [
            EvidenceFact(key="run_id", label="评测运行 ID", value=run_id),
            EvidenceFact(key="status", label="评测状态", value=_scalar(evaluation_status)),
            EvidenceFact(key="stage", label="评测阶段", value=_scalar(row.get("stage"))),
            EvidenceFact(key="started_at", label="开始时间", value=_scalar(row.get("started_at"))),
        ]
        if isinstance(model_id, str) and _SAFE_LINEAGE_ID.fullmatch(model_id):
            facts.append(EvidenceFact(key="model_id", label="模型身份", value=model_id))
        if isinstance(protocol, str) and _SAFE_REVISION.fullmatch(protocol):
            facts.append(EvidenceFact(key="protocol_version", label="评测协议", value=protocol))
        if isinstance(dataset_ids, list):
            safe_dataset_ids = [value for value in dataset_ids[:20]
                                if isinstance(value, str) and _SAFE_LINEAGE_ID.fullmatch(value)]
            if safe_dataset_ids:
                facts.append(EvidenceFact(key="dataset_ids", label="评测数据集身份", value=", ".join(safe_dataset_ids)))
        failure_reason = _failure_reason(detail, lineage)
        if status in {"failed", "partial"} and failure_reason:
            facts.append(EvidenceFact(key="failure_reason", label="失败原因", value=failure_reason))
        metrics = lineage.get("metrics") if isinstance(lineage.get("metrics"), dict) else {}
        metric_facts = 0
        for metric_id, metric in list(metrics.items())[:30]:
            if not isinstance(metric_id, str) or not _SAFE_REVISION.fullmatch(metric_id):
                continue
            if not isinstance(metric, dict):
                continue
            value = metric.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            facts.append(EvidenceFact(
                key=f"metric.{metric_id}", label=metric_id, value=value,
                unit=_scalar(metric.get("unit")) if isinstance(metric.get("unit"), str) else None,
                denominator=metric.get("denominator") if isinstance(metric.get("denominator"), int) else None,
            ))
            metric_protocol = metric.get("protocol_version")
            if isinstance(metric_protocol, str) and _SAFE_REVISION.fullmatch(metric_protocol):
                facts.append(EvidenceFact(
                    key=f"protocol.{metric_id}", label=f"{metric_id} 协议", value=metric_protocol
                ))
            comparison_key = metric.get("comparability_key")
            if isinstance(comparison_key, str) and re.fullmatch(r"[A-Fa-f0-9]{64}", comparison_key):
                facts.append(EvidenceFact(
                    key=f"comparability.{metric_id}", label=f"{metric_id} 可比性摘要", value=comparison_key
                ))
            sample_count = metric.get("sample_count")
            if isinstance(sample_count, int) and sample_count >= 0:
                facts.append(EvidenceFact(
                    key=f"coverage.{metric_id}", label=f"{metric_id} 样本覆盖", value=sample_count,
                    unit="questions",
                ))
            metric_facts += 1
        title = _scalar(row.get("run_label")) or run_id
        limitations = ["只投影评测身份、协议、覆盖分母和有限指标；不包含逐题预测或回答正文。"]
        if status in {"partial", "failed", "unknown"}:
            limitations.append("评测状态不是完整成功；不可将缺失指标解释为零分。")
        return EvidenceRef(
            evidence_id=f"evaluation:{run_id}", kind="evaluation", entity_id=run_id,
            title=f"{title} 评测", status=status,
            version=_scalar(lineage.get("lineage_schema_version")) or "legacy",
            summary=(f"评测状态 {evaluation_status or '未知'}；协议 {protocol or '未记录'}；可用指标 {metric_facts} 项。"
                     + (f"失败原因：{failure_reason}" if status in {"failed", "partial"} and failure_reason else "")),
            facts=facts, limitations=limitations,
            source_href=f"/runs/{run_id}",
        )

    @staticmethod
    def _missing_run(run_id: str) -> EvidenceRef:
        return EvidenceRef(
            evidence_id=f"run:{run_id}", kind="run", entity_id=run_id,
            title=run_id, status="missing", summary="运行记录当前不可用。",
            limitations=["记录缺失或无法安全读取；没有推断其运行状态。"],
            source_href=f"/runs/{run_id}",
        )
