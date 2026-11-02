"""Read-only evaluation of official Mini dataset evidence."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from ..lineage_validation import validate_lineage_root
from .evidence import ProcessingReport, QualityReport
from .manifest import (
    DatasetManifest,
    load_manifest,
    sha256_file,
    validate_manifest_evidence,
)
from .mini import (
    MiniDatasetProfile,
    MiniDatasetReadiness,
    MiniReadinessReport,
    ReadinessDimension,
    ReadinessIssue,
    load_mini_profile,
)
from .quality_assessment import QualityAssessmentEntry, load_quality_assessment_bundle
from .quality_review import (
    QualityReviewBundleEntry,
    classify_dataset_usage,
    load_quality_review_bundle,
)

MAX_EVIDENCE_BYTES = 8 * 1024 * 1024


def _issue(code: str, message: str, path: Path | None = None, root: Path | None = None) -> ReadinessIssue:
    rel = path.relative_to(root).as_posix() if path is not None and root is not None and path.is_relative_to(root) else None
    return ReadinessIssue(code=code, message=message, path=rel)


def _read_json(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("evidence path is a symlink")
    resolved = path.resolve(strict=True)
    resolved.relative_to(root)
    if resolved.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ValueError("evidence exceeds the read limit")
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError("expected a JSON object")
    return payload


def _manifest_paths(root: Path) -> list[Path]:
    paths = []
    for path in root.rglob("manifests/*"):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        paths.append(path)
    return sorted(paths)


def _find_manifest(root: Path, profile: MiniDatasetProfile) -> tuple[Path | None, DatasetManifest | None, list[ReadinessIssue]]:
    if profile.manifest_path:
        candidate = (root / profile.manifest_path).resolve(strict=False)
        try:
            candidate.relative_to(root.resolve(strict=True))
        except (ValueError, OSError):
            return None, None, [_issue("manifest_path_escape", "Profile manifest_path 越出 artifact root。", candidate, root)]
        if not candidate.is_file():
            return None, None, [_issue("missing_manifest", "Profile 指定的 manifest 不存在。", candidate, root)]
        try:
            manifest = load_manifest(candidate)
        except (OSError, ValueError, ValidationError) as exc:
            return None, None, [_issue("invalid_manifest", f"Manifest 无法解析：{exc}", candidate, root)]
        if manifest.stage != profile.stage:
            return None, None, [_issue("manifest_stage_mismatch", "Profile manifest stage 与目标 stage 不一致。", candidate, root)]
        return candidate, manifest, []
    matches: list[tuple[Path, DatasetManifest]] = []
    errors: list[ReadinessIssue] = []
    for path in _manifest_paths(root):
        try:
            manifest = load_manifest(path)
        except (OSError, ValueError, ValidationError) as exc:
            errors.append(_issue("invalid_manifest", f"Manifest 无法解析：{exc}", path, root))
            continue
        if manifest.stage == profile.stage:
            matches.append((path, manifest))
    if not matches:
        errors.append(_issue("missing_manifest", f"未找到阶段 {profile.stage} 的 Mini manifest。"))
        return None, None, errors
    if len(matches) > 1:
        errors.append(_issue("ambiguous_manifest", f"阶段 {profile.stage} 找到多个 manifest；profile 无法唯一选择。"))
        return None, None, errors
    return matches[0][0], matches[0][1], errors


def _load_processing(root: Path, manifest_path: Path, manifest: DatasetManifest) -> tuple[ProcessingReport | None, Path | None, list[ReadinessIssue]]:
    if manifest.processing_report is None:
        return None, None, [_issue("missing_processing_report", "Manifest 未引用 processing report。", manifest_path, root)]
    path = (manifest_path.parent / manifest.processing_report.path).resolve(strict=False)
    if not path.is_file():
        return None, path, [_issue("missing_processing_report", "Processing report 文件不存在。", path, root)]
    try:
        payload = _read_json(path, root)
        report = ProcessingReport.model_validate(payload)
        if manifest.processing_report.report_id != report.build_id:
            raise ValueError("processing report ID does not match manifest")
        output = report.outputs.get(manifest.stage)
        if not output or output.get("dataset_id") != manifest.dataset_id:
            raise ValueError("processing report output does not identify this dataset")
        if output.get("content_sha256") != manifest.hashes.get("canonical_records_sha256"):
            raise ValueError("processing report output digest does not match manifest")
        return report, path, []
    except (OSError, ValueError, ValidationError, KeyError, TypeError) as exc:
        return None, path, [_issue("invalid_processing_report", f"Processing report 无效：{exc}", path, root)]


def _load_quality(root: Path, manifest_path: Path, manifest: DatasetManifest,
                  *, bundle_root: Path | None = None,
                  bundle_entry: QualityAssessmentEntry | None = None,
                  bundle_selected: bool = False) -> tuple[QualityReport | None, Path | None, list[ReadinessIssue]]:
    if bundle_selected:
        if bundle_root is None or bundle_entry is None:
            return None, None, [_issue("invalid_quality_assessment", "复评 bundle 缺少该 dataset 的质量报告。")]
        if (bundle_entry.dataset_id != manifest.dataset_id
                or bundle_entry.content_sha256 != manifest.hashes.get("canonical_records_sha256")
                or bundle_entry.stage != manifest.stage):
            return None, None, [_issue("invalid_quality_assessment", "复评报告与 manifest 的 dataset ID、stage 或内容摘要不匹配。")]
        path = (bundle_root / bundle_entry.report_path).resolve(strict=False)
        try:
            path.relative_to(bundle_root)
            payload = _read_json(path, bundle_root)
            report = QualityReport.model_validate(payload)
            if report.quality_report_id != bundle_entry.report_id:
                raise ValueError("quality report ID does not match bundle index")
            if report.dataset_id != manifest.dataset_id:
                raise ValueError("quality report dataset ID does not match manifest")
            if report.content_sha256 != manifest.hashes.get("canonical_records_sha256"):
                raise ValueError("quality report digest does not match manifest")
            return report, path, []
        except (OSError, ValueError, ValidationError, KeyError, TypeError) as exc:
            return None, path, [_issue("invalid_quality_assessment", f"复评报告无效：{exc}", path, bundle_root)]
    if manifest.quality_report is None:
        return None, None, [_issue("missing_quality_report", "Manifest 未引用 quality report。", manifest_path, root)]
    path = (manifest_path.parent / manifest.quality_report.path).resolve(strict=False)
    if not path.is_file():
        return None, path, [_issue("missing_quality_report", "Quality report 文件不存在。", path, root)]
    try:
        payload = _read_json(path, root)
        report = QualityReport.model_validate(payload)
        if report.quality_report_id != manifest.quality_report.report_id:
            raise ValueError("quality report ID does not match manifest")
        if report.dataset_id != manifest.dataset_id:
            raise ValueError("quality report dataset ID does not match manifest")
        if report.content_sha256 != manifest.hashes.get("canonical_records_sha256"):
            raise ValueError("quality report digest does not match manifest")
        return report, path, []
    except (OSError, ValueError, ValidationError, KeyError, TypeError) as exc:
        return None, path, [_issue("invalid_quality_report", f"Quality report 无效：{exc}", path, root)]


def _profile_count_source(profile: MiniDatasetProfile, processing: ProcessingReport | None) -> tuple[int | None, str | None]:
    if processing is None:
        return None, None
    parts = profile.count_source.split(".")
    if len(parts) != 4 or parts[:2] != ["processing_report", "outputs"]:
        return None, None
    output = processing.outputs.get(parts[2])
    if parts[2] != profile.stage:
        return None, None
    if not isinstance(output, dict):
        return None, None
    return output.get(parts[3]), output.get("budget_unit") if parts[3] == "budget_count" else None


def _dimension(status: str, summary: str, *, observed: int | None = None, target: int | None = None,
               unit: str | None = None, evidence: list[str] | None = None,
               issues: list[ReadinessIssue] | None = None) -> ReadinessDimension:
    return ReadinessDimension(status=status, summary=summary, observed_count=observed, target=target,
                              unit=unit, evidence=evidence or [], issues=issues or [])


def _dataset_readiness(root: Path, profile: MiniDatasetProfile, lineage_result: dict[str, Any],
                       *, bundle_root: Path | None = None,
                       bundle_entry: QualityAssessmentEntry | None = None,
                       bundle_selected: bool = False,
                       review_bundle_root: Path | None = None,
                       review_entry: QualityReviewBundleEntry | None = None,
                       review_bundle_selected: bool = False) -> MiniDatasetReadiness:
    manifest_path, manifest, find_issues = _find_manifest(root, profile)
    if manifest is None or manifest_path is None:
        missing = _dimension("missing", "无法评估：manifest 缺失或不唯一。", issues=find_issues)
        return MiniDatasetReadiness(
            dataset_key=profile.dataset_key, stage=profile.stage, required=profile.required,
            budget=missing, quality_coverage=missing, quality_evidence=missing,
            quality_review=missing,
            processing_evidence=missing, lineage=missing,
            overall_status="not_ready" if profile.required else "not_evaluated",
            blockers=find_issues if profile.required else [],
        )

    manifest_ref = manifest_path.relative_to(root).as_posix()
    manifest_errors = validate_manifest_evidence(manifest_path)
    processing_manifest_issues = [
        _issue("manifest_evidence", message, manifest_path, root)
        for message in manifest_errors if message.startswith("processing_report")
    ]
    quality_manifest_issues = [
        _issue("manifest_evidence", message, manifest_path, root)
        for message in manifest_errors if message.startswith("quality_report")
    ]
    processing, processing_path, processing_issues = _load_processing(root, manifest_path, manifest)
    quality, quality_path, quality_issues = _load_quality(
        root, manifest_path, manifest, bundle_root=bundle_root,
        bundle_entry=bundle_entry, bundle_selected=bundle_selected,
    )
    blockers: list[ReadinessIssue] = []
    warnings: list[ReadinessIssue] = []

    observed, observed_unit = _profile_count_source(profile, processing)
    budget_issues: list[ReadinessIssue] = []
    if not profile.budget_required:
        budget = _dimension("not_required", "该留出 split 不计入训练预算。", observed=observed,
                            unit=observed_unit or profile.unit, evidence=[manifest_ref])
    elif observed is None or observed_unit is None:
        issue = _issue("budget_unavailable", "Processing report 未提供精确预算计数及单位。", processing_path, root)
        budget_issues.append(issue)
        budget = _dimension("missing", issue.message, target=profile.target, unit=profile.unit,
                            evidence=[manifest_ref], issues=budget_issues)
    elif observed_unit != profile.unit:
        issue = _issue("budget_unit_mismatch", f"期望单位 {profile.unit}，实际为 {observed_unit}。", processing_path, root)
        budget_issues.append(issue)
        budget = _dimension("invalid", issue.message, observed=observed, target=profile.target,
                            unit=observed_unit, evidence=[manifest_ref], issues=budget_issues)
    elif observed < (profile.target or 0):
        issue = _issue("budget_shortfall", f"预算不足：{observed}/{profile.target} {profile.unit}。", processing_path, root)
        budget_issues.append(issue)
        budget = _dimension("fail", issue.message, observed=observed, target=profile.target,
                            unit=profile.unit, evidence=[manifest_ref], issues=budget_issues)
    else:
        budget = _dimension("pass", f"已达预算：{observed}/{profile.target} {profile.unit}。",
                            observed=observed, target=profile.target, unit=profile.unit,
                            evidence=[manifest_ref])

    proc_evidence = [manifest_ref] + ([processing_path.relative_to(root).as_posix()] if processing_path and processing_path.is_relative_to(root) else [])
    if not profile.processing_report_required:
        proc_dimension = _dimension("not_required", "Profile 未要求 processing report。", evidence=proc_evidence)
    elif processing is None or processing_issues or processing_manifest_issues:
        issues = [*processing_issues, *processing_manifest_issues]
        proc_dimension = _dimension("invalid" if any(i.code not in {"missing_processing_report"} for i in issues) else "missing",
                                    "Processing evidence 缺失或无效。", evidence=proc_evidence, issues=issues)
    elif not processing.processing_steps:
        issue = _issue("missing_processing_steps", "Processing report 没有步骤证据。", processing_path, root)
        proc_dimension = _dimension("missing", issue.message, evidence=proc_evidence, issues=[issue])
    else:
        proc_dimension = _dimension("pass", "Processing report、dataset ID、digest 和步骤均有效。", evidence=proc_evidence)

    quality_evidence = [manifest_ref] + ([quality_path.relative_to(root).as_posix()] if quality_path and quality_path.is_relative_to(root) else [])
    if not profile.quality_report_required:
        quality_coverage = _dimension("not_required", "Profile 未要求 quality report。", evidence=quality_evidence)
        quality_dimension = _dimension("not_required", "Profile 未要求 quality report。", evidence=quality_evidence)
    elif quality is None or quality_issues or quality_manifest_issues:
        issues = [*quality_issues, *quality_manifest_issues]
        missing_only = all(
            issue.code == "missing_quality_report"
            or (issue.code == "manifest_evidence" and "quality_report is missing" in issue.message)
            for issue in issues
        )
        quality_status = "missing" if missing_only else "invalid"
        quality_coverage = _dimension(quality_status, "Quality evidence 缺失或无效。", evidence=quality_evidence, issues=issues)
        quality_dimension = _dimension(quality_status, "Quality evidence 缺失或无效。", evidence=quality_evidence, issues=issues)
    elif quality.gate.summary_status == "FAIL":
        issue = _issue("quality_gate_fail", "Quality report 包含阻断性结构检查失败。", quality_path, root)
        if quality.evidence_coverage is None:
            quality_coverage = _dimension("not_evaluated", "旧版报告未记录 metric catalog 覆盖状态。", evidence=quality_evidence)
        elif quality.evidence_coverage.status == "complete":
            quality_coverage = _dimension("pass", "每项适用指标均有显式状态（不代表指标通过）。", evidence=quality_evidence)
        else:
            coverage_issue = _issue("quality_coverage_incomplete", "质量证据目录不完整。", quality_path, root)
            quality_coverage = _dimension("warning", "质量证据目录不完整。", evidence=quality_evidence, issues=[coverage_issue])
        quality_dimension = _dimension("fail", issue.message, evidence=quality_evidence, issues=[issue])
    else:
        issue = _issue("quality_not_evaluated", "质量报告存在，但当前协议尚未完成整体语义质量评估。", quality_path, root)
        quality_dimension = _dimension("not_evaluated", issue.message, evidence=quality_evidence, issues=[issue])
        if quality.evidence_coverage is None:
            quality_coverage = _dimension("not_evaluated", "旧版报告未记录 metric catalog 覆盖状态。", evidence=quality_evidence)
        elif quality.evidence_coverage.status == "complete":
            quality_coverage = _dimension("pass", "每项适用指标均有显式状态（不代表指标通过）。", evidence=quality_evidence)
        else:
            coverage_issue = _issue("quality_coverage_incomplete", "质量证据目录不完整。", quality_path, root)
            quality_coverage = _dimension("warning", "质量证据目录不完整。", evidence=quality_evidence, issues=[coverage_issue])

    usage = classify_dataset_usage(manifest.metadata)
    if not review_bundle_selected or review_bundle_root is None or review_entry is None:
        review_issue = _issue("missing_quality_review", "缺少显式 stage-quality review bundle；质量审核未批准。")
        review_dimension = _dimension("missing", review_issue.message, evidence=[manifest_ref], issues=[review_issue])
    elif (review_entry.dataset_id != manifest.dataset_id or review_entry.stage != manifest.stage
          or review_entry.content_sha256 != manifest.hashes.get("canonical_records_sha256")
          or review_entry.usage != usage or quality is None
          or review_entry.quality_report_id != quality.quality_report_id
          or quality_path is None or sha256_file(quality_path) != review_entry.quality_report_sha256):
        review_issue = _issue("quality_review_binding_mismatch", "Review 与当前 manifest/quality report 的身份或 digest 不匹配。")
        review_dimension = _dimension("invalid", review_issue.message, evidence=[manifest_ref], issues=[review_issue])
    elif usage != "baseline":
        review_issue = _issue("non_official_usage", f"用途为 {usage}；只有显式 baseline 数据可以满足官方 Mini readiness。")
        review_dimension = _dimension("fail", review_issue.message, evidence=[manifest_ref, review_entry.review_path], issues=[review_issue])
    elif review_entry.outcome == "approved":
        review_dimension = _dimension("pass", "存在与当前数据及质量报告绑定的 approved review。",
                                       evidence=[manifest_ref, (review_bundle_root / review_entry.review_path).relative_to(root).as_posix()])
        if quality_dimension.status == "not_evaluated":
            quality_dimension = _dimension("pass", "Raw quality evidence 未单独给出总体语义结论；关联的 stage-quality review 已批准。",
                                             evidence=quality_evidence)
    elif review_entry.outcome == "rejected":
        review_issue = _issue("quality_review_rejected", "Stage-quality review 已拒绝该数据集。")
        review_dimension = _dimension("fail", review_issue.message,
                                      evidence=[manifest_ref, (review_bundle_root / review_entry.review_path).relative_to(root).as_posix()], issues=[review_issue])
    else:
        review_issue = _issue("quality_review_pending", "Stage-quality review 尚未批准。")
        review_dimension = _dimension("not_evaluated", review_issue.message,
                                      evidence=[manifest_ref, (review_bundle_root / review_entry.review_path).relative_to(root).as_posix()], issues=[review_issue])

    lineage_errors = lineage_result.get("errors", [])
    lineage_warnings = lineage_result.get("warnings", [])
    if lineage_errors:
        issues = [ReadinessIssue(code=item.get("code", "lineage_error"), message=item.get("message", "血缘校验失败"),
                                 path=item.get("path")) for item in lineage_errors]
        warnings.extend(
            ReadinessIssue(code=item.get("code", "lineage_warning"), message=item.get("message", "血缘证据有警告"),
                           path=item.get("path"))
            for item in lineage_warnings
        )
        lineage_dimension = _dimension("fail", "Artifact root 的完整血缘校验失败。", evidence=[manifest_ref], issues=issues)
    elif lineage_warnings:
        issues = [ReadinessIssue(code=item.get("code", "lineage_warning"), message=item.get("message", "血缘证据有警告"),
                                 path=item.get("path")) for item in lineage_warnings]
        lineage_dimension = _dimension("warning", "本地引用完整；部分外部证据不可用或仅警告。",
                                       evidence=[manifest_ref], issues=issues)
    else:
        lineage_dimension = _dimension("pass", "完整血缘校验通过。", evidence=[manifest_ref])

    if usage != "baseline":
        blockers.append(_issue("non_official_usage", f"用途为 {usage}；官方 Mini 只能接收 baseline 数据。", manifest_path, root))
    dimensions = [budget, quality_coverage, quality_dimension, review_dimension, proc_dimension, lineage_dimension]
    for dimension in dimensions:
        if dimension.status in {"fail", "missing", "invalid", "not_evaluated"}:
            for blocker in dimension.issues or [_issue(dimension.status, dimension.summary)]:
                if blocker not in blockers:
                    blockers.append(blocker)
        elif dimension.status == "warning":
            warnings.extend(dimension.issues)
    if not profile.required:
        blockers = []
        overall = "not_evaluated"
    elif not blockers:
        overall = "ready"
    elif any(d.status in {"fail", "missing", "invalid"} for d in dimensions):
        overall = "not_ready"
    else:
        overall = "not_evaluated"
    return MiniDatasetReadiness(
        dataset_key=profile.dataset_key, dataset_id=manifest.dataset_id, stage=manifest.stage,
        required=profile.required, usage=classify_dataset_usage(manifest.metadata),
        budget=budget, quality_coverage=quality_coverage,
        quality_evidence=quality_dimension, quality_review=review_dimension,
        processing_evidence=proc_dimension, lineage=lineage_dimension,
        overall_status=overall, blockers=blockers, warnings=warnings,
    )


def evaluate_mini_readiness(root: str | Path, profile_path: str | Path,
                            quality_assessment_path: str | Path | None = None,
                            quality_review_path: str | Path | None = None) -> MiniReadinessReport:
    """Inspect existing Mini artifacts and return a typed, non-mutating readiness report."""
    base = Path(root).resolve(strict=True)
    if not base.is_dir():
        raise ValueError("artifact root must be a directory")
    profile, profile_hash = load_mini_profile(profile_path)
    lineage_result = validate_lineage_root(base, require_complete=True)
    bundle_root: Path | None = None
    bundle_entries: dict[str, QualityAssessmentEntry] = {}
    review_bundle_root: Path | None = None
    review_entries: dict[str, QualityReviewBundleEntry] = {}
    if quality_assessment_path is not None:
        supplied = Path(quality_assessment_path).resolve(strict=True)
        supplied.relative_to(base)
        bundle_root, bundle = load_quality_assessment_bundle(supplied)
        bundle_entries = {entry.dataset_key: entry for entry in bundle.datasets}
    if quality_review_path is not None:
        supplied_review = Path(quality_review_path).resolve(strict=True)
        supplied_review.relative_to(base)
        review_bundle_root, review_bundle = load_quality_review_bundle(supplied_review)
        review_entries = {entry.dataset_key: entry for entry in review_bundle.datasets}
    datasets = [
        _dataset_readiness(
            base, item, lineage_result, bundle_root=bundle_root,
            bundle_entry=bundle_entries.get(item.dataset_key),
            bundle_selected=quality_assessment_path is not None,
            review_bundle_root=review_bundle_root,
            review_entry=review_entries.get(item.dataset_key),
            review_bundle_selected=quality_review_path is not None,
        )
        for item in profile.datasets
    ]
    required = [item for item in datasets if item.required]
    blockers = [issue for item in required for issue in item.blockers]
    warnings = [issue for item in datasets for issue in item.warnings]
    if not blockers and all(item.overall_status == "ready" for item in required):
        overall = "ready"
    elif any(item.overall_status == "not_ready" for item in required):
        overall = "not_ready"
    else:
        overall = "not_evaluated"
    return MiniReadinessReport(
        profile_id=profile.profile_id, profile_version=profile.profile_version,
        profile_sha256=profile_hash, assessment_id=f"mini-check-{uuid4().hex}", artifact_root=str(base),
        created_at=datetime.now(timezone.utc).isoformat(), datasets=datasets,
        overall_status=overall, blockers=blockers, warnings=warnings,
    )


def write_mini_readiness_report(report: MiniReadinessReport, output_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Write immutable JSON/Markdown summaries under the artifact root."""
    root = Path(report.artifact_root).resolve(strict=True)
    target = Path(output_dir) if output_dir is not None else root / "reports" / "mini-readiness" / report.assessment_id
    target = target.resolve(strict=False)
    target.relative_to(root)
    if target.exists():
        raise FileExistsError(f"Readiness report output already exists: {target}")
    target.mkdir(parents=True)
    json_path = target / "mini_readiness.json"
    markdown_path = target / "mini_readiness.md"
    payload = report.model_dump(mode="json", by_alias=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def _render_markdown(report: MiniReadinessReport) -> str:
    def status(value: str) -> str:
        return value.replace("_", " ").upper()

    lines = [
        "# Mini 数据就绪评估",
        "",
        f"- 状态：**{status(report.overall_status)}**",
        f"- Assessment：`{report.assessment_id}`",
        f"- Profile：`{report.profile_id}@{report.profile_version}` (`{report.profile_sha256}`)",
        f"- Artifact root：`{report.artifact_root}`",
        f"- 评估时间：`{report.created_at}`",
        f"- 模型效果：**{status(report.model_effect_status)}**",
        "",
        "## 阶段结果",
        "",
        "| 数据集 | 阶段 | 用途 | ID | 预算 | 质量证据 | 质量审核 | 处理证据 | 血缘 | 状态 | 来源 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    root = Path(report.artifact_root)
    output_relative = Path("reports") / "mini-readiness" / report.assessment_id
    for item in report.datasets:
        budget = f"{item.budget.observed_count}/{item.budget.target} {item.budget.unit}" if item.budget.target is not None else f"{item.budget.observed_count} {item.budget.unit or ''}".strip()
        evidence_refs = list(dict.fromkeys(ref for dimension in (
            item.budget, item.quality_evidence, item.quality_review, item.processing_evidence, item.lineage,
        ) for ref in dimension.evidence))
        links = []
        for ref in evidence_refs:
            try:
                target = root / ref
                relative = Path(os.path.relpath(target, root / output_relative)).as_posix()
                links.append(f"[{Path(ref).name}]({relative})")
            except (OSError, ValueError):
                links.append(f"`{ref}`")
        lines.append(
            f"| {item.dataset_key} | {item.stage} | {item.usage} | `{item.dataset_id or '—'}` | {budget} ({status(item.budget.status)}) "
            f"| {status(item.quality_evidence.status)} | {status(item.quality_review.status)} | {status(item.processing_evidence.status)} "
            f"| {status(item.lineage.status)} | {status(item.overall_status)} | {'<br>'.join(links) or '—'} |"
        )
    for heading, issues in (("阻断项", report.blockers), ("警告", report.warnings)):
        lines.extend(["", f"## {heading}", ""])
        if not issues:
            lines.append("无。")
        else:
            lines.extend(f"- `{issue.code}` {issue.message}" + (f" — `{issue.path}`" if issue.path else "") for issue in issues)
    lines.extend(["", "> 预算达成、质量证据和血缘完整性是独立维度；本报告不证明模型效果或清洗的因果收益。", ""])
    return "\n".join(lines)
