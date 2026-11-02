"""Versioned policy and artifact contracts for human-readable data quality review."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from pydantic import Field, ValidationError, model_validator

from ..evaluation.verification import verification_configuration
from ..synthetic.verification import assess_candidate_answers_batch
from .evidence import EvidenceModel, QualityReport
from .manifest import load_manifest, sha256_file
from .quality_assessment import (
    _load_processing_output,
    _read_rows,
    load_quality_assessment_bundle,
)

ReviewEvidenceStatus = Literal[
    "measured", "estimated", "not_measured", "unsupported", "not_applicable",
]
ReviewDecision = Literal["pass", "fail", "pending", "not_applicable"]
ReviewOutcome = Literal["approved", "rejected", "pending"]
ReviewUsage = Literal["baseline", "exploratory", "intentionally_noisy", "legacy_unknown"]
ReviewStage = Literal["cpt", "cpt_holdout", "sft", "preference", "preference_eval", "grpo", "eval"]


def classify_dataset_usage(metadata: dict[str, object] | None) -> ReviewUsage:
    """Interpret usage metadata without changing the source manifest."""
    usage = metadata.get("usage") if metadata else None
    if isinstance(usage, str) and usage in {"baseline", "exploratory", "intentionally_noisy"}:
        return usage
    return "legacy_unknown"


def quality_review_evidence_sha256(checks: list[QualityReviewCheck]) -> str:
    payload = [check.model_dump(mode="json") for check in checks]
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ReviewCheckRule(EvidenceModel):
    check_id: str = Field(min_length=1)
    metric_id: str | None = None
    method: str = Field(min_length=1)
    requirement: Literal["required", "diagnostic", "not_applicable"] = "required"
    unsupported_action: Literal["block", "pending", "not_applicable"] = "pending"
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_rule(self):
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        if self.requirement == "required" and self.unsupported_action == "not_applicable":
            raise ValueError("required checks cannot treat unsupported evidence as not applicable")
        return self


class StageQualityReviewProfile(EvidenceModel):
    checks: list[ReviewCheckRule] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_check_ids(self):
        ids = [check.check_id for check in self.checks]
        if len(ids) != len(set(ids)):
            raise ValueError("check_id values must be unique within a stage")
        return self


class QualityReviewProfile(EvidenceModel):
    schema_version: Literal["1.0"]
    profile_id: str = Field(min_length=1)
    profile_version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    stages: dict[ReviewStage, StageQualityReviewProfile] = Field(min_length=1)

    @property
    def profile_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class QualityReviewCheck(EvidenceModel):
    check_id: str
    metric_id: str | None = None
    evidence_status: ReviewEvidenceStatus
    value: int | float | None = None
    unit: str | None = None
    method: str
    requirement: Literal["required", "diagnostic", "not_applicable"] = "required"
    numerator: int | None = Field(default=None, ge=0)
    denominator: int | None = Field(default=None, ge=0)
    counts: dict[str, int] = Field(default_factory=dict)
    coverage: float | None = Field(default=None, ge=0, le=1)
    decision: ReviewDecision
    rationale: str

    @model_validator(mode="after")
    def preserve_unavailable_evidence(self):
        if self.evidence_status in {"unsupported", "not_measured"} and self.decision == "pass" and "manual" not in self.method:
            raise ValueError("unavailable automated evidence cannot satisfy a check")
        return self


class ReviewerIdentity(EvidenceModel):
    reviewer_id: str = Field(min_length=1)
    method: Literal["automated_policy", "human_review"]


class QualityReview(EvidenceModel):
    schema_version: Literal["1.0"] = "1.0"
    review_id: str
    dataset_key: str
    dataset_id: str
    stage: ReviewStage
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_report_id: str
    quality_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_id: str
    profile_version: str
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checker_name: str
    checker_version: str
    usage: ReviewUsage = "legacy_unknown"
    reviewer: ReviewerIdentity
    reviewed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    supersedes_review_id: str | None = None
    outcome: ReviewOutcome = "pending"
    checks: list[QualityReviewCheck]
    limitations: list[str] = Field(default_factory=list)
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_outcome(self):
        required = [check for check in self.checks if check.requirement == "required"]
        if self.outcome == "approved" and (
            self.reviewer.method != "human_review" or not required or any(check.decision != "pass" for check in required)
        ):
            raise ValueError("approved review requires a human reviewer and all required checks to pass")
        if self.outcome == "rejected" and not any(
            check.requirement == "required" and check.decision == "fail" for check in self.checks
        ):
            raise ValueError("rejected review requires a failed required check")
        expected_evidence_sha = quality_review_evidence_sha256(self.checks)
        if self.evidence_sha256 != expected_evidence_sha:
            raise ValueError("review evidence digest does not match its checks")
        return self


class QualityReviewBundleEntry(EvidenceModel):
    dataset_key: str
    dataset_id: str
    stage: ReviewStage
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_report_id: str
    quality_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_id: str
    review_path: str
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    usage: ReviewUsage
    outcome: ReviewOutcome


class QualityReviewBundleIndex(EvidenceModel):
    schema_version: Literal["1.0"] = "1.0"
    assessment_id: str
    created_at: str
    profile_id: str
    profile_version: str
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_quality_assessment_id: str
    datasets: list[QualityReviewBundleEntry]

    @model_validator(mode="after")
    def unique_dataset_keys(self):
        keys = [entry.dataset_key for entry in self.datasets]
        if len(keys) != len(set(keys)):
            raise ValueError("quality review bundle contains duplicate dataset keys")
        return self


class HumanReviewDecisions(EvidenceModel):
    reviewer_id: str = Field(min_length=1)
    decisions: dict[str, dict[str, dict[str, str]]]

    @model_validator(mode="after")
    def validate_decisions(self):
        for dataset_key, checks in self.decisions.items():
            if not dataset_key or not checks:
                raise ValueError("human review decisions require dataset and check entries")
            for check_id, decision in checks.items():
                if not check_id or decision.get("decision") not in {"pass", "fail", "pending"}:
                    raise ValueError("manual decision must be pass, fail, or pending")
                if not decision.get("rationale", "").strip():
                    raise ValueError("manual decisions require a non-empty rationale")
        return self


def apply_manual_review_decisions(
    checks: list[QualityReviewCheck],
    rules: list[ReviewCheckRule],
    decisions: dict[str, dict[str, str]],
) -> list[QualityReviewCheck]:
    """Apply human judgments only to declared manual checks, preserving evidence state."""
    rule_by_id = {rule.check_id: rule for rule in rules}
    check_by_id = {check.check_id: check for check in checks}
    unknown = set(decisions) - set(rule_by_id)
    if unknown:
        raise ValueError(f"manual decisions reference unknown checks: {sorted(unknown)}")
    for check_id, decision in decisions.items():
        check = check_by_id[check_id]
        if "manual" not in rule_by_id[check_id].method:
            raise ValueError(f"manual decision cannot replace automated check {check_id}")
        check.decision = decision["decision"]
        check.rationale = decision["rationale"]
    return checks


def load_quality_review_profile(path: str | Path) -> QualityReviewProfile:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    return QualityReviewProfile.model_validate(payload)


def load_quality_review_bundle(path: str | Path) -> tuple[Path, QualityReviewBundleIndex]:
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("quality review path cannot be a symlink")
    supplied = supplied.resolve(strict=True)
    index_path = supplied / "index.json" if supplied.is_dir() else supplied
    if index_path.is_symlink() or not index_path.is_file() or index_path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("quality review index must be a regular, bounded local file")
    root = index_path.parent.resolve(strict=True)
    index = QualityReviewBundleIndex.model_validate(json.loads(index_path.read_text(encoding="utf-8")))
    for entry in index.datasets:
        unresolved = root / entry.review_path
        if unresolved.is_symlink():
            raise ValueError(f"quality review path is a symlink: {entry.review_path}")
        review_path = unresolved.resolve(strict=True)
        review_path.relative_to(root)
        if not review_path.is_file() or sha256_file(review_path) != entry.review_sha256:
            raise ValueError(f"quality review digest mismatch: {entry.review_path}")
        review = QualityReview.model_validate(json.loads(review_path.read_text(encoding="utf-8")))
        expected = (entry.review_id, entry.dataset_id, entry.stage, entry.content_sha256,
                    entry.quality_report_id, entry.quality_report_sha256, entry.profile_sha256,
                    entry.usage, entry.outcome)
        actual = (review.review_id, review.dataset_id, review.stage, review.content_sha256,
                  review.quality_report_id, review.quality_report_sha256, review.profile_sha256,
                  review.usage, review.outcome)
        if actual != expected:
            raise ValueError(f"quality review binding mismatch: {entry.review_path}")
        if review.profile_id != index.profile_id or review.profile_version != index.profile_version:
            raise ValueError("quality review profile identity mismatch")
        if index.profile_sha256 != entry.profile_sha256:
            raise ValueError("quality review profile identity mismatch")
    return root, index


def _rule_check(rule: ReviewCheckRule, report: QualityReport) -> QualityReviewCheck:
    metric = report.metrics.get(rule.metric_id) if rule.metric_id else None
    if rule.requirement == "not_applicable":
        return QualityReviewCheck(check_id=rule.check_id, metric_id=rule.metric_id,
            evidence_status="not_applicable", method=rule.method, decision="not_applicable",
            rationale="该检查在当前 stage profile 中标记为不适用。")
    if not metric or metric.status not in {"measured", "estimated"} or metric.value is None:
        status = "unsupported" if metric and metric.status == "unsupported" else "not_measured"
        decision = {"block": "fail", "pending": "pending", "not_applicable": "not_applicable"}[rule.unsupported_action]
        return QualityReviewCheck(check_id=rule.check_id, metric_id=rule.metric_id,
        evidence_status=status, method=rule.method, requirement=rule.requirement, decision=decision,
            rationale="质量报告没有可用于本规则的已测量指标；保留未测状态。")
    value = metric.value
    passes = (rule.minimum is None or value >= rule.minimum) and (rule.maximum is None or value <= rule.maximum)
    decision: ReviewDecision = ("pass" if passes else "fail") if rule.minimum is not None or rule.maximum is not None else "pending"
    rationale = "指标符合 profile 阈值。" if decision == "pass" else (
        "指标未达到 profile 阈值。" if decision == "fail" else "已测量但 profile 未定义决策阈值；保持 pending。"
    )
    return QualityReviewCheck(check_id=rule.check_id, metric_id=rule.metric_id,
        evidence_status="measured" if metric.status == "measured" else "estimated", value=value,
        unit=metric.unit, method=rule.method, numerator=metric.numerator,
        denominator=metric.denominator, requirement=rule.requirement, decision=decision, rationale=rationale)


def _semantic_checks(stage: str, rows: list[dict]) -> dict[str, dict]:
    if stage not in {"sft", "preference", "preference_eval"}:
        return {}
    solutions: list[str] = []
    references: list[object | None] = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        reference = metadata.get("reference_answer")
        if stage == "sft":
            messages = row.get("messages", [])
            solution = next((item.get("content") for item in reversed(messages)
                             if isinstance(item, dict) and item.get("role") == "assistant"), "")
            solutions.append(str(solution or ""))
            references.append(reference)
        else:
            for key in ("chosen", "rejected"):
                solutions.append(str(row.get(key, "")))
                references.append(reference)
    assessments = assess_candidate_answers_batch(solutions, references)
    statuses: Counter[str] = Counter(assessment.status for assessment in assessments)
    denominator = len(assessments)
    decided = statuses["correct"] + statuses["incorrect"]
    attempted = decided + statuses["unparseable"] + statuses["timeout"] + statuses["verifier_error"]
    verifiers = sorted({f"{item.verifier}@{item.verifier_version}" for item in assessments if item.verifier})
    return {
        "counts": dict(sorted(statuses.items())), "denominator": denominator,
        "coverage": attempted / denominator if denominator else 0.0,
        "decision_coverage": decided / denominator if denominator else 0.0,
        "evidence_status": "measured" if denominator else "not_measured",
        "protocol": verification_configuration(), "verifiers": verifiers,
    }


def _canonical_check(rows: list[dict], stage: str) -> tuple[float, int, int]:
    from ..schemas import STAGE_MODELS
    model = {"cpt_holdout": STAGE_MODELS["cpt"], "preference_eval": STAGE_MODELS["preference"]}.get(stage)
    model = model or STAGE_MODELS.get(stage)
    if model is None:
        return 0.0, 0, len(rows)
    valid = 0
    for row in rows:
        try:
            model.model_validate(row)
            valid += 1
        except ValidationError:
            continue
    return valid / len(rows) if rows else 0.0, valid, len(rows)


def create_quality_review_bundle(
    artifact_root: str | Path,
    mini_profile_path: str | Path,
    quality_assessment_path: str | Path,
    review_profile_path: str | Path,
    output_dir: str | Path | None = None,
    decisions_path: str | Path | None = None,
    supersedes_bundle_path: str | Path | None = None,
) -> tuple[dict, Path]:
    """Create an offline immutable review bundle bound to each exact dataset and report."""
    from .mini import load_mini_profile

    root = Path(artifact_root).resolve(strict=True)
    profile, _ = load_mini_profile(mini_profile_path)
    review_profile = load_quality_review_profile(review_profile_path)
    missing_stages = {dataset.stage for dataset in profile.datasets} - set(review_profile.stages)
    if missing_stages:
        raise ValueError(f"review profile is missing Mini stages: {sorted(missing_stages)}")
    bundle_root, assessment_index = load_quality_assessment_bundle(quality_assessment_path)
    bundle_root.relative_to(root)
    manual_decisions = None
    if decisions_path is not None:
        decision_file = Path(decisions_path).resolve(strict=True)
        manual_decisions = HumanReviewDecisions.model_validate(json.loads(decision_file.read_text(encoding="utf-8")))
        profile_keys = {dataset.dataset_key for dataset in profile.datasets}
        unknown_keys = set(manual_decisions.decisions) - profile_keys
        if unknown_keys:
            raise ValueError(f"manual decisions reference unknown dataset keys: {sorted(unknown_keys)}")
    superseded_by_key = {}
    if supersedes_bundle_path is not None:
        prior_root, prior_index = load_quality_review_bundle(supersedes_bundle_path)
        prior_root.relative_to(root)
        superseded_by_key = {entry.dataset_key: entry.review_id for entry in prior_index.datasets}
    assessment_by_key = {item.dataset_key: item for item in assessment_index.datasets}
    output_id = f"mini-review-{uuid4().hex}"
    target = Path(output_dir).resolve(strict=False) if output_dir else root / "reports" / "mini-quality-reviews" / output_id
    target.relative_to(root)
    if target.exists():
        raise FileExistsError(f"Review output already exists: {target}")
    target.mkdir(parents=True)
    reviews_dir = target / "reviews"
    reviews_dir.mkdir()
    entries = []
    for dataset_profile in profile.datasets:
        if dataset_profile.manifest_path:
            path = (root / dataset_profile.manifest_path).resolve(strict=False)
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"{dataset_profile.dataset_key}: manifest_path escapes artifact root") from exc
            matches = [(path, load_manifest(path))] if path.is_file() else []
            if matches and matches[0][1].stage != dataset_profile.stage:
                matches = []
        else:
            matches = [(path, load_manifest(path)) for path in root.rglob("manifests/*")
                       if path.is_file() and not path.is_symlink() and path.suffix.lower() in {".yaml", ".yml", ".json"}
                       and load_manifest(path).stage == dataset_profile.stage]
        if len(matches) != 1:
            raise ValueError(f"{dataset_profile.dataset_key}: expected one manifest for {dataset_profile.stage}, found {len(matches)}")
        manifest_path, manifest = matches[0]
        entry = assessment_by_key.get(dataset_profile.dataset_key)
        if entry is None or entry.dataset_id != manifest.dataset_id or entry.stage != manifest.stage:
            raise ValueError(f"{dataset_profile.dataset_key}: quality assessment does not match dataset")
        data_path, _ = _load_processing_output(root, manifest_path, manifest)
        if sha256_file(data_path) != manifest.hashes.get("canonical_records_sha256"):
            raise ValueError(f"{manifest.dataset_id}: canonical digest mismatch")
        rows = _read_rows(data_path)
        report_path = (bundle_root / entry.report_path).resolve(strict=True)
        report = QualityReport.model_validate(json.loads(report_path.read_text(encoding="utf-8")))
        rules = review_profile.stages[manifest.stage].checks
        checks = []
        semantic = _semantic_checks(manifest.stage, rows)
        for rule in rules:
            if rule.method == "canonical_schema:v1":
                rate, numerator, denominator = _canonical_check(rows, manifest.stage)
                checks.append(QualityReviewCheck(check_id=rule.check_id, metric_id=rule.metric_id,
                    evidence_status="measured", value=rate, unit="ratio", numerator=numerator,
                    denominator=denominator, method=rule.method,
                    requirement=rule.requirement,
                    decision="pass" if rate >= (rule.minimum or 0) else "fail",
                    rationale=f"通过 canonical schema 校验 {numerator}/{denominator} 条。"))
            elif rule.method == "exact_record_id:v1":
                ids = [row.get("id") for row in rows]
                duplicates = len(ids) - len(set(ids))
                checks.append(QualityReviewCheck(check_id=rule.check_id, metric_id=rule.metric_id,
                    evidence_status="measured", value=duplicates, unit="records", numerator=duplicates,
                    denominator=len(ids), method=rule.method, requirement=rule.requirement,
                    decision="pass" if duplicates == 0 else "fail",
                    rationale=f"发现重复 record ID：{duplicates}。"))
            elif rule.check_id in {"independent_semantic_correctness", "independent_pair_ordering"}:
                evidence = semantic
                counts = evidence.get("counts", {})
                decided = counts.get("correct", 0) + counts.get("incorrect", 0)
                verifier_names = ",".join(evidence.get("verifiers", [])) or "unavailable"
                config = evidence.get("protocol", {})
                method = (
                    f"{rule.method};{config.get('protocol', 'unknown')};{verifier_names};"
                    f"batch={config.get('batch_size')};deadline={config.get('batch_timeout_seconds')}s;"
                    f"max-chars={config.get('max_answer_chars')};"
                    f"inner-timeout={config.get('math_verify_internal_timeout')}"
                )
                checks.append(QualityReviewCheck(check_id=rule.check_id, metric_id=rule.metric_id,
                    evidence_status=evidence.get("evidence_status", "not_measured"),
                    value=counts.get("correct", 0) / decided if decided else None,
                    unit="verified_response_ratio", numerator=counts.get("correct", 0),
                    denominator=decided or None, counts=counts,
                    coverage=evidence.get("decision_coverage", 0.0), method=method, requirement=rule.requirement,
                    decision="pending", rationale=(
                        f"答案核验决定覆盖率 {evidence.get('decision_coverage', 0.0):.3f}，"
                        f"尝试覆盖率 {evidence.get('coverage', 0.0):.3f}，分母 {evidence.get('denominator', 0)}；"
                        "超时、错误及未验证样本未当作错误；答案等价不代表推理质量，且无自动审批阈值。"
                    )))
            else:
                checks.append(_rule_check(rule, report))
        reviewer = ReviewerIdentity(reviewer_id="automated-policy", method="automated_policy")
        manual_by_check = (manual_decisions.decisions.get(dataset_profile.dataset_key, {})
                           if manual_decisions else {})
        apply_manual_review_decisions(checks, rules, manual_by_check)
        if manual_by_check:
            reviewer = ReviewerIdentity(reviewer_id=manual_decisions.reviewer_id, method="human_review")
        required_checks = [check for check in checks if check.requirement == "required"]
        outcome: ReviewOutcome = "approved" if required_checks and all(c.decision == "pass" for c in required_checks) else (
            "rejected" if any(c.decision == "fail" for c in required_checks) else "pending")
        usage = classify_dataset_usage(manifest.metadata)
        evidence_hash = quality_review_evidence_sha256(checks)
        review = QualityReview(
            review_id=f"{output_id}-{dataset_profile.dataset_key.replace('.', '-')}",
            dataset_key=dataset_profile.dataset_key, dataset_id=manifest.dataset_id,
            stage=manifest.stage, content_sha256=manifest.hashes["canonical_records_sha256"],
            quality_report_id=report.quality_report_id, quality_report_sha256=sha256_file(report_path),
            profile_id=review_profile.profile_id, profile_version=review_profile.profile_version,
            profile_sha256=review_profile.profile_sha256, checker_name="onegpu-posttrain-quality-review",
            checker_version="1", usage=usage,
            reviewer=reviewer, checks=checks, outcome=outcome,
            supersedes_review_id=superseded_by_key.get(dataset_profile.dataset_key),
            evidence_sha256=evidence_hash,
            limitations=["Independent answer matching does not evaluate reasoning correctness.",
                         "Unsupported/unverified responses are not counted as incorrect.",
                         "No model training or network access is performed."],
        )
        review_path = reviews_dir / f"{dataset_profile.dataset_key.replace('.', '-')}.json"
        review_path.write_text(json.dumps(review.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        entries.append({"dataset_key": dataset_profile.dataset_key, "dataset_id": manifest.dataset_id,
            "stage": manifest.stage, "content_sha256": review.content_sha256,
            "quality_report_id": report.quality_report_id,
            "quality_report_sha256": review.quality_report_sha256,
            "review_id": review.review_id, "review_path": review_path.relative_to(target).as_posix(),
            "review_sha256": sha256_file(review_path), "profile_sha256": review_profile.profile_sha256,
            "usage": usage, "outcome": review.outcome})
    index = QualityReviewBundleIndex.model_validate({"schema_version": "1.0", "assessment_id": output_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile_id": review_profile.profile_id, "profile_version": review_profile.profile_version,
        "profile_sha256": review_profile.profile_sha256,
        "source_quality_assessment_id": assessment_index.assessment_id, "datasets": entries})
    index_path = target / "index.json"
    index_path.write_text(json.dumps(index.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index.model_dump(mode="json"), index_path
