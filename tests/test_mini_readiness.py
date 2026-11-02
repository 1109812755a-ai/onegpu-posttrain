import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from onegpu_posttrain.cli import build_parser
from onegpu_posttrain.data.build import (
    _budget_evidence,
    build_finemath_artifacts,
    build_openr1_artifacts,
)
from onegpu_posttrain.data.evidence import ProcessingReport
from onegpu_posttrain.data.finemath import FineMathBuildConfig
from onegpu_posttrain.data.manifest import load_manifest, save_manifest, sha256_file
from onegpu_posttrain.data.mini import (
    MiniDataProfile,
    MiniDatasetProfile,
    MiniDatasetReadiness,
    MiniReadinessReport,
    ReadinessDimension,
    load_mini_profile,
)
from onegpu_posttrain.data.mini_readiness import (
    _find_manifest,
    _profile_count_source,
    evaluate_mini_readiness,
    write_mini_readiness_report,
)
from onegpu_posttrain.data.openr1 import OpenR1BuildConfig
from onegpu_posttrain.data.quality_assessment import (
    create_quality_assessment_bundle,
    load_quality_assessment_bundle,
)
from onegpu_posttrain.data.quality_review import (
    QualityReview,
    QualityReviewBundleEntry,
    QualityReviewBundleIndex,
    QualityReviewCheck,
    ReviewerIdentity,
    quality_review_evidence_sha256,
)
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.lineage import ArtifactRef, Digest, RunRecord

ROOT = Path(__file__).parents[1]
PROFILE = ROOT / "configs/data/mini-v1.yaml"
PROFILE_V2 = ROOT / "configs/data/mini-v2.yaml"


def test_mini_profile_declares_stage_budgets_and_units():
    profile, digest = load_mini_profile(PROFILE)
    datasets = {dataset.dataset_key: dataset for dataset in profile.datasets}

    assert profile.profile_id == "mini-data-readiness"
    assert len(digest) == 64
    assert (datasets["finemath.cpt_train"].target, datasets["finemath.cpt_train"].unit) == (5_000_000, "tokens")
    assert (datasets["openr1.sft_train"].target, datasets["openr1.sft_train"].unit) == (5_000, "examples")
    assert (datasets["openr1.preference_train"].target, datasets["openr1.preference_train"].unit) == (2_000, "preference_pairs")
    assert (datasets["openr1.grpo_train"].target, datasets["openr1.grpo_train"].unit) == (2_000, "prompts")
    assert datasets["openr1.preference_eval"].target is None
    assert datasets["openr1.preference_eval"].unit == "preference_pairs"


def test_mini_v2_profile_keeps_deepmath_and_openr1_sft_cohorts_independent():
    profile, _ = load_mini_profile(PROFILE_V2)
    datasets = {dataset.dataset_key: dataset for dataset in profile.datasets}
    assert datasets["deepmath.sft_train"].required is True
    assert datasets["deepmath.sft_train"].target == 5000
    assert datasets["deepmath.sft_train"].manifest_path.endswith("deepmath-sft/manifests/sft-train-v1.yaml")
    assert datasets["openr1.sft_train"].required is False
    assert datasets["openr1.sft_train"].target is None
    assert datasets["openr1.sft_train"].manifest_path.endswith("openr1/manifests/sft-smoke-v1.yaml")


def test_mini_profile_document_matches_profile_and_references_resolve():
    document = ROOT / "docs/implementation/data/MINI_PROFILE_zh.md"
    content = document.read_text(encoding="utf-8")
    profile, _ = load_mini_profile(PROFILE)
    targets = {dataset.dataset_key: dataset.target for dataset in profile.datasets}
    assert f"{targets['finemath.cpt_train']:,}" in content
    assert f"{targets['openr1.sft_train']:,}" in content
    assert f"{targets['openr1.preference_train']:,}" in content
    assert f"{targets['openr1.grpo_train']:,}" in content
    for target in ("../../../configs/data/mini-v1.yaml", "../../design/DATA_QUALITY_DESIGN_zh.md", "DATA_EVIDENCE.md", "../../design/LINEAGE_DESIGN_zh.md"):
        assert (document.parent / target).resolve().is_file()


def test_mini_profile_rejects_budget_without_target():
    with pytest.raises(ValidationError, match="positive target"):
        MiniDatasetProfile(
            dataset_key="openr1.preference_train",
            stage="preference",
            budget_required=True,
            target=None,
            unit="preference_pairs",
            count_source="processing_report.outputs.preference.budget_count",
        )


def test_mini_profile_rejects_duplicate_dataset_keys():
    item = MiniDatasetProfile(
        dataset_key="openr1.sft_train",
        stage="sft",
        target=5,
        unit="examples",
        count_source="processing_report.outputs.sft.budget_count",
    )
    with pytest.raises(ValidationError, match="must be unique"):
        MiniDataProfile(
            schema_version="1.0",
            profile_id="test",
            profile_version="1",
            description="fixture",
            datasets=[item, item],
        )


def test_profile_manifest_path_selects_independent_same_stage_cohorts(tmp_path):
    from onegpu_posttrain.data.manifest import (
        DatasetManifest,
        SourceSpec,
        save_manifest,
    )

    root = tmp_path / "artifact-root"
    for name in ("openr1", "deepmath"):
        save_manifest(
            root / name / "manifests" / "sft.yaml",
            DatasetManifest(
                dataset_id=f"ds-{name}", stage="sft", source=SourceSpec(repo=name, revision="r1"),
                sample_count=1,
            ),
            allow_legacy=True,
        )
    profile = MiniDatasetProfile(
        dataset_key="deepmath.sft_train", stage="sft", target=5, unit="examples",
        count_source="processing_report.outputs.sft.budget_count",
        manifest_path="deepmath/manifests/sft.yaml",
    )
    path, manifest, issues = _find_manifest(root, profile)
    assert path == (root / "deepmath/manifests/sft.yaml").resolve()
    assert manifest.dataset_id == "ds-deepmath"
    assert issues == []


def test_profile_manifest_path_cannot_escape_artifact_root(tmp_path):
    profile = MiniDatasetProfile(
        dataset_key="deepmath.sft_train", stage="sft", target=5, unit="examples",
        count_source="processing_report.outputs.sft.budget_count",
        manifest_path="../outside.yaml",
    )
    path, manifest, issues = _find_manifest(tmp_path / "root", profile)
    assert path is None and manifest is None
    assert issues[0].code == "manifest_path_escape"


@pytest.mark.parametrize(
    ("stage", "expected_unit", "count", "target"),
    [
        ("cpt", "tokens", 1234, 1000),
        ("cpt_holdout", "tokens", 234, 200),
        ("sft", "examples", 11, 10),
        ("preference", "preference_pairs", 12, 10),
        ("preference_eval", "preference_pairs", 5, 4),
        ("grpo", "prompts", 13, 10),
        ("eval", "examples", 6, 5),
    ],
)
def test_builder_budget_evidence_uses_stage_units(stage, expected_unit, count, target):
    if stage.startswith("cpt"):
        report = {
            "train" if stage == "cpt" else "holdout": {"tokens": count, "token_budget": target},
        }
    else:
        report = {
            "selected": {"holdout" if stage == "eval" else stage: count},
            "targets": {"holdout" if stage == "eval" else stage: target},
        }
    evidence = _budget_evidence(stage, report, [{}] * count if not stage.startswith("cpt") else [])
    assert evidence == {"budget_count": count, "budget_target": target, "budget_unit": expected_unit}


def test_profile_count_source_resolves_exact_stage_budget_count():
    processing = ProcessingReport(
        build_id="build-fixture",
        created_at="2026-10-17T00:00:00Z",
        processing_protocol="fixture-v1",
        source={},
        tokenizer={},
        benchmark_screening={},
        config={},
        config_sha256="a" * 64,
        stop_reason="targets_met",
        processing_steps=[],
        outputs={"preference": {"dataset_id": "ds-pref", "budget_count": 7, "budget_unit": "preference_pairs"}},
    )
    profile = MiniDatasetProfile(
        dataset_key="openr1.preference_train",
        stage="preference",
        target=5,
        unit="preference_pairs",
        count_source="processing_report.outputs.preference.budget_count",
    )
    assert _profile_count_source(profile, processing) == (7, "preference_pairs")
    mismatched = profile.model_copy(update={"count_source": "processing_report.outputs.grpo.budget_count"})
    assert _profile_count_source(mismatched, processing) == (None, None)


def _openr1_rows(count=30):
    for i in range(count):
        yield {
            "problem": f"Compute x + {i} = {i + 2}.",
            "solution": "2",
            "answer": "2",
            "problem_type": "algebra",
            "question_type": "math-word-problem",
            "source": "fixture",
            "uuid": f"mini-row-{i}",
            "is_reasoning_complete": [True, True],
            "generations": [f"Subtract {i}. Final answer: 2", f"Wrong result. Final answer: {i + 3}"],
            "correctness_math_verify": [True, False],
            "correctness_llama": [True, False],
            "finish_reasons": ["stop", "stop"],
            "correctness_count": 1,
        }


def _finemath_rows(count=100):
    for i in range(count):
        yield {"url": f"fixture://doc/{i}", "text": f"Math document {i}. " * 60, "score": 4.0, "int_score": 4}


def test_readiness_joins_all_stage_counts_to_linked_builder_artifacts(tmp_path):
    root = tmp_path / "artifacts"
    finemath = root / "finemath"
    openr1 = root / "openr1"
    build_finemath_artifacts(
        _finemath_rows(), output_dir=finemath, tokenizer=WhitespaceTokenizer(),
        config=FineMathBuildConfig(train_token_budget=100, holdout_token_budget=40, max_seq_tokens=30, holdout_permille=200),
        source_repo="fixture/finemath", source_config=None, source_revision="r1",
        tokenizer_repo="fixture/tokenizer", tokenizer_revision="t1",
    )
    build_openr1_artifacts(
        _openr1_rows(), output_dir=openr1, tokenizer=WhitespaceTokenizer(),
        config=OpenR1BuildConfig(sft_target=4, alignment_target=8, holdout_target=4,
                                 preference_target=4, preference_eval_target=2, grpo_target=4,
                                 max_raw_rows=100, sft_max_length=128, preference_max_length=128,
                                 grpo_max_prompt_length=128),
        source_repo="fixture/openr1", source_config=None, source_revision="r1",
        tokenizer_repo="fixture/tokenizer", tokenizer_revision="t1",
    )
    profile_payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    profile_payload["datasets"] = [
        {"dataset_key": "finemath.cpt_train", "stage": "cpt", "target": 1, "unit": "tokens",
         "count_source": "processing_report.outputs.cpt.budget_count"},
        {"dataset_key": "finemath.cpt_holdout", "stage": "cpt_holdout", "budget_required": False, "target": None,
         "unit": "tokens", "count_source": "processing_report.outputs.cpt_holdout.budget_count"},
        {"dataset_key": "openr1.sft_train", "stage": "sft", "target": 1, "unit": "examples",
         "count_source": "processing_report.outputs.sft.budget_count"},
        {"dataset_key": "openr1.preference_train", "stage": "preference", "target": 1, "unit": "preference_pairs",
         "count_source": "processing_report.outputs.preference.budget_count"},
        {"dataset_key": "openr1.preference_eval", "stage": "preference_eval", "budget_required": False,
         "target": None, "unit": "preference_pairs", "count_source": "processing_report.outputs.preference_eval.budget_count"},
        {"dataset_key": "openr1.grpo_train", "stage": "grpo", "target": 1, "unit": "prompts",
         "count_source": "processing_report.outputs.grpo.budget_count"},
        {"dataset_key": "openr1.internal_holdout", "stage": "eval", "budget_required": False,
         "target": None, "unit": "examples", "count_source": "processing_report.outputs.eval.budget_count"},
    ]
    profile_path = tmp_path / "test-mini-profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile_payload, sort_keys=False), encoding="utf-8")

    before = {path.relative_to(root): sha256_file(path) for path in root.rglob("*") if path.is_file()}
    report = evaluate_mini_readiness(root, profile_path)
    datasets = {item.dataset_key: item for item in report.datasets}
    assert datasets["finemath.cpt_train"].budget.observed_count >= 100
    assert datasets["finemath.cpt_train"].budget.unit == "tokens"
    assert datasets["finemath.cpt_holdout"].budget.status == "not_required"
    assert datasets["openr1.sft_train"].budget.observed_count == 4
    assert datasets["openr1.preference_train"].budget.observed_count == 4
    assert datasets["openr1.preference_eval"].budget.observed_count == 2
    assert datasets["openr1.grpo_train"].budget.observed_count == 4
    assert datasets["openr1.internal_holdout"].budget.observed_count == 4
    assert all(item.dataset_id for item in report.datasets)
    assert report.schema_version == "1.1"
    assert datasets["openr1.sft_train"].quality_coverage.status == "pass"
    assert datasets["openr1.sft_train"].quality_evidence.status == "not_evaluated"
    assert report.overall_status == "not_ready"
    assert datasets["openr1.sft_train"].usage == "legacy_unknown"
    assert datasets["openr1.sft_train"].quality_review.status == "missing"
    assert report.model_effect_status == "not_evaluated"
    json_path, markdown_path = write_mini_readiness_report(report)
    assert json_path.is_file() and markdown_path.is_file()
    serialized = json.loads(json_path.read_text(encoding="utf-8"))
    assert serialized["assessment_id"] == report.assessment_id
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "NOT EVALUATED" in markdown
    assert "../.." in markdown
    assert {path.relative_to(root): sha256_file(path) for path in root.rglob("*")
            if path.is_file() and "mini-readiness" not in path.parts} == before


def _single_sft_profile(path: Path, target: int = 1):
    profile_payload = {
        "schema_version": "1.0", "profile_id": "mini-fixture", "profile_version": "1",
        "description": "test profile", "datasets": [{
            "dataset_key": "openr1.sft_train", "stage": "sft", "target": target,
            "unit": "examples", "count_source": "processing_report.outputs.sft.budget_count",
        }],
    }
    path.write_text(yaml.safe_dump(profile_payload, sort_keys=False), encoding="utf-8")
    return path


def _build_openr1_fixture(root: Path):
    return build_openr1_artifacts(
        _openr1_rows(), output_dir=root, tokenizer=WhitespaceTokenizer(),
        config=OpenR1BuildConfig(sft_target=4, alignment_target=8, holdout_target=4,
                                 preference_target=4, preference_eval_target=2, grpo_target=4,
                                 max_raw_rows=100, sft_max_length=128, preference_max_length=128,
                                 grpo_max_prompt_length=128),
        source_repo="fixture/openr1", source_config=None, source_revision="r1",
        tokenizer_repo="fixture/tokenizer", tokenizer_revision="t1",
    )


def test_readiness_shortfall_and_unmeasured_quality_are_not_passes(tmp_path):
    root = tmp_path / "openr1"
    _build_openr1_fixture(root)
    profile = _single_sft_profile(tmp_path / "mini.yaml", target=5000)
    report = evaluate_mini_readiness(root, profile)
    item = report.datasets[0]
    manifest_path = next((root / "manifests").glob("sft-*.yaml"))
    manifest = load_manifest(manifest_path)
    quality_payload = json.loads((manifest_path.parent / manifest.quality_report.path).read_text(encoding="utf-8"))

    assert item.budget.status == "fail"
    assert item.budget.observed_count == 4
    assert item.quality_evidence.status == "not_evaluated"
    assert item.quality_coverage.status == "pass"
    assert quality_payload["metrics"]["semantic_correctness"]["status"] == "not_measured"
    assert report.overall_status == "not_ready"


@pytest.mark.parametrize("usage", ["baseline", "exploratory", "intentionally_noisy", "legacy_unknown"])
def test_official_readiness_requires_approved_review_and_explicit_baseline_usage(tmp_path, usage):
    root = tmp_path / "openr1"
    _build_openr1_fixture(root)
    profile_path = _single_sft_profile(tmp_path / "profile.yaml")
    manifest_path = next(root.rglob("manifests/sft-smoke-v1.yaml"))
    manifest = load_manifest(manifest_path)
    if usage == "legacy_unknown":
        manifest.metadata.pop("usage", None)
    else:
        manifest.metadata["usage"] = usage
    save_manifest(manifest_path, manifest)
    quality_path = (manifest_path.parent / manifest.quality_report.path).resolve()
    quality_digest = sha256_file(quality_path)
    review_path = root / "review-bundle" / "reviews/sft.json"
    review_path.parent.mkdir(parents=True)
    checks = [QualityReviewCheck(check_id="human_reasoning_review", method="manual_review",
        evidence_status="not_measured", decision="pass", rationale="Reviewed fixture records.")]
    review = QualityReview(
        review_id="review-sft-fixture", dataset_key="openr1.sft_train", dataset_id=manifest.dataset_id,
        stage="sft", content_sha256=manifest.hashes["canonical_records_sha256"],
        quality_report_id=manifest.quality_report.report_id, quality_report_sha256=quality_digest,
        profile_id="mini-stage-quality-review", profile_version="1", profile_sha256="c" * 64,
        checker_name="fixture", checker_version="1", usage=usage,
        reviewer=ReviewerIdentity(reviewer_id="reviewer", method="human_review"),
        checks=checks,
        outcome="approved", evidence_sha256=quality_review_evidence_sha256(checks),
    )
    review_path.write_text(review.model_dump_json(), encoding="utf-8")
    entry = QualityReviewBundleEntry(
        dataset_key="openr1.sft_train", dataset_id=manifest.dataset_id, stage="sft",
        content_sha256=manifest.hashes["canonical_records_sha256"],
        quality_report_id=manifest.quality_report.report_id, quality_report_sha256=quality_digest,
        review_id=review.review_id, review_path="reviews/sft.json", review_sha256=sha256_file(review_path),
        profile_sha256=review.profile_sha256, usage=usage, outcome="approved",
    )
    index = QualityReviewBundleIndex(
        assessment_id="review-bundle-fixture", created_at="2026-10-19T00:00:00Z",
        profile_id=review.profile_id, profile_version=review.profile_version,
        profile_sha256=review.profile_sha256, source_quality_assessment_id="quality-fixture",
        datasets=[entry],
    )
    index_path = review_path.parents[1] / "index.json"
    index_path.write_text(index.model_dump_json(), encoding="utf-8")
    report = evaluate_mini_readiness(root, profile_path, quality_review_path=index_path)
    dataset = report.datasets[0]
    if usage == "baseline":
        assert dataset.quality_review.status == "pass"
        assert dataset.overall_status == "ready"
    else:
        assert dataset.quality_review.status == "fail"
        assert dataset.overall_status == "not_ready"


def test_quality_reassessment_bundle_is_additive_and_digest_bound(tmp_path):
    root = tmp_path / "openr1"
    _build_openr1_fixture(root)
    profile = _single_sft_profile(tmp_path / "mini.yaml", target=4)
    before = {path.relative_to(root): sha256_file(path) for path in root.rglob("*")
              if path.is_file()}
    output = root / "reports" / "reassessments" / "fixture-assessment"
    index, index_path = create_quality_assessment_bundle(root, profile, output)

    assert index_path.is_file()
    assert len(index.datasets) == 1
    entry = index.datasets[0]
    assert entry.dataset_key == "openr1.sft_train"
    assert entry.stage == "sft"
    assert len(entry.content_sha256) == 64
    payload = json.loads((output / entry.report_path).read_text(encoding="utf-8"))
    assert payload["dataset_id"] == entry.dataset_id
    assert payload["content_sha256"] == entry.content_sha256
    assert payload["evidence_coverage"]["status"] == "complete"
    assert {path.relative_to(root): sha256_file(path) for path in root.rglob("*")
            if path.is_file() and not path.is_relative_to(output)} == before


def test_readiness_consumes_explicit_quality_bundle_and_keeps_approval_separate(tmp_path):
    root = tmp_path / "openr1"
    _build_openr1_fixture(root)
    profile = _single_sft_profile(tmp_path / "mini.yaml", target=4)
    output = root / "reports" / "reassessments" / "bundle"
    _, index_path = create_quality_assessment_bundle(root, profile, output)

    report = evaluate_mini_readiness(root, profile, index_path)
    dataset = report.datasets[0]
    assert dataset.quality_coverage.status == "pass"
    assert dataset.quality_evidence.status == "not_evaluated"
    assert report.overall_status != "ready"
    assert any("reassessments/bundle/quality/" in ref for ref in dataset.quality_evidence.evidence)


def test_quality_bundle_rejects_wrong_digest_tampered_report_and_escaping_path(tmp_path):
    root = tmp_path / "openr1"
    _build_openr1_fixture(root)
    profile = _single_sft_profile(tmp_path / "mini.yaml", target=4)
    output = root / "reports" / "reassessments" / "bundle"
    _, index_path = create_quality_assessment_bundle(root, profile, output)
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    entry = json.loads(json.dumps(index_payload["datasets"][0]))

    index_payload["datasets"][0]["content_sha256"] = "b" * 64
    index_path.write_text(json.dumps(index_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch"):
        load_quality_assessment_bundle(index_path)

    index_payload["datasets"][0] = entry
    report_path = output / entry["report_path"]
    report_path.write_text(report_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    index_path.write_text(json.dumps(index_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_quality_assessment_bundle(index_path)

    outside = output.parent / "outside.json"
    outside.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    entry["report_path"] = "../outside.json"
    entry["report_sha256"] = sha256_file(outside)
    index_payload["datasets"][0] = entry
    index_path.write_text(json.dumps(index_payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_quality_assessment_bundle(index_path)


def test_documented_quality_reassessment_cli_workflow_runs_on_local_fixture(tmp_path, capsys):
    root = tmp_path / "fixture-artifacts"
    _build_openr1_fixture(root)
    profile = _single_sft_profile(tmp_path / "mini.yaml", target=4)
    bundle_dir = root / "reports" / "mini-quality-assessments" / "fixture-review"

    assess_args = build_parser().parse_args([
        "assess-mini-data-quality", "--root", str(root), "--profile", str(profile),
        "--output", str(bundle_dir),
    ])
    assert assess_args.func(assess_args) == 0
    assess_payload = json.loads(capsys.readouterr().out)
    index_path = Path(assess_payload["index"])
    assert index_path.is_file()

    readiness_args = build_parser().parse_args([
        "validate-mini-data", "--root", str(root), "--profile", str(profile),
        "--quality-assessment", str(index_path),
    ])
    assert readiness_args.func(readiness_args) == 2
    readiness_payload = json.loads(capsys.readouterr().out)
    readiness = json.loads(Path(readiness_payload["json_report"]).read_text(encoding="utf-8"))
    dataset = readiness["datasets"][0]
    assert dataset["quality_coverage"]["status"] == "pass"
    assert dataset["quality_evidence"]["status"] == "not_evaluated"
    assert readiness["model_effect_status"] == "not_evaluated"


def test_readiness_marks_missing_and_crosslinked_quality_reports(tmp_path):
    root = tmp_path / "openr1"
    _build_openr1_fixture(root)
    profile = _single_sft_profile(tmp_path / "mini.yaml")
    manifest_path = next((root / "manifests").glob("sft-*.yaml"))
    manifest = load_manifest(manifest_path)
    quality_path = manifest_path.parent / manifest.quality_report.path
    quality_path.unlink()
    missing_report = evaluate_mini_readiness(root, profile)
    assert missing_report.datasets[0].quality_evidence.status == "missing"

    other_root = tmp_path / "openr1-mismatch"
    _build_openr1_fixture(other_root)
    other_manifest_path = next((other_root / "manifests").glob("sft-*.yaml"))
    other_manifest = load_manifest(other_manifest_path)
    other_quality_path = other_manifest_path.parent / other_manifest.quality_report.path
    payload = json.loads(other_quality_path.read_text(encoding="utf-8"))
    payload["dataset_id"] = "ds-not-the-manifest-dataset"
    other_quality_path.write_text(json.dumps(payload), encoding="utf-8")
    other_manifest.quality_report = other_manifest.quality_report.model_copy(
        update={"sha256": sha256_file(other_quality_path)}
    )
    save_manifest(other_manifest_path, other_manifest)
    mismatch_report = evaluate_mini_readiness(other_root, profile)
    assert mismatch_report.datasets[0].quality_evidence.status == "invalid"


def test_readiness_preserves_structural_quality_failure(tmp_path):
    root = tmp_path / "finemath"
    duplicate = {"url": "fixture://duplicate", "text": "mathematical text " * 30, "score": 4.0, "int_score": 4}
    build_finemath_artifacts(
        [duplicate, duplicate], output_dir=root, tokenizer=WhitespaceTokenizer(),
        config=FineMathBuildConfig(train_token_budget=1000, holdout_token_budget=0, max_seq_tokens=10, holdout_permille=0),
        source_repo="fixture/finemath", source_config=None, source_revision="r1",
        tokenizer_repo="fixture/tokenizer", tokenizer_revision="t1",
    )
    profile_payload = {
        "schema_version": "1.0", "profile_id": "mini-fixture", "profile_version": "1",
        "description": "test profile", "datasets": [{
            "dataset_key": "finemath.cpt_train", "stage": "cpt", "target": 1,
            "unit": "tokens", "count_source": "processing_report.outputs.cpt.budget_count",
        }],
    }
    profile = tmp_path / "mini-cpt.yaml"
    profile.write_text(yaml.safe_dump(profile_payload, sort_keys=False), encoding="utf-8")
    report = evaluate_mini_readiness(root, profile)
    assert report.datasets[0].quality_evidence.status == "fail"
    assert any(issue.code == "quality_gate_fail" for issue in report.datasets[0].quality_evidence.issues)


def test_readiness_lineage_keeps_integrity_errors_and_external_warnings_separate(tmp_path):
    root = tmp_path / "openr1"
    _build_openr1_fixture(root)
    profile = _single_sft_profile(tmp_path / "mini.yaml")
    duplicate_run_id = "run-fixture-duplicate"
    missing_ref = ArtifactRef(
        id="missing-file", kind="file", path=str(root / "missing.jsonl"),
        digest=Digest(value="a" * 64, status="available"),
    )
    bad_digest_ref = ArtifactRef(
        id="tampered-file", kind="file", path=str(root / "canonical/sft.jsonl"),
        digest=Digest(value="b" * 64, status="available"),
    )
    for folder, run_id, artifact in (("duplicate-a", duplicate_run_id, missing_ref),
                                     ("duplicate-b", duplicate_run_id, bad_digest_ref)):
        path = root / "runs" / folder / "lineage.json"
        path.parent.mkdir(parents=True)
        run = RunRecord(run_id=run_id, run_type="train", status="completed",
                        run_label=folder, started_at="2026-10-17T00:00:00Z", inputs=[artifact])
        path.write_text(run.model_dump_json(), encoding="utf-8")

    legacy_path = root / "runs/legacy/lineage.json"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(json.dumps({"run_id": "legacy-run", "stage": "sft", "inputs": {}}), encoding="utf-8")
    incomplete_path = root / "runs/incomplete/lineage.json"
    incomplete_path.parent.mkdir(parents=True)
    incomplete = RunRecord(run_id="run-incomplete", run_type="train", status="completed",
                           lineage_status="incomplete", run_label="incomplete",
                           started_at="2026-10-17T00:00:00Z")
    incomplete_path.write_text(incomplete.model_dump_json(), encoding="utf-8")
    external_path = root / "runs/external/lineage.json"
    external_path.parent.mkdir(parents=True)
    external = RunRecord(
        run_id="run-external", run_type="train", status="completed", run_label="external",
        started_at="2026-10-17T00:00:00Z",
        inputs=[ArtifactRef(id="model-external", kind="model", path="org/model",
                            digest=Digest(status="unavailable", reason="外部模型未下载"))],
    )
    external_path.write_text(external.model_dump_json(), encoding="utf-8")

    report = evaluate_mini_readiness(root, profile)
    lineage = report.datasets[0].lineage
    error_codes = {issue.code for issue in lineage.issues}
    warning_codes = {issue.code for issue in report.warnings}
    assert lineage.status == "fail"
    assert {"duplicate_run_id", "dangling_local_ref", "digest_mismatch", "incomplete"} <= error_codes
    assert {"legacy", "unavailable_evidence"} <= warning_codes


def _dimension(status="pass"):
    return ReadinessDimension(status=status, summary="fixture", evidence=["manifest.yaml"])


def test_readiness_report_serializes_independent_dimensions():
    dataset = MiniDatasetReadiness(
        dataset_key="openr1.sft_train",
        dataset_id="ds-sft-mini",
        stage="sft",
        budget=ReadinessDimension(status="pass", summary="5000/5000", observed_count=5000, target=5000, unit="examples"),
        quality_coverage=ReadinessDimension(status="pass", summary="all metrics enumerated"),
        quality_evidence=_dimension(),
        quality_review=_dimension(),
        processing_evidence=_dimension(),
        lineage=_dimension(),
        overall_status="ready",
    )
    report = MiniReadinessReport(
        profile_id="mini-data-readiness",
        profile_version="1",
        profile_sha256="a" * 64,
        assessment_id="mini-check-fixture",
        artifact_root="artifacts/mini",
        created_at="2026-10-17T00:00:00Z",
        datasets=[dataset],
        overall_status="ready",
    )

    payload = report.model_dump(mode="json")
    assert payload["datasets"][0]["budget"]["observed_count"] == 5000
    assert payload["datasets"][0]["lineage"]["status"] == "pass"
    assert payload["datasets"][0]["quality_coverage"]["status"] == "pass"
    assert payload["model_effect_status"] == "not_evaluated"
    assert MiniReadinessReport.model_validate(payload) == report


def test_readiness_report_cannot_call_unevaluated_quality_ready():
    with pytest.raises(ValidationError, match="all dimensions to pass"):
        MiniDatasetReadiness(
            dataset_key="openr1.sft_train",
            stage="sft",
            budget=_dimension(),
            quality_coverage=_dimension(),
            quality_evidence=_dimension("not_evaluated"),
            quality_review=_dimension(),
            processing_evidence=_dimension(),
            lineage=_dimension(),
            overall_status="ready",
        )


def test_readiness_report_rejects_claim_without_required_ready_dataset():
    dataset = MiniDatasetReadiness(
        dataset_key="openr1.sft_train",
        stage="sft",
        budget=_dimension("fail"),
        quality_evidence=_dimension(),
        quality_review=_dimension("missing"),
        processing_evidence=_dimension(),
        lineage=_dimension(),
        overall_status="not_ready",
    )
    with pytest.raises(ValidationError, match="all required datasets to be ready"):
        MiniReadinessReport(
            profile_id="mini-data-readiness",
            profile_version="1",
            profile_sha256="b" * 64,
            assessment_id="mini-check-invalid",
            artifact_root="artifacts/mini",
            created_at="2026-10-17T00:00:00Z",
            datasets=[dataset],
            overall_status="ready",
        )
