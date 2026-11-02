import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from policy_support import policy_fixture

from onegpu_posttrain.data.io import write_jsonl
from onegpu_posttrain.data.manifest import load_manifest, validate_manifest_evidence
from onegpu_posttrain.data.policy import DataPolicy, PolicyRules, exploration_preflight
from onegpu_posttrain.data.policy_build import (
    InputUnavailable,
    build_policy_artifacts,
    execute_policy,
    observe_candidates,
    select_record,
)
from onegpu_posttrain.data.quality_assessment import create_quality_assessment_bundle
from onegpu_posttrain.data.quality_review import (
    create_quality_review_bundle,
    load_quality_review_bundle,
)
from onegpu_posttrain.data.source_cache import SourceIdentity, create_source_cache
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.lineage_validation import validate_lineage_root


def test_diagnostics_disambiguate_correct_but_long_and_unknown(tmp_path):
    policy, _, rows = policy_fixture(tmp_path)
    policy.resources.max_length = 3
    counter = Counter()
    record, reason = select_record(policy, rows[0], "p", rows[0]["problem"],
                                  observe_candidates(rows[0], rows[0]["problem"], WhitespaceTokenizer()), counter, {})
    assert record is None and reason == "correct_candidates_all_over_length"
    assert counter["correct_candidates_all_over_length"] == 1
    assert counter["no_known_correct_candidate"] == 0
    policy = DataPolicy(stage="preference", tokenizer=policy.tokenizer)
    rows[0]["correctness_math_verify"][1] = None
    _, reason = select_record(policy, rows[0], "p", rows[0]["problem"],
                              observe_candidates(rows[0], rows[0]["problem"], WhitespaceTokenizer()), Counter(), {})
    assert reason == "no_known_wrong_candidate"


def test_report_only_preserves_actual_quality_states(tmp_path):
    policy, source, rows = policy_fixture(tmp_path)
    for row in rows:
        row["correctness_math_verify"] = [None, None]
        row["is_reasoning_complete"] = [False, False]
    write_jsonl(source.input_jsonl, rows)
    assert not execute_policy(policy, source)[0]
    policy.rules = PolicyRules(correctness="report_only", completeness="report_only")
    records, _ = execute_policy(policy, source)
    assert len(records) == 3
    assert all(r["metadata"]["correctness_math_verify"] is None for r in records)
    assert all(r["metadata"]["is_reasoning_complete"] is False for r in records)


def test_preview_fixed_split_and_full_prefix_agree(tmp_path):
    policy, source, _ = policy_fixture(tmp_path)
    records, report = execute_policy(policy, source, preview_rows=20)
    complete, full = execute_policy(policy, source)
    assert complete == records
    assert report["partial"] and report["stop_reason"] == "max_raw_rows"
    assert report["source_pool"] == full["source_pool"]
    policy.resources.target_count = 6
    _, other = execute_policy(policy, source, preview_rows=20)
    assert report["split_snapshot"] == other["split_snapshot"]
    assert report["source_pool"] == other["source_pool"]
    assert execute_policy(policy, source, preview_rows=20, seconds=0)[1]["stop_reason"] == "time_budget"


def test_offline_missing_inputs_never_fetch(tmp_path, monkeypatch):
    policy, source, _ = policy_fixture(tmp_path)
    def no_download(*args, **kwargs):
        pytest.fail("unexpected tokenizer load")
    monkeypatch.setattr("onegpu_posttrain.data.policy_build.HuggingFaceTokenizer", no_download)
    execute_policy(policy, source)
    source.benchmark_jsonl = str(tmp_path / "missing.jsonl")
    with pytest.raises(InputUnavailable, match="污染索引"):
        execute_policy(policy, source)


def test_real_tokenizer_missing_is_strictly_local(tmp_path, monkeypatch):
    policy, source, _ = policy_fixture(tmp_path)
    policy.tokenizer.test_only = False
    def missing(repo, revision, *, local_files_only):
        assert local_files_only is True
        raise OSError("not cached")
    monkeypatch.setattr("onegpu_posttrain.data.policy_build.HuggingFaceTokenizer", missing)
    with pytest.raises(InputUnavailable, match="不会自动下载"):
        execute_policy(policy, source)


def test_cache_reuse_and_content_pin(tmp_path):
    policy, source, rows = policy_fixture(tmp_path)
    cached, _, _ = create_source_cache(rows, cache_root=tmp_path / "cache", identity=SourceIdentity(
        repo=source.source_repo, config=None, revision="v1", split="train"))
    original = execute_policy(policy, source)[0]
    source.input_jsonl = None
    source.source_cache = str(cached)
    assert execute_policy(policy, source)[0] == original
    source.expected_sha256 = "0" * 64
    with pytest.raises(ValueError, match="摘要"):
        execute_policy(policy, source)


def test_policy_artifacts_reproducible_and_complete(tmp_path):
    policy, source, _ = policy_fixture(tmp_path / "input")
    before = Path(source.input_jsonl).read_bytes()
    a = build_policy_artifacts(policy, source, output_dir=tmp_path / "a")
    b = build_policy_artifacts(policy, source, output_dir=tmp_path / "b")
    assert a["dataset_id"] == b["dataset_id"]
    assert a["report"]["data_build_run_id"] != b["report"]["data_build_run_id"]
    assert not validate_manifest_evidence(a["manifest"])
    result = validate_lineage_root(tmp_path / "a", require_complete=True)
    assert not result["errors"], result
    assert Path(source.input_jsonl).read_bytes() == before
    manifest = load_manifest(a["manifest"])
    context = exploration_preflight(manifest, record_count=3, recipe_max_length=64)
    assert context["policy_id"] == policy.policy_id
    assert context["official_mini_acceptance"] == "not_assessed"
    with pytest.raises(ValueError, match="截断"):
        exploration_preflight(manifest, record_count=3, recipe_max_length=1)
    with pytest.raises(ValueError, match="空数据"):
        exploration_preflight(manifest, record_count=0, recipe_max_length=64)
    with pytest.raises(FileExistsError):
        build_policy_artifacts(policy, source, output_dir=tmp_path / "a")

    mini_profile = tmp_path / "sft-mini-profile.yaml"
    mini_profile.write_text(yaml.safe_dump({
        "schema_version": "1.0", "profile_id": "policy-fixture-mini", "profile_version": "1",
        "description": "Single SFT policy fixture", "datasets": [{
            "dataset_key": "openr1.sft_train", "stage": "sft", "target": 1,
            "unit": "examples", "count_source": "processing_report.outputs.sft.budget_count",
        }],
    }, sort_keys=False), encoding="utf-8")
    _, quality_index = create_quality_assessment_bundle(tmp_path / "a", mini_profile)
    _, review_index_path = create_quality_review_bundle(
        tmp_path / "a", mini_profile, quality_index,
        Path(__file__).parents[1] / "configs/data/quality-review/mini-review-v1.yaml",
    )
    _, review_index = load_quality_review_bundle(review_index_path)
    review_entry = review_index.datasets[0]
    assert review_entry.dataset_id == manifest.dataset_id
    assert review_entry.content_sha256 == manifest.hashes["canonical_records_sha256"]
    assert review_entry.usage == "exploratory"
    assert review_entry.outcome == "pending"


def test_model_generated_exploratory_sft_does_not_require_data_policy(tmp_path, monkeypatch):
    from onegpu_posttrain import cli

    artifact_root = tmp_path / "generated"
    canonical = artifact_root / "canonical" / "sft.jsonl"
    canonical.parent.mkdir(parents=True)
    canonical.write_text('{"id":"generated-1"}\n', encoding="utf-8")
    manifest_path = artifact_root / "manifests" / "sft.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("fixture: true\n", encoding="utf-8")
    manifest = SimpleNamespace(
        dataset_id="ds-generated-sft", stage="sft", sample_count=1,
        metadata={"usage": "exploratory", "source_kind": "model_generated"},
        origin=SimpleNamespace(kind="model_generated"),
    )
    monkeypatch.setattr(cli, "_dataset_lineage_ref", lambda _path: (
        SimpleNamespace(id=manifest.dataset_id, path=str(canonical)), True, [],
    ))
    monkeypatch.setattr(cli, "load_manifest", lambda _path: manifest)
    monkeypatch.setattr("onegpu_posttrain.data.manifest.validate_manifest_evidence", lambda _path: [])

    context = cli._policy_training_context(canonical, {"max_length": 1024}, "sft")

    assert context["usage"] == "exploratory"
    assert context["source_kind"] == "model_generated"
    assert context["actual_record_count"] == 1
    assert context["official_mini_acceptance"] == "not_assessed"
    assert "语义正确性未独立验证" in context["warnings"][0]


def test_existing_dataset_cannot_restore_removed_candidates(tmp_path):
    policy, source, _ = policy_fixture(tmp_path / "inputs")
    result = build_policy_artifacts(policy, source, output_dir=tmp_path / "a")
    source.input_scope = "existing_dataset"
    source.existing_stage = "sft"
    source.input_jsonl = str(tmp_path / "a/canonical/sft.jsonl")
    source.source_cache = None
    source.expected_sha256 = None
    records, report = execute_policy(policy, source, preview_rows=50)
    assert len(records) == 3
    assert report["source_pool"]["input_scope"] == "existing_dataset"
    assert report["source_pool"]["pool_id"] != result["report"]["source_pool"]["pool_id"]
    # Evaluation-only metadata cannot be laundered through canonical input.
    records[0]["metadata"]["evaluation_only"] = True
    write_jsonl(tmp_path / "eval.jsonl", records)
    source.input_jsonl = str(tmp_path / "eval.jsonl")
    with pytest.raises(ValueError, match="评测记录"):
        execute_policy(policy, source)


def test_preference_output_has_two_ancestors_and_no_unknown_label(tmp_path):
    policy, source, _ = policy_fixture(tmp_path / "input", stage="preference")
    result = build_policy_artifacts(policy, source, output_dir=tmp_path / "output")
    assert result["trainable"]
    row = json.loads((tmp_path / "output/canonical/preference.jsonl").read_text().splitlines()[0])
    assert len(row["metadata"]["parent_record_ids"]) == 2
    assert row["metadata"]["chosen_correct"] is True
    assert row["metadata"]["rejected_correct"] is False
