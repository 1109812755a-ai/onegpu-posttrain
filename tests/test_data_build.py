from __future__ import annotations

import json

import pytest

from onegpu_posttrain.data.build import build_finemath_artifacts, build_openr1_artifacts
from onegpu_posttrain.data.decontaminate import BenchmarkIndex
from onegpu_posttrain.data.evidence import ProcessingReport, QualityReport
from onegpu_posttrain.data.finemath import FineMathBuildConfig, FineMathBuilder
from onegpu_posttrain.data.manifest import (
    load_manifest,
    sha256_file,
    validate_manifest_evidence,
)
from onegpu_posttrain.data.openr1 import OpenR1BuildConfig, OpenR1Builder, _boxed_answer
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer


def _openr1_rows(n: int = 500):
    for i in range(n):
        yield {
            "problem": f"Compute the value of x in x + {i} = {i + 2}.",
            "solution": "2",
            "answer": "2",
            "problem_type": "algebra",
            "question_type": "math-word-problem",
            "source": "fixture",
            "uuid": f"row-{i}",
            "is_reasoning_complete": [True, True, False],
            "generations": [
                f"Subtract {i} from both sides. Therefore \\boxed{{2}}.",
                f"Add {i} to both sides. Therefore \\boxed{{{2 * i + 2}}}.",
                "unfinished reasoning",
            ],
            "correctness_math_verify": [True, False, False],
            "correctness_llama": [True, False, False],
            "finish_reasons": ["stop", "stop", "length"],
            "correctness_count": 1,
        }


def test_boxed_answer_handles_empty_fallback_marker():
    assert _boxed_answer("Reasoning only. Final answer:") is None
    assert _boxed_answer("Reasoning only. Answer:   \n") is None
    assert _boxed_answer("Reasoning. Final answer: 42\nExtra text") == "42"


def test_openr1_build_has_disjoint_top_level_split():
    tok = WhitespaceTokenizer()
    cfg = OpenR1BuildConfig(
        sft_target=5,
        alignment_target=10,
        holdout_target=5,
        preference_target=5,
        preference_eval_target=4,
        grpo_target=5,
        sft_max_length=128,
        preference_max_length=128,
        grpo_max_prompt_length=128,
        max_raw_rows=500,
    )
    result = OpenR1Builder(tok, cfg).build(_openr1_rows())
    assert len(result.sft) == 5
    assert len(result.preference) == 5
    assert len(result.preference_eval) == 4
    assert len(result.grpo) == 5
    assert len(result.holdout) == 5
    assert result.report["overlap"]["train_holdout"] == 0
    assert result.report["overlap"]["sft_preference"] == 0
    assert result.report["overlap"]["sft_grpo"] == 0
    assert result.report["overlap"]["preference_train_eval"] == 0
    assert result.report["overlap"]["preference_eval_internal_holdout"] <= 4
    assert all(0.5 <= r["metadata"]["length_ratio"] <= 2.0 for r in result.preference)
    assert all(r["metadata"]["evaluation_only"] is True for r in result.preference_eval)
    pair = result.preference[0]
    assert len(pair["metadata"]["parent_record_ids"]) == 2
    assert all("generation:" in record_id for record_id in pair["metadata"]["parent_record_ids"])
    assert pair["metadata"]["record_provenance"] == "record"


def test_openr1_scans_past_early_rows_until_preference_target_is_met():
    cfg = OpenR1BuildConfig(
        sft_target=0,
        alignment_target=1,
        holdout_target=0,
        preference_target=2,
        preference_eval_target=0,
        grpo_target=0,
        max_raw_rows=100,
    )
    result = OpenR1Builder(WhitespaceTokenizer(), cfg).build(_openr1_rows(100))

    assert result.report["stop_reason"] == "targets_met"
    assert result.report["selected"]["preference"] == 2
    assert result.report["counters"]["raw_rows"] > result.report["selected"]["preference"]
    assert result.report["targets"]["preference"] == 2
    assert result.report["overlap"]["train_holdout"] == 0


def test_openr1_without_stable_source_ids_does_not_fabricate_record_ancestry():
    rows = list(_openr1_rows(100))
    for row in rows:
        row.pop("uuid", None)
    cfg = OpenR1BuildConfig(sft_target=1, alignment_target=1, holdout_target=1,
                           preference_target=1, preference_eval_target=1, grpo_target=1,
                           max_raw_rows=100)
    result = OpenR1Builder(WhitespaceTokenizer(), cfg).build(rows)
    assert result.preference
    pair = result.preference[0]
    assert pair["metadata"]["parent_record_ids"] == []
    assert pair["metadata"]["record_provenance"] == "dataset_only"


def test_benchmark_contamination_before_split():
    tok = WhitespaceTokenizer()
    index = BenchmarkIndex([("bench:1", "Compute the value of x in x + 0 = 2.")])
    cfg = OpenR1BuildConfig(
        sft_target=1,
        alignment_target=1,
        holdout_target=1,
        preference_target=1,
        preference_eval_target=1,
        grpo_target=1,
        max_raw_rows=100,
        sft_max_length=128,
        preference_max_length=128,
        grpo_max_prompt_length=128,
    )
    result = OpenR1Builder(tok, cfg, index).build(_openr1_rows(100))
    assert result.report["counters"]["benchmark_contamination_removed"] >= 1
    assert any(r["hits"][0]["kind"] == "exact" for r in result.contaminated)


def test_openr1_rejects_proof_for_grpo():
    rows = list(_openr1_rows(100))
    for row in rows:
        row["question_type"] = "proof"
        row["answer"] = "proof"
    cfg = OpenR1BuildConfig(
        sft_target=1,
        alignment_target=1,
        holdout_target=1,
        preference_target=1,
        preference_eval_target=1,
        grpo_target=1,
        max_raw_rows=100,
        sft_max_length=128,
        preference_max_length=128,
        grpo_max_prompt_length=128,
    )
    result = OpenR1Builder(WhitespaceTokenizer(), cfg).build(rows)
    assert len(result.grpo) == 0
    assert result.report["counters"].get("grpo_unverifiable_ground_truth", 0) > 0


def _finemath_rows(n: int = 1000):
    for i in range(n):
        yield {
            "url": f"https://example.org/math/{i}",
            "text": " ".join([f"math{i}"] * 50),
            "token_count": 50,
            "score": 4.2,
            "int_score": 4,
        }


def test_finemath_document_split_before_chunking():
    cfg = FineMathBuildConfig(
        train_token_budget=500,
        holdout_token_budget=200,
        max_seq_tokens=25,
        holdout_permille=500,
        max_raw_rows=1000,
    )
    result = FineMathBuilder(WhitespaceTokenizer(), cfg).build(_finemath_rows())
    assert result.report["train"]["tokens"] >= 500
    assert result.report["holdout"]["tokens"] >= 200
    assert result.report["document_overlap"] == 0
    assert all(r["metadata"]["tokens"] <= 25 for r in result.train + result.holdout)


def test_finemath_distinct_text_variants_share_url_split_but_have_distinct_record_ids():
    rows = [
        {"url": "https://example.org/shared", "text": "first extraction " * 12},
        {"url": "https://example.org/shared", "text": "second extraction " * 12},
    ]
    config = FineMathBuildConfig(
        train_token_budget=1000,
        holdout_token_budget=0,
        max_seq_tokens=64,
        holdout_permille=500,
    )

    result = FineMathBuilder(WhitespaceTokenizer(), config).build(rows)

    emitted = result.train + result.holdout
    assert len(emitted) == 2
    assert len({row["id"] for row in emitted}) == 2
    assert len({row["metadata"]["document_id"] for row in emitted}) == 1
    assert sorted((len(result.train), len(result.holdout))) == [0, 2]
    assert all(row["metadata"]["data_role"] == "train_seed" for row in result.train)
    assert all(row["metadata"]["data_role"] == "final_holdout" for row in result.holdout)
    assert result.report["protocol"] == "finemath-build-v3"
    assert result.report["data_role_counts"] == {
        "train_seed": len(result.train), "final_holdout": len(result.holdout),
    }


def test_finemath_manifest_records_content_identity_protocol(tmp_path):
    report = build_finemath_artifacts(
        _finemath_rows(10),
        output_dir=tmp_path,
        tokenizer=WhitespaceTokenizer(),
        config=FineMathBuildConfig(
            train_token_budget=100,
            holdout_token_budget=0,
            max_seq_tokens=20,
            holdout_permille=0,
        ),
        source_repo="fixture",
        source_config=None,
        source_revision="rev-2",
        tokenizer_repo="fixture-tokenizer",
        tokenizer_revision="tokenizer-v1",
    )

    manifest = load_manifest(tmp_path / "manifests/cpt-smoke-v1.yaml")
    assert report["protocol"] == "finemath-build-v3"
    assert manifest.protocol_versions["record_identity"] == "v2"
    assert manifest.protocol_versions["data_role"] == "cpt-data-role-v1"
    assert manifest.metadata["build_protocol"] == "finemath-build-v3"
    assert manifest.metadata["data_role_protocol"] == "cpt-data-role-v1"
    assert manifest.metadata["data_role_counts"] == {"train_seed": manifest.sample_count}
    processing = json.loads((tmp_path / "reports/processing_report.json").read_text(encoding="utf-8"))
    assert processing["outputs"]["cpt"]["data_role_counts"] == {"train_seed": manifest.sample_count}


def test_fuzzy_contamination_index():
    index = BenchmarkIndex([("bench:1", "If Alice has 10 apples and buys 5 more, how many apples does she have?")], shingle_n=3)
    hits = index.check("If Alice has 10 apples and buys 5 more, how many apples does Alice have?", fuzzy_threshold=0.55)
    assert hits
    assert hits[0].kind == "fuzzy"


def test_finemath_contamination_uses_target_coverage_for_long_document():
    benchmark = "A train travels 120 miles in 2 hours. What is its average speed?"
    long_doc = "unrelated preface " * 20 + benchmark + " unrelated suffix" * 20
    index = BenchmarkIndex([("bench:train", benchmark)], shingle_n=3)
    cfg = FineMathBuildConfig(
        train_token_budget=20,
        holdout_token_budget=10,
        max_seq_tokens=20,
        holdout_permille=500,
        fuzzy_threshold=0.8,
        max_raw_rows=10,
    )
    rows = [{"url": "https://example.org/contaminated", "text": long_doc, "score": 4.0, "int_score": 4}]
    rows += [
        {"url": f"https://example.org/clean-{i}", "text": "clean mathematical exposition " * 20, "score": 4.0, "int_score": 4}
        for i in range(1, 10)
    ]
    result = FineMathBuilder(WhitespaceTokenizer(), cfg, index).build(rows)
    assert result.report["contamination_candidates"] >= 1


@pytest.mark.parametrize("name", ["openr1", "finemath"])
def test_artifact_reports_are_typed_linked_and_deterministic(tmp_path, name):
    common = {"tokenizer": WhitespaceTokenizer(), "source_repo": "fixture", "source_config": None,
              "source_revision": "revision-1", "tokenizer_repo": "fixture", "tokenizer_revision": "test-v1"}
    if name == "openr1":
        build = build_openr1_artifacts
        config = OpenR1BuildConfig(sft_target=3, alignment_target=4, holdout_target=3, preference_target=3,
                                  preference_eval_target=2, grpo_target=3, max_raw_rows=500)
        source = _openr1_rows
        count = 5
    else:
        build = build_finemath_artifacts
        config = FineMathBuildConfig(train_token_budget=100, holdout_token_budget=50,
                                    max_seq_tokens=25, holdout_permille=500)
        source = _finemath_rows
        count = 2
    first = tmp_path / "first"
    report = build(source(), output_dir=first, config=config, **common)
    processing_path = first / "reports" / "processing_report.json"
    processing = ProcessingReport.model_validate_json(processing_path.read_text(encoding="utf-8"))
    assert processing.build_id == report["build_id"]
    assert processing.stop_reason == "targets_met"
    if name == "openr1":
        for stage, unit in (("sft", "examples"), ("preference", "preference_pairs"),
                            ("preference_eval", "preference_pairs"), ("grpo", "prompts"), ("eval", "examples")):
            assert processing.outputs[stage]["budget_count"] == report["selected"]["holdout" if stage == "eval" else stage]
            assert processing.outputs[stage]["budget_unit"] == unit
    else:
        assert processing.outputs["cpt"]["budget_count"] == report["train"]["tokens"]
        assert processing.outputs["cpt"]["budget_target"] == report["train"]["token_budget"]
        assert processing.outputs["cpt"]["budget_unit"] == "tokens"
    assert "| Step | Status | Input | Output |" in processing_path.with_suffix(".md").read_text(encoding="utf-8")
    screening = next(s for s in processing.processing_steps if s.operation == "decontaminate")
    assert screening.status == "skipped"
    assert screening.limitations
    manifests = list((first / "manifests").glob("*.yaml"))
    assert len(manifests) == count
    for path in manifests:
        manifest = load_manifest(path)
        assert validate_manifest_evidence(path) == []
        for ref in (manifest.processing_report, manifest.quality_report):
            assert sha256_file(path.parent / ref.path) == ref.sha256
        quality = QualityReport.model_validate_json((path.parent / manifest.quality_report.path).read_text(encoding="utf-8"))
        assert quality.dataset_id == manifest.dataset_id
        assert manifest.lineage.produced_by_run_id == report["data_build_run_id"]
        assert manifest.lineage.content_sha256 == manifest.hashes["canonical_records_sha256"]
        assert quality.run_id
        quality_run = json.loads((first / "runs" / quality.run_id / "lineage.json").read_text(encoding="utf-8"))
        assert quality_run["run_type"] == "quality_assess"
        assert quality_run["inputs"][0]["id"] == manifest.dataset_id
        assert quality.content_sha256 == manifest.hashes["canonical_records_sha256"]
        assert quality.metrics["schema_valid_rate"].value == 1.0
        if manifest.stage == "eval":
            assert "semantic_correctness" not in quality.metrics
            assert quality.metrics["scoring_coverage"].status == "not_measured"
        else:
            assert quality.metrics["semantic_correctness"].status == "not_measured"
        assert quality.gate.summary_status == "NOT_EVALUATED"
    second = build(source(), output_dir=tmp_path / "second", config=config, **common)
    assert second["dataset_ids"] == report["dataset_ids"]
    assert second["build_id"] != report["build_id"]
    assert second["processing_steps"] == report["processing_steps"]
    revised = build(source(), output_dir=tmp_path / "revision", config=config, **{**common, "source_revision": "revision-2"})
    assert revised["dataset_ids"] != report["dataset_ids"]


def test_openr1_step_rejections_and_before_after_rates():
    examples = list(_openr1_rows(5))
    examples[1]["problem"] = "Contaminated example with an unmistakable unrelated question?"
    examples[2]["correctness_math_verify"] = [False, False, False]
    examples[3]["generations"] = ["Complete but no final answer marker"]
    examples[3]["is_reasoning_complete"] = [True]
    examples[3]["finish_reasons"] = ["stop"]
    rows = [{"problem": " "}, examples[0], examples[0], *examples[1:]]
    index = BenchmarkIndex([("benchmark", examples[1]["problem"])])
    cfg = OpenR1BuildConfig(sft_target=50, alignment_target=50, holdout_target=50, max_raw_rows=100)
    result = OpenR1Builder(WhitespaceTokenizer(), cfg, index).build(rows)
    steps = result.report["processing_steps"]
    assert steps[0]["first_rejection_reasons"]["empty_problem"] == 1
    assert steps[1]["first_rejection_reasons"]["exact_duplicate_problem"] == 1
    assert steps[2]["rejected_count"] >= 1
    assert steps[4]["rejected_count"] >= 2
    assert result.report["stop_reason"] == "source_exhausted"
    assert result.report["counters"]["raw_rows"] == len(rows)
    assert result.report["selected"]["preference"] < result.report["targets"]["preference"]
    assert result.report["selected"]["preference_eval"] < result.report["targets"]["preference_eval"]
    assert result.report["counters"]["benchmark_contamination_removed"] == 1
    assert result.report["counters"]["no_verifier_correct_trace"] >= 1
    for item in steps:
        if item["output_is_subset"]:
            assert item["input_count"]["value"] == item["output_count"]["value"] + item["rejected_count"] + item["unselected_count"]
    trace = steps[3]
    assert trace["metrics_before"]["answer_parse_rate"]["value"] < 1
    assert trace["metrics_after"]["answer_parse_rate"]["value"] == 1


def test_openr1_cap_is_explicit_and_empty_rates_are_unavailable():
    cfg = OpenR1BuildConfig(max_raw_rows=5)
    result = OpenR1Builder(WhitespaceTokenizer(), cfg).build({"problem": ""} for _ in range(20))
    assert result.report["stop_reason"] == "max_raw_rows"
    assert result.report["counters"]["raw_rows"] == 5
    trace = result.report["processing_steps"][3]
    assert trace["metrics_before"]["answer_parse_rate"]["status"] == "not_measured"
    assert trace["metrics_before"]["answer_parse_rate"]["value"] is None


def test_openr1_cli_returns_nonzero_when_source_exhausts_before_targets(tmp_path, capsys):
    from onegpu_posttrain.cli import build_parser

    source_path = tmp_path / "openr1.jsonl"
    source_path.write_text("\n".join(json.dumps(row) for row in _openr1_rows(5)), encoding="utf-8")
    config_path = tmp_path / "source.yaml"
    config_path.write_text("source:\n  repo: fixture\n  revision: r1\n", encoding="utf-8")
    args = build_parser().parse_args([
        "build-openr1", "--config", str(config_path), "--input-jsonl", str(source_path),
        "--output", str(tmp_path / "output"), "--test-tokenizer", "--sft-target", "50",
        "--alignment-target", "50", "--holdout-target", "50", "--preference-target", "50",
        "--preference-eval-target", "50", "--grpo-target", "50",
    ])

    assert args.func(args) == 3
    capsys.readouterr()
    report = json.loads((tmp_path / "output/reports/openr1_data_report.json").read_text(encoding="utf-8"))
    assert report["stop_reason"] == "source_exhausted"
    assert report["counters"]["raw_rows"] == 5
    assert report["selected"]["preference"] < report["targets"]["preference"]


def test_finemath_budget_exclusion_is_not_a_quality_rejection():
    config = FineMathBuildConfig(train_token_budget=10, holdout_token_budget=10,
                                max_seq_tokens=10, holdout_permille=500)
    result = FineMathBuilder(WhitespaceTokenizer(), config).build(_finemath_rows())
    steps = result.report["processing_steps"]
    budget = steps[-1]
    assert budget["unselected_count"] > 0
    assert budget["rejected_count"] == 0
    assert budget["input_count"]["value"] == budget["output_count"]["value"] + budget["unselected_count"]


def test_empty_artifacts_do_not_claim_perfect_quality(tmp_path):
    build_finemath_artifacts([], output_dir=tmp_path, tokenizer=WhitespaceTokenizer(), config=FineMathBuildConfig(),
                             source_repo="fixture", source_config=None, source_revision="r1",
                             tokenizer_repo="fixture", tokenizer_revision="v1")
    report = QualityReport.model_validate_json((tmp_path / "reports/quality/cpt.json").read_text(encoding="utf-8"))
    assert report.metrics["schema_valid_rate"].status == "not_measured"
    assert report.metrics["schema_valid_rate"].value is None
    empty_openr1 = build_openr1_artifacts([], output_dir=tmp_path / "openr1", tokenizer=WhitespaceTokenizer(),
                                         config=OpenR1BuildConfig(), source_repo="fixture", source_config=None,
                                         source_revision="r1", tokenizer_repo="fixture", tokenizer_revision="v1")
    assert all(value is None for value in empty_openr1["retention"].values())


def test_duplicate_cpt_ids_fail_structural_acceptance(tmp_path, capsys):
    from onegpu_posttrain.cli import build_parser

    row = {"url": "fixture:duplicate", "text": "math " * 10}
    result = build_finemath_artifacts([row, row], output_dir=tmp_path, tokenizer=WhitespaceTokenizer(),
                                     config=FineMathBuildConfig(train_token_budget=20, holdout_token_budget=0,
                                                              max_seq_tokens=10, holdout_permille=0),
                                     source_repo="fixture", source_config=None, source_revision="r1",
                                     tokenizer_repo="fixture", tokenizer_revision="v1")
    assert result["quality_gate_failures"] == ["cpt"]
    quality = QualityReport.model_validate_json((tmp_path / "reports/quality/cpt.json").read_text(encoding="utf-8"))
    assert quality.metrics["duplicate_record_ids"].value == 1
    assert quality.gate.summary_status == "FAIL"
    source_path = tmp_path / "raw.jsonl"
    source_path.write_text("\n".join(json.dumps(row) for _ in range(2)), encoding="utf-8")
    config_path = tmp_path / "source.yaml"
    config_path.write_text("source:\n  repo: fixture\n  revision: r1\n", encoding="utf-8")
    args = build_parser().parse_args([
        "build-finemath", "--config", str(config_path), "--input-jsonl", str(source_path),
        "--output", str(tmp_path / "cli"), "--test-tokenizer", "--train-token-budget", "20",
        "--holdout-token-budget", "0", "--max-seq-tokens", "10", "--holdout-permille", "0",
    ])
    assert args.func(args) == 3
    capsys.readouterr()


def test_benchmark_index_fingerprint_and_empty_index_status(tmp_path):
    common = {"tokenizer": WhitespaceTokenizer(), "config": FineMathBuildConfig(train_token_budget=10, holdout_token_budget=0,
                                                                                max_seq_tokens=10, holdout_permille=0),
              "source_repo": "fixture", "source_config": None, "source_revision": "r1",
              "tokenizer_repo": "fixture", "tokenizer_revision": "v1"}
    empty = build_finemath_artifacts(_finemath_rows(), output_dir=tmp_path / "empty", benchmark_index=BenchmarkIndex([]), **common)
    assert empty["processing_steps"][2]["status"] == "skipped"
    nonempty = build_finemath_artifacts(_finemath_rows(), output_dir=tmp_path / "checked",
                                      benchmark_index=BenchmarkIndex([("b1", "An unrelated benchmark question")]), **common)
    processing = ProcessingReport.model_validate_json((tmp_path / "checked/reports/processing_report.json").read_text(encoding="utf-8"))
    assert processing.benchmark_screening["indexed_items"] == 1
    assert len(processing.benchmark_screening["normalized_index_sha256"]) == 64
    assert nonempty["dataset_ids"] != empty["dataset_ids"]


def test_manifest_evidence_validator_detects_report_tampering(tmp_path):
    output = tmp_path / "lineage"
    build_finemath_artifacts(
        _finemath_rows(10), output_dir=output, tokenizer=WhitespaceTokenizer(),
        config=FineMathBuildConfig(train_token_budget=20, holdout_token_budget=10,
                                   max_seq_tokens=10, holdout_permille=500),
        source_repo="fixture", source_config=None, source_revision="r1",
        tokenizer_repo="fixture", tokenizer_revision="v1",
    )
    manifest = next((output / "manifests").glob("*.yaml"))
    assert validate_manifest_evidence(manifest) == []
    report = output / "reports" / "processing_report.json"
    report.write_text(report.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert any("sha256 mismatch" in error for error in validate_manifest_evidence(manifest))
