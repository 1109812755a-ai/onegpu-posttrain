from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from onegpu_posttrain.data.sft_calibration import (
    CalibrationError,
    _parquet_inventory,
    _sample_rows,
    calibrate_sft_lengths,
)
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.evaluation.verification import AnswerVerificationResult


def _write_source(path, *, rows=None, columns=None):
    rows = rows or [{
        "question": "What is 2 + 2?",
        "final_answer": "4",
        "r1_solution_1": "Compute carefully. \\boxed{4}",
        "r1_solution_2": "2 plus 2 is 4. \\boxed{4}",
        "r1_solution_3": "The result is 5. \\boxed{5}",
    }]
    table = pa.Table.from_pylist(rows, schema=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_missing_and_malformed_sources_fail_offline(tmp_path):
    with pytest.raises(CalibrationError, match="不会访问网络"):
        calibrate_sft_lengths(source=tmp_path / "missing", output=tmp_path / "out")

    bad = tmp_path / "bad"
    _write_source(bad / "bad.parquet", rows=[{"question": "x"}])
    _, inventory, _ = _parquet_inventory(bad)
    from onegpu_posttrain.data.sft_calibration import _iter_parquet_rows
    with pytest.raises(CalibrationError, match="缺少 DeepMath 必需列"):
        list(_iter_parquet_rows(inventory))


def test_seeded_hash_sampling_is_stable_across_file_order(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    rows = [
        {"question": f"Question {i}", "final_answer": str(i),
         "r1_solution_1": f"\\boxed{{{i}}}", "r1_solution_2": "", "r1_solution_3": ""}
        for i in range(8)
    ]
    _write_source(first / "a.parquet", rows=rows[:4])
    _write_source(first / "b.parquet", rows=rows[4:])
    _write_source(second / "b.parquet", rows=rows[4:])
    _write_source(second / "a.parquet", rows=rows[:4])
    _, inventory_a, _ = _parquet_inventory(first)
    _, inventory_b, _ = _parquet_inventory(second)
    sample_a, total_a = _sample_rows(inventory_a, limit=3, seed=17)
    sample_b, total_b = _sample_rows(inventory_b, limit=3, seed=17)
    # Inventory is sorted by filename, so physical creation order has no effect.
    assert total_a == total_b == 8
    assert [row["source_row_id"] for row in sample_a] == [row["source_row_id"] for row in sample_b]


def test_calibration_separates_length_and_answer_agreement_and_is_repeatable(tmp_path):
    source = tmp_path / "source"
    rows = [
        {"question": f"Question {i}", "final_answer": str(i),
         "r1_solution_1": ("step " * 1200 + f"\\boxed{{{i}}}") if i == 0 else f"Short reasoning. \\boxed{{{i}}}",
         "r1_solution_2": f"Longer reasoning with several extra words. \\boxed{{{i + 5 if i == 0 else i}}}",
         "r1_solution_3": f"Wrong answer. \\boxed{{{i + 1}}}"}
        for i in range(4)
    ]
    _write_source(source / "train.parquet", rows=rows)
    tokenizer = WhitespaceTokenizer()
    output_a = tmp_path / "calibration-a"
    output_b = tmp_path / "calibration-b"
    report_a = calibrate_sft_lengths(source=source, output=output_a, sample_limit=3, seed=11, tokenizer=tokenizer)
    report_b = calibrate_sft_lengths(source=source, output=output_b, sample_limit=3, seed=11, tokenizer=tokenizer)

    assert [item["max_length"] for item in report_a["candidates"]] == [1024, 2048, 4096]
    assert report_a["source"]["snapshot_sha256"] == report_b["source"]["snapshot_sha256"]
    assert report_a["sampling"]["selected_source_row_ids_sha256"] == report_b["sampling"]["selected_source_row_ids_sha256"]
    assert report_a["quality_evidence"]["semantic_verification"] == "not_measured"
    assert report_a["quality_evidence"]["answer_agreement"] == "exact normalized string agreement with final_answer; not semantic verification"
    assert report_a["quality_evidence"]["parsed_answer_mismatch_count"] > 0
    assert report_a["quality_evidence"]["answer_unparseable_count"] == 0
    assert report_a["candidates"][0]["training_examples"] == 2
    assert report_a["candidates"][1]["training_examples"] == 3
    assert report_a["candidates"][0]["content_sha256"] != report_a["candidates"][1]["content_sha256"]

    for candidate in report_a["candidates"]:
        canonical_lines = (output_a / candidate["canonical_path"]).read_text(encoding="utf-8").splitlines()
        assert candidate["rendered_length_tokens"]["count"] == 9
        for line in canonical_lines:
            record = json.loads(line)
            rendered = tokenizer.render_messages(record["messages"])
            direct = tokenizer.count_text(rendered)
            assert record["metadata"]["rendered_tokens"] == direct
            assert record["metadata"]["semantic_verification"] == "not_measured"
        assert candidate["content_sha256"]
        manifest = json.loads((output_a / candidate["manifest_path"]).read_text(encoding="utf-8"))
        assert manifest["eligibility"]["overlength_action"] == "reject; never truncate"
        assert manifest["quality_evidence"]["quality_approval"] == "not_evaluated"
        native = (output_a / candidate["native_path"]).read_text(encoding="utf-8").splitlines()
        assert len(native) == candidate["training_examples"]


def test_existing_output_and_official_mini_output_are_refused(tmp_path, monkeypatch):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="全新路径"):
        calibrate_sft_lengths(source=tmp_path / "missing", output=existing, tokenizer=WhitespaceTokenizer())


def test_unparseable_trace_and_missing_reference_stay_unverified(tmp_path):
    source = tmp_path / "source"
    _write_source(source / "train.parquet", rows=[{
        "question": "Explain the result.",
        "final_answer": None,
        "r1_solution_1": "A derivation without a final numeric answer.",
        "r1_solution_2": "A symbolic result is described here.",
        "r1_solution_3": "",
    }])
    report = calibrate_sft_lengths(
        source=source, output=tmp_path / "calibration", sample_limit=1, tokenizer=WhitespaceTokenizer(),
    )
    evidence = report["quality_evidence"]
    assert evidence["answer_unparseable_count"] == 2
    assert evidence["reference_answer_missing_count"] == 2
    assert evidence["semantic_verification"] == "not_measured"
    assert evidence["quality_approval"] == "not_evaluated"
    assert report["candidates"][0]["training_examples"] == 0


def test_semantic_mode_selects_only_typed_correct_and_changes_artifact_identity(tmp_path, monkeypatch):
    import onegpu_posttrain.data.sft_calibration as calibration

    source = tmp_path / "source"
    _write_source(source / "train.parquet", rows=[
        {
            "question": "Reduce one half.", "final_answer": "1/2",
            "r1_solution_1": r"\boxed{2/4}",
            "r1_solution_2": r"\boxed{0}",
            "r1_solution_3": r"\boxed{9}",
        },
        {
            "question": "One plus one.", "final_answer": "2",
            "r1_solution_1": r"\boxed{2}",
            "r1_solution_2": r"\boxed{3}",
            "r1_solution_3": "",
        },
    ])

    def fake_verify(pairs):
        by_prediction = {"2/4": "correct", "0": "incorrect", "9": "timeout", "3": "verifier_error"}
        return [AnswerVerificationResult(
            status=by_prediction[prediction], method="test-verifier:v1",
            verifier="test-verifier", verifier_version="1",
        ) for prediction, _ in pairs]

    monkeypatch.setattr(calibration, "verify_answer_pairs", fake_verify)
    exact = calibrate_sft_lengths(
        source=source, output=tmp_path / "exact", sample_limit=2, candidates=(64,),
        tokenizer=WhitespaceTokenizer(),
    )
    semantic = calibrate_sft_lengths(
        source=source, output=tmp_path / "semantic", sample_limit=2, candidates=(64,),
        tokenizer=WhitespaceTokenizer(), answer_verification="semantic",
    )

    candidate = semantic["candidates"][0]
    manifest = json.loads((tmp_path / "semantic" / candidate["manifest_path"]).read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (tmp_path / "semantic" / candidate["canonical_path"]).read_text(encoding="utf-8").splitlines()]
    assert candidate["training_examples"] == 2
    assert candidate["answer_agreement_examples"] == 1
    assert {record["metadata"]["answer_verification_status"] for record in records} == {"correct"}
    assert semantic["quality_evidence"]["answer_verification"]["denominator"] == 5
    assert semantic["quality_evidence"]["answer_verification"]["counts"] == {
        "correct": 2, "incorrect": 1, "timeout": 1, "verifier_error": 1,
    }
    assert semantic["quality_evidence"]["answer_verification"]["verifier"] == "test-verifier"
    assert manifest["source"]["snapshot_sha256"] == semantic["source"]["snapshot_sha256"]
    assert manifest["hashes"]["canonical_sha256"] == candidate["content_sha256"]
    assert semantic["calibration_id"] != exact["calibration_id"]
    assert candidate["content_sha256"] != exact["candidates"][0]["content_sha256"]

