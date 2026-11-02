from __future__ import annotations

import json
from pathlib import Path

import pytest

from onegpu_posttrain.data.decontaminate import BenchmarkIndex
from onegpu_posttrain.data.deepmath_mini import (
    PINNED_DEEPMATH_REVISION,
    DeepMathMiniBuildError,
    _split,
    build_deepmath_mini_sft,
)
from onegpu_posttrain.data.io import read_jsonl
from onegpu_posttrain.data.manifest import load_manifest, validate_manifest_evidence
from onegpu_posttrain.data.normalize import problem_id
from onegpu_posttrain.lineage_validation import validate_lineage_root


class CharacterTokenizer:
    name = "test-character-tokenizer"

    class _Tokenizer:
        chat_template = "fixture-template-v1"

    tokenizer = _Tokenizer()

    def count_messages(self, messages):
        return sum(len(message["content"]) for message in messages)


def _snapshot(tmp_path: Path, *, license_id="mit") -> Path:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "snapshots" / PINNED_DEEPMATH_REVISION
    (root / "data").mkdir(parents=True)
    (root / "README.md").write_text(f"---\nlicense: {license_id}\n---\n# DeepMath\n", encoding="utf-8")
    rows = []
    for index in range(40):
        question = f"Find the value for fixture problem number {index}."
        rows.append({
            "question": question,
            "final_answer": "1",
            "difficulty": float(index % 5),
            "topic": "algebra" if index % 2 else "number theory",
            "r1_solution_1": f"Work for problem {index}. \\boxed{{1}}",
            "r1_solution_2": "",
            "r1_solution_3": "",
        })
    rows.append({"question": "A too long problem", "final_answer": "1", "difficulty": 1.0, "topic": "algebra",
                 "r1_solution_1": "x" * 3000 + " \\boxed{1}", "r1_solution_2": "", "r1_solution_3": ""})
    table = pyarrow.Table.from_pylist(rows)
    parquet.write_table(table, root / "data" / "train-00000.parquet")
    return root


def test_deepmath_builder_is_deterministic_and_keeps_quality_claims_separate(tmp_path):
    source = _snapshot(tmp_path)
    result1 = build_deepmath_mini_sft(
        source=source, output_dir=tmp_path / "build-a", tokenizer=CharacterTokenizer(),
        seed=42, max_length=2048, target_examples=10,
    )
    result2 = build_deepmath_mini_sft(
        source=source, output_dir=tmp_path / "build-b", tokenizer=CharacterTokenizer(),
        seed=42, max_length=2048, target_examples=10,
    )
    assert result1["count"] == result2["count"] == 10
    assert result1["canonical_sha256"] == result2["canonical_sha256"]
    rows = list(read_jsonl(result1["canonical"]))
    assert all(row["metadata"]["rendered_tokens"] <= 2048 for row in rows)
    assert all(row["metadata"]["reasoning_review"] == "not_measured" for row in rows)
    adapter = json.loads(Path(result1["native"]).with_suffix(".jsonl.adapter.json").read_text(encoding="utf-8"))
    assert adapter["input_sha256"] == result1["canonical_sha256"]
    assert adapter["output_sha256"] == result1["native_sha256"]
    quality = json.loads(Path(result1["quality_report"]).read_text(encoding="utf-8"))
    assert quality["answer_check"]["method"] == "exact normalized string agreement"
    assert quality["reasoning_review"]["status"] == "not_measured"
    assert result1["build_report"]["split"]["train_problems"] + result1["build_report"]["split"]["eval_problems"] == 41
    manifest = load_manifest(result1["manifest"])
    assert manifest.metadata["license"] == "mit"
    assert manifest.metadata["usage"] == "exploratory"
    assert manifest.metadata["data_policy"]["config"]["stage"] == "sft"
    assert manifest.metadata["data_policy"]["config"]["resources"]["max_length"] == 2048
    assert manifest.metadata["policy_integrity"] == "incomplete"
    assert manifest.lineage.record_provenance == "record"
    assert validate_manifest_evidence(result1["manifest"]) == []
    assert validate_lineage_root(Path(result1["manifest"]).parent.parent)["errors"] == []


def test_deepmath_builder_splits_before_selection_and_excludes_local_overlap(tmp_path):
    source = _snapshot(tmp_path)
    overlap_question = next(f"Find the value for fixture problem number {i}." for i in range(40)
                            if _split(problem_id(f"Find the value for fixture problem number {i}."), 42) == "train")
    index = BenchmarkIndex([("known-holdout", overlap_question)])
    result = build_deepmath_mini_sft(
        source=source, output_dir=tmp_path / "build", tokenizer=CharacterTokenizer(),
        benchmark_index=index, seed=42, target_examples=50,
    )
    ids = {row["metadata"]["problem_id"] for row in read_jsonl(result["canonical"])}
    assert problem_id(overlap_question) not in ids
    assert result["build_report"]["counters"]["contamination"] == 1
    manifest = load_manifest(result["manifest"])
    assert manifest.metadata["policy_integrity"] == "passed"
    assert manifest.metadata["data_policy"]["policy_id"].startswith("policy-sft-")


def test_deepmath_builder_fails_closed_for_license_or_revision_mismatch(tmp_path):
    source = _snapshot(tmp_path, license_id="unknown")
    with pytest.raises(DeepMathMiniBuildError, match="license metadata"):
        build_deepmath_mini_sft(source=source, output_dir=tmp_path / "bad-license", tokenizer=CharacterTokenizer())
    other = source.parent / "not-pinned"
    other.mkdir()
    (other / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")
    with pytest.raises(DeepMathMiniBuildError, match="revision 不匹配"):
        build_deepmath_mini_sft(source=other, output_dir=tmp_path / "bad-revision", tokenizer=CharacterTokenizer())
