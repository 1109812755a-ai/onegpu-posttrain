import json
from pathlib import Path

from onegpu_posttrain.reporting.stage_compare import (
    compare_eval_files,
    write_stage_comparison,
)


def _dump(path: Path, stage: str, gsm: float, math: float, internal: float, ppl: float):
    path.write_text(json.dumps({
        "run_id": stage,
        "stage": stage,
        "model": stage,
        "evalscope_reports": {"gsm8k": {"score": gsm}, "math_500": {"score": math}},
        "internal_math": {"accuracy": internal},
        "domain_perplexity": {"perplexity": ppl},
    }), encoding="utf-8")


def test_compare_eval_files(tmp_path: Path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _dump(a, "base", 0.1, 0.02, 0.2, 10.0)
    _dump(b, "cpt", 0.12, 0.03, 0.25, 8.0)
    r = compare_eval_files(a, b)
    assert r["delta_candidate_minus_base"]["gsm8k"] == 0.01999999999999999
    assert r["delta_candidate_minus_base"]["domain_ppl"] == -2.0
    out = tmp_path / "out"
    write_stage_comparison(a, b, out)
    assert (out / "stage_comparison.md").exists()
    assert (out / "stage_comparison.json").exists()


def test_lineage_warns_on_protocol_and_dataset_mismatch_but_keeps_legacy_scores(tmp_path: Path):
    base_dir, candidate_dir = tmp_path / "base", tmp_path / "candidate"
    base_dir.mkdir()
    candidate_dir.mkdir()
    a, b = base_dir / "eval.json", candidate_dir / "eval.json"
    _dump(a, "base", 0.1, 0.02, 0.2, 10.0)
    _dump(b, "cleaned", 0.12, 0.03, 0.25, 8.0)
    (base_dir / "evaluation_lineage.json").write_text(json.dumps({
        "protocol_version": "math-v1", "dataset_ids": ["benchmark-gsm8k", "ds-before"],
    }), encoding="utf-8")
    (candidate_dir / "evaluation_lineage.json").write_text(json.dumps({
        "protocol_version": "math-v2", "dataset_ids": ["benchmark-gsm8k", "ds-after"],
    }), encoding="utf-8")

    result = compare_eval_files(a, b)
    assert result["candidate"]["metrics"]["gsm8k"] == 0.12
    assert result["delta_candidate_minus_base"]["gsm8k"] == 0.01999999999999999
    assert result["comparability"]["status"] == "warning"
    assert len(result["comparability"]["warnings"]) == 2
    assert "not a causal claim" in result["notes"]["causality"]
    output = tmp_path / "comparison"
    write_stage_comparison(a, b, output)
    report = (output / "stage_comparison.md").read_text(encoding="utf-8")
    assert "not a causal claim" in report
