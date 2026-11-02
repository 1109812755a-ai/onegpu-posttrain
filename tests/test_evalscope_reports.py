import json
from pathlib import Path

from onegpu_posttrain.evaluation.evalscope_reports import collect_evalscope_reports


def test_collect_evalscope_reports(tmp_path: Path):
    p = tmp_path / "reports" / "Qwen" / "gsm8k.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"dataset_name": "gsm8k", "model_name": "Qwen", "score": 0.42,
                             "perf_metrics": {"summary": {"n_samples": 4}}}), encoding="utf-8")
    reports = collect_evalscope_reports(tmp_path)
    assert reports["gsm8k"]["score"] == 0.42
    assert reports["gsm8k"]["model_name"] == "Qwen"
    assert reports["gsm8k"]["samples"] == 4


def test_collect_evalscope_schema_v2_metric_score(tmp_path: Path):
    p = tmp_path / "reports" / "Qwen" / "math_500.json"
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps({"schema_version": 2, "dataset_name": "math_500", "metrics": [{"score": 0.2, "num": 3}]}),
        encoding="utf-8",
    )
    reports = collect_evalscope_reports(tmp_path)
    assert reports["math_500"]["score"] == 0.2
    assert reports["math_500"]["samples"] == 3
