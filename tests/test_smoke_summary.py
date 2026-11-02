import json
from pathlib import Path

from onegpu_posttrain.reporting.smoke_summary import write_smoke_summary


def test_smoke_summary_is_partial_safe(tmp_path: Path):
    root = tmp_path / "runs"
    d = root / "base-smoke-s42" / "evaluation"
    d.mkdir(parents=True)
    (d / "eval.json").write_text(json.dumps({"evalscope_reports": {"gsm8k": {"score": .1}}}), encoding="utf-8")
    result = write_smoke_summary(root, tmp_path / "summary")
    assert len(result["stages"]) == 7
    assert result["stages"][0]["stage"] == "base"
    assert (tmp_path / "summary" / "smoke_summary.md").exists()
