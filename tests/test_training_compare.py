import json
from pathlib import Path

from onegpu_posttrain.reporting.training_compare import (
    write_training_feasibility_comparison,
)


def test_training_feasibility_comparison_records_oom(tmp_path: Path):
    peft = tmp_path / "peft.json"
    full = tmp_path / "full.json"
    peft.write_text(json.dumps({"train": {"status": "PASS", "wall_seconds": 10, "peak_gpu_memory_mib_sampled": 4000}}))
    full.write_text(json.dumps({"execution": {"status": "OOM", "wall_seconds": 2, "peak_gpu_memory_mib_sampled": 7900}}))
    result = write_training_feasibility_comparison(peft, full, tmp_path / "out")
    assert result["full"]["status"] == "OOM"
    assert "official 8GB path" in result["interpretation"]
    assert (tmp_path / "out" / "cpt_feasibility.md").exists()
