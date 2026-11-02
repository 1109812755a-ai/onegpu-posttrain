from pathlib import Path

import pytest

from onegpu_posttrain.data.importer import validate_canonical_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stage", ["cpt", "sft", "preference", "grpo"])
def test_documented_stage_example_matches_canonical_schema(stage):
    path = ROOT / "docs" / "examples" / "datasets" / f"{stage}.jsonl"
    with path.open("rb") as stream:
        report = validate_canonical_jsonl(stream, stage)
    assert report.valid, report.report()
    assert report.record_count == 1


def test_workbench_guide_documents_registered_adapter_and_each_training_path():
    guide = (ROOT / "docs/guides/DATASET_WORKBENCH_zh.md").read_text(encoding="utf-8")
    for command in ("adapt-registered-dataset", "run-peft-cpt", "run-sft", "run-dpo", "run-simpo", "run-grpo"):
        assert command in guide
    assert "dataset_id" in guide and "license_status" in guide
