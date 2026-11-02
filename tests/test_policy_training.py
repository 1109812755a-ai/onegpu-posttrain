import json

import pytest
from policy_support import policy_fixture

from onegpu_posttrain.adapters.ms_swift import adapt_file
from onegpu_posttrain.cli import (
    _policy_training_context,
    _write_training_run,
    build_parser,
)
from onegpu_posttrain.data.policy_build import build_policy_artifacts


def test_actual_training_preflight_and_lineage_keep_policy(tmp_path):
    policy, source, _ = policy_fixture(tmp_path / "input")
    built = build_policy_artifacts(policy, source, output_dir=tmp_path / "data")
    native = tmp_path / "native.jsonl"
    adapt_file("sft", tmp_path / "data/canonical/sft.jsonl", native)
    context = _policy_training_context(native, {"max_length": 64}, "sft")
    assert context["actual_record_count"] == 3
    assert context["policy_id"] == policy.policy_id
    with pytest.raises(ValueError, match="截断"):
        _policy_training_context(native, {"max_length": 1}, "sft")
    result = {"train": {"status": "FAILED"}, "data_policy_context": context}
    output = tmp_path / "run"
    _write_training_run(output=output, result=result, run_id="run-policy-test-1", run_label="fixture",
                        experiment_id="exp-fixture", stage="sft", model="unavailable-fixture-model",
                        dataset=native, config_path="recipe.yaml", resolved_config={"training": {"max_length": 64}},
                        started_at="2026-10-17T00:00:00Z")
    lineage = json.loads((output / "lineage.json").read_text(encoding="utf-8"))
    assert any(item["id"] == policy.policy_id and item["role"] == "data_policy" for item in lineage["inputs"])
    assert json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))["data_policy_context"] == context
    assert built["official_mini_acceptance"] == "not_assessed"
    native.write_text(native.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="adapter 输出摘要"):
        _policy_training_context(native, {"max_length": 64}, "sft")


def test_policy_cli_fixture_and_missing_source(tmp_path, capsys):
    args = build_parser().parse_args(["preview-data-policy", "--policy", "configs/data/policies/sft-example.yaml",
                                      "--source", "configs/data/policy-sources/example.yaml", "--rows", "8"])
    assert args.func(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["counters"]["raw_rows"] == 8
    assert report["preview"] is True
    args.source = str(tmp_path / "missing.yaml")
    assert args.func(args) == 2
