import json
from argparse import Namespace
from pathlib import Path

from onegpu_posttrain import cli
from onegpu_posttrain.data.build import build_finemath_artifacts
from onegpu_posttrain.data.finemath import FineMathBuildConfig
from onegpu_posttrain.data.tokenization import WhitespaceTokenizer
from onegpu_posttrain.lineage import new_run_id
from onegpu_posttrain.lineage_validation import validate_lineage_root


def test_dataset_to_training_merge_evaluation_lineage_chain(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    data_root = root / "data/processed/finemath"
    rows = [
        {"url": f"fixture:{index}", "text": f"A mathematical statement with unique number {index}. " * 8}
        for index in range(12)
    ]
    build_finemath_artifacts(
        rows, output_dir=data_root, tokenizer=WhitespaceTokenizer(),
        config=FineMathBuildConfig(train_token_budget=20, holdout_token_budget=10,
                                   max_seq_tokens=10, holdout_permille=500),
        source_repo="fixture", source_config=None, source_revision="rev-1",
        tokenizer_repo="fixture-tokenizer", tokenizer_revision="rev-1",
    )
    base = root / "models/base"
    base.mkdir(parents=True)
    (base / "weights.bin").write_bytes(b"base-model")
    adapter = root / "runs/sft/model/checkpoint-1"
    adapter.mkdir(parents=True)
    (adapter / "adapter.bin").write_bytes(b"adapter")
    checkpoint = data_root / "canonical/cpt_train.jsonl"
    train_output = root / "runs/sft"
    train_id = new_run_id("train", label="sft-e2e")
    train_result = {"train": {"status": "PASS"}, "checkpoint": str(adapter)}
    cli._write_training_run(
        output=train_output, result=train_result, run_id=train_id, run_label="sft-e2e",
        experiment_id="exp-e2e", stage="sft", model=base, dataset=checkpoint,
        config_path="recipes/mini/sft.yaml", resolved_config={"seed": 42, "max_steps": 1},
        started_at="2026-10-17T00:00:00Z",
    )

    merged = root / "models/sft-merged"
    def merge(*args, **kwargs):
        merged.mkdir(parents=True)
        (merged / "weights.bin").write_bytes(b"merged-model")
        return {"returncode": 0}
    monkeypatch.setattr(cli, "run_command", merge)
    merge_result = cli._run_merge_with_lineage(
        output=train_output, checkpoint=adapter, parent_model=base, merged_dir=merged,
        run_label="sft-e2e-merge", experiment_id="exp-e2e", stage="sft",
    )
    def fake_evalscope(cmd, dry_run=False, stream_output=False):
        work_dir = Path(cmd[cmd.index("--work-dir") + 1])
        assert (work_dir / "reviews" / merged.name).is_dir()
        return {"returncode": 0}
    monkeypatch.setattr(cli, "run_evalscope", fake_evalscope)
    monkeypatch.setattr(cli, "collect_evalscope_reports", lambda path: {
        "gsm8k": {"score": 0.5, "accuracy": 0.5, "samples": 4},
        "math_500": {"score": 0.25, "accuracy": 0.25, "samples": 4}})
    eval_args = Namespace(
        config="recipes/smoke/00-base-eval.yaml", model=str(merged), revision=None,
        output=str(root / "runs/evaluation"), run_label="sft-e2e-eval", stage="sft",
        limit=4, dry_run=False, internal_holdout=None, internal_limit=20, max_new_tokens=32,
        domain_holdout=None, domain_limit=20, ppl_max_length=128, dtype="float16",
    )
    assert cli.cmd_run_model_eval(eval_args) == 0
    evaluation = json.loads((root / "runs/evaluation/evaluation_lineage.json").read_text(encoding="utf-8"))
    assert evaluation["produced_by_run_id"]
    assert evaluation["model_id"] == json.loads(
        (train_output / "model_artifacts" / f"{json.loads((train_output / 'merge_runs' / merge_result['run_id'] / 'lineage.json').read_text(encoding='utf-8'))['outputs'][0]['id']}.json").read_text(encoding="utf-8")
    )["model_id"]

    assert evaluation["dataset_artifacts"]
    assert all(item["availability"] == "unavailable" for item in evaluation["dataset_artifacts"])
    eval_run = json.loads((root / "runs/evaluation/evaluation_run.json").read_text(encoding="utf-8"))
    assert eval_run["lineage_status"] == "incomplete"
    assert any("benchmark" in warning for warning in eval_run["lineage_warnings"])
    checked = validate_lineage_root(root, require_complete=False)
    assert checked["valid"], checked["errors"]
    assert checked["summary"]["runs"] >= 6  # build, quality, train, merge, evaluation
    assert any(item["code"] == "unavailable_evidence" for item in checked["warnings"])
    strict = validate_lineage_root(root, require_complete=True)
    assert not strict["valid"]
    assert any(item["code"] == "incomplete" for item in strict["errors"])
