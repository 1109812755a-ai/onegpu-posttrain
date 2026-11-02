from __future__ import annotations

import json

from onegpu_posttrain.reporting.cpt_effect import build_cpt_effect_report, write_cpt_effect_report


def put(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def fixture(tmp_path, *, holdout="holdout-v1", status="completed", train_status="PASS"):
    run_dir = tmp_path / "runs" / "cpt"
    put(run_dir / "run.json", {"stage": "peft_cpt", "train": {
        "status": train_status, "wall_seconds": 33.5, "peak_gpu_memory_mib_sampled": 7100,
    }, "merge": {"run_id": "merge-cpt-1", "execution": {"wall_seconds": 4.5}}})
    put(run_dir / "lineage.json", {"run_id": "train-cpt-1", "inputs": [
        {"id": "base-model", "kind": "model", "role": "parent"},
        {"id": "ds-cpt-generated-1", "kind": "dataset", "role": "train"}],
        "outputs": [{"id": "adapter-model", "kind": "model"}]})
    base = tmp_path / "eval-base" / "eval.json"
    after = tmp_path / "eval-cpt" / "eval.json"
    put(base, {"eval_id": "eval-base-1", "model": "base", "domain_perplexity": {"perplexity": 3.0},
               "evalscope_reports": {"gsm8k": {"score": 0.21}}})
    put(after, {"eval_id": "eval-cpt-1", "model": "cpt", "domain_perplexity": {},
                "evalscope_reports": {"gsm8k": {"score": 0.24}}})
    for path, evalid, modelid, producer in (
        (base, "eval-base-1", "base-model", "base-run"),
        (after, "eval-cpt-1", "merged-model", "merge-cpt-1"),
    ):
        put(path.with_name("evaluation_lineage.json"), {
            "eval_id": evalid, "model_id": modelid, "model_produced_by_run_id": producer,
            "dataset_ids": [holdout], "protocol_version": "unified-v1", "status": status,
        })
    return run_dir, base, after


def test_cpt_effect_report_computes_only_observed_metrics_and_resources(tmp_path):
    run, base, after = fixture(tmp_path)
    report = build_cpt_effect_report(run, base, after)
    assert report["dataset_id"] == "ds-cpt-generated-1"
    assert report["metrics"]["gsm8k"] == {
        "before": 0.21, "after": 0.24, "delta": 0.03,
        "status": "measured", "direction": "higher_is_better",
    }
    assert report["metrics"]["domain_ppl"]["before"] == 3.0
    assert report["metrics"]["domain_ppl"]["after"] is None
    assert report["metrics"]["domain_ppl"]["delta"] is None
    assert report["metrics"]["domain_ppl"]["status"] == "not_measured"
    assert report["resources"]["peak_gpu_memory_mib_sampled"] == 7100
    assert report["causal_claim"] is False
    destination = tmp_path / "effect.json"
    write_cpt_effect_report(run, base, after, destination)
    assert json.loads(destination.read_text())["dataset_id"] == report["dataset_id"]


def test_mismatched_holdout_or_incomplete_evaluation_keeps_deltas_unknown(tmp_path):
    run, base, after = fixture(tmp_path, status="partial")
    report = build_cpt_effect_report(run, base, after)
    assert report["metrics"]["gsm8k"]["delta"] is None
    assert report["comparison_status"] == "observational_or_incomplete"

    run, base, after = fixture(tmp_path / "different", holdout="other-holdout")
    # Change only one side to simulate a holdout mismatch.
    sidecar = after.with_name("evaluation_lineage.json")
    payload = json.loads(sidecar.read_text())
    payload["dataset_ids"] = ["different-holdout"]
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    mismatch = build_cpt_effect_report(run, base, after)
    assert mismatch["metrics"]["gsm8k"]["delta"] is None
    assert any("数据集集合不一致" in warning for warning in mismatch["warnings"])


def test_failed_training_never_claims_effect_delta(tmp_path):
    run, base, after = fixture(tmp_path, train_status="OOM")
    report = build_cpt_effect_report(run, base, after)
    assert report["metrics"]["gsm8k"]["delta"] is None
    assert any("没有 PASS" in warning for warning in report["warnings"])
