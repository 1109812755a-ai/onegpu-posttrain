import json
from pathlib import Path

from onegpu_posttrain.lineage import sha256_json
from onegpu_posttrain.reporting.alignment_compare import write_alignment_comparison


def _write(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_alignment_comparison_uses_common_preference_metrics(tmp_path: Path):
    eval_payloads = {
        "sft": {"evalscope_reports": {"gsm8k": {"score": 0.2}, "math_500": {"score": 0.1}}, "internal_math": {"accuracy": 0.3}},
        "dpo": {"evalscope_reports": {"gsm8k": {"score": 0.22}, "math_500": {"score": 0.11}}, "internal_math": {"accuracy": 0.31}},
        "simpo": {"evalscope_reports": {"gsm8k": {"score": 0.21}, "math_500": {"score": 0.12}}, "internal_math": {"accuracy": 0.32}},
    }
    pref_payloads = {
        "sft": {"policy_pair_accuracy": 0.5, "policy_pair_margin": 0.0, "chosen_mean_logprob": -2.0, "rejected_mean_logprob": -2.0},
        "dpo": {"policy_pair_accuracy": 0.7, "policy_pair_margin": 0.2, "chosen_mean_logprob": -1.8, "rejected_mean_logprob": -2.0},
        "simpo": {"policy_pair_accuracy": 0.8, "policy_pair_margin": 0.3, "chosen_mean_logprob": -1.7, "rejected_mean_logprob": -2.0},
    }
    paths = {}
    for stage, payload in eval_payloads.items():
        p = tmp_path / f"{stage}-eval.json"
        _write(p, payload)
        paths[f"{stage}_eval"] = p
    for stage, payload in pref_payloads.items():
        p = tmp_path / f"{stage}-pref.json"
        _write(p, payload)
        paths[f"{stage}_pref"] = p

    result = write_alignment_comparison(
        sft_eval=paths["sft_eval"],
        dpo_eval=paths["dpo_eval"],
        simpo_eval=paths["simpo_eval"],
        sft_preference=paths["sft_pref"],
        dpo_preference=paths["dpo_pref"],
        simpo_preference=paths["simpo_pref"],
        output_dir=tmp_path / "out",
    )
    assert result["stages"]["simpo"]["preference"]["policy_pair_accuracy"] == 0.8
    assert result["stages"]["simpo"]["preference"]["policy_pair_accuracy_evidence"]["status"] == "measured"
    assert result["comparability"]["dpo_vs_simpo"]["status"] == "observational_only"
    assert result["guardrails"]["status"] == "not_configured"
    assert result["training_cost"]["status"] == "not_measured"
    assert (tmp_path / "out" / "alignment_comparison.md").exists()


def _write_run(root: Path, *, run_id: str, algorithm: str, data_digest: str = "d" * 64, seed: int = 42):
    root.mkdir()
    resolved = {"algorithm": algorithm, "training": {"seed": seed, "max_steps": 3}}
    _write(root / "resolved_config.json", resolved)
    _write(root / "lineage.json", {
        "run_id": run_id, "status": "completed", "resolved_config_sha256": sha256_json(resolved),
        "inputs": [
            {"id": "model-m2", "kind": "model", "digest": {"value": "a" * 64}},
            {"id": "dataset-pref", "kind": "dataset", "digest": {"value": data_digest}},
        ],
    })


def test_alignment_marks_only_evidence_matched_dpo_simpo_as_ab_and_preserves_missing_vs_zero(tmp_path):
    paths = {}
    for stage in ("sft", "dpo", "simpo"):
        eval_path = tmp_path / f"{stage}-eval.json"
        _write(eval_path, {"evalscope_reports": {}, "internal_math": {}})
        paths[f"{stage}_eval"] = eval_path
        pref_path = tmp_path / f"{stage}-pref.json"
        payload = {"policy_pair_accuracy": 0.0, "policy_pair_margin": None, "pairs": 0}
        if stage == "dpo":
            payload["metric_statuses"] = {
                "preference.accuracy": {"status": "failed", "denominator": 0, "reason": "model load failed"},
                "preference.margin": {"status": "not_measured", "denominator": 0, "reason": "not run"},
            }
            payload["policy_pair_accuracy"] = None
        _write(pref_path, payload)
        paths[f"{stage}_preference"] = pref_path

    dpo_run, simpo_run = tmp_path / "dpo-run", tmp_path / "simpo-run"
    _write_run(dpo_run, run_id="run-dpo", algorithm="dpo")
    _write_run(simpo_run, run_id="run-simpo", algorithm="simpo")
    result = write_alignment_comparison(**paths, output_dir=tmp_path / "matched",
                                       dpo_run=dpo_run, simpo_run=simpo_run)
    assert result["comparability"]["dpo_vs_simpo"]["status"] == "matched_ab"
    assert result["stages"]["dpo"]["preference"]["policy_pair_accuracy"] is None
    assert result["stages"]["dpo"]["preference"]["policy_pair_accuracy_evidence"]["status"] == "failed"
    assert result["stages"]["sft"]["preference"]["policy_pair_accuracy"] == 0.0

    mismatched = tmp_path / "simpo-run-mismatch"
    _write_run(mismatched, run_id="run-simpo-other-data", algorithm="simpo", data_digest="e" * 64)
    result = write_alignment_comparison(**paths, output_dir=tmp_path / "mismatch",
                                       dpo_run=dpo_run, simpo_run=mismatched)
    assert result["comparability"]["dpo_vs_simpo"]["status"] == "observational_only"
    assert any("dataset" in reason for reason in result["comparability"]["dpo_vs_simpo"]["reasons"])

    altered_config = json.loads((simpo_run / "resolved_config.json").read_text(encoding="utf-8"))
    altered_config["training"]["max_steps"] = 99
    _write(simpo_run / "resolved_config.json", altered_config)
    result = write_alignment_comparison(**paths, output_dir=tmp_path / "config-mismatch",
                                       dpo_run=dpo_run, simpo_run=simpo_run)
    assert result["comparability"]["dpo_vs_simpo"]["status"] == "observational_only"
    assert "resolved config hash mismatch" in result["comparability"]["dpo_vs_simpo"]["reasons"][0]
