import json
from pathlib import Path

from onegpu_posttrain.reporting.grpo_compare import write_grpo_comparison


def _eval(path: Path, gsm=0.1, math=0.2, internal=0.3):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "evalscope_reports": {"gsm8k": {"score": gsm}, "math_500": {"score": math}},
        "internal_math": {"accuracy": internal},
    }), encoding="utf-8")


def _run(path: Path, status="PASS", beta=0.04):
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "latest": {"reward": 0.5, "reward_std": 0.2, "frac_reward_zero_std": 0.1, "kl": 0.03 if beta else None, "completions/mean_length": 100},
        "reward_components": {
            "onegpu_accuracy": {"latest_mean": 0.45},
            "onegpu_boxed_format": {"latest_mean": 0.7},
        },
    }
    path.write_text(json.dumps({"train": {"status": status, "wall_seconds": 10, "peak_gpu_memory_mib_sampled": 7000}, "grpo_metrics": metrics}), encoding="utf-8")


def test_grpo_comparison_handles_profiles(tmp_path: Path):
    sft = tmp_path / "sft.json"
    se = tmp_path / "std_eval.json"
    le = tmp_path / "low_eval.json"
    sr = tmp_path / "std_run.json"
    lr = tmp_path / "low_run.json"
    _eval(sft)
    _eval(se, .2, .3, .4)
    _eval(le, .15, .25, .35)
    _run(sr, beta=.04)
    _run(lr, beta=0)
    result = write_grpo_comparison(
        sft_eval=sft, standard_run=sr, standard_eval=se, lowmem_run=lr, lowmem_eval=le, output_dir=tmp_path / "out"
    )
    assert result["standard"]["training"]["status"] == "PASS"
    assert result["standard"]["training"]["accuracy_reward"] == 0.45
    assert (tmp_path / "out" / "grpo_comparison.md").exists()


def test_grpo_comparison_allows_standard_oom(tmp_path: Path):
    sft = tmp_path / "sft.json"; _eval(sft)
    sr = tmp_path / "std_run.json"; _run(sr, status="OOM")
    lr = tmp_path / "low_run.json"; _run(lr, status="PASS", beta=0)
    le = tmp_path / "low_eval.json"; _eval(le)
    result = write_grpo_comparison(sft_eval=sft, standard_run=sr, standard_eval=None, lowmem_run=lr, lowmem_eval=le, output_dir=tmp_path / "out")
    assert result["standard"]["training"]["status"] == "OOM"
    assert result["standard"]["task"] is None
