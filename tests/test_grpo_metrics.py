import json
from pathlib import Path

from onegpu_posttrain.evaluation.grpo_metrics import collect_grpo_training_metrics


def test_collect_grpo_metrics(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-20"
    ckpt.mkdir()
    state = {
        "log_history": [
            {
                "step": 10,
                "reward": 0.4,
                "reward_std": 0.2,
                "frac_reward_zero_std": 0.25,
                "kl": 0.03,
                "completions/mean_length": 100,
                "rewards/onegpu_accuracy/mean": 0.35,
                "rewards/onegpu_boxed_format/mean": 0.5,
            },
            {
                "step": 20,
                "reward": 0.6,
                "reward_std": 0.3,
                "frac_reward_zero_std": 0.1,
                "kl": 0.02,
                "completions/mean_length": 120,
                "rewards/onegpu_accuracy/mean": 0.55,
                "rewards/onegpu_boxed_format/mean": 0.7,
            },
        ]
    }
    (ckpt / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    result = collect_grpo_training_metrics(tmp_path, beta=0.04)
    assert result["optimizer_step"] == 20
    assert result["latest"]["reward"] == 0.6
    assert result["mean"]["reward"] == 0.5
    assert result["latest"]["kl"] == 0.02
    assert result["reward_components"]["onegpu_accuracy"]["latest_mean"] == 0.55


def test_lowmem_marks_kl_na(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-2"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text(json.dumps({"log_history": [{"step": 2, "reward": 0.5, "kl": 99.0}]}), encoding="utf-8")
    result = collect_grpo_training_metrics(tmp_path, beta=0.0)
    assert result["kl_applicable"] is False
    assert result["latest"]["kl"] is None
    assert result["metric_statuses"]["kl"]["status"] == "not_applicable"


def test_unknown_reward_metrics_are_not_inferred_from_training_steps(tmp_path: Path):
    ckpt = tmp_path / "checkpoint-3"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"step": 3, "loss": 0.7}]}), encoding="utf-8")
    result = collect_grpo_training_metrics(tmp_path, beta=0.04)
    assert result["status"] == "not_measured"
    assert result["latest"]["reward"] is None
    assert result["metric_statuses"]["reward"]["status"] == "not_measured"
    assert result["metric_statuses"]["kl"]["status"] == "not_measured"
