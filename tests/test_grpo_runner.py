
import pytest

from onegpu_posttrain.training.ms_swift_runner import build_grpo_command


def _training(beta=0.04, ref=True):
    return {
        "tuner_type": "lora",
        "torch_dtype": "float16",
        "learning_rate": 1e-5,
        "lora_rank": 8,
        "lora_alpha": 32,
        "target_modules": "all-linear",
        "gradient_checkpointing": True,
        "beta": beta,
        "reference_enabled": ref,
        "use_vllm": False,
        "num_generations": 2,
        "generation_batch_size_completions": 4,
        "per_device_train_batch_size_completions": 1,
        "gradient_accumulation_steps": 4,
        "max_prompt_length_candidate": 512,
        "max_completion_length_candidate": 512,
        "max_steps": 20,
        "seed": 42,
    }


def test_grpo_standard_command_contract():
    training = _training()
    training["logging_dir"] = "out/tensorboard"
    cmd = build_grpo_command(
        model="M2", dataset="grpo.jsonl", output_dir="out", training=training,
        reward={"accuracy": 1.0, "format": 0.1}, external_plugin="plugins/grpo_rewards.py",
    )
    joined = " ".join(cmd)
    assert "--rlhf_type grpo" in joined
    assert "--beta 0.04" in joined
    assert "--use_vllm false" in joined
    assert "--generation_batch_size 4" in joined
    assert "--num_generations 2" in joined
    assert "--per_device_train_batch_size 1" in joined
    assert "--gradient_accumulation_steps 4" in joined
    assert "--max_length 512" in joined
    assert "--max_completion_length 512" in joined
    assert "--truncation_strategy delete" in joined
    assert "--reward_funcs onegpu_accuracy onegpu_boxed_format" in joined
    assert "--reward_weights 1.0 0.1" in joined
    assert "--external_plugins plugins/grpo_rewards.py" in joined
    assert "--logging_dir out/tensorboard" in joined


def test_grpo_lowmem_beta_zero_contract():
    cmd = build_grpo_command(
        model="M2", dataset="grpo.jsonl", output_dir="out", training=_training(0.0, False),
        reward={"accuracy": 1.0, "format": 0.1},
    )
    assert cmd[cmd.index("--beta") + 1] == "0.0"


def test_grpo_rejects_mismatched_reference_semantics():
    with pytest.raises(ValueError, match="reference_enabled"):
        build_grpo_command(
            model="M2", dataset="grpo.jsonl", output_dir="out", training=_training(0.0, True),
            reward={"accuracy": 1.0, "format": 0.1},
        )


def test_grpo_rejects_invalid_generation_batch():
    t = _training()
    t["generation_batch_size_completions"] = 3
    with pytest.raises(ValueError, match="divisible by num_generations"):
        build_grpo_command(model="M2", dataset="d", output_dir="o", training=t, reward={"accuracy": 1, "format": .1})
