from pathlib import Path

import pytest

from onegpu_posttrain.training.ms_swift_runner import (
    build_merge_command,
    build_peft_cpt_command,
    discover_latest_checkpoint,
)


def test_build_peft_cpt_command_contains_stage_contract(tmp_path: Path):
    data = tmp_path / "cpt.jsonl"
    data.write_text('{"messages":[{"role":"assistant","content":"math"}]}\n', encoding="utf-8")
    cmd = build_peft_cpt_command(
        model="Qwen/Qwen3-0.6B-Base",
        model_revision="abc123",
        dataset=data,
        output_dir=tmp_path / "out",
        training={
            "tuner_type": "lora",
            "torch_dtype": "float16",
            "max_steps": 20,
            "lora_rank": 8,
            "seed": 42,
        },
    )
    assert cmd[:2] == ["swift", "pt"]
    assert "--model_revision" in cmd
    assert "abc123" in cmd
    assert "--tuner_type" in cmd and "lora" in cmd
    assert "--max_steps" in cmd and "20" in cmd
    assert "--split_dataset_ratio" in cmd and "0" in cmd
    assert cmd[cmd.index("--dataloader_persistent_workers") + 1] == "false"


def test_build_peft_cpt_command_qlora_profile(tmp_path: Path):
    cmd = build_peft_cpt_command(
        model="m",
        model_revision=None,
        dataset=tmp_path / "d.jsonl",
        output_dir=tmp_path / "out",
        training={"quant_bits": 4, "quant_method": "bnb"},
    )
    assert "--quant_method" in cmd and "bnb" in cmd
    assert "--quant_bits" in cmd and "4" in cmd


def test_build_peft_cpt_command_supports_one_epoch_without_smoke_step_cap(tmp_path: Path):
    cmd = build_peft_cpt_command(
        model="Qwen/Qwen3-0.6B-Base",
        model_revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        dataset=tmp_path / "cpt.jsonl",
        output_dir=tmp_path / "mini-cpt",
        training={"num_train_epochs": 1, "seed": 42, "gradient_checkpointing": True},
    )
    assert "--num_train_epochs" in cmd
    assert cmd[cmd.index("--num_train_epochs") + 1] == "1"
    assert "--max_steps" not in cmd


def test_discover_latest_checkpoint(tmp_path: Path):
    (tmp_path / "v0" / "checkpoint-10").mkdir(parents=True)
    (tmp_path / "v0" / "checkpoint-20").mkdir(parents=True)
    (tmp_path / "v0" / "checkpoint-20-merged").mkdir(parents=True)
    assert discover_latest_checkpoint(tmp_path).name == "checkpoint-20"


def test_discover_best_checkpoint_uses_trainer_state_and_keeps_final(tmp_path: Path):
    import json

    from onegpu_posttrain.training.ms_swift_runner import discover_checkpoint_selection

    best = tmp_path / "checkpoint-10"
    final = tmp_path / "checkpoint-20"
    best.mkdir(); final.mkdir()
    (best / "adapter.bin").write_bytes(b"best")
    (final / "adapter.bin").write_bytes(b"final")
    (tmp_path / "trainer_state.json").write_text(json.dumps({
        "best_model_checkpoint": str(best), "best_metric": 0.25,
    }), encoding="utf-8")
    selected = discover_checkpoint_selection(tmp_path, "best")
    assert Path(selected["selected"]) == best
    assert Path(selected["best"]) == best
    assert Path(selected["final"]) == final
    assert selected["metric_value"] == 0.25


def test_merge_command_explicit_output():
    cmd = build_merge_command(checkpoint="checkpoint-20", output_dir="merged/M1", model="base-model")
    assert cmd[:2] == ["swift", "export"]
    assert cmd[cmd.index("--adapters") + 1] == "checkpoint-20"
    assert cmd[cmd.index("--model") + 1] == "base-model"
    assert cmd[cmd.index("--output_dir") + 1] == "merged/M1"
    assert cmd[cmd.index("--merge_lora") + 1] == "true"
    assert cmd[cmd.index("--load_args") + 1] == "false"


def test_build_full_cpt_probe_command_is_full_parameter(tmp_path: Path):
    from onegpu_posttrain.training.ms_swift_runner import build_full_cpt_probe_command

    cmd = build_full_cpt_probe_command(
        model="Qwen/Qwen3-0.6B-Base",
        model_revision="rev",
        dataset=tmp_path / "cpt.jsonl",
        output_dir=tmp_path / "out",
        training={"max_steps": 5, "max_length": 512, "optim": "adamw_torch"},
    )
    assert cmd[:2] == ["swift", "pt"]
    assert cmd[cmd.index("--tuner_type") + 1] == "full"
    assert cmd[cmd.index("--max_steps") + 1] == "5"
    assert cmd[cmd.index("--max_length") + 1] == "512"
    assert cmd[cmd.index("--optim") + 1] == "adamw_torch"
    assert "--lora_rank" not in cmd


def test_build_sft_command_uses_fresh_lora_on_parent(tmp_path: Path):
    from onegpu_posttrain.training.ms_swift_runner import build_sft_command

    cmd = build_sft_command(
        model=tmp_path / "M1-CPT",
        dataset=tmp_path / "sft.jsonl",
        output_dir=tmp_path / "out",
        training={"max_steps": 20, "max_length": 1024, "lora_rank": 8},
    )
    assert cmd[:2] == ["swift", "sft"]
    assert cmd[cmd.index("--model") + 1].endswith("M1-CPT")
    assert cmd[cmd.index("--tuner_type") + 1] == "lora"
    assert cmd[cmd.index("--max_steps") + 1] == "20"
    assert cmd[cmd.index("--max_length") + 1] == "1024"
    assert "--report_to" not in cmd

    probe_cmd = build_sft_command(
        model=tmp_path / "M1-CPT",
        dataset=tmp_path / "sft.jsonl",
        output_dir=tmp_path / "probe",
        training={"report_to": "none"},
    )
    assert probe_cmd[probe_cmd.index("--report_to") + 1] == "none"


def test_build_sft_command_supports_complete_epoch_budget():
    from onegpu_posttrain.training.ms_swift_runner import build_sft_command

    cmd = build_sft_command(
        model="M1-CPT", dataset="deepmath-sft.jsonl", output_dir="sft-run",
        training={"num_train_epochs": 1, "max_length": 2048, "gradient_checkpointing": True},
    )
    assert cmd[cmd.index("--num_train_epochs") + 1] == "1"
    assert "--max_steps" not in cmd


def test_sft_command_emits_only_approved_qlora_flags_and_keeps_lora_unchanged():
    from onegpu_posttrain.training.ms_swift_runner import build_sft_command

    common = {"quant_method": "bnb", "quant_bits": 4, "bnb_4bit_compute_dtype": "float16",
              "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True}
    qlora = build_sft_command(model="base", dataset="data.jsonl", output_dir="out", training=common)
    for flag, value in [("--quant_method", "bnb"), ("--quant_bits", "4"),
                        ("--bnb_4bit_compute_dtype", "float16"), ("--bnb_4bit_quant_type", "nf4"),
                        ("--bnb_4bit_use_double_quant", "true")]:
        assert qlora[qlora.index(flag) + 1] == value
    lora = build_sft_command(model="base", dataset="data.jsonl", output_dir="out", training={})
    assert not any(arg.startswith("--quant") or arg.startswith("--bnb") for arg in lora)
    with pytest.raises(ValueError, match="仅允许批准的 QLoRA"):
        build_sft_command(model="base", dataset="data.jsonl", output_dir="out",
                          training={**common, "quant_bits": 8})


def test_failure_status_distinguishes_oom():
    from onegpu_posttrain.training.ms_swift_runner import _failure_status

    assert _failure_status("torch.OutOfMemoryError: CUDA out of memory") == "OOM"
    assert _failure_status("some unrelated traceback") == "FAILED"


def test_build_dpo_command_contract(tmp_path: Path):
    from onegpu_posttrain.training.ms_swift_runner import build_preference_rlhf_command

    cmd = build_preference_rlhf_command(
        algorithm="dpo",
        model=tmp_path / "M2-SFT",
        dataset=tmp_path / "preference.jsonl",
        output_dir=tmp_path / "out",
        training={"max_steps": 20, "beta": 0.1, "loss_type": "sigmoid", "lora_rank": 8},
    )
    assert cmd[:2] == ["swift", "rlhf"]
    assert cmd[cmd.index("--rlhf_type") + 1] == "dpo"
    assert cmd[cmd.index("--model") + 1].endswith("M2-SFT")
    assert cmd[cmd.index("--beta") + 1] == "0.1"
    assert cmd[cmd.index("--loss_type") + 1] == "sigmoid"
    assert "--simpo_gamma" not in cmd


def test_build_original_simpo_command_contract(tmp_path: Path):
    from onegpu_posttrain.training.ms_swift_runner import build_preference_rlhf_command

    cmd = build_preference_rlhf_command(
        algorithm="simpo",
        model=tmp_path / "M2-SFT",
        dataset=tmp_path / "preference.jsonl",
        output_dir=tmp_path / "out",
        training={"max_steps": 20, "beta": 2.0, "simpo_gamma": 1.0, "cpo_alpha": 0.0},
    )
    assert cmd[cmd.index("--rlhf_type") + 1] == "simpo"
    assert cmd[cmd.index("--beta") + 1] == "2.0"
    assert cmd[cmd.index("--simpo_gamma") + 1] == "1.0"
    assert cmd[cmd.index("--cpo_alpha") + 1] == "0.0"
    assert "--loss_type" not in cmd
