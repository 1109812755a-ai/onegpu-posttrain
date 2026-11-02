from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from ..subprocess_utils import child_process_env, resolve_console_command


def command_string(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def build_peft_cpt_command(
    *,
    model: str,
    model_revision: str | None,
    dataset: str | Path,
    output_dir: str | Path,
    training: dict[str, Any],
    validation_dataset: str | Path | None = None,
) -> list[str]:
    """Build a conservative single-GPU `swift pt` LoRA command.

    The dataset must already be converted to ms-swift pre-training native format.
    """
    cmd = [
        "swift", "pt",
        "--model", str(model),
        "--dataset", str(dataset),
        "--output_dir", str(output_dir),
        "--tuner_type", str(training.get("tuner_type", "lora")),
        "--torch_dtype", str(training.get("torch_dtype", "float16")),
        "--per_device_train_batch_size", str(training.get("per_device_train_batch_size", 1)),
        "--gradient_accumulation_steps", str(training.get("gradient_accumulation_steps", 8)),
        "--learning_rate", str(training.get("learning_rate", 1e-4)),
        "--lora_rank", str(training.get("lora_rank", 8)),
        "--lora_alpha", str(training.get("lora_alpha", 32)),
        "--target_modules", str(training.get("target_modules", "all-linear")),
        "--max_length", str(training.get("max_length", 1024)),
        "--gradient_checkpointing", str(bool(training.get("gradient_checkpointing", True))).lower(),
        "--warmup_ratio", str(training.get("warmup_ratio", 0.05)),
        "--logging_steps", str(training.get("logging_steps", 1)),
        "--save_steps", str(training.get("save_steps", 10)),
        "--save_total_limit", str(training.get("save_total_limit", 2)),
        "--split_dataset_ratio", str(training.get("split_dataset_ratio", 0)),
        "--dataset_num_proc", str(training.get("dataset_num_proc", 1)),
        "--dataloader_num_workers", str(training.get("dataloader_num_workers", 0)),
        "--dataloader_persistent_workers", str(bool(training.get("dataloader_persistent_workers", False))).lower(),
        "--use_hf", str(bool(training.get("use_hf", True))).lower(),
        "--seed", str(training.get("seed", 42)),
    ]
    if model_revision:
        cmd += ["--model_revision", str(model_revision)]
    _append_validation_args(cmd, validation_dataset, training)

    # A recipe may choose an epoch-based budget for a complete dataset pass.
    # Keep the historical fixed-step default for Smoke and older recipes.
    if training.get("num_train_epochs") is not None:
        cmd += ["--num_train_epochs", str(training["num_train_epochs"])]
    else:
        cmd += ["--max_steps", str(training.get("max_steps", 20))]

    # Optional QLoRA profile. For Qwen3-0.6B the default smoke profile stays
    # non-quantized LoRA because it is simpler and should fit 8 GB; QLoRA can
    # be enabled without changing the runner contract.
    if training.get("quant_bits"):
        cmd += [
            "--quant_method", str(training.get("quant_method", "bnb")),
            "--quant_bits", str(training["quant_bits"]),
            "--bnb_4bit_compute_dtype", str(training.get("bnb_4bit_compute_dtype", training.get("torch_dtype", "float16"))),
            "--bnb_4bit_quant_type", str(training.get("bnb_4bit_quant_type", "nf4")),
            "--bnb_4bit_use_double_quant", str(bool(training.get("bnb_4bit_use_double_quant", True))).lower(),
        ]
    if training.get("weight_decay") is not None:
        cmd += ["--weight_decay", str(training["weight_decay"])]
    if training.get("max_grad_norm") is not None:
        cmd += ["--max_grad_norm", str(training["max_grad_norm"])]
    if training.get("save_only_model") is not None:
        cmd += ["--save_only_model", str(bool(training["save_only_model"])).lower()]
    return cmd



def build_full_cpt_probe_command(
    *,
    model: str,
    model_revision: str | None,
    dataset: str | Path,
    output_dir: str | Path,
    training: dict[str, Any],
) -> list[str]:
    """Build a deliberately small full-parameter CPT feasibility probe.

    This is diagnostic: the goal is to answer whether a full-parameter update can
    execute on the target GPU, not to produce a useful checkpoint.
    """
    cmd = [
        "swift", "pt",
        "--model", str(model),
        "--dataset", str(dataset),
        "--output_dir", str(output_dir),
        "--tuner_type", "full",
        "--torch_dtype", str(training.get("torch_dtype", "float16")),
        "--per_device_train_batch_size", str(training.get("per_device_train_batch_size", 1)),
        "--gradient_accumulation_steps", str(training.get("gradient_accumulation_steps", 1)),
        "--learning_rate", str(training.get("learning_rate", 1e-5)),
        "--optim", str(training.get("optim", "adamw_torch")),
        "--max_length", str(training.get("max_length", 512)),
        "--gradient_checkpointing", str(bool(training.get("gradient_checkpointing", True))).lower(),
        "--max_steps", str(training.get("max_steps", 5)),
        "--warmup_ratio", str(training.get("warmup_ratio", 0.0)),
        "--logging_steps", str(training.get("logging_steps", 1)),
        "--save_steps", str(training.get("save_steps", training.get("max_steps", 5))),
        "--save_total_limit", str(training.get("save_total_limit", 1)),
        "--split_dataset_ratio", str(training.get("split_dataset_ratio", 0)),
        "--dataset_num_proc", str(training.get("dataset_num_proc", 1)),
        "--dataloader_num_workers", str(training.get("dataloader_num_workers", 0)),
        "--dataloader_persistent_workers", str(bool(training.get("dataloader_persistent_workers", False))).lower(),
        "--save_only_model", str(bool(training.get("save_only_model", True))).lower(),
        "--use_hf", str(bool(training.get("use_hf", True))).lower(),
        "--seed", str(training.get("seed", 42)),
    ]
    if model_revision:
        cmd += ["--model_revision", str(model_revision)]
    return cmd


def build_sft_command(
    *,
    model: str | Path,
    dataset: str | Path,
    output_dir: str | Path,
    training: dict[str, Any],
    validation_dataset: str | Path | None = None,
) -> list[str]:
    """Build the single-GPU LoRA SFT smoke command for merged M1."""
    cmd = [
        "swift", "sft",
        "--model", str(model),
        "--dataset", str(dataset),
        "--output_dir", str(output_dir),
        "--tuner_type", str(training.get("tuner_type", "lora")),
        "--torch_dtype", str(training.get("torch_dtype", "float16")),
        "--per_device_train_batch_size", str(training.get("per_device_train_batch_size", 1)),
        "--gradient_accumulation_steps", str(training.get("gradient_accumulation_steps", 8)),
        "--learning_rate", str(training.get("learning_rate", 1e-4)),
        "--lora_rank", str(training.get("lora_rank", 8)),
        "--lora_alpha", str(training.get("lora_alpha", 32)),
        "--target_modules", str(training.get("target_modules", "all-linear")),
        "--max_length", str(training.get("max_length", 1024)),
        "--gradient_checkpointing", str(bool(training.get("gradient_checkpointing", True))).lower(),
        "--warmup_ratio", str(training.get("warmup_ratio", 0.05)),
        "--logging_steps", str(training.get("logging_steps", 1)),
        "--save_steps", str(training.get("save_steps", 10)),
        "--save_total_limit", str(training.get("save_total_limit", 2)),
        "--split_dataset_ratio", str(training.get("split_dataset_ratio", 0)),
        "--dataset_num_proc", str(training.get("dataset_num_proc", 1)),
        "--dataloader_num_workers", str(training.get("dataloader_num_workers", 0)),
        "--dataloader_persistent_workers", str(bool(training.get("dataloader_persistent_workers", False))).lower(),
        "--use_hf", str(bool(training.get("use_hf", True))).lower(),
        "--seed", str(training.get("seed", 42)),
    ]
    _append_validation_args(cmd, validation_dataset, training)
    if training.get("weight_decay") is not None:
        cmd += ["--weight_decay", str(training["weight_decay"])]
    if training.get("max_grad_norm") is not None:
        cmd += ["--max_grad_norm", str(training["max_grad_norm"])]
    if training.get("save_only_model") is not None:
        cmd += ["--save_only_model", str(bool(training["save_only_model"])).lower()]
    if training.get("report_to") is not None:
        cmd += ["--report_to", str(training["report_to"])]
    if training.get("quant_bits") is not None:
        approved_qlora = {
            "quant_method": "bnb", "quant_bits": 4,
            "bnb_4bit_compute_dtype": "float16", "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        }
        if any(training.get(key) != value for key, value in approved_qlora.items()):
            raise ValueError("SFT 仅允许批准的 QLoRA 配置：BNB 4-bit NF4、FP16 compute、double quant")
        cmd += ["--quant_method", "bnb", "--quant_bits", "4",
                "--bnb_4bit_compute_dtype", "float16", "--bnb_4bit_quant_type", "nf4",
                "--bnb_4bit_use_double_quant", "true"]
    if training.get("num_train_epochs") is not None:
        cmd += ["--num_train_epochs", str(training["num_train_epochs"])]
    else:
        cmd += ["--max_steps", str(training.get("max_steps", 20))]
    return cmd


def build_preference_rlhf_command(
    *,
    algorithm: str,
    model: str | Path,
    dataset: str | Path,
    output_dir: str | Path,
    training: dict[str, Any],
    validation_dataset: str | Path | None = None,
) -> list[str]:
    """Build a single-GPU DPO/SimPO LoRA command from merged M2.

    DPO and SimPO share the same dataset/parent/budget contract. Algorithm-only
    hyperparameters are appended explicitly so the experiment artifact shows
    exactly what changed.
    """
    algorithm = algorithm.lower()
    if algorithm not in {"dpo", "simpo"}:
        raise ValueError(f"Unsupported preference algorithm: {algorithm}")
    cmd = [
        "swift", "rlhf",
        "--rlhf_type", algorithm,
        "--model", str(model),
        "--dataset", str(dataset),
        "--output_dir", str(output_dir),
        "--tuner_type", str(training.get("tuner_type", "lora")),
        "--torch_dtype", str(training.get("torch_dtype", "float16")),
        "--per_device_train_batch_size", str(training.get("per_device_train_batch_size", 1)),
        "--gradient_accumulation_steps", str(training.get("gradient_accumulation_steps", 8)),
        "--learning_rate", str(training.get("learning_rate", 1e-4)),
        "--lora_rank", str(training.get("lora_rank", 8)),
        "--lora_alpha", str(training.get("lora_alpha", 32)),
        "--target_modules", str(training.get("target_modules", "all-linear")),
        "--max_length", str(training.get("max_length", 1024)),
        "--gradient_checkpointing", str(bool(training.get("gradient_checkpointing", True))).lower(),
        "--max_steps", str(training.get("max_steps", 20)),
        "--warmup_ratio", str(training.get("warmup_ratio", 0.05)),
        "--logging_steps", str(training.get("logging_steps", 1)),
        "--save_steps", str(training.get("save_steps", 10)),
        "--save_total_limit", str(training.get("save_total_limit", 2)),
        "--split_dataset_ratio", str(training.get("split_dataset_ratio", 0)),
        "--dataset_num_proc", str(training.get("dataset_num_proc", 1)),
        "--dataloader_num_workers", str(training.get("dataloader_num_workers", 0)),
        "--dataloader_persistent_workers", str(bool(training.get("dataloader_persistent_workers", False))).lower(),
        "--use_hf", str(bool(training.get("use_hf", True))).lower(),
        "--seed", str(training.get("seed", 42)),
        "--beta", str(training.get("beta", 0.1 if algorithm == "dpo" else 2.0)),
    ]
    _append_validation_args(cmd, validation_dataset, training)
    if algorithm == "dpo":
        cmd += ["--loss_type", str(training.get("loss_type", "sigmoid"))]
        if training.get("label_smoothing") is not None:
            cmd += ["--label_smoothing", str(training.get("label_smoothing", 0.0))]
    else:
        cmd += [
            "--simpo_gamma", str(training.get("simpo_gamma", 1.0)),
            "--cpo_alpha", str(training.get("cpo_alpha", 0.0)),
        ]
    if training.get("weight_decay") is not None:
        cmd += ["--weight_decay", str(training["weight_decay"])]
    if training.get("max_grad_norm") is not None:
        cmd += ["--max_grad_norm", str(training["max_grad_norm"])]
    return cmd



def build_grpo_command(
    *,
    model: str | Path,
    dataset: str | Path,
    output_dir: str | Path,
    training: dict[str, Any],
    reward: dict[str, Any],
    external_plugin: str | Path = "plugins/grpo_rewards.py",
    validation_dataset: str | Path | None = None,
) -> list[str]:
    """Build the single-GPU LoRA GRPO smoke command from merged M2.

    v0.1 intentionally keeps rollout generation in-process (``use_vllm=false``)
    and treats batches in completion units.  The builder validates the frozen
    completion-budget semantics before emitting a command.
    """
    per_device = int(training.get("per_device_train_batch_size_completions", 1))
    grad_acc = int(training.get("gradient_accumulation_steps", 4))
    generation_batch = int(training.get("generation_batch_size_completions", per_device * grad_acc))
    num_generations = int(training.get("num_generations", 2))
    if per_device < 1 or grad_acc < 1 or generation_batch < 1 or num_generations < 2:
        raise ValueError("GRPO batch values must be positive and num_generations >= 2")
    if generation_batch % num_generations != 0:
        raise ValueError("generation_batch_size must be divisible by num_generations")
    if generation_batch % per_device != 0:
        raise ValueError("generation_batch_size must be divisible by per_device_train_batch_size")
    # Reference 1-GPU recipe freezes generation batch to the effective optimizer
    # completion batch so G=2/G=4 experiments hold completion budget constant.
    if bool(training.get("enforce_equal_effective_generation_batch", True)):
        effective = per_device * grad_acc
        if generation_batch != effective:
            raise ValueError(
                f"v0.1 single-GPU GRPO requires generation_batch_size ({generation_batch}) "
                f"== effective completion batch ({effective})"
            )

    beta = float(training.get("beta", 0.04))
    ref_enabled = bool(training.get("reference_enabled", beta != 0.0))
    if (beta != 0.0) != ref_enabled:
        raise ValueError("reference_enabled must match beta!=0 for the frozen standard/lowmem profiles")

    accuracy_weight = float(reward.get("accuracy", 1.0))
    format_weight = float(reward.get("format", 0.1))
    reward_funcs = list(training.get("reward_funcs", ["onegpu_accuracy", "onegpu_boxed_format"]))
    reward_weights = list(training.get("reward_weights", [accuracy_weight, format_weight]))
    if len(reward_funcs) != len(reward_weights):
        raise ValueError("reward_funcs and reward_weights must have equal length")

    cmd = [
        "swift", "rlhf",
        "--rlhf_type", "grpo",
        "--model", str(model),
        "--dataset", str(dataset),
        "--output_dir", str(output_dir),
        "--tuner_type", str(training.get("tuner_type", "lora")),
        "--torch_dtype", str(training.get("torch_dtype", "float16")),
        "--per_device_train_batch_size", str(per_device),
        "--gradient_accumulation_steps", str(grad_acc),
        "--generation_batch_size", str(generation_batch),
        "--num_generations", str(num_generations),
        "--learning_rate", str(training.get("learning_rate", 1e-5)),
        "--lora_rank", str(training.get("lora_rank", 8)),
        "--lora_alpha", str(training.get("lora_alpha", 32)),
        "--target_modules", str(training.get("target_modules", "all-linear")),
        "--max_length", str(training.get("max_prompt_length_candidate", 512)),
        "--max_completion_length", str(training.get("max_completion_length_candidate", 512)),
        # The data layer already applies a length gate. Deleting an unexpected
        # over-limit prompt is safer than silently left-truncating it.
        "--truncation_strategy", str(training.get("truncation_strategy", "delete")),
        "--gradient_checkpointing", str(bool(training.get("gradient_checkpointing", True))).lower(),
        "--max_steps", str(training.get("max_steps", 20)),
        "--warmup_ratio", str(training.get("warmup_ratio", 0.05)),
        "--logging_steps", str(training.get("logging_steps", 1)),
        "--save_steps", str(training.get("save_steps", 10)),
        "--save_total_limit", str(training.get("save_total_limit", 2)),
        "--split_dataset_ratio", str(training.get("split_dataset_ratio", 0)),
        "--dataset_num_proc", str(training.get("dataset_num_proc", 1)),
        "--dataloader_num_workers", str(training.get("dataloader_num_workers", 0)),
        "--dataloader_persistent_workers", str(bool(training.get("dataloader_persistent_workers", False))).lower(),
        "--use_hf", str(bool(training.get("use_hf", True))).lower(),
        "--use_vllm", str(bool(training.get("use_vllm", False))).lower(),
        "--temperature", str(training.get("temperature", 1.0)),
        "--top_p", str(training.get("top_p", 1.0)),
        "--num_iterations", str(training.get("num_iterations", 1)),
        "--loss_type", str(training.get("loss_type", "grpo")),
        "--beta", str(beta),
        "--external_plugins", str(external_plugin),
        "--reward_funcs", *[str(x) for x in reward_funcs],
        "--reward_weights", *[str(x) for x in reward_weights],
        "--log_completions", str(bool(training.get("log_completions", True))).lower(),
        "--seed", str(training.get("seed", 42)),
    ]
    _append_validation_args(cmd, validation_dataset, training)
    if training.get("weight_decay") is not None:
        cmd += ["--weight_decay", str(training["weight_decay"])]
    if training.get("max_grad_norm") is not None:
        cmd += ["--max_grad_norm", str(training["max_grad_norm"])]
    if training.get("logging_dir") is not None:
        cmd += ["--logging_dir", str(training["logging_dir"])]
    return cmd


def _append_validation_args(cmd: list[str], validation_dataset: str | Path | None,
                            training: dict[str, Any]) -> None:
    """Add ms-swift native evaluation input without enabling its implicit random split."""
    if validation_dataset is None:
        if training.get("load_best_model_at_end"):
            raise ValueError("best checkpoint selection requires --val_dataset")
        return
    if training.get("split_dataset_ratio", 0) not in (0, 0.0, None):
        raise ValueError("不能同时使用 --val_dataset 与非零 --split_dataset_ratio")
    strategy = training.get("eval_strategy", "epoch")
    if strategy not in {"epoch", "steps"}:
        raise ValueError("eval_strategy 仅允许 epoch 或 steps")
    cmd.extend(["--val_dataset", str(validation_dataset), "--eval_strategy", str(strategy)])
    save_strategy = training.get("save_strategy", strategy if training.get("load_best_model_at_end") else "steps")
    if training.get("load_best_model_at_end") and save_strategy != strategy:
        raise ValueError("best checkpoint selection 要求 save_strategy 与 eval_strategy 一致")
    if training.get("load_best_model_at_end"):
        cmd.extend(["--save_strategy", str(save_strategy), "--load_best_model_at_end", "true",
                    "--metric_for_best_model", str(training.get("metric_for_best_model", "loss")),
                    "--greater_is_better", str(bool(training.get("greater_is_better", False))).lower()])
    if strategy == "steps":
        eval_steps = training.get("eval_steps", 10)
        if isinstance(eval_steps, bool) or not isinstance(eval_steps, (int, float)) or eval_steps <= 0:
            raise ValueError("eval_steps 必须是正数")
        cmd.extend(["--eval_steps", str(eval_steps)])
        if training.get("load_best_model_at_end"):
            save_steps = training.get("save_steps", 10)
            if isinstance(save_steps, bool) or not isinstance(save_steps, (int, float)) or save_steps <= 0 or save_steps % eval_steps:
                raise ValueError("best checkpoint selection 要求 save_steps 是 eval_steps 的正整数倍")


def discover_checkpoint_selection(output_dir: str | Path, policy: str = "final") -> dict[str, Any]:
    """Resolve latest/final/best checkpoint paths from ms-swift trainer evidence."""
    if policy not in {"final", "best"}:
        raise ValueError("checkpoint policy 仅允许 final 或 best")
    root = Path(output_dir).resolve(strict=True)
    latest = discover_latest_checkpoint(root).resolve(strict=True)
    selection: dict[str, Any] = {"policy": policy, "latest": str(latest), "final": str(latest)}
    if policy == "final":
        selection.update({"selected": str(latest), "selected_step": _checkpoint_step(latest),
                          "best": None, "metric": None, "metric_value": None})
        return selection
    state_files = [path for path in root.rglob("trainer_state.json")
                   if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)]
    if not state_files:
        raise FileNotFoundError("best policy 缺少 ms-swift trainer_state.json")
    state_file = max(state_files, key=lambda path: path.stat().st_mtime_ns)
    state = json.loads(state_file.read_text(encoding="utf-8"))
    best_value = state.get("best_model_checkpoint")
    if not isinstance(best_value, str) or not best_value.strip():
        raise ValueError("训练未记录 best_model_checkpoint")
    candidate = Path(best_value)
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve(strict=True)
    else:
        candidate = candidate.resolve(strict=True)
    if (not candidate.is_dir() or not candidate.is_relative_to(root)
            or _checkpoint_step(candidate) < 0):
        raise ValueError("训练记录的 best checkpoint 不存在或越界")
    if not any(candidate.iterdir()):
        raise ValueError("训练记录的 best checkpoint 为空")
    selection.update({"best": str(candidate), "selected": str(candidate),
                      "selected_step": _checkpoint_step(candidate),
                      "metric": state.get("metric_for_best_model"),
                      "metric_value": state.get("best_metric")})
    return selection

def build_merge_command(*, checkpoint: str | Path, output_dir: str | Path,
                        model: str | Path) -> list[str]:
    return [
        "swift", "export",
        "--model", str(model),
        "--adapters", str(checkpoint),
        "--merge_lora", "true",
        # Do not inherit quantization settings from a QLoRA checkpoint. The
        # adapter was trained against this exact parent, and the handoff model
        # must be a normal inference-ready merged checkpoint, not re-quantized.
        "--load_args", "false",
        "--output_dir", str(output_dir),
        "--exist_ok", "true",
    ]



def _sample_gpu_memory_mib() -> float | None:
    """Best-effort system-level GPU memory sample via nvidia-smi.

    This measures total memory used on GPU 0, not process-exclusive allocation.
    It is therefore diagnostic metadata, not a replacement for framework-native
    peak memory metrics.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            text=True, capture_output=True, timeout=2, check=False,
        )
        if proc.returncode != 0:
            return None
        first = proc.stdout.strip().splitlines()[0]
        return float(first.strip())
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def _failure_status(stderr_text: str, stdout_text: str = "") -> str:
    text = (stderr_text + "\n" + stdout_text).lower()
    oom_markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "outofmemoryerror",
        "cublas_status_alloc_failed",
        "failed to allocate memory",
    )
    return "OOM" if any(marker in text for marker in oom_markers) else "FAILED"


def run_monitored_command(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    log_dir: str | Path,
    dry_run: bool = False,
    allow_failure: bool = False,
    sample_interval_seconds: float = 0.5,
) -> dict[str, Any]:
    """Run a subprocess while sampling GPU memory with nvidia-smi.

    `allow_failure=True` is used by feasibility probes so an OOM becomes a
    recorded experimental outcome rather than an exception that destroys the
    artifact trail.
    """
    payload: dict[str, Any] = {
        "command": cmd,
        "command_string": command_string(cmd),
        "dry_run": dry_run,
        "gpu_memory_metric": "nvidia-smi total memory.used on GPU0",
    }
    if dry_run:
        payload["status"] = "PLANNED"
        return payload

    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    stdout_path = d / "stdout.log"
    stderr_path = d / "stderr.log"
    baseline = _sample_gpu_memory_mib()
    peak = baseline
    start = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            resolve_console_command(cmd), cwd=cwd, text=True, stdout=out, stderr=err, env=child_process_env()
        )
        while proc.poll() is None:
            sample = _sample_gpu_memory_mib()
            if sample is not None:
                peak = sample if peak is None else max(peak, sample)
            time.sleep(sample_interval_seconds)
        returncode = proc.wait()
    elapsed = time.perf_counter() - start
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    status = "PASS" if returncode == 0 else _failure_status(stderr_text, stdout_text)
    payload.update({
        "returncode": returncode,
        "status": status,
        "wall_seconds": elapsed,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "gpu_memory_baseline_mib": baseline,
        "peak_gpu_memory_mib_sampled": peak,
        "delta_gpu_memory_mib_sampled": (peak - baseline) if peak is not None and baseline is not None else None,
    })
    if returncode != 0 and not allow_failure:
        tail = (stderr_text or stdout_text)[-3000:]
        raise RuntimeError(f"Command failed ({returncode}): {tail}")
    return payload


def run_command(cmd: list[str], *, cwd: str | Path | None = None, log_dir: str | Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "command": cmd,
        "command_string": command_string(cmd),
        "dry_run": dry_run,
    }
    if dry_run:
        return payload
    start = time.perf_counter()
    proc = subprocess.run(
        resolve_console_command(cmd), cwd=cwd, text=True, capture_output=True, env=child_process_env(), check=False
    )
    elapsed = time.perf_counter() - start
    payload.update({"returncode": proc.returncode, "wall_seconds": elapsed})
    if log_dir is not None:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (d / "stderr.log").write_text(proc.stderr, encoding="utf-8")
        payload["stdout_log"] = str(d / "stdout.log")
        payload["stderr_log"] = str(d / "stderr.log")
    else:
        payload["stdout"] = proc.stdout
        payload["stderr"] = proc.stderr
    if proc.returncode != 0:
        tail = proc.stderr[-3000:] if proc.stderr else proc.stdout[-3000:]
        raise RuntimeError(f"Command failed ({proc.returncode}): {tail}")
    return payload


def _checkpoint_step(path: Path) -> int:
    m = re.search(r"checkpoint-(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def discover_latest_checkpoint(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    candidates = [p for p in root.rglob("checkpoint-*") if p.is_dir() and not p.name.endswith("-merged")]
    candidates = [p for p in candidates if _checkpoint_step(p) >= 0]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directory found under {root}")
    return max(candidates, key=lambda p: (_checkpoint_step(p), p.stat().st_mtime_ns))


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ms-swift": package_version("ms-swift"),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
        "datasets": package_version("datasets"),
        "bitsandbytes": package_version("bitsandbytes"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env.update({
                "gpu": torch.cuda.get_device_name(0),
                "vram_gib": round(props.total_memory / 1024**3, 3),
                "cuda_runtime": torch.version.cuda,
            })
    except (ImportError, OSError, RuntimeError) as exc:  # diagnostic only
        env["cuda_available"] = False
        env["torch_probe_error"] = repr(exc)
    return env


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
