from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROTOCOL = "grpo-completion-calibration-v1"
DEFAULT_CANDIDATES = (512, 768, 1024, 1536)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_grpo_calibration(config_path: str | Path, *, output_root: str | Path | None = None) -> dict[str, Any]:
    """Expand one frozen GRPO recipe into controlled completion-length candidates."""
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    recipe_path = Path(config["base_recipe"])
    base_recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8")) or {}
    model = Path(config["model"])
    dataset = Path(config["dataset"])
    lengths = tuple(int(value) for value in config.get("candidate_lengths", DEFAULT_CANDIDATES))
    if lengths != DEFAULT_CANDIDATES:
        raise ValueError(f"candidate_lengths must be exactly {DEFAULT_CANDIDATES}, got {lengths}")
    if not dataset.is_file():
        raise FileNotFoundError(f"GRPO prompt dataset not found: {dataset}")
    if not base_recipe.get("experiment_id") and not config.get("experiment_id"):
        raise ValueError("experiment_id is required for a shared calibration lineage")

    experiment_id = str(config.get("experiment_id", base_recipe.get("experiment_id")))
    training = deepcopy(base_recipe.get("training", {}))
    seed = int(training.get("seed", base_recipe.get("seed", 42)))
    profile = str(base_recipe.get("profile", "standard"))
    prompt_ids: list[str] = []
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("problem_id") is not None:
                prompt_ids.append(str(row["problem_id"]))
    if not prompt_ids:
        raise ValueError("dataset must expose prompt problem_id values for controlled calibration")

    root = Path(output_root or config.get("output_root", "runs/dev7/grpo-completion-calibration"))
    fixed_training = deepcopy(training)
    fixed_training.pop("max_completion_length_candidate", None)
    candidates: list[dict[str, Any]] = []
    for length in lengths:
        label = f"grpo-completion-{length}-s{seed}"
        recipe = deepcopy(base_recipe)
        recipe["experiment_id"] = experiment_id
        recipe["run_label"] = label
        recipe.setdefault("training", {})["max_completion_length_candidate"] = length
        recipe_file = root / "recipes" / f"completion-{length}.yaml"
        recipe_file.parent.mkdir(parents=True, exist_ok=True)
        recipe_file.write_text(yaml.safe_dump(recipe, sort_keys=False, allow_unicode=True), encoding="utf-8")
        recipe_sha256 = _digest_file(recipe_file)
        run_dir = root / "candidates" / f"completion-{length}"
        command = [
            "onegpu-posttrain", "run-grpo", "--config", str(recipe_file),
            "--model", str(model), "--dataset", str(dataset), "--output", str(run_dir), "--no-merge",
        ]
        candidates.append({
            "completion_length": length,
            "run_label": label,
            "run_id": None,
            "recipe": str(recipe_file),
            "recipe_sha256": recipe_sha256,
            "output_dir": str(run_dir),
            "command": command,
            "status": "planned",
        })

    manifest = {
        "protocol": PROTOCOL,
        "experiment_id": experiment_id,
        "profile": profile,
        "parent_model": str(model),
        "dataset": str(dataset),
        "dataset_sha256": _digest_file(dataset),
        "base_recipe_sha256": _digest_file(recipe_path),
        "prompt_ids": prompt_ids,
        "prompt_ids_sha256": hashlib.sha256("\n".join(prompt_ids).encode("utf-8")).hexdigest(),
        "seed": seed,
        "fixed_training": fixed_training,
        "reward": deepcopy(base_recipe.get("reward", {})),
        "budget": {
            "type": training.get("budget_type", "completions"),
            "max_steps": training.get("max_steps"),
            "num_generations": training.get("num_generations"),
            "generation_batch_size_completions": training.get("generation_batch_size_completions"),
            "planned_training_completion_instances": (
                int(training.get("per_device_train_batch_size_completions", 1))
                * int(training.get("gradient_accumulation_steps", 1))
                * int(training.get("max_steps", 0))
            ),
        },
        "candidate_lengths": list(lengths),
        "candidates": candidates,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def summarize_grpo_calibration(manifest_path: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for candidate in manifest["candidates"]:
        run_dir = Path(candidate["output_dir"])
        run = _read_json(run_dir / "run.json")
        metrics = _read_json(run_dir / "grpo_metrics.json")
        train = (run or {}).get("train") or {}
        mean_metrics = (metrics or {}).get("mean") or {}
        components = (metrics or {}).get("reward_components") or {}

        def measured(value: Any) -> dict[str, Any]:
            return {"status": "available" if value is not None else "unavailable", "value": value}

        def component_mean(name: str, components: dict[str, Any] = components) -> Any:
            entry = components.get(name) or {}
            return entry.get("mean_mean")

        status = str(train.get("status", "missing")).lower()
        rows.append({
            "completion_length": candidate["completion_length"],
            "run_label": candidate["run_label"],
            "run_id": (run or {}).get("run_id"),
            "experiment_id": (run or {}).get("experiment_id"),
            "status": status,
            "metrics": {
                "actual_generated_length": measured(mean_metrics.get("completions/mean_length")),
                "completion_clipped_ratio": measured(mean_metrics.get("completions/clipped_ratio")),
                "reward": measured(mean_metrics.get("reward")),
                "reward_std": measured(mean_metrics.get("reward_std")),
                "accuracy_reward": measured(component_mean("OneGPUAccuracyReward")),
                "format_reward": measured(component_mean("OneGPUBoxedFormatReward")),
                "wall_seconds": measured(train.get("wall_seconds")),
                "peak_gpu_memory_mib_sampled": measured(train.get("peak_gpu_memory_mib_sampled")),
            },
        })

    supported = [
        row for row in rows
        if row["status"] == "pass"
        and row["metrics"]["completion_clipped_ratio"]["status"] == "available"
    ]
    best = min(supported, key=lambda row: row["metrics"]["completion_clipped_ratio"]["value"]) if supported else None
    best_clip = best["metrics"]["completion_clipped_ratio"]["value"] if best else None
    executed = [row for row in rows if row["run_id"] is not None]
    run_ids = [row["run_id"] for row in executed]
    control_validation = {
        "executed_runs_share_experiment_id": all(
            row["experiment_id"] == manifest["experiment_id"] for row in executed
        ),
        "executed_run_ids_unique": len(run_ids) == len(set(run_ids)),
        "executed_runs": len(executed),
    }
    result = {
        "protocol": PROTOCOL,
        "experiment_id": manifest["experiment_id"],
        "control_validation": control_validation,
        "controls": {key: manifest[key] for key in (
            "profile", "parent_model", "dataset", "dataset_sha256", "prompt_ids_sha256", "seed", "fixed_training", "reward", "budget"
        )},
        "target_clipping_ratio": 0.30,
        "candidates": rows,
        "best_supported_completion_length": best["completion_length"] if best else None,
        "best_supported_clipping_ratio": best_clip,
        "target_met": best_clip < 0.30 if best_clip is not None else None,
    }
    out = Path(output_dir or manifest_file.parent)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grpo_calibration_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# GRPO Completion-Length Calibration", "",
        f"Experiment: `{manifest['experiment_id']}`; profile: `{manifest['profile']}`; clipping target: `< 30%`.",
        "", "| Length | Status | Mean generated tokens | Clipped ratio | Reward | Accuracy reward | Wall seconds | Peak VRAM MiB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        m = row["metrics"]
        values = [m[key]["value"] for key in (
            "actual_generated_length", "completion_clipped_ratio", "reward", "accuracy_reward", "wall_seconds", "peak_gpu_memory_mib_sampled"
        )]
        formatted = ["N/A" if value is None else f"{value:.4g}" for value in values]
        lines.append(f"| {row['completion_length']} | {row['status']} | " + " | ".join(formatted) + " |")
    lines.extend(["", f"Best supported length: `{result['best_supported_completion_length']}`; target met: `{result['target_met']}`.", ""])
    (out / "grpo_calibration_report.md").write_text("\n".join(lines), encoding="utf-8")
    return result
