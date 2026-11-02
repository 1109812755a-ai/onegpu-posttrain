from __future__ import annotations

import json
from pathlib import Path

import yaml

from onegpu_posttrain.reporting.grpo_calibration import (
    prepare_grpo_calibration,
    summarize_grpo_calibration,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_prepare_calibration_changes_only_completion_length_and_shares_controls(tmp_path):
    recipe = {
        "run_label": "base",
        "profile": "lowmem",
        "seed": 42,
        "training": {
            "max_prompt_length_candidate": 512,
            "max_completion_length_candidate": 512,
            "max_steps": 20,
            "num_generations": 2,
            "gradient_accumulation_steps": 4,
            "per_device_train_batch_size_completions": 1,
            "beta": 0.0,
        },
        "reward": {"accuracy": 1.0, "format": 0.1},
    }
    base_recipe = tmp_path / "base.yaml"
    base_recipe.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    dataset = tmp_path / "prompts.jsonl"
    dataset.write_text('\n'.join(
        json.dumps({"problem_id": f"p{i}", "messages": []}) for i in range(3)
    ) + "\n", encoding="utf-8")
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump({
        "experiment_id": "exp-grpo-calibration-test",
        "base_recipe": str(base_recipe),
        "model": str(tmp_path / "M2-SFT"),
        "dataset": str(dataset),
        "output_root": str(tmp_path / "calibration"),
    }), encoding="utf-8")

    manifest = prepare_grpo_calibration(config)

    assert manifest["experiment_id"] == "exp-grpo-calibration-test"
    assert manifest["prompt_ids"] == ["p0", "p1", "p2"]
    assert manifest["budget"]["planned_training_completion_instances"] == 80
    recipes = [yaml.safe_load(Path(item["recipe"]).read_text(encoding="utf-8")) for item in manifest["candidates"]]
    assert [r["training"]["max_completion_length_candidate"] for r in recipes] == [512, 768, 1024, 1536]
    assert {r["experiment_id"] for r in recipes} == {manifest["experiment_id"]}
    assert len({r["run_label"] for r in recipes}) == 4
    for candidate_recipe in recipes:
        fixed = dict(candidate_recipe["training"])
        fixed.pop("max_completion_length_candidate")
        assert fixed == manifest["fixed_training"]
        assert candidate_recipe["profile"] == "lowmem"


def test_summary_preserves_failure_and_marks_unmeasured_metrics_unavailable(tmp_path):
    manifest_path = tmp_path / "calibration_manifest.json"
    candidates = []
    for length in (512, 768, 1024, 1536):
        run_dir = tmp_path / f"candidate-{length}"
        candidates.append({
            "completion_length": length,
            "run_label": f"run-{length}",
            "output_dir": str(run_dir),
        })
    manifest = {
        "protocol": "grpo-completion-calibration-v1",
        "experiment_id": "exp-test",
        "profile": "lowmem",
        "parent_model": "M2-SFT",
        "dataset": "prompts.jsonl",
        "dataset_sha256": "sha-ds",
        "prompt_ids_sha256": "sha-prompts",
        "seed": 42,
        "fixed_training": {"max_steps": 20},
        "reward": {"accuracy": 1.0},
        "budget": {"type": "completions", "max_steps": 20},
        "candidates": candidates,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _write_json(tmp_path / "candidate-512" / "run.json", {
        "run_id": "run-512", "experiment_id": "exp-test", "train": {
            "status": "PASS", "wall_seconds": 120.0, "peak_gpu_memory_mib_sampled": 4000,
        },
    })
    _write_json(tmp_path / "candidate-512" / "grpo_metrics.json", {
        "mean": {"completions/mean_length": 500, "completions/clipped_ratio": 0.4, "reward": 0.2},
        "reward_components": {"OneGPUAccuracyReward": {"mean_mean": 0.1}},
    })
    _write_json(tmp_path / "candidate-768" / "run.json", {
        "run_id": "run-768", "experiment_id": "exp-test", "train": {"status": "OOM"},
    })

    report = summarize_grpo_calibration(manifest_path)

    assert [row["status"] for row in report["candidates"]] == ["pass", "oom", "missing", "missing"]
    assert report["candidates"][0]["metrics"]["completion_clipped_ratio"]["value"] == 0.4
    assert report["candidates"][1]["metrics"]["reward"]["status"] == "unavailable"
    assert report["candidates"][2]["metrics"]["actual_generated_length"]["value"] is None
    assert report["best_supported_completion_length"] == 512
    assert report["target_met"] is False
    assert report["control_validation"]["executed_run_ids_unique"]
    assert (tmp_path / "grpo_calibration_report.md").is_file()
