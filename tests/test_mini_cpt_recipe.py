from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from onegpu_posttrain.data.mini_cpt_preflight import (
    MiniCPTPreflightError,
    preflight_mini_cpt,
)

ROOT = Path(__file__).resolve().parents[1]
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


def _base_model(root: Path) -> Path:
    model = root / "models--Qwen--Qwen3-0.6B-Base" / "snapshots" / REVISION
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture-weights")
    return model


def test_mini_cpt_recipe_resolves_pinned_finemath_budget_and_base(tmp_path):
    recipe = yaml.safe_load((ROOT / "recipes/mini/10-peft-cpt.yaml").read_text(encoding="utf-8"))
    model = yaml.safe_load((ROOT / recipe["model_config"]).read_text(encoding="utf-8"))
    manifest = yaml.safe_load((ROOT / recipe["dataset"]["manifest"]).read_text(encoding="utf-8"))
    review = json.loads((ROOT / "data/processed/mini/2026-10-17-build-02/reports/mini-quality-reviews/remediation-v1/reviews/finemath-cpt_train.json").read_text(encoding="utf-8"))

    assert recipe["dataset"]["dataset_id"] == manifest["dataset_id"]
    assert manifest["hashes"]["canonical_records_sha256"] == "f8849745f6f15901b5294f223bff1b90ef7c48ef73a20c47fb2b726b319e106d"
    assert recipe["dataset"]["expected_budget_count"] == 5_000_368
    assert recipe["dataset"]["expected_budget_unit"] == "tokens"
    assert recipe["dataset"]["usage"] == "exploratory"
    assert review["outcome"] == "pending"
    assert review["usage"] == "legacy_unknown"
    assert model["model"]["repo"] == "Qwen/Qwen3-0.6B-Base"
    assert model["model"]["revision"] == "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    assert recipe["parent_model_id"] == f"{model['model']['repo']}@{model['model']['revision']}"
    native_path = ROOT / recipe["dataset"]["native"]
    adapter = json.loads(native_path.with_suffix(".jsonl.adapter.json").read_text(encoding="utf-8"))
    assert adapter["input_sha256"] == manifest["hashes"]["canonical_records_sha256"]
    assert adapter["records"] == manifest["sample_count"]
    assert recipe["seed"] == 42
    assert recipe["training"]["num_train_epochs"] == 1
    assert recipe["training"]["tuner_type"] == "lora"
    assert recipe["training"]["per_device_train_batch_size"] == 1
    assert recipe["training"]["gradient_checkpointing"] is True
    model_snapshot = _base_model(tmp_path)
    preflight = preflight_mini_cpt(recipe, recipe["dataset"]["native"], root=ROOT, model=model_snapshot)
    assert preflight["status"] == "pass"
    assert preflight["usage"] == "exploratory"
    assert preflight["budget_count"] == 5_000_368
    assert preflight["downloads_performed"] is False
    assert preflight["base_model"]["directory_sha256"]


def test_mini_cpt_preflight_fails_closed_for_missing_native_input(tmp_path):
    recipe = yaml.safe_load((ROOT / "recipes/mini/10-peft-cpt.yaml").read_text(encoding="utf-8"))
    model_snapshot = _base_model(tmp_path)
    with pytest.raises(MiniCPTPreflightError, match="不会联网下载"):
        preflight_mini_cpt(recipe, recipe["dataset"]["native"], root=tmp_path, model=model_snapshot)


def test_mini_cpt_preflight_rejects_unpinned_dataset_path(tmp_path):
    recipe = yaml.safe_load((ROOT / "recipes/mini/10-peft-cpt.yaml").read_text(encoding="utf-8"))
    model_snapshot = _base_model(tmp_path)
    with pytest.raises(MiniCPTPreflightError, match="配方不一致"):
        preflight_mini_cpt(recipe, "data/elsewhere.jsonl", root=ROOT, model=model_snapshot)


def test_mini_cpt_preflight_rejects_incomplete_base_snapshot(tmp_path):
    recipe = yaml.safe_load((ROOT / "recipes/mini/10-peft-cpt.yaml").read_text(encoding="utf-8"))
    model_snapshot = tmp_path / "models--Qwen--Qwen3-0.6B-Base" / "snapshots" / REVISION
    model_snapshot.mkdir(parents=True)
    (model_snapshot / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    with pytest.raises(MiniCPTPreflightError, match="缺少模型权重"):
        preflight_mini_cpt(recipe, recipe["dataset"]["native"], root=ROOT, model=model_snapshot)


def test_smoke_cpt_recipe_remains_fixed_step_1024():
    recipe = yaml.safe_load((ROOT / "recipes/smoke/10-peft-cpt.yaml").read_text(encoding="utf-8"))
    assert recipe["training"]["max_length"] == 1024
    assert recipe["training"]["max_steps"] == 20


def test_mini_sft_recipe_is_separate_2048_deepmath_one_epoch():
    recipe = yaml.safe_load((ROOT / "recipes/mini/20-sft.yaml").read_text(encoding="utf-8"))
    dataset = yaml.safe_load((ROOT / recipe["dataset"]["manifest"]).read_text(encoding="utf-8"))
    assert recipe["dataset"]["dataset_id"] == dataset["dataset_id"]
    assert dataset["sample_count"] == recipe["dataset"]["expected_count"] == 5000
    assert recipe["dataset"]["dataset_key"] == "deepmath.sft_train"
    assert recipe["parent_model_path"].endswith("merged/M1-CPT")
    assert recipe["training"]["max_length"] == 2048
    assert recipe["training"]["num_train_epochs"] == 1
    assert recipe["training"]["per_device_train_batch_size"] == 1
    assert recipe["training"]["gradient_checkpointing"] is True
    smoke = yaml.safe_load((ROOT / "recipes/smoke/20-sft.yaml").read_text(encoding="utf-8"))
    assert smoke["training"]["max_length"] == 1024
