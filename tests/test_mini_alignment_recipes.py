from __future__ import annotations

import json
import argparse
from pathlib import Path

import yaml

from onegpu_posttrain.adapters.ms_swift import adapt_file
from onegpu_posttrain.cli import _dataset_lineage_ref, cmd_run_dpo, cmd_run_grpo


ROOT = Path(__file__).resolve().parents[1]
RECIPES = ROOT / "recipes" / "mini"
BUILD = ROOT / "data/processed/mini/2026-10-17-build-02/openr1"
M2 = ROOT / "runs/mini/sft-deepmath-1024-subset-full-v1/merged/M2-SFT"


def _load(name: str) -> dict:
    return yaml.safe_load((RECIPES / name).read_text(encoding="utf-8"))


def test_mini_alignment_recipes_bind_existing_data_and_shared_parent() -> None:
    names = [
        "30-dpo-exploratory-4pairs.yaml",
        "31-simpo-exploratory-4pairs.yaml",
        "40-grpo-standard-exploratory.yaml",
        "41-grpo-lowmem-exploratory.yaml",
    ]
    recipes = [_load(name) for name in names]
    assert M2.is_dir()
    assert {item["experiment_id"] for item in recipes} == {"mini-alignment-existing-data-v1"}
    assert {item["parent"] for item in recipes} == {"runs/mini/sft-deepmath-1024-subset-full-v1/merged/M2-SFT"}
    assert {item["usage"] for item in recipes} == {"exploratory"}
    for item in recipes:
        manifest_path = ROOT / item["dataset"]["manifest"]
        canonical_path = BUILD / "canonical" / (
            "preference.jsonl" if item["stage"] in {"dpo", "simpo"} else "grpo.jsonl"
        )
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        assert canonical_path.is_file()
        assert manifest["dataset_id"]
        assert manifest["sample_count"] == item["dataset"]["actual_records"]
        assert item["dataset"]["quality_status"] == "NOT_EVALUATED"
        assert item["dataset"]["quality_report"]
    assert [item["dataset"]["actual_records"] for item in recipes] == [4, 4, 2000, 2000]


def test_mini_simpo_recipe_is_original_simpo_and_smoke_recipe_is_unchanged() -> None:
    simpo = _load("31-simpo-exploratory-4pairs.yaml")
    assert simpo["training"]["cpo_alpha"] == 0.0
    smoke = yaml.safe_load((ROOT / "recipes/smoke/31-simpo-original.yaml").read_text(encoding="utf-8"))
    assert smoke["training"]["cpo_alpha"] == 0.0
    assert smoke["training"]["max_steps"] == 20


def test_canonical_adapter_retains_manifest_lineage_and_excludes_smoke_fixture(tmp_path: Path) -> None:
    canonical = BUILD / "canonical" / "preference.jsonl"
    native = tmp_path / "preference.jsonl"
    report = adapt_file("preference", canonical, native)
    ref, complete, warnings = _dataset_lineage_ref(native)
    manifest = yaml.safe_load((BUILD / "manifests/preference-smoke-v1.yaml").read_text(encoding="utf-8"))
    assert report["records"] == 4
    assert complete is True
    assert warnings == []
    assert ref.id == manifest["dataset_id"]
    assert json.loads(native.with_suffix(".jsonl.adapter.json").read_text(encoding="utf-8"))["input_sha256"] == manifest["hashes"]["canonical_records_sha256"]
    smoke_fixture = ROOT / "artifacts/native/ms-swift/smoke/preference.jsonl"
    if smoke_fixture.is_file():
        assert smoke_fixture.resolve() != native.resolve()


def test_grpo_profiles_are_bounded_and_standard_vs_lowmem_is_explicit() -> None:
    standard = _load("40-grpo-standard-exploratory.yaml")
    lowmem = _load("41-grpo-lowmem-exploratory.yaml")
    assert standard["training"]["max_steps"] == lowmem["training"]["max_steps"] == 3
    assert standard["training"]["reference_enabled"] is True
    assert standard["training"]["beta"] > 0
    assert lowmem["training"]["reference_enabled"] is False
    assert lowmem["training"]["beta"] == 0


def test_dry_runs_disclose_exploratory_usage_quality_and_actual_count(tmp_path: Path) -> None:
    preference_native = tmp_path / "preference.jsonl"
    grpo_native = tmp_path / "grpo.jsonl"
    adapt_file("preference", BUILD / "canonical/preference.jsonl", preference_native)
    adapt_file("grpo", BUILD / "canonical/grpo.jsonl", grpo_native)
    dpo_out = tmp_path / "dpo"
    grpo_out = tmp_path / "grpo"
    common = {"model": str(M2), "merge": False, "dry_run": True}
    cmd_run_dpo(argparse.Namespace(
        **common, config=str(RECIPES / "30-dpo-exploratory-4pairs.yaml"),
        dataset=str(preference_native), output=str(dpo_out),
    ))
    cmd_run_grpo(argparse.Namespace(
        **common, config=str(RECIPES / "40-grpo-standard-exploratory.yaml"),
        dataset=str(grpo_native), output=str(grpo_out), external_plugin="plugins/grpo_rewards.py",
    ))
    for output, count, unit in [(dpo_out, 4, "preference_pairs"), (grpo_out, 2000, "prompts")]:
        run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        assert run["exploration_context"] == {
            "usage": "exploratory",
            "actual_record_count": count,
            "count_unit": unit,
            "quality_report": run["exploration_context"]["quality_report"],
            "quality_status": "NOT_EVALUATED",
            "official_mini_acceptance": "not_assessed",
        }
