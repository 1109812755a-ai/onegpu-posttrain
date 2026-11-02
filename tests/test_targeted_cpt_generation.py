from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from onegpu_posttrain.data.decontaminate import BenchmarkIndex
from onegpu_posttrain.data.manifest import load_manifest
from onegpu_posttrain.data.targeted_cpt import generate_targeted_cpt_dataset
from onegpu_posttrain.data.cpt_training import prepare_registered_cpt_training, preflight_registered_cpt
from onegpu_posttrain.data.targeted_stage_plan import TargetedStagePlan
from onegpu_posttrain import cli
from onegpu_posttrain.lineage import sha256_file
from onegpu_posttrain.synthetic.providers import FakeProvider, ProviderProfile


SHA = "a" * 64


def plan(*, expected=8, requests=3):
    return TargetedStagePlan.model_validate({
        "target": {"stage": "cpt", "domain": "mathematics", "subtopics": ["algebra"]},
        "training_seed": {"dataset_id": "train-seed", "content_sha256": SHA, "role": "train_seed"},
        "starting_model": {"model_id": "base", "content_sha256": SHA},
        "diagnostic_evaluation": {"dataset_id": "diagnostic", "dataset_sha256": "d" * 64,
                                  "protocol_version": "eval-v1", "config_sha256": "e" * 64,
                                  "role": "diagnostic_dev"},
        "final_evaluation": {"dataset_id": "holdout", "dataset_sha256": "b" * 64,
                             "protocol_version": "eval-v1", "config_sha256": "c" * 64,
                             "role": "final_holdout"},
        "provider_profile_id": "fake-cpt",
        "budget": {"unit": "tokens", "expected_usable_count": expected,
                   "max_candidates": requests, "max_requests": requests,
                   "max_output_tokens": 128},
        "training_budget": 1, "seed": 7,
    })


def response(text, subtopic="algebra"):
    return json.dumps({"text": text, "subtopic": subtopic})


def call(provider, root, *, expected=8, requests=3, benchmark=()):
    p = plan(expected=expected, requests=requests)
    return generate_targeted_cpt_dataset(
        p, provider=provider,
        profile=ProviderProfile(profile_id="fake-cpt", adapter="fake", model="fixture"),
        benchmark_index=BenchmarkIndex(benchmark), benchmark_index_id="holdout-index-v1",
        benchmark_index_sha256=hashlib.sha256(b"fixture-index").hexdigest(),
        tokenizer_id="whitespace-test-v1", count_tokens=lambda text: len(text.split()),
        output_root=root / "artifacts" / "datasets", workspace=root,
    )


def test_generated_cpt_partial_artifact_is_registered_with_content_bound_report(tmp_path):
    provider = FakeProvider([response("A clear explanation of polynomial equations and their roots.")])
    result = call(provider, tmp_path, expected=40, requests=1)
    assert result["status"] == "partial"
    assert result["registered"] is True
    assert result["record_count"] == 1
    assert result["factuality"] == "not_verified"
    assert len(result["canonical_sha256"]) == 64
    assert result["quality_report_id"]
    artifact = tmp_path / "artifacts" / "datasets" / result["dataset_id"]
    canonical = artifact / "canonical" / "cpt.jsonl"
    generated = json.loads(canonical.read_text(encoding="utf-8"))
    assert generated["metadata"]["data_role"] == "train_seed"
    report = json.loads((artifact / "targeted_generation_report.json").read_text(encoding="utf-8"))
    manifest = load_manifest(artifact / "manifests" / "cpt-model-generated-v1.yaml")
    assert report["canonical_sha256"] == sha256_file(canonical) == result["canonical_sha256"]
    assert report["dataset_id"] == result["dataset_id"]
    assert manifest.dataset_id == report["dataset_id"]
    assert manifest.hashes["canonical_records_sha256"] == report["canonical_sha256"]
    assert manifest.metadata["targeted_plan_id"] == report["plan_id"]
    assert manifest.quality_report.report_id == report["quality_report_id"]


def test_generation_deduplicates_and_rejects_contaminated_content(tmp_path):
    repeated = "This passage explains how fractions represent equal parts of a whole."
    provider = FakeProvider([response(repeated), response(repeated), response("A new detailed discussion of geometric transformations and symmetry.")])
    result = call(provider, tmp_path, expected=100, requests=3,
                  benchmark=[("bench-1", "This passage explains how fractions represent equal parts of a whole.")])
    # First is rejected by contamination; the repeated second candidate is too,
    # and only the distinct final candidate may enter the dataset.
    assert result["record_count"] == 1
    assert result["rejections"]["benchmark_contamination"] == 2
    rows = (tmp_path / "artifacts" / "datasets" / result["dataset_id"] / "canonical" / "cpt.jsonl").read_text().splitlines()
    assert len(rows) == 1


def test_generation_with_no_usable_candidate_does_not_register_empty_dataset(tmp_path):
    provider = FakeProvider(["not json"])
    result = call(provider, tmp_path, requests=1)
    assert result["status"] == "failed"
    assert result["registered"] is False
    assert not (tmp_path / "artifacts" / "datasets").exists()


def test_registered_cpt_preflight_and_native_adapter_are_offline_and_lineage_bound(tmp_path):
    provider = FakeProvider([response("A clear explanation of polynomial equations and their roots.")])
    generated = call(provider, tmp_path, expected=4, requests=1)
    model = tmp_path / "models" / "base"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture-only-not-a-real-model")
    manifest = Path(generated["artifact_path"]) / "manifests" / "cpt-model-generated-v1.yaml"
    preflight = preflight_registered_cpt(manifest, model=model)
    assert preflight["status"] == "pass"
    assert preflight["downloads_performed"] is False
    prepared = prepare_registered_cpt_training(
        manifest, model=model, revision=None, adapter_output=tmp_path / "adapter-output",
    )
    assert Path(prepared["dataset"]).is_file()
    sidecar = json.loads(Path(prepared["dataset"]).with_suffix(".jsonl.adapter.json").read_text())
    assert sidecar["input_sha256"] == preflight["dataset_sha256"]


@pytest.mark.parametrize(("train_status", "expected_status", "expected_exit"), [
    ("PASS", "completed", 0), ("FAILED", "failed", 4), ("OOM", "oom", 4),
])
def test_registered_cpt_cli_records_truthful_lineage_for_terminal_states(
    tmp_path, monkeypatch, capsys, train_status, expected_status, expected_exit,
):
    provider = FakeProvider([response("A clear explanation of polynomial equations and their roots.")])
    generated = call(provider, tmp_path, expected=4, requests=1)
    model = tmp_path / "models" / "base"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture-only-not-a-real-model")
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model:\n  repo: fixture/base\n  revision: fixture-revision\n", encoding="utf-8")
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(json.dumps({"run_label": "targeted-cpt-test", "model_config": str(model_config),
                                  "training": {"max_steps": 1}}), encoding="utf-8")

    def fake_monitored(command, *, log_dir, dry_run, allow_failure):
        output_dir = Path(command[command.index("--output_dir") + 1])
        if train_status == "PASS":
            (output_dir / "checkpoint-1").mkdir(parents=True)
        return {"status": train_status, "log_dir": str(log_dir), "dry_run": dry_run,
                "allow_failure": allow_failure, "exit_code": 0 if train_status == "PASS" else 1}

    monkeypatch.setattr(cli, "run_monitored_command", fake_monitored)
    rc = cli.cmd_run_registered_cpt(type("Args", (), {
        "workspace": str(tmp_path),
        "manifest": str(Path(generated["artifact_path"]) / "manifests" / "cpt-model-generated-v1.yaml"),
        "config": str(recipe), "model": str(model), "revision": None,
        "output": "runs/cpt-test", "run_id": "run-ui-cpt-test", "merge": False, "dry_run": False,
    })())
    assert rc == expected_exit
    capsys.readouterr()
    run_dir = tmp_path / "runs/cpt-test"
    lineage = json.loads((run_dir / "lineage.json").read_text(encoding="utf-8"))
    assert any(item["id"] == generated["dataset_id"] for item in lineage["inputs"])
    assert lineage["status"] == expected_status
    assert lineage["run_id"] == "run-ui-cpt-test"
    if train_status != "PASS":
        assert lineage["outputs"] == []
    assert (run_dir / "model_artifacts").exists() is (train_status == "PASS")
    if train_status == "PASS":
        assert lineage["outputs"][0]["kind"] == "model"
        assert json.loads((run_dir / "model_artifacts" / f"{lineage['outputs'][0]['id']}.json").read_text())["stage"] == "cpt"
