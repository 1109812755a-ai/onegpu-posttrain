from __future__ import annotations

import json
import io
import time
from pathlib import Path

import yaml
from fastapi.testclient import TestClient
from test_sft_generation_workflow import _responses
from test_sft_workflow_bundle import _inputs
from test_targeted_sft_bridge import make_shared_plan

from onegpu_posttrain.lineage import sha256_directory
from onegpu_posttrain.synthetic.providers import FakeProvider
from onegpu_posttrain.synthetic.sft_workflow import execute_sft_generation
from onegpu_posttrain.synthetic.workflow_bundle import register_sft_workflow_bundle
from onegpu_posttrain.data.targeted_stage_plan import register_targeted_stage_plan
from onegpu_posttrain.ui.server import create_app


def test_offline_generation_review_registration_launcher_preflight_lineage(tmp_path, monkeypatch):
    workspace, inputs = _inputs(tmp_path)
    bundle = register_sft_workflow_bundle(workspace, inputs=inputs)
    shared_stage_plan = make_shared_plan(inputs)
    shared_plan = register_targeted_stage_plan(workspace, shared_stage_plan)
    provider = FakeProvider(_responses(inputs))

    import onegpu_posttrain.cli as cli_module
    import onegpu_posttrain.ui.launcher as launcher_module
    import onegpu_posttrain.ui.sft_generation_jobs as job_module

    monkeypatch.setattr(job_module, "execute_sft_generation", lambda root, bundle_id, **kwargs:
                        execute_sft_generation(root, bundle_id, provider=provider, **kwargs))
    monkeypatch.setattr(launcher_module, "RECIPE", "recipes/mini/test-sft.yaml")
    monkeypatch.setattr(launcher_module, "MODEL_ID", "model-offline-test")
    monkeypatch.setattr(launcher_module, "DATASET_ID", "ds-fixed-test")
    monkeypatch.setattr(launcher_module, "_sft_runtime_available", lambda: True)

    model_dir = workspace / "models" / "test-m1"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    model_digest = sha256_directory(model_dir)
    model_registry = workspace / "runs/mini/cpt-finemath-5m-s42-v3/model_artifacts/model-offline-test.json"
    model_registry.parent.mkdir(parents=True)
    model_registry.write_text(json.dumps({
        "model_id": "model-offline-test", "kind": "merged",
        "digest": {"algorithm": "sha256", "value": model_digest, "status": "available"},
    }), encoding="utf-8")
    recipe_path = workspace / "recipes/mini/test-sft.yaml"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(yaml.safe_dump({
        "stage": "sft", "parent_model_id": "model-offline-test",
        "parent_model_path": "models/test-m1",
        "dataset": {"dataset_id": "ds-fixed-test"},
        "training": {"max_length": 1024}, "seed": 42,
    }), encoding="utf-8")

    def fake_plan(args):
        (Path(args.output) / "train_plan.json").write_text(json.dumps({"stage": "sft"}), encoding="utf-8")
        return 0

    monkeypatch.setattr(cli_module, "cmd_plan_sft", fake_plan)

    with TestClient(create_app(workspace, enable_synthetic_generation=True,
                               enable_experiment_launcher=True)) as client:
        generation_caps = client.get("/api/synthetic/sft/capabilities").json()
        generation_token = generation_caps["token"]
        generation_headers = {"X-Local-Synthetic-Token": generation_token}
        payload = {"bundle_id": bundle.bundle_id, "usable_target": 1,
                   "max_requests": 1, "target_candidates": 1,
                   "stage_plan_id": shared_plan["plan_id"], "plan_version": shared_plan["plan_version"]}
        assert any(item["plan_id"] == shared_plan["plan_id"]
                   for row in generation_caps["bundles"] for item in row.get("shared_stage_plans", []))
        preflight = client.post("/api/synthetic/sft/preflight", json=payload, headers=generation_headers)
        assert preflight.status_code == 200
        assert preflight.json()["provider_calls"] == 0
        assert preflight.json()["targeted_stage_plan"]["plan_id"] == shared_plan["plan_id"]
        assert provider.calls == 0
        stage_preflight = client.post("/api/synthetic/stage/preflight",
                                      json=payload | {"stage": "sft", "mode": "targeted"},
                                      headers=generation_headers)
        assert stage_preflight.status_code == 200
        assert stage_preflight.json()["bundle_checks"]["targeted_stage_plan"]["plan_id"] == shared_plan["plan_id"]
        # The shared real-provider gate does not pretend the fake fixture is a real provider.
        assert stage_preflight.json()["status"] == "blocked"
        assert provider.calls == 0
        started = client.post("/api/synthetic/sft/start", json=payload | {"confirm_start": True},
                              headers=generation_headers)
        assert started.status_code == 202
        execution_id = started.json()["execution_id"]
        for _ in range(100):
            state = client.get(f"/api/synthetic/sft/runs/{execution_id}", headers=generation_headers).json()
            if state["status"] in {"completed", "partial", "failed", "cancelled"}:
                break
            time.sleep(0.02)
        assert state["status"] == "completed"
        assert provider.calls == 1 and state["result"]["network_used"] is False

        reviewed = client.get(f"/api/synthetic/sft/runs/{execution_id}/preview", headers=generation_headers)
        assert reviewed.status_code == 200 and reviewed.json()["records"]
        registered = client.post(f"/api/synthetic/sft/runs/{execution_id}/register",
                                 json={"confirm_reviewed": True}, headers=generation_headers)
        assert registered.status_code == 200
        dataset_id = registered.json()["registered_dataset_id"]

        launcher_caps = client.get("/api/launcher/capabilities").json()
        launcher_headers = {"X-Local-Launcher-Token": launcher_caps["token"]}
        launcher_preflight = client.post("/api/launcher/preflight", json={"dataset_id": dataset_id},
                                         headers=launcher_headers)
        assert launcher_preflight.status_code == 200, launcher_preflight.text
        launch_identity = launcher_preflight.json()
        assert launch_identity["ok"] is True
        assert launch_identity["dataset"]["dataset_id"] == dataset_id
        assert launch_identity["dataset"]["canonical_sha256"] == registered.json()["registration"]["canonical_sha256"]
        assert provider.calls == 1  # no real network was used

        class FakeTrainingProcess:
            """Run the real lineage writer while replacing only the GPU trainer."""

            def __init__(self, argv, **kwargs):
                output = Path(argv[argv.index("--output") + 1])
                run_id = argv[argv.index("--run-id") + 1]
                checkpoint = output / "fixture-checkpoint"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter.bin").write_bytes(b"offline adapter fixture")
                cli_module._write_training_run(
                    output=output,
                    result={"train": {"status": "PASS", "wall_seconds": 4.0,
                                      "peak_gpu_memory_mib_sampled": 2048},
                            "checkpoint": str(checkpoint)},
                    run_id=run_id, run_label="targeted-sft-offline", experiment_id=None,
                    stage="sft", model=argv[argv.index("--model") + 1],
                    dataset=argv[argv.index("--dataset") + 1],
                    config_path=argv[argv.index("--config") + 1],
                    resolved_config={"training": {"max_length": 1024}},
                    started_at="2026-10-24T00:00:00+00:00",
                    targeted_plan_id=shared_plan["plan_id"],
                    targeted_plan_version=shared_plan["plan_version"],
                )
                self.stdout = io.StringIO("step=1/1\n")
                self.returncode = 0

            def poll(self): return self.returncode
            def wait(self, timeout=None): return self.returncode

        monkeypatch.setattr(launcher_module.subprocess, "Popen", FakeTrainingProcess)
        started = client.post("/api/launcher/start",
                              json={"confirmation_token": launch_identity["confirmation_token"],
                                    "acknowledged_dataset_id": dataset_id},
                              headers=launcher_headers)
        assert started.status_code == 200, started.text
        training_run_id = started.json()["run_id"]
        for _ in range(100):
            run_status = client.get(f"/api/launcher/runs/{training_run_id}").json()
            if run_status["status"] in {"completed", "failed", "oom"}:
                break
            time.sleep(0.02)
        assert run_status["status"] == "completed", run_status
        train_lineage = json.loads((workspace / run_status["output"] / "lineage.json").read_text())
        assert train_lineage["targeted_plan_id"] == shared_plan["plan_id"]
        assert train_lineage["targeted_plan_version"] == shared_plan["plan_version"]
        train_dataset_refs = [item for item in train_lineage["inputs"] if item["kind"] == "dataset"]
        assert [(item["id"], item["digest"]["value"]) for item in train_dataset_refs] == [
            (dataset_id, registered.json()["registration"]["canonical_sha256"])
        ]

    from onegpu_posttrain.data.manifest import load_manifest
    registered_manifest = workspace / registered.json()["registration"]["manifest_path"]
    manifest = load_manifest(registered_manifest)
    assert manifest.metadata["targeted_stage_plan_id"] == shared_plan["plan_id"]
    assert manifest.metadata["targeted_stage_plan_version"] == shared_plan["plan_version"]
    assert manifest.lineage.parents == [state["result"]["source_dataset_id"]]
    assert manifest.lineage.source_evidence["generation_execution_id"] == execution_id
    artifact_root = registered_manifest.parent.parent
    build_lineage = json.loads((artifact_root / "runs" / manifest.lineage.produced_by_run_id / "lineage.json").read_text())
    input_ids = {item["id"] for item in build_lineage["inputs"]}
    output_ids = {item["id"] for item in build_lineage["outputs"]}
    assert state["result"]["source_dataset_id"] in input_ids
    assert execution_id in input_ids
    assert dataset_id in output_ids
    assert build_lineage["targeted_plan_id"] == shared_plan["plan_id"]
    registration_runs = [json.loads(path.read_text()) for path in (artifact_root / "runs").glob("*/lineage.json")]
    assert {item["run_type"] for item in registration_runs} >= {"data_build", "quality_assess"}
    assert all(item["targeted_plan_id"] == shared_plan["plan_id"]
               and item["targeted_plan_version"] == shared_plan["plan_version"]
               for item in registration_runs)

    # Exercise evaluation lineage and effect reporting using deterministic offline evaluation output.
    from argparse import Namespace
    from onegpu_posttrain.data.targeted import MetricTarget
    from onegpu_posttrain.synthetic.effects import TrainingResourceEvidence, build_before_after_report, save_effect_report
    from test_synthetic_effects import _report

    adapter_record_path = next((workspace / run_status["output"] / "model_artifacts").glob("model-*.json"))
    adapter_record = json.loads(adapter_record_path.read_text())
    eval_output = workspace / "runs" / "evaluations" / "run-targeted-sft-offline-eval"
    monkeypatch.setattr(cli_module, "run_evalscope", lambda *args, **kwargs: {"returncode": 0})
    monkeypatch.setattr(cli_module, "collect_evalscope_reports", lambda _path: {
        "gsm8k": {"score": 0.5, "accuracy": 0.5, "samples": 2},
        "math_500": {"score": 0.5, "accuracy": 0.5, "samples": 2},
    })
    eval_args = Namespace(
        config="recipes/mini/30-eval-math-v1.yaml", model=adapter_record["path"], revision=None,
        output=str(eval_output), run_label="targeted-sft-offline-eval", stage="sft", limit=2,
        dry_run=False, internal_holdout=None, internal_limit=20, max_new_tokens=32,
        domain_holdout=None, domain_limit=20, ppl_max_length=128, dtype="float16",
        run_id="run-targeted-sft-offline-eval", stream_output=True,
        targeted_plan_id=shared_plan["plan_id"], targeted_plan_version=shared_plan["plan_version"],
    )
    assert cli_module.cmd_run_model_eval(eval_args) == 0
    evaluation = json.loads((eval_output / "evaluation_lineage.json").read_text(encoding="utf-8"))
    assert evaluation["targeted_plan_id"] == train_lineage["targeted_plan_id"]
    assert evaluation["targeted_plan_version"] == train_lineage["targeted_plan_version"]
    assert evaluation["model_produced_by_run_id"] == training_run_id
    assert evaluation["model_id"] == adapter_record["model_id"]

    before = _report("eval-targeted-before", "model-before", correct=1)
    after = before.model_copy(update={"eval_id": evaluation["eval_id"], "model_id": evaluation["model_id"]})
    registration_info = registered.json()["registration"]
    effect = build_before_after_report(
        before=before, after=after, target_scenario_id=shared_stage_plan.target.scenario_id,
        target=MetricTarget(name="accuracy", direction="higher", threshold=0.5,
                            threshold_source="offline fixture", minimum_samples=1),
        guardrails=[], before_guard_metrics={}, after_guard_metrics={},
        training_dataset_id=dataset_id, training_dataset_sha256=registration_info["canonical_sha256"],
        actual_added_records=registration_info["count"], planned_training_budget=shared_stage_plan.training_budget,
        actual_training_budget=registration_info["count"], cost_status="unknown",
        targeted_plan_id=shared_plan["plan_id"], targeted_plan_version=shared_plan["plan_version"],
        training_run_id=training_run_id,
        training_resources=TrainingResourceEvidence(actual_training_records=registration_info["count"],
                                                     wall_seconds=4.0, peak_gpu_memory_mib=2048),
    )
    assert effect.training_dataset_id == train_dataset_refs[0]["id"] == dataset_id
    assert effect.training_run_id == training_run_id and effect.after_eval_id == evaluation["eval_id"]
    assert effect.targeted_plan_id == evaluation["targeted_plan_id"]
    effect_path = workspace / "reports" / f"{effect.report_id}.json"
    save_effect_report(effect, effect_path)
    effect_payload = json.loads(effect_path.read_text())
    assert effect_payload["actual_added_records"] == registration_info["count"]
    assert effect_payload["training_resources"]["peak_gpu_memory_mib"] == 2048
