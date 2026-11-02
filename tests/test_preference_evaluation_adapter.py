import io
import json
import time
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from onegpu_posttrain import cli
from onegpu_posttrain.data.dataset_import import register_jsonl
from onegpu_posttrain.data.manifest import load_manifest, save_manifest, sha256_file
from onegpu_posttrain.data.registry import DataSource
from onegpu_posttrain.evaluation.preference import PreferenceEvalResult
from onegpu_posttrain.lineage import (
    ArtifactRef,
    ModelLineage,
    RunRecord,
    sha256_directory,
)
from onegpu_posttrain.ui.launcher import SFTLauncher
from onegpu_posttrain.ui.preference_evaluation_adapter import (
    PreferenceEvaluationAdapter,
    PreferenceEvaluationRequest,
)
from onegpu_posttrain.ui.run_workspace import RunWorkspace
from onegpu_posttrain.ui.stage_evaluation import (
    StageEvaluationAdapter,
    StageEvaluationRequest,
)


def _register_preference(root: Path, label: str, problem_ids: list[str], *, heldout: bool):
    rows = []
    for index, problem_id in enumerate(problem_ids):
        rows.append({
            "id": f"{label}-{index}", "raw_problem": f"Question {problem_id}",
            "prompt": [{"role": "user", "content": f"Question {problem_id}"}],
            "chosen": "correct answer", "rejected": "wrong answer",
            "prompt_template_id": "math-v1",
            "metadata": {"problem_id": problem_id, "dataset_role": "holdout" if heldout else "train_pairs",
                         "evaluation_only": heldout, "chosen_correct": True,
                         "rejected_correct": False},
        })
    body = "".join(json.dumps(row) + "\n" for row in rows).encode()
    result = register_jsonl(
        io.BytesIO(body), workspace=root,
        output_root=root / "data/processed" / label, stage="preference",
        origin=DataSource(source_id=f"source-{label}", kind="local_upload", name=label,
                          license_status="not_provided", declared_by="user"),
    )
    if not heldout:
        return result.dataset_id, result.path / "manifests/preference-user-upload-v1.yaml"
    manifest_path = result.path / "manifests/preference-user-upload-v1.yaml"
    manifest = load_manifest(manifest_path)
    report_path = (manifest_path.parent / manifest.processing_report.path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["outputs"]["preference_eval"] = report["outputs"].pop("preference")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest.stage = "preference_eval"
    manifest.processing_report.sha256 = sha256_file(report_path)
    save_manifest(manifest_path, manifest)
    return result.dataset_id, manifest_path


def _workspace(root: Path, train_problem_ids: list[str], eval_problem_ids: list[str], *, training_stage: str = "dpo"):
    recipe = root / "recipes/smoke/30-dpo.yaml"
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text(yaml.safe_dump({"stage": "dpo"}), encoding="utf-8")
    (root / "recipes/smoke/31-simpo-original.yaml").write_text("stage: simpo\n", encoding="utf-8")
    for relative in ("recipes/smoke/10-peft-cpt.yaml", "recipes/mini/23-sft-1024-fallback.yaml",
                     "recipes/smoke/40-grpo-standard.yaml", "recipes/smoke/41-grpo-lowmem.yaml"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    (root / "plugins").mkdir(exist_ok=True)
    (root / "plugins/grpo_rewards.py").write_text("# fixture\n", encoding="utf-8")
    train_id, train_manifest_path = _register_preference(root, "train-data", train_problem_ids, heldout=False)
    eval_id, _ = _register_preference(root, "eval-data", eval_problem_ids, heldout=True)
    train_manifest = load_manifest(train_manifest_path)
    train_canonical = train_manifest_path.parent.parent / "canonical/preference.jsonl"

    run_id = f"run-{training_stage}-training-fixture"
    run_root = root / "runs" / run_id
    (run_root / "model_artifacts").mkdir(parents=True)
    adapter_path = run_root / "train/adapter"
    adapter_path.mkdir(parents=True)
    (adapter_path / "adapter.bin").write_bytes(b"adapter")
    adapter_digest = sha256_directory(adapter_path)
    adapter_id = f"model-{adapter_digest[:20]}"
    adapter_model = ModelLineage(
        model_id=adapter_id, kind="adapter", path=str(adapter_path),
        digest={"value": adapter_digest, "status": "available"}, stage=training_stage,
        produced_by_run_id=run_id,
    )
    (run_root / "model_artifacts" / f"{adapter_id}.json").write_text(adapter_model.model_dump_json(), encoding="utf-8")
    merged_path = run_root / f"merged/M3-{training_stage.upper()}"
    merged_path.mkdir(parents=True)
    (merged_path / "weights.bin").write_bytes(b"merged")
    merged_digest = sha256_directory(merged_path)
    merged_id = f"model-{merged_digest[:20]}"
    merge_run_id = f"run-{training_stage}-merge-fixture"
    merged_model = ModelLineage(
        model_id=merged_id, kind="merged", path=str(merged_path),
        digest={"value": merged_digest, "status": "available"}, stage=training_stage,
        adapter_id=adapter_id, produced_by_run_id=merge_run_id,
    )
    (run_root / "model_artifacts" / f"{merged_id}.json").write_text(merged_model.model_dump_json(), encoding="utf-8")
    (run_root / "merge_runs" / merge_run_id).mkdir(parents=True)
    train = RunRecord(
        run_id=run_id, run_type="train", status="completed", started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:01:00Z",
        inputs=[ArtifactRef(id=train_id, kind="dataset", path=str(train_canonical), role="train",
                            digest={"value": train_manifest.hashes["canonical_records_sha256"], "status": "available"})],
        outputs=[ArtifactRef(id=adapter_id, kind="model", path=str(adapter_path), role="adapter",
                             digest={"value": adapter_digest, "status": "available"})],
    )
    (run_root / "lineage.json").write_text(train.model_dump_json(), encoding="utf-8")
    merge = RunRecord(
        run_id=merge_run_id, run_type="merge", status="completed", started_at="2026-01-01T00:01:00Z",
        completed_at="2026-01-01T00:02:00Z",
        inputs=[ArtifactRef(id=adapter_id, kind="model", path=str(adapter_path), role="adapter",
                            digest={"value": adapter_digest, "status": "available"})],
        outputs=[ArtifactRef(id=merged_id, kind="model", path=str(merged_path), role="merged",
                             digest={"value": merged_digest, "status": "available"})],
    )
    (run_root / "merge_runs" / merge_run_id / "lineage.json").write_text(merge.model_dump_json(), encoding="utf-8")
    return train_id, eval_id, merged_id, run_root


def test_preflight_requires_registered_holdout_and_disjoint_problem_ids(tmp_path):
    train_id, eval_id, model_id, _ = _workspace(tmp_path, ["p1", "p2"], ["p3"])
    adapter = PreferenceEvaluationAdapter(tmp_path)
    valid = adapter.preflight(PreferenceEvaluationRequest(
        profile_id="preference-heldout-v1", model_id=model_id, dataset_id=eval_id))
    same_train = adapter.preflight(PreferenceEvaluationRequest(
        profile_id="preference-heldout-v1", model_id=model_id, dataset_id=train_id))
    assert valid["status"] == "ready_for_cli"
    assert valid["shared_task_benchmark"] == "separate-evaluation-profile"
    assert same_train["status"] == "blocked"
    assert any("preference_eval" in blocker for blocker in same_train["blockers"])


def test_preflight_rejects_problem_id_overlap(tmp_path):
    _, eval_id, model_id, _ = _workspace(tmp_path, ["shared"], ["shared"])
    result = PreferenceEvaluationAdapter(tmp_path).preflight(PreferenceEvaluationRequest(
        profile_id="preference-heldout-v1", model_id=model_id, dataset_id=eval_id))
    assert result["status"] == "blocked"
    assert any("problem_id" in blocker for blocker in result["blockers"])


@pytest.mark.parametrize("training_stage", ["dpo", "simpo"])
def test_stage_evaluation_profile_resolves_registered_model_and_builds_fixed_preference_command(tmp_path, training_stage):
    _train_id, eval_id, model_id, _ = _workspace(tmp_path, ["p1", "p2"], ["p3"], training_stage=training_stage)
    adapter = StageEvaluationAdapter(tmp_path)
    summary = adapter.preflight(StageEvaluationRequest(
        training_run_id=f"run-{training_stage}-training-fixture", profile_id="preference-heldout-v1", dataset_id=eval_id,
    ))
    assert summary["ok"] is True, summary
    assert summary["model"]["model_id"] == model_id
    assert summary["dataset"]["dataset_id"] == eval_id
    command = adapter.build_command(StageEvaluationRequest(
        training_run_id=f"run-{training_stage}-training-fixture", profile_id="preference-heldout-v1", dataset_id=eval_id,
    ), "run-evaluation-fixture")
    assert command[command.index("--model-id") + 1] == model_id
    assert command[command.index("--dataset-id") + 1] == eval_id
    assert command[command.index("--output") + 1] == "runs/evaluations/run-evaluation-fixture"


def test_cpt_stage_evaluation_accepts_registered_cpt_holdout_stage(tmp_path, monkeypatch):
    _train_id, _unused_eval_id, model_id, _ = _workspace(
        tmp_path, ["train-1"], ["unused-1"], training_stage="cpt")
    holdout = register_jsonl(
        io.BytesIO((json.dumps({"id": "holdout-1", "text": "Held-out domain text.",
                                "source": "fixture", "metadata": {"tokens": 4}}) + "\n").encode()),
        workspace=tmp_path, output_root=tmp_path / "data/processed/cpt-holdout",
        stage="cpt",
        origin=DataSource(source_id="cpt-holdout-fixture", kind="local_upload", name="CPT holdout fixture",
                          license_status="not_provided", declared_by="user"),
    )
    holdout_manifest_path = holdout.path / "manifests/cpt-user-upload-v1.yaml"
    holdout_manifest = load_manifest(holdout_manifest_path)
    holdout_manifest.stage = "cpt_holdout"
    holdout_manifest.metadata["dataset_role"] = "cpt_holdout"
    save_manifest(holdout_manifest_path, holdout_manifest)
    monkeypatch.setattr("onegpu_posttrain.ui.stage_evaluation.shutil.which", lambda *_a, **_kw: "evalscope.exe")
    adapter = StageEvaluationAdapter(tmp_path)
    request = StageEvaluationRequest(
        training_run_id="run-cpt-training-fixture", profile_id="cpt-math-v1", dataset_id=holdout.dataset_id)

    summary = adapter.preflight(request)
    command = adapter.build_command(request, "run-cpt-eval-fixture") if summary["ok"] else []

    assert summary["ok"] is True, summary
    assert summary["model"]["model_id"] == model_id
    assert summary["dataset"]["dataset_id"] == holdout.dataset_id
    assert summary["dataset"]["role"] == "cpt_holdout"
    holdout_path = Path(command[command.index("--domain-holdout") + 1])
    assert holdout_path.name == "cpt.jsonl" and holdout_path.parent.name == "canonical"
    assert command[command.index("--domain-limit") + 1] == "20"


def test_stage_evaluation_profile_blocks_train_dataset_and_wrong_stage_profile(tmp_path):
    train_id, _eval_id, _model_id, _ = _workspace(tmp_path, ["p1"], ["p2"])
    adapter = StageEvaluationAdapter(tmp_path)
    train_data = adapter.preflight(StageEvaluationRequest(
        training_run_id="run-dpo-training-fixture", profile_id="preference-heldout-v1", dataset_id=train_id,
    ))
    wrong_profile = adapter.preflight(StageEvaluationRequest(
        training_run_id="run-dpo-training-fixture", profile_id="cpt-math-v1",
    ))
    assert train_data["ok"] is False
    assert any("preference_eval" in item or "held-out" in item for item in train_data["blockers"])
    assert wrong_profile["ok"] is False
    assert any("不适用于" in item for item in wrong_profile["blockers"])


def test_stage_evaluation_is_a_separate_observable_child_run_and_preserves_training_status(tmp_path, monkeypatch):
    _train_id, eval_id, _model_id, training_root = _workspace(tmp_path, ["p1"], ["p2"])
    training_lineage_path = training_root / "lineage.json"
    original_training_lineage = training_lineage_path.read_bytes()
    (training_root / "run.json").write_text(json.dumps({
        "run_id": "run-dpo-training-fixture", "stage": "dpo", "status": "completed",
    }), encoding="utf-8")
    launcher = SFTLauncher(tmp_path)
    launcher._write_state({"run_id": "run-dpo-training-fixture", "run_kind": "stage-training",
                           "stage": "dpo", "status": "completed", "evaluation_status": "pending",
                           "output": training_root.relative_to(tmp_path).as_posix(),
                           "started_at": "2026-01-01T00:00:00+00:00"})

    class FinishedEvaluation:
        returncode = 0
        stdout = io.StringIO("preference eval complete\n")

        def __init__(self, argv, **_kwargs):
            parent_state = launcher._read_state("run-dpo-training-fixture")
            assert parent_state["evaluation_status"] == "starting"
            output = tmp_path / argv[argv.index("--output") + 1]
            assert not output.exists(), "Launcher must leave output creation to the evaluator CLI"
            output.mkdir(parents=True)
            run_id = argv[argv.index("--run-id") + 1]
            (output / "eval.json").write_text(json.dumps({
                "protocol": "response-mean-logprob-v1", "model_id": "model-fixture",
                "metric_statuses": {"preference.accuracy": {"value": 0.75, "denominator": 4,
                    "denominator_unit": "pairs", "status": "measured", "reason": "4 pairs scored"}},
            }), encoding="utf-8")
            (output / "evaluation_run.json").write_text(json.dumps({"run_id": run_id, "status": "completed"}), encoding="utf-8")
            (output / "evaluation_lineage.json").write_text(json.dumps({"eval_id": "eval-fixture", "produced_by_run_id": run_id}), encoding="utf-8")

        def poll(self): return self.returncode
        def wait(self, timeout=None): return self.returncode

    monkeypatch.setattr("onegpu_posttrain.ui.launcher.subprocess.Popen", FinishedEvaluation)
    preflight = launcher.stage_evaluation_preflight({
        "training_run_id": "run-dpo-training-fixture", "profile_id": "preference-heldout-v1", "dataset_id": eval_id,
    })
    assert preflight["ok"] is True
    started = launcher.start_stage_evaluation(preflight["confirmation_token"])
    for _ in range(100):
        state = launcher._read_state(started["run_id"])
        parent_state = launcher._read_state("run-dpo-training-fixture")
        if (state and state["status"] == "completed" and parent_state
                and parent_state["evaluation_status"] == "completed"):
            break
        time.sleep(0.01)
    assert state["run_kind"] == "evaluation" and state["parent_run_id"] == "run-dpo-training-fixture"
    assert state["log_path"].startswith(".onegpu-posttrain/launcher/logs/")
    assert state["log_path"].endswith(".eval.log")
    assert not (tmp_path / state["output"] / "launcher.log").exists()
    parent_state = launcher._read_state("run-dpo-training-fixture")
    assert parent_state["evaluation_status"] == "completed"
    assert parent_state["evaluation_run_id"] == started["run_id"]
    detail = RunWorkspace(tmp_path).run_detail(started["run_id"])
    assert detail["status"] == "completed" and detail["parent_run_id"] == "run-dpo-training-fixture"
    assert detail["metrics"][0]["denominator"] == 4 and detail["metrics"][0]["denominator_unit"] == "pairs"
    parent = RunWorkspace(tmp_path).run_detail("run-dpo-training-fixture")
    assert parent["status"] == "completed" and parent["evaluations"][0]["run_id"] == started["run_id"]
    assert training_lineage_path.read_bytes() == original_training_lineage


def test_get_parent_training_run_exposes_running_stage_evaluation_log_tail(tmp_path):
    launcher = SFTLauncher(tmp_path)
    parent_id = "run-dpo-training-fixture"
    eval_id = "run-dpo-evaluation-fixture"
    log_path = tmp_path / ".onegpu-posttrain" / "launcher" / "logs" / f"{eval_id}.eval.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("evaluation step 1/3\\nevaluation step 2/3\\n", encoding="utf-8")
    launcher._write_state({"run_id": parent_id, "status": "completed", "run_kind": "stage-training",
                           "evaluation_status": "running", "evaluation_run_id": eval_id,
                           "output": "runs/dpo/training", "started_at": "2026-01-01T00:00:00+00:00"})
    launcher._write_state({"run_id": eval_id, "run_kind": "evaluation", "parent_run_id": parent_id,
                           "status": "running", "evaluation_status": "running",
                           "output": f"runs/evaluations/{eval_id}",
                           "log_path": log_path.relative_to(tmp_path).as_posix()})

    run = launcher.get_run(parent_id)

    assert run["evaluation_status"] == "running"
    assert "evaluation step 2/3" in run["evaluation_log_tail"]


@pytest.mark.parametrize("training_stage", ["dpo", "simpo"])
def test_registered_preference_eval_records_pair_denominator_and_separate_run(tmp_path, monkeypatch, training_stage):
    train_id, eval_id, model_id, training_root = _workspace(
        tmp_path, ["p1"], ["p2", "p3"], training_stage=training_stage,
    )
    args = Namespace(workspace=str(tmp_path), profile_id="preference-heldout-v1", model_id=model_id,
                     dataset_id=eval_id, run_id="run-preference-eval-ok",
                     output="runs/evaluations/run-preference-eval-ok")
    monkeypatch.setattr(cli, "evaluate_preference_pairs", lambda **kwargs: PreferenceEvalResult(
        pairs=2, policy_pair_accuracy=0.5, policy_pair_margin=0.125,
        chosen_mean_logprob=-0.5, rejected_mean_logprob=-0.625, ties=0,
        max_response_tokens_observed=12))

    assert cli.cmd_run_registered_preference_eval(args) == 0
    output = tmp_path / args.output
    report = json.loads((output / "eval.json").read_text(encoding="utf-8"))
    run = json.loads((output / "evaluation_run.json").read_text(encoding="utf-8"))
    lineage = json.loads((output / "evaluation_lineage.json").read_text(encoding="utf-8"))
    assert report["metric_statuses"]["preference.accuracy"]["denominator"] == 2
    assert report["shared_task_benchmark"]["gsm8k.accuracy"]["status"] == "not_measured"
    assert run["run_id"] == lineage["produced_by_run_id"] == args.run_id
    assert run["inputs"][1]["id"] == eval_id and run["inputs"][1]["role"] == "preference_eval"
    assert lineage["model_produced_by_run_id"] == f"run-{training_stage}-merge-fixture"
    assert (training_root / "lineage.json").is_file()  # evaluation is a new Run, not a rewrite of training
    assert train_id != eval_id


def test_partial_and_failed_preference_eval_keep_truthful_metric_status(tmp_path, monkeypatch):
    _, eval_id, model_id, _ = _workspace(tmp_path, ["p1"], ["p2", "p3"])
    args = Namespace(workspace=str(tmp_path), profile_id="preference-heldout-v1", model_id=model_id,
                     dataset_id=eval_id, run_id="run-preference-eval-partial",
                     output="runs/evaluations/run-preference-eval-partial")
    monkeypatch.setattr(cli, "evaluate_preference_pairs", lambda **kwargs: PreferenceEvalResult(
        pairs=1, policy_pair_accuracy=1.0, policy_pair_margin=0.25,
        chosen_mean_logprob=-0.5, rejected_mean_logprob=-0.75, ties=0,
        max_response_tokens_observed=8))
    assert cli.cmd_run_registered_preference_eval(args) == 4
    partial = json.loads((tmp_path / args.output / "evaluation_run.json").read_text(encoding="utf-8"))
    assert partial["status"] == "partial"

    args.run_id = "run-preference-eval-failed"
    args.output = "runs/evaluations/run-preference-eval-failed"
    monkeypatch.setattr(cli, "evaluate_preference_pairs", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("model load failed")))
    assert cli.cmd_run_registered_preference_eval(args) == 4
    output = tmp_path / args.output
    report = json.loads((output / "eval.json").read_text(encoding="utf-8"))
    lineage = json.loads((output / "evaluation_lineage.json").read_text(encoding="utf-8"))
    assert report["status"] == lineage["status"] == "failed"
    assert report["metric_statuses"]["preference.margin"]["status"] == "not_measured"
