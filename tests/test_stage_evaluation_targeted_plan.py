from __future__ import annotations

from onegpu_posttrain.lineage import Digest, ModelLineage, RunRecord
from onegpu_posttrain.ui.stage_evaluation import StageEvaluationAdapter, StageEvaluationRequest


def test_stage_evaluation_command_carries_training_plan_identity(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "merged"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}\n", encoding="utf-8")
    digest = "a" * 64
    model = ModelLineage(
        model_id="model-0123456789abcdefabcd", kind="merged", path=str(model_path),
        digest=Digest(value=digest, status="available"), stage="sft",
    )
    training = RunRecord(
        run_id="run-targeted-sft-offline-001", run_type="train", status="completed",
        started_at="2026-10-24T00:00:00Z", targeted_plan_id="plan-targeted-0123456789abcdefabcd",
        targeted_plan_version=2,
    )
    adapter = StageEvaluationAdapter(tmp_path)
    monkeypatch.setattr(adapter, "preflight", lambda request: {"ok": True})
    monkeypatch.setattr(adapter, "_model_for_training_run", lambda run_id: (model, training))
    monkeypatch.setattr(adapter, "_model_path", lambda value: model_path)

    command = adapter.build_command(
        StageEvaluationRequest(training_run_id=training.run_id, profile_id="sft-math-eval-v1"),
        "run-targeted-sft-eval-001",
    )
    assert command[command.index("--targeted-plan-id") + 1] == training.targeted_plan_id
    assert command[command.index("--targeted-plan-version") + 1] == str(training.targeted_plan_version)
