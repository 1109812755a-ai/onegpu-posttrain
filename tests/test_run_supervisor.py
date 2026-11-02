import pytest

from onegpu_posttrain.ui.run_supervisor import LocalRunSupervisor, parse_training_progress


def test_lifecycle_transitions_enforce_terminal_states_and_one_active_run(tmp_path):
    supervisor = LocalRunSupervisor(tmp_path)
    created = supervisor.create(run_id="run-sft-1", stage="sft", profile_id="sft-mini-1024-v1",
                                output_location="runs/sft/sft-mini-1024-v1/confirm-a")
    assert created["status"] == "queued"
    with pytest.raises(RuntimeError, match="已有本地 GPU Run"):
        supervisor.create(run_id="run-sft-2", stage="sft", profile_id="sft-mini-1024-v1",
                           output_location="runs/sft/second")

    assert supervisor.transition("run-sft-1", "starting")["started_at"]
    assert supervisor.transition("run-sft-1", "running")["status"] == "running"
    completed = supervisor.transition("run-sft-1", "completed", exit_code=0)
    assert completed["completed_at"]
    assert supervisor.active_run() is None
    with pytest.raises(ValueError, match="非法 Run 状态转换"):
        supervisor.transition("run-sft-1", "running")


def test_lifecycle_persists_status_and_marks_unknown_old_process_interrupted(tmp_path):
    supervisor = LocalRunSupervisor(tmp_path)
    supervisor.create(run_id="run-grpo-1", stage="grpo", profile_id="grpo-standard-smoke-v1",
                      output_location="runs/grpo/grpo-standard-smoke-v1/confirm-b")
    supervisor.transition("run-grpo-1", "starting")
    supervisor.transition("run-grpo-1", "running")

    recovered = LocalRunSupervisor(tmp_path).get("run-grpo-1")

    assert recovered["status"] == "interrupted"
    assert "不会自动恢复" in recovered["detail"]


def test_logs_are_bounded_and_progress_is_never_invented(tmp_path):
    supervisor = LocalRunSupervisor(tmp_path, max_log_bytes=256)
    supervisor.create(run_id="run-cpt-1", stage="cpt", profile_id="cpt-smoke-v1",
                      output_location="runs/cpt/cpt-smoke-v1/confirm-c")
    supervisor.append_log("run-cpt-1", "x" * 400)
    record = supervisor.get("run-cpt-1")

    assert record["log_bytes"] <= 256
    assert "保留尾部" in supervisor.read_log("run-cpt-1")
    assert parse_training_progress("loss=0.8, step 12/40") == {"step": 12, "total": 40}
    assert parse_training_progress("loss=0.8; progress unavailable") is None
    assert supervisor.update_progress("run-cpt-1", "no progress evidence") ["progress_status"] == "unknown"


def test_output_paths_and_run_ids_are_restricted(tmp_path):
    supervisor = LocalRunSupervisor(tmp_path)
    with pytest.raises(ValueError, match="run_id"):
        supervisor.create(run_id="../escape", stage="sft", profile_id="fixed", output_location="runs/out")
    with pytest.raises(ValueError, match="相对路径"):
        supervisor.create(run_id="run-sft-1", stage="sft", profile_id="fixed", output_location="../outside")
