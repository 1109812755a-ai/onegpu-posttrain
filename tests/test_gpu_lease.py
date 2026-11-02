from concurrent.futures import ThreadPoolExecutor

import pytest

from onegpu_posttrain.ui.gpu_lease import (
    GpuLeaseBusy,
    GpuLeaseRecoveryRequired,
    WorkspaceGpuLease,
)


def test_gpu_lease_is_mutually_exclusive_across_instances(tmp_path):
    managers = [WorkspaceGpuLease(tmp_path), WorkspaceGpuLease(tmp_path)]

    def attempt(index):
        try:
            return managers[index].acquire(task_type="chat", task_id=f"job-{index}")
        except GpuLeaseBusy:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        handles = list(pool.map(attempt, range(2)))

    acquired = [handle for handle in handles if handle is not None]
    assert len(acquired) == 1
    with pytest.raises(GpuLeaseBusy) as error:
        managers[0].acquire(task_type="training", task_id="run-another")
    assert error.value.owner["task_id"] == acquired[0].task_id
    assert acquired[0].release() is True
    assert managers[0].inspect()["status"] == "available"


def test_gpu_lease_owner_metadata_and_heartbeat(tmp_path):
    manager = WorkspaceGpuLease(tmp_path)
    lease = manager.acquire(task_type="evaluation", task_id="eval-1", run_id="run-eval-1")

    state = manager.inspect()
    assert state["status"] == "busy"
    assert state["owner"]["task_type"] == "evaluation"
    assert state["owner"]["task_id"] == "eval-1"
    assert state["owner"]["run_id"] == "run-eval-1"
    assert lease.heartbeat() is True
    assert lease.release() is True


def test_dead_owner_is_reported_and_requires_explicit_token_recovery(tmp_path):
    manager = WorkspaceGpuLease(tmp_path)
    lease = manager.acquire(task_type="training", task_id="run-crashed")
    lease.set_owner_pid(9_999_999)

    state = manager.inspect()
    assert state["status"] == "stale"
    with pytest.raises(GpuLeaseBusy):
        manager.acquire(task_type="chat", task_id="chat-1")
    with pytest.raises(GpuLeaseRecoveryRequired):
        manager.recover_stale(state["owner"]["recovery_token"])
    assert manager.recover_stale(state["owner"]["recovery_token"], confirm=True) is True
    assert manager.inspect()["status"] == "available"


def test_only_current_owner_can_release(tmp_path):
    manager = WorkspaceGpuLease(tmp_path)
    lease = manager.acquire(task_type="chat", task_id="chat-1")

    assert manager.release("not-the-owner") is False
    assert manager.inspect()["status"] == "busy"
    assert lease.release() is True
