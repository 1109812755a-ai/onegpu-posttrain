import io
import os
import threading
import time

import pytest

from onegpu_posttrain.ui.gpu_lease import GpuLeaseBusy, WorkspaceGpuLease
from onegpu_posttrain.ui.launcher import SFTLauncher
import onegpu_posttrain.ui.launcher as launcher_module


def test_training_start_holds_workspace_gpu_lease_until_child_stops(tmp_path, monkeypatch):
    root = tmp_path.resolve()

    class SlowProcess:
        def __init__(self, *_args, **_kwargs):
            self.pid = os.getpid()
            self.stdout = io.StringIO("")
            self.returncode = None
            self.finished = threading.Event()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.finished.wait(timeout)
            return self.returncode

        def send_signal(self, _signal):
            self.returncode = -2
            self.finished.set()

        def kill(self):
            self.returncode = -9
            self.finished.set()

    monkeypatch.setattr(launcher_module.subprocess, "Popen", lambda *args, **kwargs: SlowProcess())
    first = SFTLauncher(root)
    started = first.start_stage_cli(
        run_id="run-sft-one", stage="sft", profile_id="profile-one",
        output_location="runs/sft/profile-one/run-sft-one", argv=["python", "-c", "pass"],
        dataset_id="ds-one", parent_model_id="model-one",
    )
    assert started["status"] == "running"
    lease = WorkspaceGpuLease(root)
    assert lease.inspect()["status"] == "busy"

    second = SFTLauncher(root)
    second._confirmations["evaluation-token"] = {
        "kind": "evaluation", "expires": time.time() + 60, "training_run_id": "run-source",
    }
    with pytest.raises(GpuLeaseBusy):
        second.start_evaluation("evaluation-token")
    second._confirmations["stage-evaluation-token"] = {
        "kind": "stage_evaluation", "expires": time.time() + 60,
    }
    with pytest.raises(GpuLeaseBusy):
        second.start_stage_evaluation("stage-evaluation-token")
    with pytest.raises(GpuLeaseBusy):
        second.start_stage_cli(
            run_id="run-sft-two", stage="sft", profile_id="profile-two",
            output_location="runs/sft/profile-two/run-sft-two", argv=["python", "-c", "pass"],
            dataset_id="ds-two", parent_model_id="model-two",
        )
    assert not (root / "runs/sft/profile-two/run-sft-two").exists()

    first.shutdown()
    assert lease.inspect()["status"] == "available"
