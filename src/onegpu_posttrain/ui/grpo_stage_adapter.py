"""Allowlisted GRPO standard/low-memory CLI adapter for registered inputs."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from .stage_profiles import stage_profile_catalog
from .training_preflight import StagePreflightRequest, preflight_stage

_RUN_ID = re.compile(r"^run-[a-z0-9][a-z0-9-]{0,119}$")


class GRPOStageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1, max_length=100)
    dataset_id: str = Field(min_length=1, max_length=200)
    parent_model_id: str = Field(min_length=1, max_length=200)
    parameter_overrides: dict[str, StrictInt | StrictFloat] = Field(default_factory=dict, max_length=16)
    memory_mode_id: StrictStr | None = Field(default=None, max_length=64)


class GRPOStageAdapter:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve(strict=True)
        self.profiles = {profile.profile_id: profile
                         for profile in stage_profile_catalog(self.workspace).profiles
                         if profile.stage == "grpo"}

    def preflight(self, request: GRPOStageRequest) -> dict[str, Any]:
        profile = self.profiles.get(request.profile_id)
        if profile is None:
            return {"schema_version": "1.0", "read_only": True, "status": "blocked",
                    "blockers": ["profile_id 不是批准的 GRPO profile"], "execution_started": False}
        generic = preflight_stage(
            self.workspace,
            StagePreflightRequest(stage="grpo", profile_id=profile.profile_id,
                                  dataset_ids=[request.dataset_id], parent_model_id=request.parent_model_id,
                                  parameter_overrides=request.parameter_overrides,
                                  memory_mode_id=request.memory_mode_id),
            launcher_enabled=True, allow_cli_adapter=True,
        )
        return {**generic, "status": "blocked" if generic["blockers"] else "ready_for_cli",
                "notice": "仅使用批准的 GRPO profile、contract-checked registered dataset 与 registered SFT merged model。"}

    def build_command(self, request: GRPOStageRequest, *, run_id: str,
                      config_path: str | None = None) -> list[str]:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("无效 GRPO run_id")
        preflight = self.preflight(request)
        if preflight["status"] != "ready_for_cli":
            raise ValueError("GRPO 预检阻断：" + "; ".join(preflight["blockers"]))
        profile = self.profiles[request.profile_id]
        output = self.workspace / "runs" / "grpo" / profile.profile_id / run_id
        if output.exists() or output.is_symlink():
            raise FileExistsError("GRPO Run 输出已存在；历史 Run 不覆盖")
        command = [sys.executable, "-m", "onegpu_posttrain", "run-registered-grpo",
                "--workspace", str(self.workspace), "--profile-id", profile.profile_id,
                "--dataset-id", request.dataset_id, "--parent-model-id", request.parent_model_id,
                "--run-id", run_id, "--output", output.relative_to(self.workspace).as_posix()]
        if config_path is not None:
            expected = f".onegpu-posttrain/launcher/configs/{run_id}.yaml"
            if config_path != expected or (self.workspace / config_path).is_symlink():
                raise ValueError("GRPO resolved recipe 必须属于当前已确认 Run")
            command.extend(["--config", config_path])
        return command
