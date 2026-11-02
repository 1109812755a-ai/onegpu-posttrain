"""Allowlisted DPO/SimPO adapters for registered preference datasets."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from .stage_profiles import stage_profile_catalog
from .training_preflight import StagePreflightRequest, preflight_stage

_RUN_ID = re.compile(r"^run-[a-z0-9][a-z0-9-]{0,119}$")


class PreferenceStageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1, max_length=100)
    dataset_id: str = Field(min_length=1, max_length=200)
    parent_model_id: str = Field(min_length=1, max_length=200)
    parameter_overrides: dict[str, StrictInt | StrictFloat] = Field(default_factory=dict, max_length=16)
    memory_mode_id: StrictStr | None = Field(default=None, max_length=64)


class PreferenceStageAdapter:
    """Resolve only registered IDs; recipes and algorithm are server-owned."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve(strict=True)
        self.profiles = {profile.profile_id: profile for profile in stage_profile_catalog(self.workspace).profiles}

    def preflight(self, request: PreferenceStageRequest, *, launcher_enabled: bool = True) -> dict[str, Any]:
        profile = self.profiles.get(request.profile_id)
        if profile is None or profile.stage not in {"dpo", "simpo"}:
            return {"schema_version": "1.0", "read_only": True, "status": "blocked",
                    "blockers": ["profile_id 不是批准的 DPO/SimPO profile"], "execution_started": False}
        generic = preflight_stage(
            self.workspace,
            StagePreflightRequest(stage=profile.stage, profile_id=profile.profile_id,
                                  dataset_ids=[request.dataset_id], parent_model_id=request.parent_model_id,
                                  parameter_overrides=request.parameter_overrides,
                                  memory_mode_id=request.memory_mode_id),
            launcher_enabled=launcher_enabled,
            allow_cli_adapter=True,
        )
        return {**generic, "status": "blocked" if generic["blockers"] else "ready_for_cli",
                "notice": "只使用批准 recipe、已登记 Preference dataset 和登记 SFT merged model；未启动训练。"}

    def build_command(self, request: PreferenceStageRequest, *, run_id: str,
                      config_path: str | None = None) -> list[str]:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("无效 run_id")
        preflight = self.preflight(request)
        if preflight["status"] != "ready_for_cli":
            raise ValueError("Preference 阶段预检阻断：" + "; ".join(preflight["blockers"]))
        profile = self.profiles[request.profile_id]
        output = f"runs/{profile.stage}/{profile.profile_id}/{run_id}"
        target = self.workspace / output
        if target.exists() or target.is_symlink():
            raise FileExistsError("Run 输出位置已存在；历史 Run 不覆盖")
        command = [
            sys.executable, "-m", "onegpu_posttrain", "run-registered-preference",
            "--workspace", str(self.workspace), "--profile-id", profile.profile_id,
            "--dataset-id", request.dataset_id, "--parent-model-id", request.parent_model_id,
            "--run-id", run_id, "--output", output,
        ]
        if config_path is not None:
            expected = f".onegpu-posttrain/launcher/configs/{run_id}.yaml"
            if config_path != expected or (self.workspace / config_path).is_symlink():
                raise ValueError("Preference resolved recipe 必须属于当前已确认 Run")
            command.extend(["--config", config_path])
        return command
