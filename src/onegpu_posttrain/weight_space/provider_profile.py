"""Load only the fixed Bailian editor profile; never return credential values."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..synthetic.providers import ProviderProfile

PROFILE_RELATIVE_PATH = Path("configs/synthetic/provider.dashscope-qwen3.8-max.yaml")
PROFILE_ID = "dashscope-qwen3.8-max-weight-space"
PROFILE_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def load_editor_profile(workspace: str | Path) -> ProviderProfile:
    root = Path(workspace).resolve(strict=True)
    profile = ProviderProfile.model_validate(
        yaml.safe_load((root / PROFILE_RELATIVE_PATH).read_text(encoding="utf-8"))
    )
    if (profile.profile_id != PROFILE_ID or profile.model != "qwen3.8-max"
            or profile.endpoint != PROFILE_ENDPOINT or profile.api_key_env != "ONEGPU_SYNTHETIC_API_KEY"
            or profile.adapter != "openai_compatible"):
        raise ValueError("weight-space editor profile identity changed")
    return profile
