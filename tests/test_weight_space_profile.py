"""The dedicated editor profile has fixed public fields and no secret payload."""

from pathlib import Path

import pytest

from onegpu_posttrain.weight_space.provider_profile import (
    PROFILE_RELATIVE_PATH,
    load_editor_profile,
)


def test_editor_profile_keeps_secret_out_of_return_value(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / PROFILE_RELATIVE_PATH
    target = tmp_path / PROFILE_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_bytes())
    monkeypatch.setenv("ONEGPU_SYNTHETIC_API_KEY", "sensitive-fixture-key")
    profile = load_editor_profile(tmp_path)
    assert profile.model == "qwen3.8-max"
    assert profile.input_cost_per_million_tokens is None
    assert "sensitive-fixture-key" not in profile.model_dump_json()


def test_editor_profile_rejects_model_override(tmp_path):
    target = tmp_path / PROFILE_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_text("profile_id: dashscope-qwen3.8-max-weight-space\n"
                      "adapter: openai_compatible\n"
                      "endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions\n"
                      "model: another-model\napi_key_env: ONEGPU_SYNTHETIC_API_KEY\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        load_editor_profile(tmp_path)
