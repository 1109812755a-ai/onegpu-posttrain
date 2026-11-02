from __future__ import annotations

import os

import pytest

from onegpu_posttrain.synthetic.secrets import (
    LocalCredentialFileError,
    load_local_provider_credential,
)


def test_loads_plain_export_and_quoted_values_only_for_requested_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ONEGPU_SYNTHETIC_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "# local secret\nexport ONEGPU_SYNTHETIC_API_KEY='new-key'\n",
        encoding="utf-8",
    )

    assert load_local_provider_credential("ONEGPU_SYNTHETIC_API_KEY", env_file) is True
    assert os.environ["ONEGPU_SYNTHETIC_API_KEY"] == "new-key"


def test_process_environment_wins_without_reading_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ONEGPU_SYNTHETIC_API_KEY", "process-key")
    env_file = tmp_path / ".env.local"
    env_file.write_text("not a valid line containing secret-value", encoding="utf-8")

    assert load_local_provider_credential("ONEGPU_SYNTHETIC_API_KEY", env_file) is False
    assert os.environ["ONEGPU_SYNTHETIC_API_KEY"] == "process-key"


@pytest.mark.parametrize(
    "contents, expected_fragment",
    [
        ("ONEGPU_SYNTHETIC_API_KEY=secret-value\nONEGPU_SYNTHETIC_API_KEY=other-secret", "重复"),
        ("ONEGPU_SYNTHETIC_API_KEY=secret-value\nOTHER_TOKEN=other-secret", "不允许"),
        ("broken-line-with-secret-value", "格式无效"),
        ('ONEGPU_SYNTHETIC_API_KEY="secret-value', "引号不匹配"),
    ],
)
def test_invalid_file_fails_closed_without_echoing_values(tmp_path, monkeypatch, contents, expected_fragment):
    monkeypatch.delenv("ONEGPU_SYNTHETIC_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text(contents, encoding="utf-8")

    with pytest.raises(LocalCredentialFileError) as exc:
        load_local_provider_credential("ONEGPU_SYNTHETIC_API_KEY", env_file)
    assert expected_fragment in str(exc.value)
    assert "secret-value" not in str(exc.value)
    assert "other-secret" not in str(exc.value)


def test_missing_file_leaves_credential_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("ONEGPU_SYNTHETIC_API_KEY", raising=False)
    assert load_local_provider_credential("ONEGPU_SYNTHETIC_API_KEY", tmp_path / ".env.local") is False
    assert "ONEGPU_SYNTHETIC_API_KEY" not in os.environ
