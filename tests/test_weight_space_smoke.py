"""The optional smoke reports identity and usage without prompt or secrets."""

from types import SimpleNamespace

from onegpu_posttrain.weight_space import smoke


def test_smoke_report_does_not_include_credential_or_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "load_editor_profile", lambda _: SimpleNamespace(
        model="qwen3.8-max", api_key_env="ONEGPU_SYNTHETIC_API_KEY"))
    monkeypatch.setattr(smoke, "load_local_provider_credential", lambda *_: True)

    class FakeClient:
        def complete(self, *_args, **_kwargs):
            return SimpleNamespace(prompt_tokens=4, completion_tokens=2, calls=[], provider_request_id="req-1")

    report = smoke.live_editor_smoke(tmp_path, client=FakeClient())
    assert report["status"] == "success"
    assert report["model"] == "qwen3.8-max" and report["prompt_tokens"] == 4
    assert "OK" not in str(report) and "ONEGPU_SYNTHETIC_API_KEY" not in str(report)
