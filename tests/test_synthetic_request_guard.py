import multiprocessing

import pytest

from onegpu_posttrain.synthetic.providers import (
    GenerationRequest,
    GenerationResponse,
    MissingCredentialError,
    OutputProtocol,
    ProviderError,
)
from onegpu_posttrain.synthetic.request_guard import (
    RequestGuard,
    UncertainProviderRequest,
    acknowledge_uncertain_attempt,
    unresolved_attempts,
)


def _hold_request(bundle_dir, entered, release, result):
    class BlockingProvider:
        def generate(self, profile, request):
            entered.set()
            if not release.wait(15):
                raise RuntimeError("test timeout")
            return GenerationResponse(text="{}", finish_reason="stop")

    try:
        guard = RequestGuard(bundle_dir, "child", BlockingProvider())
        guard.generate(None, GenerationRequest(prompt="test", max_output_tokens=1,
                       output_protocol=OutputProtocol(protocol_id="test", version="1", required_fields=["value"])))
        result.put("done")
    except BaseException as exc:
        result.put(type(exc).__name__)


def test_crash_leaves_pending_and_future_execution_is_blocked(tmp_path):
    class CrashingProvider:
        calls = 0

        def generate(self, profile, request):
            self.calls += 1
            raise KeyboardInterrupt()

    provider = CrashingProvider()
    guard = RequestGuard(tmp_path, "execution-1", provider)
    with pytest.raises(KeyboardInterrupt):
        guard.generate(None, GenerationRequest(prompt="secret prompt", max_output_tokens=1,
                       output_protocol=OutputProtocol(protocol_id="test", version="1", required_fields=["value"])))
    assert provider.calls == 1
    assert unresolved_attempts(tmp_path)[0]["state"] == "pending"
    with pytest.raises(UncertainProviderRequest):
        RequestGuard(tmp_path, "execution-2", provider)
    assert provider.calls == 1
    assert "secret prompt" not in (tmp_path / "generation-requests.jsonl").read_text(encoding="utf-8")


def test_malformed_ledger_fails_closed(tmp_path):
    (tmp_path / "generation-requests.jsonl").write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ledger is malformed"):
        RequestGuard(tmp_path, "execution-1", object())


def test_concurrent_process_cannot_send_or_review_inflight_request(tmp_path):
    context = multiprocessing.get_context("spawn")
    entered, release, result = context.Event(), context.Event(), context.Queue()
    guard = RequestGuard(tmp_path, "parent", object())
    child = context.Process(target=_hold_request, args=(tmp_path, entered, release, result))
    child.start()
    try:
        assert entered.wait(15)
        request = GenerationRequest(prompt="test", max_output_tokens=1,
                                    output_protocol=OutputProtocol(protocol_id="test", version="1", required_fields=["value"]))
        with pytest.raises(UncertainProviderRequest, match="正在执行"):
            guard.generate(None, request)
        attempt_id = unresolved_attempts(tmp_path)[0]["attempt_id"]
        with pytest.raises(UncertainProviderRequest, match="正在执行"):
            acknowledge_uncertain_attempt(tmp_path, attempt_id)
    finally:
        release.set()
        child.join(15)
    assert not child.is_alive()
    assert result.get(timeout=5) == "done"
    assert unresolved_attempts(tmp_path) == []


def test_identical_prompts_in_independent_attempts_have_distinct_verified_caches(tmp_path):
    import json

    from onegpu_posttrain.synthetic.cache import load_response_cache

    class Provider:
        calls = 0
        def generate(self, profile, request):
            self.calls += 1
            return GenerationResponse(text=f"response {self.calls}", finish_reason="stop")

    provider = Provider()
    guard = RequestGuard(tmp_path, "execution-1", provider)
    request = GenerationRequest(prompt="same prompt", max_output_tokens=1,
                                output_protocol=OutputProtocol(protocol_id="test", version="1", required_fields=["value"]))
    guard.generate(None, request)
    guard.generate(None, request)
    events = [json.loads(line) for line in (tmp_path / "generation-requests.jsonl").read_text(encoding="utf-8").splitlines()]
    completed = [event for event in events if event["state"] == "completed"]
    assert len(completed) == 2 and completed[0]["cache_key"] != completed[1]["cache_key"]
    assert [load_response_cache(tmp_path, item["response_cache_path"], item["cache_key"], item["response_sha256"]).text
            for item in completed] == ["response 1", "response 2"]
    assert "response 1" not in (tmp_path / "generation-requests.jsonl").read_text(encoding="utf-8")


def test_provider_failure_ledger_keeps_only_safe_diagnostic_fields(tmp_path):
    import json

    class TimedOutProvider:
        def generate(self, profile, request):
            raise ProviderError("timeout", "provider request timed out", retryable=True, uncertain=True,
                                timeout_seconds=request.timeout_seconds)

    guard = RequestGuard(tmp_path, "execution-timeout", TimedOutProvider())
    request = GenerationRequest(prompt="private prompt content", max_output_tokens=1, timeout_seconds=143,
                                output_protocol=OutputProtocol(protocol_id="test", version="1", required_fields=["value"]))
    with pytest.raises(ProviderError):
        guard.generate(None, request)
    row = json.loads((tmp_path / "generation-requests.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["error_category"] == "timeout"
    assert row["timeout_seconds"] == 143
    assert row["state"] == "uncertain"
    assert "private prompt content" not in json.dumps(row)
    assert guard.provider_diagnostics()[0]["timeout_seconds"] == 143


def test_missing_credential_is_recorded_as_not_sent_and_does_not_block_retry(tmp_path):
    import json

    class MissingCredentialProvider:
        def generate(self, profile, request):
            raise MissingCredentialError("credential is not configured")

    request = GenerationRequest(prompt="private prompt", max_output_tokens=1,
                                output_protocol=OutputProtocol(protocol_id="test", version="1", required_fields=["value"]))
    guard = RequestGuard(tmp_path, "execution-missing-credential", MissingCredentialProvider())
    with pytest.raises(ProviderError) as exc:
        guard.generate(None, request)

    assert exc.value.category == "missing_credential"
    assert exc.value.uncertain is False
    row = json.loads((tmp_path / "generation-requests.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["state"] == "failed"
    assert row["error_category"] == "missing_credential"
    assert unresolved_attempts(tmp_path) == []
    assert "credential is not configured" not in json.dumps(row)
