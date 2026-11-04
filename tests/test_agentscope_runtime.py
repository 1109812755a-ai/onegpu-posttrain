"""AgentScope tool loop and workspace boundary checks without a network call."""

from pathlib import Path
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("agentscope")

from agentscope.model import ChatResponse
from onegpu_posttrain.assistant.agentscope_runtime import AgentScopeResult, AgentScopeRuntime
from onegpu_posttrain.assistant.agentscope_tools import WorkspaceAgentTools
from onegpu_posttrain.assistant.contracts import (
    AssistantAnswer, ContextSummary, EvidenceFact, EvidenceRef, WorkspaceMemory,
)
from onegpu_posttrain.assistant.model_adapter import AssistantModelError
from onegpu_posttrain.assistant.evidence_reader import WorkspaceEvidenceReader
from onegpu_posttrain.assistant.profiles import AssistantProfileCatalog
from onegpu_posttrain.assistant.service import WorkspaceAssistantService
from onegpu_posttrain.assistant.service import classify_question
from onegpu_posttrain.assistant.store import WorkspaceAssistantStore
from onegpu_posttrain.assistant.langfuse_observer import LangfuseObserver
from onegpu_posttrain.assistant.contracts import ObservationEvent
from onegpu_posttrain.ui.run_workspace import RunWorkspace
from onegpu_posttrain.ui.server import create_app
from fastapi.testclient import TestClient


def test_toolkit_has_only_read_only_tools(tmp_path):
    tools = WorkspaceAgentTools(tmp_path)
    names = {entry["function"]["name"] for entry in tools.toolkit().get_json_schemas()}
    assert names == {"investigate_workspace"}
    with pytest.raises(ValueError):
        tools.read_run("../../etc/passwd")
    with pytest.raises(ValueError):
        tools.search_runs(stage="../../outside")


def test_failed_evaluation_diagnosis_is_not_routed_to_comparison():
    assert classify_question("这个评测为什么失败？ run-base-choice-accuracy-v1-fixture") == "recent_failed_runs"
    assert classify_question("比较两次评测结果，为什么最近一次失败？") == "compare_evaluations"


def test_specific_failed_run_returns_structured_local_diagnosis_without_bailian(tmp_path):
    reason = "ValueError: checkpoint model type qwen3_5 but Transformers does not recognize this architecture"
    run_id = "run-base-choice-accuracy-v1-eac504a318be"
    refs = [
        EvidenceRef(evidence_id=f"run:{run_id}", kind="run", entity_id=run_id,
                    title="fixture", status="available", summary="评测失败", source_href=f"/runs/{run_id}"),
        EvidenceRef(evidence_id=f"evaluation:{run_id}", kind="evaluation", entity_id=run_id,
                    title="fixture 评测", status="failed", summary=reason,
                    facts=[{"key": "failure_reason", "label": "失败原因", "value": reason}],
                    source_href=f"/runs/{run_id}"),
    ]
    with patch.object(WorkspaceEvidenceReader, "search", return_value=refs):
        result = AgentScopeRuntime(tmp_path, SimpleNamespace(profile=None)).run(
            f"请定位评测运行 {run_id} 的失败原因", "只读分析", [],
        )
    assert "Transformers does not recognize" in result.answer.conclusion
    assert "模型加载阶段" in result.answer.interpretations[0].text
    assert result.answer.facts[0].evidence_ids == [f"evaluation:{run_id}"]
    assert result.model_calls == 0
    assert result.fallback_reason and "本地结构化失败证据" in result.fallback_reason


def test_agentscope_calls_tools_and_returns_cited_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_SCOPE_KEY", "fixture-key")
    profile = SimpleNamespace(model="test-model", endpoint="https://example.invalid/v1/chat/completions",
                              api_key_env="TEST_AGENT_SCOPE_KEY", adapter="openai_compatible")
    model = SimpleNamespace(profile=profile)
    ref = EvidenceRef(evidence_id="run:run-fixture", kind="run", entity_id="run-fixture",
                      title="fixture", status="available", summary="训练失败", source_href="/runs/run-fixture")

    def investigate(self, question):
        self._tick()
        self.tool_names.add("investigate_workspace")
        self._remember([ref])
        return {"kind": "recent_failed_runs", "evidence": [ref.model_dump(mode="json")]}

    class FakeProvider:
        def __init__(self, *args, **kwargs):
            pass

        async def __call__(self, messages, **kwargs):
            answer = {
                "conclusion": "找到了失败运行。",
                "facts": [{"text": "运行失败。", "evidence_ids": ["run:run-fixture"]}],
                "interpretations": [], "limitations": [], "clarification_options": [],
            }
            return ChatResponse(content=[{"type": "text", "text": json.dumps(answer, ensure_ascii=False)}])

    with patch("agentscope.model.OpenAIChatModel", FakeProvider), \
         patch.object(WorkspaceAgentTools, "investigate_workspace", investigate):
        result = AgentScopeRuntime(Path(tmp_path), model).run("查找失败运行", "只读分析", [])
    assert result.tool_calls == 1
    assert result.model_calls == 1
    assert result.answer.facts[0].evidence_ids == ["run:run-fixture"]
    assert [item.evidence_id for item in result.evidence] == ["run:run-fixture"]


def test_service_requires_session_consent_and_persists_agent_answer(tmp_path):
    store = WorkspaceAssistantStore(tmp_path)
    ref = EvidenceRef(evidence_id="run:run-fixture", kind="run", entity_id="run-fixture",
                      title="fixture", status="available", summary="训练失败", source_href="/runs/run-fixture")
    answer = AssistantAnswer(conclusion="找到了失败运行。", facts=[{"text": "运行失败。", "evidence_ids": [ref.evidence_id]}])
    class FakeRuntime:
        calls = 0
        def run(self, question, instruction, history):
            self.calls += 1
            return AgentScopeResult(answer, [ref], [], 2, 3)
    runtime = FakeRuntime()
    service = WorkspaceAssistantService(store=store, evidence_reader=None,
                                        profiles=AssistantProfileCatalog(store), model=None,
                                        agent_runtime=runtime)
    first = service.ask("最近有哪些失败运行？")
    assert first.consent_required and runtime.calls == 0
    session_id = first.task.session_id
    still_unconfirmed = service.ask("最近有哪些失败运行？", session_id=session_id)
    assert still_unconfirmed.consent_required and runtime.calls == 0
    service.confirm_data_egress(session_id)
    second = service.ask("最近有哪些失败运行？", session_id=session_id)
    assert runtime.calls == 1
    assert second.task.status == "succeeded"
    assert second.task.metadata["runtime"] == "agentscope"
    assert second.evidence[0].evidence_id == ref.evidence_id
    messages = store.get_session(session_id).messages
    assert len(messages) == 3 and all(message.message_kind == "chat" for message in messages)


def test_server_uses_agentscope_by_default_and_legacy_only_when_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("ONEGPU_ASSISTANT_RUNTIME", raising=False)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/assistant/status").json()["runtime"] == "agentscope"
    monkeypatch.setenv("ONEGPU_ASSISTANT_RUNTIME", "legacy")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/assistant/status").json()["runtime"] == "legacy"


def test_existing_session_is_readable_after_store_reopen(tmp_path):
    original = WorkspaceAssistantStore(tmp_path)
    session = original.create_session("旧版对话", "workspace-analyst")
    reopened = WorkspaceAssistantStore(tmp_path)
    restored = reopened.get_session(session.session_id)
    assert restored is not None and restored.session_id == session.session_id
    assert restored.assistant_profile_id == "workspace-analyst"


def test_langfuse_receives_allowlisted_metadata_and_degrades_on_failure(tmp_path):
    class Client:
        metadata = None
        trace_id = "a" * 32
        def start_as_current_observation(self, **kwargs):
            self.metadata = kwargs["metadata"]
            return self
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def get_trace_url(self, **_kwargs): return "https://langfuse.local/trace"
        def flush(self): pass
    client = Client()
    observer = LangfuseObserver(tmp_path, client=client)
    event = ObservationEvent(event_name="assistant_task_finished", task_id="task-fixture",
                             occurred_at=datetime.now(timezone.utc),
                             attributes={"runtime": "agentscope", "tool_names": "read_lineage",
                                         "tool_calls": 1, "model_calls": 3,
                                         "question": "private question", "answer": "private answer"})
    trace_id, trace_url = observer.emit(event)
    assert trace_id == "a" * 32 and trace_url == "https://langfuse.local/trace"
    assert client.metadata == {"runtime": "agentscope", "tool_names": "read_lineage",
                               "tool_calls": 1, "model_calls": 3,
                               "task_id": "task-fixture", "event_name": "assistant_task_finished"}

    class BrokenClient(Client):
        def start_as_current_observation(self, **_kwargs): raise RuntimeError("secret")
    broken = LangfuseObserver(tmp_path, client=BrokenClient())
    assert broken.emit(event) == (None, None)
    assert broken.status()["state"] == "unavailable"


def test_long_term_memory_and_summary_are_scoped_to_agent_context(tmp_path):
    store = WorkspaceAssistantStore(tmp_path)
    session = store.create_session("实验追问", "workspace-analyst")
    store.grant_data_egress_consent(session.session_id, version="test")
    now = datetime.now(timezone.utc)
    store.save_memory(WorkspaceMemory(memory_id="memory-prefer-denominators", content="优先列出指标分母",
                                      created_at=now, updated_at=now))
    summary = ContextSummary(summary_id="summary-experiment-history", session_id=session.session_id, revision=1,
                             summary="用户在追问最近一次 SFT 评测。", source_message_ids=["message-history"],
                             created_at=now, updated_at=now)
    store.save_summary(summary)
    ref = EvidenceRef(evidence_id="run:run-memory-fixture", kind="run", entity_id="run-memory-fixture",
                      title="fixture", status="available", summary="运行记录", source_href="/runs/run-memory-fixture")
    answer = AssistantAnswer(conclusion="找到运行记录。", facts=[{
        "text": "运行记录已读取。", "evidence_ids": [ref.evidence_id],
    }])
    class CaptureRuntime:
        instruction = ""
        history = None
        def run(self, _question, instruction, history):
            self.instruction, self.history = instruction, history
            return AgentScopeResult(answer, [ref], [], 1, 1)
    runtime = CaptureRuntime()
    service = WorkspaceAssistantService(store=store, evidence_reader=None,
                                        profiles=AssistantProfileCatalog(store), model=None,
                                        agent_runtime=runtime)
    result = service.ask("SFT 的分母是多少？", session_id=session.session_id)
    assert "优先列出指标分母" in runtime.instruction
    assert any("用户在追问最近一次 SFT 评测" in item["content"] for item in runtime.history)
    assert result.task.memory_ids == ["memory-prefer-denominators"]
    assert result.task.context_summary_id == summary.summary_id


def test_langfuse_is_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ONEGPU_ASSISTANT_LANGFUSE_ENABLED", raising=False)
    observer = LangfuseObserver(tmp_path)
    assert not observer.configured
    assert observer.status()["state"] == "disabled"


def test_search_runs_filters_and_limits(tmp_path):
    tools = WorkspaceAgentTools(tmp_path)
    class FakeRuns:
        def list_runs(self, **kwargs):
            assert kwargs["stage"] == "sft" and kwargs["status"] == "failed"
            return {"runs": [{"run_id": "run-a", "run_kind": "training", "stage": "sft", "status": "failed"},
                             {"run_id": "run-b", "run_kind": "training", "stage": "sft", "status": "failed"},
                             {"run_id": "run-c", "run_kind": "training", "stage": "sft", "status": "failed"},
                             {"run_id": "run-d", "run_kind": "evaluation", "stage": "sft", "status": "failed"}],
                    "total": 4, "partial_scan": False}
    tools.runs = FakeRuns()
    assert tools.search_runs("sft", "failed", "training")["runs"][0]["run_id"] == "run-a"
    second_page = tools.search_runs("sft", "failed", "training", offset=1, limit=1)
    assert second_page["runs"][0]["run_id"] == "run-b" and second_page["more"]
    assert tools.search_runs("sft", "failed", "training", offset=2, limit=1)["runs"][0]["run_id"] == "run-c"
    with pytest.raises(ValueError):
        tools.search_runs(kind="shell")


def test_tool_call_budget_is_enforced(tmp_path):
    tools = WorkspaceAgentTools(tmp_path)
    tools.max_calls = 1
    tools._tick()
    with pytest.raises(ValueError, match="上限"):
        tools._tick()


def test_runtime_bounds_old_history_without_trimming_current_evidence():
    history = [{"role": "user", "content": "很长的历史问题。" * 1000},
               {"role": "assistant", "content": "旧回答。" * 1000},
               {"role": "user", "content": "最近问题保留。"}]
    bounded, dropped = AgentScopeRuntime._bounded_history(history)
    assert dropped == 2
    assert bounded == history[-1:]


def test_service_rejects_fabricated_citation(tmp_path):
    store = WorkspaceAssistantStore(tmp_path)
    ref = EvidenceRef(evidence_id="run:run-fixture", kind="run", entity_id="run-fixture",
                      title="fixture", status="available", summary="失败", source_href="/runs/run-fixture")
    answer = AssistantAnswer(conclusion="运行失败。", facts=[{"text": "失败。", "evidence_ids": ["run:other"]}])
    class FakeRuntime:
        def run(self, question, instruction, history):
            return AgentScopeResult(answer, [ref], [], 1, 1)
    service = WorkspaceAssistantService(store=store, evidence_reader=None,
                                        profiles=AssistantProfileCatalog(store), model=None,
                                        agent_runtime=FakeRuntime())
    first = service.ask("近期失败运行")
    service.confirm_data_egress(first.task.session_id)
    result = service.ask("近期失败运行", session_id=first.task.session_id)
    assert result.task.status == "failed"
    assert result.task.error_code == "invalid_citation"


def test_comparison_rejects_different_protocols(tmp_path):
    tools = WorkspaceAgentTools(tmp_path)
    def evaluation(run_id, protocol):
        return EvidenceRef(evidence_id=f"evaluation:{run_id}", kind="evaluation", entity_id=run_id,
                           title=run_id, status="available", source_href=f"/runs/{run_id}", facts=[
                               EvidenceFact(key="model_id", label="模型", value="model-1"),
                               EvidenceFact(key="dataset_ids", label="数据集", value="ds-1"),
                               EvidenceFact(key="protocol_version", label="协议", value=protocol),
                           ])
    refs = {"run-a": evaluation("run-a", "v1"), "run-b": evaluation("run-b", "v2")}
    class FakeReader:
        def search(self, query):
            return [refs[query.selected_entity_ids[0]]]
    tools.reader = FakeReader()
    result = tools.compare("run-a", "run-b")
    assert result["comparison"]["status"] == "incompatible"
    assert result["comparison"]["metrics"] == []
    assert set(tools.evidence) == {"evaluation:run-a", "evaluation:run-b"}


def test_comparison_keeps_partial_and_missing_results_non_numeric(tmp_path):
    tools = WorkspaceAgentTools(tmp_path)
    def evaluation(run_id, status="available"):
        return EvidenceRef(evidence_id=f"evaluation:{run_id}", kind="evaluation", entity_id=run_id,
                           title=run_id, status=status, source_href=f"/runs/{run_id}", facts=[
                               EvidenceFact(key="model_id", label="模型", value="model-1"),
                               EvidenceFact(key="dataset_ids", label="数据集", value="ds-1"),
                               EvidenceFact(key="protocol_version", label="协议", value="v1"),
                           ])
    available = evaluation("run-a")
    partial = evaluation("run-b", "partial")
    class PartialReader:
        def search(self, query):
            return [available if query.selected_entity_ids[0] == "run-a" else partial]
    tools.reader = PartialReader()
    result = tools.compare("run-a", "run-b")["comparison"]
    assert result["status"] == "partial" and result["metrics"] == []
    class MissingReader:
        def search(self, query):
            return [available] if query.selected_entity_ids[0] == "run-a" else []
    tools.reader = MissingReader()
    result = tools.compare("run-a", "run-c")
    assert result["status"] == "missing" and len(result["evidence"]) == 1


def test_dataset_lineage_quality_legacy_and_missing_are_explicit(tmp_path):
    reader = WorkspaceEvidenceReader(tmp_path)
    dataset = {"dataset_id": "ds-fixture", "stage": "sft", "count": 12, "unit": "records",
               "source": {"kind": "local_upload"}, "manifest": {"lineage": {"parents": ["ds-parent"]}},
               "quality": {"gate": {"summary_status": "FAIL"}, "quality_protocol": "quality-v1",
                           "metrics": {"coverage": {"value": 0.75, "denominator": 12}}}}
    class Catalog:
        scanned_at = None
        builds = {"build-fixture": {"id": "build-fixture", "mode": "legacy", "datasets": [dataset]}}
        def scan(self): self.scanned_at = "scanned"
        def detail(self, _build_id): return {"datasets": [dataset]}
    reader._catalog = Catalog()
    refs = reader._search_datasets({"ds-fixture"}, 10)
    by_kind = {item.kind: item for item in refs}
    assert by_kind["dataset"].status == "legacy"
    assert by_kind["dataset"].source_href == "/data#dataset-ds-fixture"
    assert by_kind["quality_report"].status == "available"
    assert next(f.value for f in by_kind["quality_report"].facts if f.key == "quality_status") == "FAIL"
    assert by_kind["lineage"].status == "available"
    assert "ds-parent" in by_kind["lineage"].summary
    missing = reader._search_datasets({"ds-absent"}, 10)
    assert [item.status for item in missing] == ["missing", "missing"]


def test_runtime_missing_credentials_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_MISSING_AGENT_KEY", raising=False)
    profile = SimpleNamespace(model="test-model", endpoint="https://example.invalid/v1/chat/completions",
                              api_key_env="TEST_MISSING_AGENT_KEY", adapter="openai_compatible")
    with pytest.raises(AssistantModelError) as exc:
        AgentScopeRuntime(tmp_path, SimpleNamespace(profile=profile)).run("问题", "只读", [])
    assert exc.value.code == "not_configured"
    assert "TEST_MISSING_AGENT_KEY" not in str(exc.value)


@pytest.mark.parametrize("question", ["帮我执行 shell 命令", "帮我运行 Python", "帮我启动训练", "请联网搜索训练方案", "请创建文件", "请修改数据"])
def test_mutation_requests_never_invoke_agent(tmp_path, question):
    store = WorkspaceAssistantStore(tmp_path)
    class MustNotRun:
        def run(self, *args):
            pytest.fail("read-only guard must reject execution requests")
    service = WorkspaceAssistantService(store=store, evidence_reader=None,
                                        profiles=AssistantProfileCatalog(store), model=None,
                                        agent_runtime=MustNotRun())
    first = service.ask(question)
    service.confirm_data_egress(first.task.session_id)
    result = service.ask(question, session_id=first.task.session_id)
    assert result.task.error_code == "read_only_capability_boundary"


def test_runtime_provider_error_does_not_expose_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_SCOPE_KEY", "sensitive-fixture")
    profile = SimpleNamespace(model="test-model", endpoint="https://example.invalid/v1/chat/completions",
                              api_key_env="TEST_AGENT_SCOPE_KEY", adapter="openai_compatible")
    class BrokenProvider:
        def __init__(self, *args, **kwargs):
            pass
        async def __call__(self, *args, **kwargs):
            raise RuntimeError("sensitive-fixture")
    ref = EvidenceRef(evidence_id="run:run-fixture", kind="run", entity_id="run-fixture",
                      title="fixture", status="available", summary="fixture", source_href="/runs/run-fixture")
    with patch("agentscope.model.OpenAIChatModel", BrokenProvider), \
         patch.object(WorkspaceAgentTools, "investigate_workspace", lambda self, question: self._remember([ref])):
        with pytest.raises(AssistantModelError) as exc:
            AgentScopeRuntime(tmp_path, SimpleNamespace(profile=profile)).run("问题", "只读", [])
    assert exc.value.code == "agentscope_error"
    assert "sensitive-fixture" not in str(exc.value)


def test_lineage_lookup_uses_bounded_run_row_without_rescanning(tmp_path, monkeypatch):
    output = tmp_path / "runs" / "fixture"
    output.mkdir(parents=True)
    (output / "lineage.json").write_text(json.dumps({"run_id": "run-fixture", "outputs": []}), encoding="utf-8")
    workspace = RunWorkspace(tmp_path)
    monkeypatch.setattr(workspace, "_candidates", lambda: pytest.fail("unexpected full scan"))
    row = {"run_id": "run-fixture", "source_refs": ["runs/fixture"]}
    assert workspace.run_lineage_from_row("run-fixture", row)["run_id"] == "run-fixture"
    assert workspace.run_lineage_from_row("run-other", row) is None
    assert workspace.run_lineage_from_row("run-fixture", {"run_id": "run-fixture", "source_refs": ["../outside"]}) is None


def test_runtime_timeout_returns_safe_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_SCOPE_KEY", "fixture-key")
    profile = SimpleNamespace(model="test-model", endpoint="https://example.invalid/v1/chat/completions",
                              api_key_env="TEST_AGENT_SCOPE_KEY", adapter="openai_compatible")
    class DummyProvider:
        def __init__(self, *args, **kwargs):
            pass
    real_wait_for = asyncio.wait_for
    async def expire(coroutine, timeout):
        if timeout == 30:
            return await real_wait_for(coroutine, timeout)
        coroutine.close()
        raise asyncio.TimeoutError()
    ref = EvidenceRef(evidence_id="run:run-fixture", kind="run", entity_id="run-fixture",
                      title="fixture", status="available", summary="fixture", source_href="/runs/run-fixture")
    with patch("agentscope.model.OpenAIChatModel", DummyProvider), patch("asyncio.wait_for", expire):
        with patch.object(WorkspaceAgentTools, "investigate_workspace", lambda self, question: self._remember([ref])):
            with pytest.raises(AssistantModelError) as exc:
                AgentScopeRuntime(tmp_path, SimpleNamespace(profile=profile)).run("问题", "只读", [])
    assert exc.value.code == "timeout"
    assert exc.value.model_calls == 0 and exc.value.tool_calls == 0
    assert exc.value.model_attempts == 0


def test_runtime_returns_clarification_when_no_evidence_and_no_structured_answer(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_AGENT_SCOPE_KEY", "fixture-key")
    profile = SimpleNamespace(model="test-model", endpoint="https://example.invalid/v1/chat/completions",
                              api_key_env="TEST_AGENT_SCOPE_KEY", adapter="openai_compatible")
    class NoAnswerAgent:
        def __init__(self, **kwargs): pass
        async def __call__(self, *args, **kwargs): return SimpleNamespace(metadata=None)
    class DummyProvider:
        def __init__(self, *args, **kwargs): pass
    with patch("agentscope.agent.ReActAgent", NoAnswerAgent), \
         patch("agentscope.model.OpenAIChatModel", DummyProvider), \
         patch("onegpu_posttrain.assistant.agentscope_runtime.WorkspaceAgentTools.search_runs", return_value={"runs": []}):
        result = AgentScopeRuntime(tmp_path, SimpleNamespace(profile=profile)).run("找一个不存在的运行", "只读", [])
    assert result.answer.clarification_options
    assert result.answer.conclusion == "本轮没有找到可用于回答的工作区证据。"
    assert result.evidence == []


def test_verified_merged_model_resolves_training_dataset(tmp_path):
    reader = WorkspaceEvidenceReader(tmp_path)
    row = {"run_id": "run-training", "run_kind": "training"}
    lineage = {"inputs": [{"kind": "dataset", "id": "ds-fixture"}],
               "outputs": [{"kind": "model", "id": "model-adapter"}]}
    class FakeRuns:
        def list_runs(self, **kwargs): return {"runs": [row]}
        def run_lineage_from_row(self, run_id, selected): return lineage
        def run_detail(self, run_id, **kwargs): return {"lineage": lineage}
    reader._runs = FakeRuns()
    verified = {"model-merged": {"source_run_id": "run-training", "produced_by_run_id": "run-merge"}}
    with patch("onegpu_posttrain.ui.model_playground_catalog.ModelPlaygroundCatalog._verified_sources", return_value=verified), \
         patch.object(reader, "_project_run", return_value=[]):
        refs = reader._search_model_lineage({"model-merged"}, 1)
    facts = {item.key: item.value for item in refs[0].facts}
    assert refs[0].status == "available"
    assert facts["dataset_ids"] == "ds-fixture" and facts["merge_run_id"] == "run-merge"
