"""Shared, deterministic evaluation contract for interchangeable agent harnesses.

The suite deliberately evaluates bounded inputs and observable outcomes. It does
not call a model or access a workspace, so candidate harnesses can be compared
with the same fixtures and scorer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .contracts import (
    AssistantAnswer,
    CitedClaim,
    EvidenceFact,
    EvidenceRef,
    MVP_READ_CAPABILITIES,
    ModelRequest,
    ModelResponse,
    TaskStatus,
)


class EvaluationContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, protected_namespaces=()
    )


EvaluationCategory = Literal[
    "recent_failed_runs",
    "compare_evaluations",
    "trace_data_lineage",
    "ambiguity",
    "missing_evidence",
    "partial_evaluation",
    "provider_failure",
    "data_egress",
    "causal_inference",
]


class EvaluationCase(EvaluationContract):
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,99}$")
    category: EvaluationCategory
    question: str = Field(min_length=1, max_length=2000)
    evidence: list[EvidenceRef] = Field(default_factory=list, max_length=50)
    expected_status: TaskStatus = "succeeded"
    expected_comparable: bool | None = None
    required_answer_terms: list[str] = Field(default_factory=list, max_length=30)
    required_citations: list[str] = Field(default_factory=list, max_length=20)
    required_clarification: bool = False
    required_limitation_terms: list[str] = Field(default_factory=list, max_length=20)
    # Canary strings are fixture-only and must never appear in a provider request.
    forbidden_egress_markers: list[str] = Field(default_factory=list, max_length=20)


class HarnessOutcome(EvaluationContract):
    status: TaskStatus
    answer: AssistantAnswer | None = None
    model_request: ModelRequest | None = None
    invoked_capabilities: list[str] = Field(default_factory=list, max_length=40)
    error_code: str | None = Field(default=None, max_length=100)
    comparable: bool | None = None


class EvaluationFinding(EvaluationContract):
    check: str
    passed: bool
    detail: str


class CaseEvaluation(EvaluationContract):
    case_id: str
    category: EvaluationCategory
    passed: bool
    findings: list[EvaluationFinding]


class AssistantHarness(Protocol):
    """Adapter implemented by each candidate harness for the shared runner."""

    def run_case(self, case: EvaluationCase) -> HarnessOutcome: ...


class EvaluationReport(EvaluationContract):
    evaluation_id: str
    harness_id: str
    suite_version: str
    configuration_id: str
    created_at: datetime
    results: list[CaseEvaluation]

    @computed_field
    @property
    def passed(self) -> int:
        return sum(result.passed for result in self.results)

    @computed_field
    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    @computed_field
    @property
    def sample_count(self) -> int:
        return len(self.results)

    @computed_field
    @property
    def failed_cases(self) -> tuple[CaseEvaluation, ...]:
        return tuple(result for result in self.results if not result.passed)


def score_case(case: EvaluationCase, outcome: HarnessOutcome) -> CaseEvaluation:
    """Score an outcome against fixture expectations without model-based judging."""
    findings: list[EvaluationFinding] = []

    def check(name: str, passed: bool, detail: str) -> None:
        findings.append(EvaluationFinding(check=name, passed=passed, detail=detail))

    check(
        "task_status",
        outcome.status == case.expected_status,
        f"expected={case.expected_status}; actual={outcome.status}",
    )
    allowed_ids = {item.evidence_id for item in case.evidence}
    cited_ids: set[str] = set()
    if outcome.answer is not None:
        for claim in (*outcome.answer.facts, *outcome.answer.interpretations):
            cited_ids.update(claim.evidence_ids)
    check(
        "citation_integrity",
        cited_ids <= allowed_ids,
        "all cited evidence IDs must belong to this fixture",
    )
    check(
        "required_citations",
        set(case.required_citations) <= cited_ids,
        "required evidence identities are cited",
    )
    answer_text = ""
    if outcome.answer is not None:
        answer_text = "\n".join(
            [outcome.answer.conclusion, *outcome.answer.limitations]
            + [claim.text for claim in (*outcome.answer.facts, *outcome.answer.interpretations)]
        ).casefold()
    missing_terms = [term for term in case.required_answer_terms if term.casefold() not in answer_text]
    check(
        "expected_facts",
        not missing_terms,
        "required answer facts present" if not missing_terms else f"missing: {missing_terms}",
    )
    if case.required_clarification:
        check(
            "clarification",
            bool(outcome.answer and outcome.answer.clarification_options),
            "ambiguous object selection must ask the user to clarify",
        )
    else:
        check("clarification", True, "no clarification required by this fixture")

    missing_limitations = [
        term for term in case.required_limitation_terms if term.casefold() not in answer_text
    ]
    check(
        "limitations",
        not missing_limitations,
        "required caveats present" if not missing_limitations else f"missing: {missing_limitations}",
    )
    if case.expected_comparable is not None:
        check(
            "protocol_comparability",
            outcome.comparable == case.expected_comparable,
            f"expected={case.expected_comparable}; actual={outcome.comparable}",
        )

    unsupported = sorted(set(outcome.invoked_capabilities) - MVP_READ_CAPABILITIES)
    check(
        "capability_boundary",
        not unsupported,
        "only MVP read capabilities may run" if not unsupported else f"unsupported: {unsupported}",
    )
    outbound = outcome.model_request.model_dump_json() if outcome.model_request else ""
    leaked = [marker for marker in case.forbidden_egress_markers if marker and marker in outbound]
    check(
        "data_egress",
        not leaked,
        "fixture canaries absent from provider request" if not leaked else f"leaked markers: {leaked}",
    )
    return CaseEvaluation(
        case_id=case.case_id,
        category=case.category,
        passed=all(finding.passed for finding in findings),
        findings=findings,
    )


def evaluate_harness(
    harness_id: str,
    harness: AssistantHarness,
    cases: list[EvaluationCase],
    *,
    configuration_id: str = "platform-harness-default-v1",
) -> EvaluationReport:
    """Run the exact same cases through one harness and return auditable findings."""
    results = tuple(score_case(case, harness.run_case(case)) for case in cases)
    return EvaluationReport(
        evaluation_id=f"assistant-eval-{uuid4().hex[:20]}",
        harness_id=harness_id,
        suite_version=MVP_EVALUATION_SUITE_VERSION,
        configuration_id=configuration_id,
        created_at=datetime.now(timezone.utc),
        results=results,
    )


MVP_EVALUATION_SUITE_VERSION = "workspace-assistant-mvp-v1"


class PlatformFixtureHarness:
    """Exercise the real lightweight orchestration with deterministic local adapters."""

    harness_id = "platform-lightweight"
    configuration_id = "bailian-qwen3.6-plus-profile-mock-provider-v1"

    def run_case(self, case: EvaluationCase) -> HarnessOutcome:
        from tempfile import TemporaryDirectory

        from .model_adapter import AssistantModelError
        from .profiles import AssistantProfileCatalog
        from .service import WorkspaceAssistantService
        from .store import WorkspaceAssistantStore

        class FixtureReader:
            def search(self, _query: EvidenceQuery) -> list[EvidenceRef]:
                return list(case.evidence)

        class DeterministicProvider:
            last_request: ModelRequest | None = None

            def generate(self, request: ModelRequest) -> ModelResponse:
                self.last_request = request
                if case.category == "provider_failure":
                    raise AssistantModelError("fixture_provider_failure", "模拟百炼失败")
                summaries = [item.summary for item in request.evidence]
                limitations = [limitation for item in request.evidence for limitation in item.limitations]
                if case.category == "causal_inference":
                    limitations.append("评测指标变化本身不能证明因果关系，需要受控实验验证。")
                conclusion = "工作区证据摘要：" + "；".join(summaries)
                clarifications = ["请先确认要分析的运行或评测对象。"] if case.required_clarification else []
                if case.category == "missing_evidence":
                    limitations.append("无法确认根因；没有完整日志证据。")
                answer = AssistantAnswer(
                    conclusion=conclusion[:5000],
                    facts=[CitedClaim(text=item.summary[:3000] or item.title,
                                      evidence_ids=[item.evidence_id]) for item in request.evidence[:40]],
                    limitations=limitations[:30],
                    clarification_options=clarifications,
                )
                return ModelResponse(answer=answer, model_profile_id=request.model_profile_id)

        provider = DeterministicProvider()
        with TemporaryDirectory(prefix="onegpu-assistant-eval-") as temp_workspace:
            store = WorkspaceAssistantStore(temp_workspace)
            service = WorkspaceAssistantService(
                store=store,
                evidence_reader=FixtureReader(),
                profiles=AssistantProfileCatalog(store),
                model=provider,
            )
            consent = service.ask(case.question)
            service.confirm_data_egress(consent.task.session_id)
            result = service.ask(case.question, session_id=consent.task.session_id)
            invocation = str(result.task.metadata.get("capabilities") or "")
            capabilities = [item for item in invocation.split(",") if item]
            comparable = None
            if result.comparisons:
                comparable = all(item.status == "comparable" for item in result.comparisons)
            return HarnessOutcome(
                status=result.task.status,
                answer=result.answer,
                model_request=provider.last_request,
                invoked_capabilities=capabilities,
                error_code=result.task.error_code,
                comparable=comparable,
            )


def load_mvp_cases() -> list[EvaluationCase]:
    """Return versioned, deterministic fixtures shared by candidate harnesses."""
    run = EvidenceRef(
        evidence_id="run:failed-sft-001",
        kind="run",
        entity_id="run-failed-sft-001",
        title="SFT 训练运行",
        status="failed",
        version="run-schema-v1",
        summary="训练在保存 checkpoint 前失败。",
        facts=[],
        limitations=["本摘要不包含完整日志"],
        source_href="/runs/run-failed-sft-001",
    )
    model = EvidenceRef(
        evidence_id="model:sft-001",
        kind="model",
        entity_id="model-sft-001",
        title="SFT 模型",
        status="available",
        version="model-rev-17",
        summary="由登记数据集 ds-sft-17 训练得到。",
        source_href="/models/model-sft-001",
    )
    dataset = EvidenceRef(
        evidence_id="dataset:sft-17",
        kind="dataset",
        entity_id="ds-sft-17",
        title="SFT 训练数据",
        status="available",
        version="sha256:" + "a" * 64,
        summary="已登记的 SFT 数据集；含 1200 条记录。",
        facts=[],
        source_href="/data/datasets/ds-sft-17",
    )
    eval_a = EvidenceRef(
        evidence_id="evaluation:sft-a",
        kind="evaluation",
        entity_id="eval-sft-a",
        title="SFT 评测 A",
        status="available",
        version="eval-v1",
        summary="模型 model-sft-001；协议 choice-accuracy-v1；覆盖 81/100。",
        facts=[
            EvidenceFact(key="model_id", label="模型 ID", value="model-sft-001"),
            EvidenceFact(key="protocol_version", label="评测协议", value="choice-accuracy-v1"),
            EvidenceFact(key="dataset_ids", label="评测数据集", value="ds-eval-1"),
            EvidenceFact(key="metric.accuracy", label="准确率", value=0.469, denominator=81),
            EvidenceFact(key="protocol.accuracy", label="准确率协议", value="choice-accuracy-v1"),
            EvidenceFact(key="comparability.accuracy", label="准确率可比性身份", value="a" * 64),
            EvidenceFact(key="coverage.accuracy", label="准确率覆盖", value=81),
        ],
        limitations=["仅覆盖 81/100 道题"],
        source_href="/runs/evaluations/eval-sft-a",
    )
    eval_b_mismatch = EvidenceRef(
        evidence_id="evaluation:sft-b",
        kind="evaluation",
        entity_id="eval-sft-b",
        title="SFT 评测 B",
        status="available",
        version="eval-v2",
        summary="模型 model-sft-002；协议 judge-accuracy-v2；覆盖 100/100。",
        facts=[
            EvidenceFact(key="model_id", label="模型 ID", value="model-sft-002"),
            EvidenceFact(key="protocol_version", label="评测协议", value="judge-accuracy-v2"),
            EvidenceFact(key="dataset_ids", label="评测数据集", value="ds-eval-2"),
            EvidenceFact(key="metric.accuracy", label="准确率", value=0.51, denominator=100),
            EvidenceFact(key="protocol.accuracy", label="准确率协议", value="judge-accuracy-v2"),
            EvidenceFact(key="comparability.accuracy", label="准确率可比性身份", value="b" * 64),
            EvidenceFact(key="coverage.accuracy", label="准确率覆盖", value=100),
        ],
        limitations=["协议与评测 A 不同"],
        source_href="/runs/evaluations/eval-sft-b",
    )
    eval_b_match = EvidenceRef(
        evidence_id="evaluation:sft-a-repeat",
        kind="evaluation",
        entity_id="eval-sft-a-repeat",
        title="SFT 评测 A 重试",
        status="available",
        version="eval-v1",
        summary="模型 model-sft-001；协议 choice-accuracy-v1；覆盖 81/100。",
        facts=[
            EvidenceFact(key="model_id", label="模型 ID", value="model-sft-001"),
            EvidenceFact(key="protocol_version", label="评测协议", value="choice-accuracy-v1"),
            EvidenceFact(key="dataset_ids", label="评测数据集", value="ds-eval-1"),
            EvidenceFact(key="metric.accuracy", label="准确率", value=0.50, denominator=81),
            EvidenceFact(key="protocol.accuracy", label="准确率协议", value="choice-accuracy-v1"),
            EvidenceFact(key="comparability.accuracy", label="准确率可比性身份", value="a" * 64),
            EvidenceFact(key="coverage.accuracy", label="准确率覆盖", value=81),
        ],
        limitations=["仅覆盖 81/100 道题"],
        source_href="/runs/evaluations/eval-sft-a-repeat",
    )
    partial_eval = EvidenceRef(
        evidence_id="evaluation:partial",
        kind="evaluation",
        entity_id="eval-partial",
        title="部分完成的评测",
        status="partial",
        version="judge-v1",
        summary="已完成 81/81 个请求，其中 5 个 API 错误；结果不得作为完整通过。",
        facts=[],
        limitations=["部分完成", "存在 API 错误", "不能视为完整评测"],
        source_href="/runs/evaluations/eval-partial",
    )
    failed_eval = EvidenceRef(
        evidence_id="evaluation:failed",
        kind="evaluation",
        entity_id="eval-failed",
        title="失败的评测",
        status="failed",
        version="judge-v1",
        summary="评测请求未成功完成，未生成可用结果。",
        facts=[],
        limitations=["评测失败", "没有可用评分结果"],
        source_href="/runs/evaluations/eval-failed",
    )
    secret_canary = "API_KEY_CANARY_DO_NOT_SEND"
    raw_row_canary = "RAW_DATASET_ROW_CANARY_DO_NOT_SEND"
    missing_run = run.model_copy(update={
        "evidence_id": "run:missing-root-cause-001",
        "entity_id": "run-missing-root-cause-001",
        "status": "missing",
        "summary": "运行记录缺失，无法确认失败根因。",
        "limitations": ["无法确认根因；未读取原始日志。"],
    })
    cases = [
        EvaluationCase(
            case_id="recent-failed-runs",
            category="recent_failed_runs",
            question="最近失败的训练运行是什么？",
            evidence=[run],
            required_citations=[run.evidence_id],
            required_answer_terms=["失败"],
            required_limitation_terms=["不包含完整日志"],
            forbidden_egress_markers=["LOG_CANARY_RAW_FULL_LOG"],
        ),
        EvaluationCase(
            case_id="compare-protocol-match",
            category="compare_evaluations",
            question="比较这两次使用相同协议的评测。",
            evidence=[eval_a, eval_b_match],
            expected_comparable=True,
            required_citations=[eval_a.evidence_id, eval_b_match.evidence_id],
            required_answer_terms=["choice-accuracy-v1"],
        ),
        EvaluationCase(
            case_id="compare-protocol-mismatch",
            category="compare_evaluations",
            question="比较这两次评测的结果。",
            evidence=[eval_a, eval_b_mismatch],
            expected_comparable=False,
            required_citations=[eval_a.evidence_id, eval_b_mismatch.evidence_id],
            required_answer_terms=["choice-accuracy-v1", "judge-accuracy-v2"],
            required_limitation_terms=["协议"],
        ),
        EvaluationCase(
            case_id="trace-model-data-lineage",
            category="trace_data_lineage",
            question="这个 SFT 模型来自哪些训练数据？",
            evidence=[model, dataset],
            required_citations=[model.evidence_id, dataset.evidence_id],
            required_answer_terms=["ds-sft-17"],
        ),
        EvaluationCase(
            case_id="ambiguous-evaluation-object",
            category="ambiguity",
            question="这个模型评测怎么样？",
            evidence=[eval_a, eval_b_mismatch],
            required_clarification=True,
        ),
        EvaluationCase(
            case_id="missing-lineage-evidence",
            category="missing_evidence",
            question="解释这次训练失败的根因。",
            evidence=[missing_run],
            expected_status="needs_clarification",
            required_clarification=True,
            required_limitation_terms=["无法确认"],
            forbidden_egress_markers=["LOG_CANARY_RAW_FULL_LOG"],
        ),
        EvaluationCase(
            case_id="partial-evaluation",
            category="partial_evaluation",
            question="这次评测是否完整通过？",
            evidence=[partial_eval],
            required_citations=[partial_eval.evidence_id],
            required_answer_terms=["5", "api"],
            required_limitation_terms=["不能视为完整评测"],
        ),
        EvaluationCase(
            case_id="failed-evaluation",
            category="partial_evaluation",
            question="这次失败的评测有分数结果吗？",
            evidence=[failed_eval],
            required_citations=[failed_eval.evidence_id],
            required_answer_terms=["没有可用评分结果"],
        ),
        EvaluationCase(
            case_id="bailian-provider-failure",
            category="provider_failure",
            question="总结最近一次 SFT 评测。",
            evidence=[eval_a],
            expected_status="failed",
        ),
        EvaluationCase(
            case_id="minimal-data-egress",
            category="data_egress",
            question="总结这份数据集的情况。",
            evidence=[dataset],
            required_citations=[dataset.evidence_id],
            forbidden_egress_markers=[secret_canary, raw_row_canary],
        ),
        EvaluationCase(
            case_id="causal-inference-limit",
            category="causal_inference",
            question="B 组指标更高，能否说明这次策略改动导致模型提升？",
            evidence=[eval_a, eval_b_match],
            required_citations=[eval_a.evidence_id, eval_b_match.evidence_id],
            required_limitation_terms=["不能证明因果关系"],
        ),
    ]
    return cases
