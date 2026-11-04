"""Bounded workspace evidence tools for the AgentScope assistant."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from ..ui.run_workspace import RunWorkspace
from .comparison import compare_evaluations
from .contracts import EvidenceQuery, EvidenceRef
from .evidence_reader import WorkspaceEvidenceReader

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_STAGES = {"base", "cpt", "sft", "dpo", "simpo", "grpo"}
_STATUSES = {"completed", "failed", "partial", "running", "interrupted", "oom"}
_LOG = logging.getLogger(__name__)


class WorkspaceAgentTools:
    """Expose only typed projections, with no arbitrary path or mutation operations."""

    def __init__(self, workspace: Path, reader: WorkspaceEvidenceReader | None = None):
        self.workspace = Path(workspace).resolve(strict=True)
        self.runs = RunWorkspace(self.workspace)
        self.reader = reader or WorkspaceEvidenceReader(self.workspace)
        self.evidence: dict[str, EvidenceRef] = {}
        self.comparisons = []
        self.calls = 0
        self.max_calls = 12
        self.tool_names: set[str] = set()
        self.tool_duration_ms = 0

    def _tick(self) -> None:
        self.calls += 1
        if self.calls > self.max_calls:
            raise ValueError("本轮工具调用达到上限，请缩小问题范围。")

    @staticmethod
    def _id(value: str) -> str:
        if not _ID.fullmatch(value):
            raise ValueError("对象 ID 格式无效。")
        return value

    def _remember(self, refs: list[EvidenceRef]) -> list[dict]:
        result = []
        for ref in refs:
            self.evidence[ref.evidence_id] = ref
            result.append({
                "evidence_id": ref.evidence_id, "kind": ref.kind, "entity_id": ref.entity_id,
                "title": ref.title, "status": ref.status, "summary": ref.summary,
                "facts": [{k: v for k, v in fact.model_dump(mode="json", exclude_none=True).items() if k != "label"}
                          for fact in ref.facts],
                "limitations": ref.limitations, "source_href": ref.source_href,
            })
        return result

    def search_runs(self, stage: str = "", status: str = "", kind: str = "", offset: int = 0, limit: int = 3) -> dict:
        self._tick()
        stage, status, kind = stage.lower().strip(), status.lower().strip(), kind.lower().strip()
        if stage and stage not in _STAGES or status and status not in _STATUSES:
            raise ValueError("不支持的阶段或状态。")
        if kind not in {"", "training", "evaluation", "merge"} or offset < 0 or offset > 200:
            raise ValueError("不支持的运行类型或分页。")
        limit = max(1, min(int(limit), 3))
        rows = self.runs.list_runs(stage=stage or None, status=status or None, limit=200)
        matching = [row for row in rows["runs"] if not kind or row.get("run_kind") == kind]
        safe = [{key: row.get(key) for key in ("run_id", "run_label", "run_kind", "stage", "status", "started_at")}
                for row in matching[offset:offset + limit]]
        return {"runs": safe, "offset": offset, "returned": len(safe), "more": offset + len(safe) < len(matching),
                "partial_scan": bool(rows.get("partial_scan") or rows.get("total", 0) > 200)}

    def read_run(self, run_id: str, section: str = "summary") -> list[dict]:
        self._tick()
        run_id = self._id(run_id)
        sections = {"summary": {"run", "log"}, "configuration": {"configuration"},
                    "lineage": {"lineage"}, "log": {"log"}}
        if section not in sections:
            raise ValueError("不支持的运行详情类型。")
        refs = self.reader.search(EvidenceQuery(kind="general", query=run_id, selected_entity_ids=[run_id], limit=1))
        return self._remember([r for r in refs if r.kind in sections[section]])

    def read_evaluation(self, run_id: str) -> list[dict]:
        self._tick()
        run_id = self._id(run_id)
        refs = self.reader.search(EvidenceQuery(kind="compare_evaluations", query=run_id, selected_entity_ids=[run_id], limit=1))
        return self._remember([r for r in refs if r.kind == "evaluation"])

    def read_lineage(self, entity_id: str) -> list[dict]:
        self._tick()
        entity_id = self._id(entity_id)
        refs = self.reader.search(EvidenceQuery(kind="trace_data_lineage", query=entity_id, selected_entity_ids=[entity_id], limit=1))
        return self._remember([r for r in refs if r.kind in {"run", "lineage", "model", "dataset", "quality_report"}])

    def compare(self, left_id: str, right_id: str) -> dict:
        self._tick()
        left_id, right_id = self._id(left_id), self._id(right_id)
        if left_id == right_id:
            raise ValueError("请选择两次不同的评测。")
        refs = []
        for entity_id in (left_id, right_id):
            matches = self.reader.search(EvidenceQuery(kind="compare_evaluations", query=entity_id, selected_entity_ids=[entity_id], limit=1))
            refs.extend(r for r in matches if r.kind == "evaluation")
        if len(refs) != 2:
            return {"status": "missing", "reason": "未找到两次完整的评测记录。", "evidence": self._remember(refs)}
        comparison = compare_evaluations(refs[0], refs[1])
        self.comparisons.append(comparison)
        self._remember(refs)
        return {"comparison": comparison.model_dump(mode="json"), "evidence": [
            {"evidence_id": ref.evidence_id, "entity_id": ref.entity_id, "status": ref.status,
             "title": ref.title, "source_href": ref.source_href} for ref in refs
        ]}

    def investigate_workspace(self, question: str) -> dict:
        """Route one read-only investigation through bounded local evidence readers."""
        self._tick()
        self.tool_names.add("investigate_workspace")
        from .service import classify_question

        kind = classify_question(question)
        selected = sorted(set(re.findall(
            r"(?<![A-Za-z0-9])(?:run|eval|model|ds|dataset)-[A-Za-z0-9][A-Za-z0-9._-]{0,158}"
            r"[A-Za-z0-9](?![A-Za-z0-9._-])", question, re.IGNORECASE,
        )))[:20]
        if kind == "compare_evaluations":
            refs = self.reader.search(EvidenceQuery(
                kind="compare_evaluations", query=question, selected_entity_ids=selected, limit=50,
            ))
            evaluations = [ref for ref in refs if ref.kind == "evaluation"]
            text = question.casefold()
            requested = next((markers for markers in (
                ("sft", "监督微调"), ("cpt", "预训练"), ("dpo", "偏好优化"), ("grpo", "强化学习"),
            ) if any(marker in text for marker in markers)), None)
            if requested:
                evaluations = [ref for ref in evaluations if any(
                    marker in f"{ref.title} {ref.entity_id}".casefold() for marker in requested
                )]
            # The comparison contract treats left as the baseline and right as
            # the newer result. The reader returns newest first, so reverse the
            # selected pair before computing deltas.
            evaluations = sorted(
                evaluations,
                key=lambda ref: next((str(fact.value) for fact in ref.facts if fact.key == "started_at"), ""),
                reverse=True,
            )[:2]
            if len(evaluations) == 2:
                baseline, latest = evaluations[1], evaluations[0]
                comparison = compare_evaluations(baseline, latest)
                self.comparisons.append(comparison)
            # Do not retain every search candidate in the model context. Only
            # the selected pair is relevant and has passed stage filtering.
            return {"kind": kind, "comparison": self.comparisons[-1].model_dump(mode="json") if len(evaluations) == 2 else None,
                    "evidence": self._remember(evaluations),
                    "needs_clarification": len(evaluations) < 2}

        if kind == "trace_data_lineage":
            refs = self.reader.search(EvidenceQuery(
                kind=kind, query=question, selected_entity_ids=selected, limit=10,
            ))
            return {"kind": kind, "evidence": self._remember(refs)}

        if selected:
            refs = self.reader.search(EvidenceQuery(
                kind=kind, query=question, selected_entity_ids=selected, limit=10,
            ))
            return {"kind": kind, "evidence": self._remember(refs)}

        stage = next((name for name in ("cpt", "sft", "dpo", "simpo", "grpo", "base")
                      if name in question.casefold()), "")
        status = "failed" if kind == "recent_failed_runs" else ""
        run_kind = "evaluation" if "评测" in question or "evaluation" in question.casefold() else ""
        runs = self.search_runs(stage=stage, status=status, kind=run_kind, limit=3)
        refs = []
        for run in runs["runs"]:
            run_id = run.get("run_id")
            if not isinstance(run_id, str):
                continue
            refs.extend(self.read_evaluation(run_id) if run.get("run_kind") == "evaluation"
                        else self.read_run(run_id))
        return {"kind": kind, "evidence": self._remember(refs), "more": runs.get("more", False)}

    def toolkit(self):
        """Build an explicit allowlist; never register AgentScope built-in tools."""
        from agentscope.message import TextBlock
        from agentscope.tool import Toolkit, ToolResponse

        async def invoke(method, *args):
            started = time.perf_counter()
            try:
                result = await asyncio.to_thread(method, *args)
                text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
            except (ValueError, TypeError):
                text = json.dumps({"error": "工具参数无效或调用次数超限。"}, ensure_ascii=False)
            except Exception:
                text = json.dumps({"error": "工作区记录读取失败，请缩小范围或重试。"}, ensure_ascii=False)
            finally:
                self.tool_names.add(method.__name__)
                self.tool_duration_ms += int((time.perf_counter() - started) * 1000)
            _LOG.info("assistant tool %s result_chars=%d", method.__name__, len(text))
            return ToolResponse(content=[TextBlock(type="text", text=text)])

        async def investigate_workspace(question: str) -> ToolResponse:
            """根据用户问题在工作区只读搜索运行、评测结果或数据/模型血缘。

            Args:
                question: 用户的实验问题原文，用于确定要查询的工作区证据。
            """
            return await invoke(self.investigate_workspace, question)

        toolkit = Toolkit()
        toolkit.register_tool_function(investigate_workspace)
        return toolkit
