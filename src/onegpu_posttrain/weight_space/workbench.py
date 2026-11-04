"""Loopback workbench orchestration for one confirmed editor iteration."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..synthetic.secrets import load_local_provider_credential
from .candidate_registry import register_candidate
from .candidates import (
    apply_candidate_edit,
    candidate_state,
    require_finalizable_candidate,
    transition_candidate,
)
from .editor_loop import run_editor_iteration
from .evaluation import compare_final_results, evaluate_in_docker
from .manifest import MODEL_ID
from .provider_profile import load_editor_profile
from .registry import resolve_inputs
from .tool_provider import ToolCall
from .weight_tools import dispatch_weight_tool

EXPERIMENT_ID = "exp-weight-space-baseline01"
BASELINE_ROOT = Path("runs/weight-space/exp-weight-space-baseline01/evaluations")


class WeightSpaceWorkbench:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self._lock = threading.RLock()
        self._confirmation: str | None = None
        self._confirmation_expires_at = 0.0
        self._job: dict | None = None

    def _baseline(self, role: str) -> tuple[Path, dict]:
        name = "practice-baseline-v2-20261103" if role == "practice" else "final-baseline-v2-20261103"
        path = self.workspace / BASELINE_ROOT / name
        report = json.loads((path / "eval.json").read_text(encoding="utf-8"))
        if report["role"] != role or report["metrics"]["overall"]["status"] != "measured":
            raise ValueError("pinned baseline report is incomplete")
        return path, report

    def _candidate_rows(self) -> list[dict]:
        root = self.workspace / "runs" / "weight-space" / EXPERIMENT_ID / "candidates"
        rows = []
        for path in sorted(root.glob("candidate-*/edit.json")):
            candidate_id = path.parent.name
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                state = candidate_state(self.workspace, EXPERIMENT_ID, candidate_id)
                evaluations = self.workspace / "runs" / "weight-space" / EXPERIMENT_ID / "evaluations"
                practice_path = evaluations / f"{candidate_id}-practice" / "eval.json"
                final_path = evaluations / f"{candidate_id}-final" / "eval.json"
                practice = json.loads(practice_path.read_text(encoding="utf-8")) if practice_path.exists() else None
                final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else None
                rows.append({"candidate_id": candidate_id, "model_id": f"model-{candidate_id[10:]}",
                             "candidate_sha256": record["candidate_sha256"], "parent_sha256": record["parent_sha256"],
                             "tensor_name": record["tensor_name"], "changed_elements": record["changed_elements"],
                             "state": state["state"], "practice": practice, "final": final})
            except (ValueError, OSError, KeyError, json.JSONDecodeError):
                rows.append({"candidate_id": candidate_id, "state": "invalid"})
        return rows

    def snapshot(self) -> dict:
        with self._lock:
            job = dict(self._job) if self._job else None
        try:
            inputs = resolve_inputs(self.workspace, EXPERIMENT_ID)
            profile = load_editor_profile(self.workspace)
            load_local_provider_credential(profile.api_key_env or "", self.workspace / ".env.local")
            credential_ready = bool(os.environ.get(profile.api_key_env or ""))
            practice_path, practice = self._baseline("practice")
            _, final = self._baseline("final_holdout")
            return {"ready": True, "experiment": inputs.manifest.model_dump(mode="json"),
                    "provider": {"model": profile.model, "profile_id": profile.profile_id,
                                 "credential_ready": credential_ready, "pricing": "unknown",
                                 "max_provider_calls": 4, "max_tool_calls": 8,
                                 "max_output_tokens_per_call": 512},
                    "baseline": {"practice": practice, "final": final},
                    "candidates": self._candidate_rows(), "job": job,
                    "practice_predictions_available": (practice_path / "choice_predictions.jsonl").is_file()}
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
            return {"ready": False, "reason": str(exc)[:240], "job": job,
                    "candidates": self._candidate_rows()}

    def preflight(self) -> dict:
        state = self.snapshot()
        if not state["ready"] or not state["provider"]["credential_ready"] or not state["practice_predictions_available"]:
            raise ValueError("model, dataset, baseline, or editor credential is not ready")
        with self._lock:
            if self._job and self._job["status"] == "running":
                raise ValueError("an experiment action is already running")
            self._confirmation = uuid4().hex
            self._confirmation_expires_at = time.monotonic() + 300
            return {"confirmation_token": self._confirmation,
                    "provider_model": state["provider"]["model"],
                    "max_provider_calls": 4, "max_tool_calls": 8,
                    "max_output_tokens_per_call": 512, "pricing": "unknown",
                    "egress": "Up to 30 practice questions, their answers and model outputs, plus bounded tensor values, are sent to Bailian. Final-holdout questions are excluded."}

    def _start(self, kind: str, worker, *, confirmation_token: str | None = None) -> dict:
        with self._lock:
            if self._job and self._job["status"] == "running":
                raise ValueError("an experiment action is already running")
            if kind == "iteration":
                if (not confirmation_token or confirmation_token != self._confirmation
                        or time.monotonic() > self._confirmation_expires_at):
                    raise ValueError("this iteration needs a fresh confirmation")
                self._confirmation = None
                self._confirmation_expires_at = 0.0
            self._job = {"id": uuid4().hex, "kind": kind, "status": "running"}
            job_id = self._job["id"]

        def run() -> None:
            try:
                result = worker()
                update = {"status": "completed", "result": result}
            except Exception as exc:  # noqa: BLE001 - worker failures must reach the status view
                update = {"status": "failed", "error_category": type(exc).__name__}
            with self._lock:
                if self._job and self._job["id"] == job_id:
                    self._job.update(update)

        threading.Thread(target=run, name=f"weight-space-{kind}", daemon=True).start()
        return dict(self._job)

    def start_iteration(self, confirmation_token: str) -> dict:
        return self._start("iteration", self._run_iteration, confirmation_token=confirmation_token)

    def _run_iteration(self) -> dict:
        inputs = resolve_inputs(self.workspace, EXPERIMENT_ID)
        profile = load_editor_profile(self.workspace)
        load_local_provider_credential(profile.api_key_env or "", self.workspace / ".env.local")
        practice_path, _ = self._baseline("practice")
        rows = [json.loads(line) for line in (practice_path / "choice_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        wrong = [row for row in rows if not row["correct"]]
        selected = (wrong or rows)[:20]
        predictions = {row["id"]: row["raw_output"] for row in selected}
        proposals: list[ToolCall] = []

        def execute(call: ToolCall) -> dict:
            result = dispatch_weight_tool(call, inputs.model_path, inputs.manifest.target.checkpoint_sha256)
            if call.name == "propose_edit":
                if proposals:
                    raise ValueError("one edit per iteration")
                proposals.append(call)
            return result

        result = run_editor_iteration(inputs, predictions, profile, execute,
                                      parent_sha256=inputs.manifest.target.checkpoint_sha256)
        iterations = self.workspace / "runs" / "weight-space" / EXPERIMENT_ID / "iterations"
        iterations.mkdir(parents=True, exist_ok=True)
        index = len(list(iterations.glob("iteration-*/record.json"))) + 1
        record_dir = iterations / f"iteration-{index:04d}"
        record_dir.mkdir(exist_ok=False)
        record_path = record_dir / "record.json"
        record = {"schema_version": "1.0", "experiment_id": EXPERIMENT_ID, "iteration": index,
                  "provider_profile_id": profile.profile_id, "provider_model": profile.model,
                  "parent_model_id": MODEL_ID, "parent_sha256": inputs.manifest.target.checkpoint_sha256,
                  "practice_dataset_id": inputs.manifest.practice.dataset_id,
                  "practice_dataset_sha256": inputs.manifest.practice.canonical_sha256,
                  "provider_calls": result.provider_calls, "tool_calls": result.tool_calls,
                  "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
                  "events": result.events, "error_category": result.error_category,
                  "recorded_at": datetime.now(timezone.utc).isoformat()}

        def save(status: str, **fields: object) -> None:
            payload = {**record, "status": status, **fields}
            record_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if result.status != "completed" or not proposals:
            save(result.status if result.status == "failed" else "no_edit")
            return {"status": result.status if result.status == "failed" else "no_edit",
                    "error_category": result.error_category, "provider_calls": result.provider_calls,
                    "tool_calls": result.tool_calls, "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens, "events": result.events}
        artifact = apply_candidate_edit(self.workspace, EXPERIMENT_ID, inputs.model_path,
                                        inputs.manifest.target.checkpoint_sha256, proposals[0])
        save("candidate_created", candidate_id=artifact.candidate_id,
             candidate_sha256=artifact.candidate_sha256, edit_artifact_path=str(artifact.metadata_path))
        model = register_candidate(self.workspace, artifact, MODEL_ID)
        output = self.workspace / "runs" / "weight-space" / EXPERIMENT_ID / "evaluations" / f"{artifact.candidate_id}-practice"
        try:
            practice = evaluate_in_docker(self.workspace, inputs, role="practice", model_path=artifact.model_path,
                                          model_sha256=artifact.candidate_sha256, output_dir=output,
                                          candidate_id=artifact.candidate_id, iteration=index,
                                          parent_sha256=artifact.parent_sha256)
        except Exception:
            save("practice_failed", candidate_id=artifact.candidate_id,
                 candidate_sha256=artifact.candidate_sha256, edit_artifact_path=str(artifact.metadata_path),
                 practice_output_path=str(output))
            raise
        save("completed", candidate_id=artifact.candidate_id, candidate_sha256=artifact.candidate_sha256,
             model_id=model.model_id, edit_artifact_path=str(artifact.metadata_path),
             practice_output_path=str(output), practice_accuracy=practice["metrics"]["overall"]["accuracy"])
        return {"status": "completed", "candidate_id": artifact.candidate_id,
                "model_id": model.model_id, "provider_calls": result.provider_calls,
                "tool_calls": result.tool_calls, "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "practice_accuracy": practice["metrics"]["overall"]["accuracy"]}

    def transition(self, candidate_id: str, action: str) -> dict:
        with self._lock:
            if self._job and self._job["status"] == "running":
                raise ValueError("an experiment action is already running")
        return transition_candidate(self.workspace, EXPERIMENT_ID, candidate_id, action)

    def start_final(self, candidate_id: str) -> dict:
        artifact = require_finalizable_candidate(self.workspace, EXPERIMENT_ID, candidate_id)
        evaluations = self.workspace / "runs" / "weight-space" / EXPERIMENT_ID / "evaluations"
        practice_path = evaluations / f"{candidate_id}-practice" / "eval.json"
        if not practice_path.is_file():
            raise ValueError("candidate practice evaluation must finish first")
        practice = json.loads(practice_path.read_text(encoding="utf-8"))
        if practice.get("role") != "practice" or practice.get("metrics", {}).get("overall", {}).get("status") != "measured":
            raise ValueError("candidate practice evaluation is incomplete")
        if (evaluations / f"{candidate_id}-final").exists():
            raise ValueError("candidate final evaluation already exists")

        def worker() -> dict:
            inputs = resolve_inputs(self.workspace, EXPERIMENT_ID)
            final = evaluate_in_docker(self.workspace, inputs, role="final_holdout",
                                       model_path=artifact.model_path, model_sha256=artifact.candidate_sha256,
                                       output_dir=evaluations / f"{candidate_id}-final",
                                       candidate_id=candidate_id, iteration=next((int(p.parent.name.split("-")[1]) for p in
                                           (self.workspace / "runs" / "weight-space" / EXPERIMENT_ID / "iterations").glob("iteration-*/record.json")
                                           if json.loads(p.read_text(encoding="utf-8")).get("candidate_id") == candidate_id), None),
                                       parent_sha256=artifact.parent_sha256)
            _, baseline = self._baseline("final_holdout")
            return {"candidate_id": candidate_id, "comparison": compare_final_results(baseline, final)}

        return self._start("final", worker)
