"""Container-based practice and final scoring with pinned comparison identity."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ..lineage import (
    ArtifactRef,
    Digest,
    EvaluationLineage,
    RunRecord,
    new_run_id,
    sha256_directory,
    sha256_file,
    sha256_json,
)
from .manifest import DOCKER_IMAGE, MODEL_REVISION
from .registry import ResolvedInputs
from .runtime import docker_host_path, docker_prefix

EvaluationRole = Literal["practice", "final_holdout"]


def comparison_identity(inputs: ResolvedInputs, dataset_sha256: str) -> str:
    return sha256_json({"dataset_sha256": dataset_sha256, "sample_count": (
        inputs.manifest.practice.record_count if dataset_sha256 == inputs.manifest.practice.canonical_sha256
        else inputs.manifest.final_holdout.record_count),
        "model_revision": MODEL_REVISION, "processor_sha256": inputs.manifest.target.processor_sha256,
        "runtime_image": DOCKER_IMAGE, "runtime_torch": inputs.manifest.runtime.torch_version,
        "runtime_transformers": inputs.manifest.runtime.transformers_version,
        "inference": inputs.manifest.inference.model_dump(mode="json")})


def compare_final_results(baseline: dict, candidate: dict) -> dict:
    """Compare only the same pinned holdout, runtime, processor, and coverage."""
    fields = ("comparison_identity", "dataset_id", "dataset_sha256", "sample_count",
              "protocol", "model_revision", "processor_sha256", "runtime_image", "inference")
    if (baseline.get("role") != "final_holdout" or candidate.get("role") != "final_holdout"
            or any(baseline.get(key) != candidate.get(key) for key in fields)):
        raise ValueError("baseline and candidate evaluation identities are not comparable")
    base = baseline["metrics"]["overall"]
    new = candidate["metrics"]["overall"]
    if (base["denominator"] != baseline["sample_count"]
            or new["denominator"] != candidate["sample_count"]):
        raise ValueError("baseline and candidate sample coverage differs")
    return {"baseline_accuracy": base["accuracy"], "candidate_accuracy": new["accuracy"],
            "delta_accuracy": new["accuracy"] - base["accuracy"],
            "denominator": base["denominator"],
            "comparison_identity": baseline["comparison_identity"]}


def evaluate_in_docker(workspace: str | Path, inputs: ResolvedInputs, *, role: EvaluationRole,
                       model_path: str | Path, model_sha256: str, output_dir: str | Path,
                       candidate_id: str | None = None, iteration: int | None = None,
                       parent_sha256: str | None = None, timeout_seconds: int = 900) -> dict:
    root = Path(workspace).resolve(strict=True)
    model = Path(model_path).resolve(strict=True)
    dataset = inputs.practice_path if role == "practice" else inputs.final_path
    identity = inputs.manifest.practice if role == "practice" else inputs.manifest.final_holdout
    output = Path(output_dir).resolve(strict=False)
    if not output.is_relative_to(root / "runs" / "weight-space") or output.exists():
        raise ValueError("evaluation output must be a new weight-space run directory")
    if sha256_directory(model) != model_sha256 or sha256_file(dataset) != identity.canonical_sha256:
        raise ValueError("model or evaluation dataset digest changed")
    output.mkdir(parents=True, exist_ok=False)
    mounts = [(root / "src", "/workspace/src", True), (model, "/model", True),
              (dataset, "/suite/eval.jsonl", True), (output, "/output", False)]
    command = [*docker_prefix(), "run", "--rm", "--network", "none", "--gpus", "all",
               "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=512m",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "256"]
    for source, target, read_only in mounts:
        command.extend(["--mount", f"type=bind,source={docker_host_path(source)},target={target}"
                        + (",readonly" if read_only else "")])
    command.extend(["--env", "PYTHONPATH=/workspace/src", "--env", "TRANSFORMERS_OFFLINE=1",
                    "--env", "HF_HUB_OFFLINE=1", DOCKER_IMAGE, "python", "-m",
                    "onegpu_posttrain.weight_space.docker_choice_worker"])
    result = subprocess.run(command, capture_output=True, timeout=timeout_seconds, check=False)
    if result.returncode:
        (output / "failure.json").write_text(json.dumps({"status": "failed", "exit_code": result.returncode}),
                                              encoding="utf-8")
        raise RuntimeError(f"Qwen3.5 Docker evaluation failed with exit code {result.returncode}")
    report = json.loads((output / "choice_metrics.json").read_text(encoding="utf-8"))
    if (report.get("sample_count") != identity.record_count or report.get("protocol") != "choice-accuracy-v1"
            or report.get("runtime", {}).get("torch") != inputs.manifest.runtime.torch_version
            or report.get("runtime", {}).get("transformers") != inputs.manifest.runtime.transformers_version):
        raise RuntimeError("Docker evaluator result does not match pinned input and runtime")
    lineage = {"schema_version": "1.0", "experiment_id": inputs.manifest.experiment_id,
               "role": role, "iteration": iteration, "candidate_id": candidate_id,
               "model_id": inputs.manifest.target.model_id, "model_revision": MODEL_REVISION,
               "model_sha256": model_sha256, "parent_sha256": parent_sha256,
               "dataset_id": identity.dataset_id, "dataset_sha256": identity.canonical_sha256,
               "sample_count": identity.record_count, "protocol": "choice-accuracy-v1",
               "comparison_identity": comparison_identity(inputs, identity.canonical_sha256),
               "processor_sha256": inputs.manifest.target.processor_sha256,
               "runtime_image": DOCKER_IMAGE, "runtime": report["runtime"],
               "inference": inputs.manifest.inference.model_dump(mode="json"),
               "metrics": report["metrics"]}
    lineage["metric_statuses"] = {
        f"choice_accuracy.{kind}": {"value": item["accuracy"], "correct": item["correct"],
                                     "denominator": item["denominator"], "denominator_unit": "questions",
                                     "parse_failures": item["parse_failures"], "status": item["status"]}
        for kind, item in report["metrics"].items()
    }
    run_id = new_run_id("evaluate", label=f"weight-space-{role}")
    eval_id = f"eval-{run_id.removeprefix('run-')}"
    registered_model_id = (f"model-{candidate_id.removeprefix('candidate-')}"
                           if candidate_id else inputs.manifest.target.model_id)
    normalized_metrics = {
        f"choice_accuracy.{kind}": {"value": item["accuracy"], "correct": item["correct"],
                                     "denominator": item["denominator"],
                                     "parse_failures": item["parse_failures"]}
        for kind, item in report["metrics"].items()
    }
    recorded_at = datetime.now(timezone.utc).isoformat()
    typed_lineage = EvaluationLineage(
        eval_id=eval_id, produced_by_run_id=run_id,
        model_id=registered_model_id,
        dataset_ids=[identity.dataset_id],
        dataset_artifacts=[ArtifactRef(id=identity.dataset_id, kind="dataset", role=role,
                                       digest=Digest(status="available", value=identity.canonical_sha256))],
        protocol_version="choice-accuracy-v1", config_sha256=lineage["comparison_identity"],
        status="completed", metrics=normalized_metrics,
        experiment_id=inputs.manifest.experiment_id, iteration=iteration, dataset_role=role,
        candidate_id=candidate_id, candidate_sha256=model_sha256,
        parent_candidate_sha256=parent_sha256, target_model_revision=MODEL_REVISION,
        processor_sha256=inputs.manifest.target.processor_sha256,
        runtime_image=DOCKER_IMAGE,
        runtime_versions={"torch": report["runtime"]["torch"],
                          "transformers": report["runtime"]["transformers"]},
        inference_config=inputs.manifest.inference.model_dump(mode="json"),
        comparison_identity=lineage["comparison_identity"],
    )
    code_revision = {"status": "available", "files": {
        "docker_choice_worker.py": sha256_file(root / "src/onegpu_posttrain/weight_space/docker_choice_worker.py"),
        "choice_accuracy.py": sha256_file(root / "src/onegpu_posttrain/evaluation/choice_accuracy.py"),
    }}
    run_record = RunRecord(
        run_id=run_id, run_type="evaluate", status="completed", started_at=recorded_at,
        completed_at=recorded_at, experiment_id=inputs.manifest.experiment_id,
        run_label=f"weight-space {role}", code_revision=code_revision,
        inputs=[ArtifactRef(id=registered_model_id,
                            kind="model", digest=Digest(status="available", value=model_sha256)),
                ArtifactRef(id=identity.dataset_id, kind="dataset", role=role,
                            digest=Digest(status="available", value=identity.canonical_sha256))],
        outputs=[ArtifactRef(id=eval_id, kind="evaluation", path=str(output.relative_to(root)))],
    )
    (output / "evaluation_lineage.json").write_text(typed_lineage.model_dump_json(indent=2) + "\n",
                                                     encoding="utf-8")
    (output / "evaluation_run.json").write_text(run_record.model_dump_json(indent=2) + "\n",
                                                 encoding="utf-8")
    (output / "eval.json").write_text(json.dumps(lineage, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    return lineage
