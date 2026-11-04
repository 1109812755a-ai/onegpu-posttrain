"""Resolve the already registered target and disjoint objective suites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.manifest import (
    load_manifest,
    resolve_canonical_path,
    validate_manifest_evidence,
)
from ..lineage import sha256_file
from ..ui.stage_evaluation import StageEvaluationAdapter
from ..ui.training_preflight import _find_manifest
from .manifest import (
    DOCKER_IMAGE,
    FINAL_DATASET_ID,
    FINAL_DATASET_SHA256,
    MODEL_CHECKPOINT_SHA256,
    MODEL_ID,
    MODEL_PROCESSOR_SHA256,
    MODEL_REPO,
    MODEL_REVISION,
    PRACTICE_DATASET_ID,
    PRACTICE_DATASET_SHA256,
    Baseline,
    Dataset,
    ExperimentManifest,
    Inference,
    Runtime,
    TargetModel,
)


@dataclass(frozen=True)
class ResolvedInputs:
    manifest: ExperimentManifest
    model_path: Path
    practice_path: Path
    final_path: Path


def _dataset(workspace: Path, dataset_id: str, digest: str, count: int, role: str) -> tuple[Dataset, Path]:
    manifest_path = _find_manifest(workspace, dataset_id)
    manifest = load_manifest(manifest_path)
    evidence_errors = validate_manifest_evidence(manifest_path)
    if evidence_errors:
        raise ValueError(f"{dataset_id}: registered evidence invalid: {'; '.join(evidence_errors)}")
    canonical = resolve_canonical_path(manifest_path, manifest)
    if (manifest.dataset_id != dataset_id or manifest.stage != "eval"
            or manifest.sample_count != count
            or manifest.hashes.get("canonical_records_sha256") != digest):
        raise ValueError(f"{dataset_id}: registered dataset identity changed")
    metadata = manifest.metadata if isinstance(manifest.metadata, dict) else {}
    if metadata.get("dataset_role") != "evaluation_only":
        raise ValueError(f"{dataset_id}: registered dataset role changed")
    return Dataset(dataset_id=dataset_id, role=role, canonical_sha256=digest, record_count=count), canonical


def resolve_inputs(workspace: str | Path, experiment_id: str) -> ResolvedInputs:
    """Reject changed or unavailable registered artifacts before creating an experiment."""
    root = Path(workspace).resolve(strict=True)
    adapter = StageEvaluationAdapter(root)
    model, _ = adapter._registered_base_model(MODEL_ID)
    if model.digest.value != MODEL_CHECKPOINT_SHA256:
        raise ValueError("registered Qwen3.5 checkpoint digest changed")
    model_path = adapter._model_path(model)
    processor = model_path / "chat_template.jinja"
    if not processor.is_file() or sha256_file(processor) != MODEL_PROCESSOR_SHA256:
        raise ValueError("registered Qwen3.5 processor digest changed")
    practice, practice_path = _dataset(root, PRACTICE_DATASET_ID, PRACTICE_DATASET_SHA256, 300, "practice")
    final, final_path = _dataset(root, FINAL_DATASET_ID, FINAL_DATASET_SHA256, 81, "final_holdout")
    manifest = ExperimentManifest(
        schema_version="1.0", experiment_id=experiment_id,
        target=TargetModel(model_id=MODEL_ID, repository=MODEL_REPO, revision=MODEL_REVISION,
                           checkpoint_sha256=MODEL_CHECKPOINT_SHA256,
                           processor_sha256=MODEL_PROCESSOR_SHA256),
        runtime=Runtime(image=DOCKER_IMAGE, torch_version="2.6.0+cu124",
                        transformers_version="5.8.0",
                        model_loader="AutoProcessor+AutoModelForMultimodalLM"),
        practice=practice, final_holdout=final,
        inference=Inference(prompt_template="qwen3.5-processor-chat-template",
                            scoring_protocol="choice-accuracy-v1", max_new_tokens=16,
                            do_sample=False, skip_special_tokens=True),
        baseline=Baseline(model_id=MODEL_ID, dataset_id=FINAL_DATASET_ID,
                          input_sha256=FINAL_DATASET_SHA256, correct=33,
                          denominator=81, parse_failures=0, evidence="observed_container_run"),
        editor_model="qwen3.8-max",
    )
    return ResolvedInputs(manifest, model_path, practice_path, final_path)
