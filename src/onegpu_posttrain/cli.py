from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from . import __version__
from .adapters.ms_swift import adapt_file
from .adapters.registered_dataset import adapt_registered_dataset
from .data.benchmarks import benchmark_index_from_hf, benchmark_index_from_jsonl
from .data.build import build_finemath_artifacts, build_openr1_artifacts
from .data.calibration_registration import register_sft_calibration_candidate
from .data.cpt_training import prepare_registered_cpt_training
from .data.cpt_role_repair import repair_legacy_cpt_roles
from .data.deepmath_mini import build_deepmath_mini_sft
from .data.finemath import FineMathBuildConfig
from .data.manifest import load_manifest
from .data.mini_cpt_preflight import preflight_mini_cpt
from .data.mini_readiness import evaluate_mini_readiness, write_mini_readiness_report
from .data.openr1 import OpenR1BuildConfig
from .data.policy import exploration_preflight
from .data.quality_assessment import create_quality_assessment_bundle
from .data.quality_review import create_quality_review_bundle
from .data.sft_calibration import calibrate_sft_lengths
from .data.sft_derivation import derive_sft_length_subset
from .data.source_cache import SourceIdentity, create_source_cache, open_source_cache
from .data.sources import iter_hf_dataset, iter_local_jsonl
from .subprocess_utils import console_command_available
from .data.targeted_plan import load_targeted_plan, save_targeted_plan
from .data.targeted_stage_plan import (
    load_targeted_stage_plan,
    register_targeted_stage_plan,
)
from .data.targeted_training_preflight import preflight_targeted_training
from .data.tokenization import HuggingFaceTokenizer, WhitespaceTokenizer
from .data.validate import validate_jsonl
from .evaluation.evalscope_reports import collect_evalscope_reports
from .evaluation.evalscope_runner import (
    build_evalscope_command,
    run_evalscope,
    write_plan,
)
from .evaluation.grpo_metrics import collect_grpo_training_metrics
from .evaluation.internal_math import TransformersGenerator, evaluate_internal_math
from .evaluation.perplexity import evaluate_cpt_perplexity
from .evaluation.preference import evaluate_preference_pairs
from .evaluation.scenarios import (
    build_scenario_report,
    load_scenario_definition,
    write_scenario_report,
)
from .lineage import (
    ArtifactRef,
    Digest,
    EvaluationLineage,
    ModelLineage,
    RunRecord,
    new_run_id,
    sha256_directory,
    sha256_file,
    sha256_json,
)
from .lineage_validation import validate_lineage_root
from .reporting.alignment_compare import write_alignment_comparison
from .reporting.cpt_effect import write_cpt_effect_report
from .reporting.evaluation_matrix import build_evaluation_matrix
from .reporting.grpo_calibration import (
    prepare_grpo_calibration,
    summarize_grpo_calibration,
)
from .reporting.grpo_compare import write_grpo_comparison
from .reporting.smoke_summary import write_smoke_summary
from .reporting.stage_compare import write_stage_comparison
from .reporting.training_compare import write_training_feasibility_comparison
from .synthetic.generation import (
    execute_targeted_generation,
    prepare_generation,
)
from .synthetic.secrets import load_local_provider_credential
from .synthetic.workflow_bundle import register_sft_workflow_bundle
from .training.ms_swift_runner import (
    build_full_cpt_probe_command,
    build_grpo_command,
    build_merge_command,
    build_peft_cpt_command,
    build_preference_rlhf_command,
    build_sft_command,
    capture_environment,
    discover_checkpoint_selection,
    discover_latest_checkpoint,
    run_command,
    run_monitored_command,
    write_json,
)

EXAMPLE_MAP = {
    "cpt.jsonl": "cpt",
    "sft.jsonl": "sft",
    "preference.jsonl": "preference",
    "grpo.jsonl": "grpo",
    "eval.jsonl": "eval",
}


def _select_training_checkpoint(train_dir: Path, recipe: dict) -> dict[str, object]:
    ui_config = recipe.get("_onegpu_ui") if isinstance(recipe.get("_onegpu_ui"), dict) else {}
    selection_config = ui_config.get("checkpoint_selection") if isinstance(ui_config.get("checkpoint_selection"), dict) else {}
    policy = selection_config.get("policy", "best" if recipe.get("training", {}).get("load_best_model_at_end") else "final")
    if policy == "final":
        final = discover_latest_checkpoint(train_dir)
        selected = {"policy": "final", "latest": str(final), "final": str(final), "best": None,
                    "selected": str(final), "selected_step": int(re.search(r"checkpoint-(\d+)$", final.name).group(1)),
                    "metric": None, "metric_value": None}
    else:
        selected = discover_checkpoint_selection(train_dir, str(policy))
    digests: dict[str, str] = {}
    for key in ("latest", "final", "best", "selected"):
        value = selected.get(key)
        if isinstance(value, str) and value and Path(value).is_dir():
            digests[key] = sha256_directory(Path(value))
    selected["digests"] = digests
    selected["metric"] = selection_config.get("metric") or selected.get("metric")
    return selected


def _version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _make_tokenizer(args: argparse.Namespace):
    if args.test_tokenizer:
        return WhitespaceTokenizer()
    return HuggingFaceTokenizer(args.tokenizer_repo, args.tokenizer_revision)


def _source_iter(args: argparse.Namespace, source_cfg: dict):
    source = source_cfg["source"]
    identity = SourceIdentity(
        repo=source["repo"],
        config=source.get("config"),
        revision=source["revision"],
        split="train",
    )
    if args.input_jsonl:
        return iter_local_jsonl(args.input_jsonl), {
            "mode": "local_jsonl",
            "content_sha256": sha256_file(args.input_jsonl),
        }
    if args.source_cache:
        manifest, rows, manifest_sha256 = open_source_cache(args.source_cache, expected=identity)
        return rows, {
            "mode": "verified_cache",
            "cache_key": manifest.cache_key,
            "manifest_sha256": manifest_sha256,
            "content_sha256": manifest.content_sha256,
            "row_count": manifest.row_count,
            "byte_count": manifest.byte_count,
        }
    return iter_hf_dataset(
        identity.repo,
        config=identity.config,
        split=identity.split,
        revision=identity.revision,
        streaming=True,
    ), {"mode": "remote_stream"}


def _benchmark_index(args: argparse.Namespace):
    paths = args.benchmark_jsonl or []
    if paths:
        metadata = [{"kind": "local_jsonl", "path": str(p)} for p in paths]
        return benchmark_index_from_jsonl(paths), metadata
    if args.benchmark_config:
        cfg = _load_yaml(args.benchmark_config)
        specs = cfg.get("benchmarks", [])
        metadata = [
            {k: spec.get(k) for k in ("id", "repo", "config", "split", "field", "revision")}
            for spec in specs
        ]
        return benchmark_index_from_hf(specs), metadata
    return None, []


def _resolve_model_config(path: str | Path) -> tuple[str, str | None]:
    cfg = _load_yaml(path)
    section = cfg.get("model", cfg)
    if isinstance(section, str):
        return section, None
    model = section.get("model_id") or section.get("repo") or section.get("model")
    if not model:
        raise ValueError(f"model config {path} must contain model_id/repo/model")
    return str(model), section.get("revision")


def _assert_new_run_directory(output: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    if (output / "run.json").exists() or (output / "lineage.json").exists():
        raise FileExistsError(f"run output already contains an execution record: {output}; choose a fresh output directory")


def _dataset_lineage_ref(dataset: str | Path, *, role: str = "train") -> tuple[ArtifactRef, bool, list[str]]:
    """Resolve an adapter JSONL to its canonical dataset manifest when evidence exists."""
    path = Path(dataset).resolve()
    candidate = path
    expected_sha = sha256_file(path) if path.is_file() else None
    warnings: list[str] = []
    adapter_report = path.with_suffix(path.suffix + ".adapter.json")
    if adapter_report.is_file():
        report = json.loads(adapter_report.read_text(encoding="utf-8"))
        source = Path(report.get("input", ""))
        if not source.is_absolute():
            source = (Path.cwd() / source).resolve()
        if source.is_file():
            actual = sha256_file(source)
            if report.get("input_sha256") and report["input_sha256"] != actual:
                raise ValueError(f"adapter source digest mismatch: {source}")
            candidate, expected_sha = source, actual
        else:
            warnings.append("adapter source JSONL is unavailable; using native adapter file evidence")
    manifests: list[Path] = []
    for parent in [candidate.parent, *list(candidate.parents)[:5]]:
        directory = parent / "manifests"
        if directory.is_dir():
            manifests.extend(directory.glob("*.yaml"))
    for manifest_path in manifests:
        manifest = load_manifest(manifest_path)
        if expected_sha and manifest.hashes.get("canonical_records_sha256") == expected_sha:
            ref = ArtifactRef(
                id=manifest.dataset_id, kind="dataset", path=str(candidate),
                digest=Digest(value=expected_sha, status="available"), role=role,
            )
            return ref, True, warnings
        if expected_sha and manifest.metadata.get("native_sha256") == expected_sha:
            native_ref = manifest.metadata.get("native_path")
            canonical_ref = manifest.metadata.get("canonical_path", f"../canonical/{candidate.name}")
            artifact_root = manifest_path.parent.parent.resolve()
            native_candidate = (artifact_root / native_ref).resolve() if native_ref else None
            canonical_candidate = (artifact_root / canonical_ref).resolve() if canonical_ref else None
            if native_candidate == candidate.resolve() and canonical_candidate and canonical_candidate.is_file():
                canonical_digest = sha256_file(canonical_candidate)
                if canonical_digest != manifest.hashes.get("canonical_records_sha256"):
                    raise ValueError(f"native dataset canonical digest does not match its manifest: {canonical_candidate}")
                ref = ArtifactRef(
                    id=manifest.dataset_id, kind="dataset", path=str(canonical_candidate),
                    digest=Digest(value=canonical_digest, status="available"), role=role,
                )
                return ref, True, warnings
    if manifests and expected_sha:
        stage = candidate.stem.replace("_", "-").replace(".jsonl", "")
        related = [p for p in manifests if p.stem.lower().startswith(stage.lower())]
        if related:
            raise ValueError(f"dataset content does not match its manifest: {candidate}")
    file_hash = sha256_file(path) if path.is_file() else None
    if not manifests:
        warnings.append("no dataset manifest was found; JSONL provenance is incomplete")
    else:
        warnings.append("no manifest matched the consumed dataset content; JSONL provenance is incomplete")
    digest = Digest(value=file_hash, status="available") if file_hash else Digest(
        status="unavailable", reason="dataset file is not a local file")
    artifact_id = f"file-{file_hash[:24]}" if file_hash else f"file-unavailable-{sha256_json(str(dataset))[:12]}"
    return ArtifactRef(id=artifact_id, kind="file", path=str(path), digest=digest, role=role), False, warnings


def _dataset_targeted_plan(dataset: str | Path) -> tuple[str | None, int | None]:
    """Read plan identity only from a content-matching registered manifest."""
    path = Path(dataset).resolve()
    candidate = path
    expected_sha = sha256_file(path) if path.is_file() else None
    adapter_report = path.with_suffix(path.suffix + ".adapter.json")
    if adapter_report.is_file():
        try:
            report = json.loads(adapter_report.read_text(encoding="utf-8"))
            source = Path(report.get("input", ""))
            if not source.is_absolute():
                source = (Path.cwd() / source).resolve()
            if source.is_file():
                candidate, expected_sha = source, sha256_file(source)
        except (OSError, ValueError, TypeError):
            return None, None
    if not expected_sha:
        return None, None
    for parent in [candidate.parent, *list(candidate.parents)[:6]]:
        manifest_dir = parent / "manifests"
        if not manifest_dir.is_dir() or manifest_dir.is_symlink():
            continue
        for manifest_path in manifest_dir.glob("*.yaml"):
            if manifest_path.is_symlink() or not manifest_path.is_file():
                continue
            try:
                manifest = load_manifest(manifest_path)
                matches = manifest.hashes.get("canonical_records_sha256") == expected_sha
                if not matches:
                    continue
                # Generated SFT registration uses the explicit ``targeted_stage_*``
                # names; older imported manifests may still use ``targeted_plan_*``.
                plan_id = manifest.metadata.get("targeted_stage_plan_id")
                plan_version = manifest.metadata.get("targeted_stage_plan_version")
                if plan_id is None and plan_version is None:
                    plan_id = manifest.metadata.get("targeted_plan_id")
                    plan_version = manifest.metadata.get("targeted_plan_version")
                if plan_id is None and plan_version is None:
                    return None, None
                if (not isinstance(plan_id, str) or not re.fullmatch(r"plan-targeted-[a-f0-9]{20}", plan_id)
                        or not isinstance(plan_version, int) or plan_version < 1):
                    raise ValueError("registered dataset has invalid targeted plan identity")
                return plan_id, plan_version
            except (OSError, ValueError, TypeError):
                continue
    return None, None


def _model_lineage_ref(model: str | Path, revision: str | None = None) -> ArtifactRef:
    path = Path(model)
    if path.is_dir():
        try:
            digest = Digest(value=sha256_directory(path), status="available")
            model_id = f"model-{digest.value[:20]}"
        except OSError as exc:
            digest = Digest(status="not_computed", reason=f"model directory hashing failed: {exc}")
            model_id = f"model-ref-{sha256_json(str(path.resolve()))[:20]}"
    elif path.is_file():
        digest = Digest(value=sha256_file(path), status="available")
        model_id = f"model-{digest.value[:20]}"
    else:
        digest = Digest(status="unavailable", reason="parent model content digest was not captured")
        model_id = f"model-ref-{sha256_json(str(model))[:20]}"
    return ArtifactRef(id=model_id, kind="model", path=str(path), revision=revision, digest=digest, role="parent")


def _registered_model_producer(model: str | Path, model_ref: ArtifactRef) -> str | None:
    """Resolve a producer from an adjacent or workspace-local, digest-matching registration."""
    path = Path(model)
    if not path.exists() or path.is_symlink():
        return None
    try:
        resolved_path = path.resolve(strict=True)
    except OSError:
        return None
    registries = [parent / "model_artifacts" for parent in list(path.parents)[:6]]
    workspace_roots = [parent for parent in list(path.parents)[:6] if (parent / "runs").is_dir()]
    workspace_root = workspace_roots[0] if workspace_roots else None
    registries.extend(root / "runs" for root in workspace_roots[:1])
    scanned = 0
    for registry_root in registries:
        if registry_root.is_symlink() or not registry_root.is_dir():
            continue
        artifact_paths = (registry_root.glob("**/model_artifacts/model-*.json")
                          if registry_root.name == "runs" else registry_root.glob("model-*.json"))
        for artifact_path in artifact_paths:
            if artifact_path.is_symlink() or not artifact_path.is_file():
                continue
            scanned += 1
            if scanned > 2000:
                return None
            try:
                if artifact_path.stat().st_size > 64 * 1024:
                    continue
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                registered_path = Path(str(artifact.get("path", "")))
                if not registered_path.is_absolute():
                    if workspace_root is None:
                        continue
                    registered_path = workspace_root / registered_path
                registered_digest = artifact.get("digest", {}).get("value")
                if (artifact.get("model_id") == model_ref.id
                        and registered_path.resolve(strict=True) == resolved_path
                        and registered_digest == (model_ref.digest.value if model_ref.digest else None)):
                    produced_by = artifact.get("produced_by_run_id")
                    return produced_by if isinstance(produced_by, str) else None
            except (OSError, ValueError, TypeError):
                continue
    return None


def _policy_training_context(dataset: str | Path, training: dict, stage: str) -> dict:
    """Check policy data before creating a plan or launching a training process."""
    path = Path(dataset)
    if not path.is_file():
        return {}
    ref, _, _ = _dataset_lineage_ref(path)
    canonical = Path(ref.path)
    for parent in [canonical.parent, *list(canonical.parents)[:5]]:
        directory = parent / "manifests"
        if not directory.is_dir():
            continue
        for manifest_path in directory.glob("*.yaml"):
            manifest = load_manifest(manifest_path)
            if manifest.dataset_id != ref.id or manifest.metadata.get("usage") != "exploratory":
                continue
            from .data.io import read_jsonl
            from .data.manifest import validate_manifest_evidence
            failures = validate_manifest_evidence(manifest_path)
            if failures:
                raise ValueError("探索数据证据校验失败：" + "; ".join(failures))
            if manifest.stage != stage:
                raise ValueError("探索数据阶段与训练阶段不一致")
            count = sum(1 for _ in read_jsonl(path))
            if count != manifest.sample_count:
                raise ValueError("实际训练记录数与 canonical manifest 不一致")
            sidecar = path.with_suffix(path.suffix + ".adapter.json")
            if sidecar.is_file():
                adapter = json.loads(sidecar.read_text(encoding="utf-8"))
                if adapter.get("output_sha256") != sha256_file(path):
                    raise ValueError("训练 adapter 输出摘要不一致")
            # Calibration cohorts are explicitly exploratory, data-only lineage
            # artifacts. They intentionally have no policy approval; preserve the
            # exploratory warning while still enforcing rendered-length safety.
            if manifest.metadata.get("provenance_type") in {"sft_length_calibration", "sft_length_subset"}:
                maximum = manifest.metadata.get("max_rendered_tokens")
                recipe_max_length = training.get("max_length")
                if recipe_max_length is None or maximum is None:
                    raise ValueError("长度校准数据需要明确的训练 max_length 与实测渲染长度")
                if int(maximum) > recipe_max_length:
                    raise ValueError("数据完整长度超过 recipe max_length；不能静默截断完整答案")
                return {
                    "usage": "exploratory",
                    "actual_record_count": count,
                    "official_mini_acceptance": "not_assessed",
                    "warnings": [
                        "长度校准候选未绑定 DataPolicy；质量审批与语义推理正确性仍未评估",
                    ],
                }
            if "data_policy" not in manifest.metadata:
                origin_kind = manifest.origin.kind if manifest.origin else manifest.metadata.get("source_kind")
                if origin_kind == "model_generated":
                    return {
                        "usage": "exploratory",
                        "source_kind": "model_generated",
                        "actual_record_count": count,
                        "official_mini_acceptance": "not_assessed",
                        "warnings": [
                            "模型生成数据未绑定 DataPolicy；canonical/schema 与登记证据有效，但语义正确性未独立验证",
                        ],
                    }
                raise ValueError("探索数据缺少 DataPolicy 登记；不能按策略实验语义启动训练")
            return exploration_preflight(manifest, record_count=count, recipe_max_length=training.get("max_length"))
    return {}


def _write_training_run(*, output: Path, result: dict[str, object], run_id: str | None,
                        run_label: str | None, experiment_id: str | None, stage: str,
                        model: str | Path, dataset: str | Path, config_path: str | Path,
                        resolved_config: dict, started_at: str, model_ref: ArtifactRef | None = None,
                        validation_dataset: str | Path | None = None,
                        validation_metadata: dict | None = None,
                        targeted_plan_id: str | None = None, targeted_plan_version: int | None = None) -> None:
    """Write execution record and normalized lineage for completed real invocations."""
    write_json(output / "run.json", result)
    if run_id is None:
        return
    manifest_plan_id, manifest_plan_version = _dataset_targeted_plan(dataset)
    if targeted_plan_id is None:
        targeted_plan_id, targeted_plan_version = manifest_plan_id, manifest_plan_version
    elif (targeted_plan_id, targeted_plan_version) != (manifest_plan_id, manifest_plan_version):
        raise ValueError("CLI targeted plan identity does not match the content-matched dataset manifest")
    if targeted_plan_id is not None:
        resolved_config = {**resolved_config, "targeted_stage_plan": {
            "plan_id": targeted_plan_id, "plan_version": targeted_plan_version,
        }}
    if isinstance(result.get("checkpoint_selection"), dict):
        resolved_config = {**resolved_config, "checkpoint_selection": result["checkpoint_selection"]}
    dataset_ref, dataset_complete, lineage_warnings = _dataset_lineage_ref(dataset)
    inputs = [model_ref or _model_lineage_ref(model), dataset_ref]
    validation_complete = True
    if validation_dataset is not None:
        validation_ref, validation_complete, validation_warnings = _dataset_lineage_ref(
            validation_dataset, role="training_validation",
        )
        inputs.append(validation_ref)
        lineage_warnings.extend(validation_warnings)
        if isinstance(validation_metadata, dict) and validation_metadata.get("mode") == "random_split":
            membership_path = Path(validation_metadata.get("membership_manifest_path")
                                   or Path(validation_dataset).parent / "split_manifest.json").resolve(strict=True)
            membership_digest = sha256_file(membership_path)
            expected_membership_digest = validation_metadata.get("membership_manifest_sha256")
            if expected_membership_digest and membership_digest != expected_membership_digest:
                raise ValueError("materialized training-validation membership manifest digest mismatch")
            validation_metadata = {**validation_metadata,
                                   "membership_manifest_path": str(membership_path),
                                   "membership_manifest_sha256": membership_digest}
            result["training_validation"] = validation_metadata
            inputs.append(ArtifactRef(
                id=f"training-validation-members-{membership_digest[:24]}", kind="file",
                path=str(membership_path), digest=Digest(value=membership_digest, status="available"),
                role="training_validation_membership_manifest",
            ))
        resolved_config = {**resolved_config,
                           "training_validation": validation_metadata or result.get("training_validation") or {},
                           "validation_dataset": str(validation_dataset)}
    elif validation_metadata:
        resolved_config = {**resolved_config, "training_validation": validation_metadata}
    if isinstance(validation_metadata, dict) and validation_metadata.get("mode") == "random_split":
        source_id = validation_metadata.get("source_dataset_id")
        source_digest = validation_metadata.get("source_content_sha256")
        if isinstance(source_id, str) and source_id and isinstance(source_digest, str) and re.fullmatch(r"[0-9a-f]{64}", source_digest):
            inputs.append(ArtifactRef(
                id=source_id, kind="dataset", digest=Digest(value=source_digest, status="available"),
                role="training_source_before_validation_split",
            ))
    policy_context = result.get("data_policy_context")
    if policy_context:
        resolved_config = {**resolved_config, "data_policy_context": policy_context}
        if policy_context.get("policy_id"):
            inputs.append(ArtifactRef(id=policy_context["policy_id"], kind="file", role="data_policy",
                                      revision=policy_context["policy_config_sha256"]))
        lineage_warnings.extend(policy_context.get("warnings", []))
    write_json(output / "resolved_config.json", resolved_config)
    checkpoint = result.get("checkpoint")
    outputs: list[ArtifactRef] = []
    if isinstance(checkpoint, str) and Path(checkpoint).exists():
        checkpoint_path = Path(checkpoint)
        output_digest = Digest(value=sha256_directory(checkpoint_path), status="available") if checkpoint_path.is_dir() else Digest(
            value=sha256_file(checkpoint_path), status="available")
        adapter_id = f"model-{output_digest.value[:20]}"
        outputs.append(ArtifactRef(id=adapter_id, kind="model",
                                   path=checkpoint, digest=output_digest,
                                   role="adapter"))
        adapter_manifest = ModelLineage(
            model_id=adapter_id, kind="adapter", path=str(checkpoint_path), digest=output_digest,
            parent_model_ids=[inputs[0].id], produced_by_run_id=run_id, stage=stage,
        )
        write_json(output / "model_artifacts" / f"{adapter_id}.json", adapter_manifest.model_dump(mode="json"))
    selection = result.get("checkpoint_selection")
    if isinstance(selection, dict):
        seen_checkpoint_paths = {str(checkpoint)} if isinstance(checkpoint, str) else set()
        for label in ("latest", "final", "best"):
            checkpoint_path = selection.get(label)
            if not isinstance(checkpoint_path, str) or checkpoint_path in seen_checkpoint_paths:
                continue
            path = Path(checkpoint_path)
            digest = (selection.get("digests") or {}).get(label)
            if path.is_dir() and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
                outputs.append(ArtifactRef(id=f"checkpoint-{label}-{digest[:24]}", kind="file",
                                           path=checkpoint_path, digest=Digest(value=digest, status="available"),
                                           role=f"training_checkpoint_{label}"))
                seen_checkpoint_paths.add(checkpoint_path)
    model_digest_available = inputs[0].digest is not None and inputs[0].digest.status == "available"
    # Random partitions are derived files rather than separately registered datasets;
    # their source membership and hashes are committed in the resolved config and
    # the file itself is represented as a digest-bearing lineage input.
    validation_has_audit_identity = bool(
        validation_metadata and (
            validation_metadata.get("mode") == "none"
            or validation_metadata.get("split_identity_sha256")
            or validation_metadata.get("content_sha256")
        )
    )
    incomplete = not dataset_complete or not model_digest_available or (
        validation_dataset is not None and not validation_complete and not validation_has_audit_identity
    )
    execution = result.get("train", result.get("execution", {}))
    execution_status = execution.get("status") if isinstance(execution, dict) else None
    status = {"PASS": "completed", "OOM": "oom", "FAILED": "failed"}.get(execution_status, "failed")
    run = RunRecord(
        run_id=run_id, run_type="train", status=status,
        lineage_status="incomplete" if incomplete else "complete", run_label=run_label, experiment_id=experiment_id,
        targeted_plan_id=targeted_plan_id, targeted_plan_version=targeted_plan_version,
        started_at=started_at, completed_at=datetime.now().astimezone().isoformat(), inputs=inputs, outputs=outputs,
        recipe_path=str(config_path), resolved_config_sha256=sha256_json(resolved_config),
        training_memory_mode=(result.get("training_memory_mode")
                              if isinstance(result.get("training_memory_mode"), dict) else None),
        lineage_warnings=lineage_warnings + ([] if model_digest_available else ["parent model content digest is unavailable"]),
    )
    result["run_id"] = run_id
    result["run_label"] = run_label
    result["experiment_id"] = experiment_id
    write_json(output / "run.json", result)
    write_json(output / "lineage.json", run.model_dump(mode="json"))


def _run_merge_with_lineage(*, output: Path, checkpoint: str | Path, parent_model: str | Path,
                            merged_dir: str | Path, run_label: str, experiment_id: str | None,
                            stage: str, dry_run: bool = False) -> dict[str, object]:
    """Execute adapter merge as its own typed Run and record its actual result."""
    command = build_merge_command(checkpoint=checkpoint, output_dir=merged_dir, model=parent_model)
    if dry_run:
        return {"planned": True, "command": command}
    run_id = new_run_id("merge", label=run_label)
    started_at = datetime.now().astimezone().isoformat()
    parent_ref = _model_lineage_ref(parent_model)
    adapter_path = Path(checkpoint)
    adapter_digest = Digest(value=sha256_directory(adapter_path), status="available") if adapter_path.is_dir() else Digest(
        value=sha256_file(adapter_path), status="available")
    adapter_ref = ArtifactRef(id=f"model-{adapter_digest.value[:20]}", kind="model", path=str(adapter_path),
                              digest=adapter_digest, role="adapter")
    merged_path = Path(merged_dir)
    try:
        execution = run_command(command, log_dir=output / "logs" / "merge")
        status = "completed"
        errors: list[str] = []
    except RuntimeError as exc:
        execution = {"status": "FAILED", "error": str(exc)}
        status = "failed"
        errors = [str(exc)]
    outputs: list[ArtifactRef] = []
    if status == "completed" and merged_path.is_dir():
        merged_digest = Digest(value=sha256_directory(merged_path), status="available")
        model_id = f"model-{merged_digest.value[:20]}"
        merged_model = ModelLineage(
            model_id=model_id, kind="merged", path=str(merged_path), digest=merged_digest,
            parent_model_ids=[parent_ref.id], adapter_id=adapter_ref.id, produced_by_run_id=run_id, stage=stage,
        )
        write_json(output / "model_artifacts" / f"{model_id}.json", merged_model.model_dump(mode="json"))
        outputs.append(ArtifactRef(id=model_id, kind="model", path=str(merged_path), digest=merged_digest, role="merged"))
    elif status == "completed":
        status = "failed"
        errors = [f"merge command completed without producing model directory: {merged_path}"]
        execution = {**execution, "status": "FAILED", "error": errors[0]}
    run = RunRecord(
        run_id=run_id, run_type="merge", status=status, run_label=run_label, experiment_id=experiment_id,
        started_at=started_at, completed_at=datetime.now().astimezone().isoformat(),
        inputs=[parent_ref, adapter_ref], outputs=outputs, errors=errors,
    )
    write_json(output / "merge_runs" / run_id / "lineage.json", run.model_dump(mode="json"))
    return {"run_id": run_id, "execution": execution, "merged_model": str(merged_path) if outputs else None}


def _build_eval_context(config_path: str | Path, model_override: str | None = None, revision_override: str | None = None):
    cfg = _load_yaml(config_path)
    model_cfg_path = cfg["model_config"]
    model, revision = _resolve_model_config(model_cfg_path)
    if model_override:
        model = model_override
        # Local merged checkpoints should not inherit the base HF revision.
        if Path(model_override).exists():
            revision = None
    if revision_override is not None:
        revision = revision_override or None
    eval_cfg = _load_yaml(cfg["evaluation_config"])
    return cfg, model, revision, eval_cfg


def _eval_command(*, cfg: dict, model: str, revision: str | None, eval_cfg: dict, output: Path,
                  limit: int | None, datasets: list[str] | None = None,
                  dataset_args: dict[str, object] | None = None) -> list[str]:
    datasets = datasets or list(eval_cfg.get("benchmarks", ["gsm8k", "math_500"]))
    extra = ["--no-timestamp", "--seed", str(cfg.get("seed", 42))]
    model_args: dict[str, object] = {"precision": "torch.float16", "device_map": "auto"}
    if revision:
        model_args["revision"] = revision
    extra += ["--model-args", json.dumps(model_args)]
    if eval_cfg.get("generation"):
        extra += ["--generation-config", json.dumps(eval_cfg["generation"])]
    if dataset_args:
        extra += ["--dataset-args", json.dumps({datasets[0]: dataset_args})]
    return build_evalscope_command(
        model=model,
        datasets=datasets,
        output_dir=output / "evalscope",
        limit=limit,
        extra_args=extra,
    )


def _eval_commands(*, cfg: dict, model: str, revision: str | None, eval_cfg: dict,
                   output: Path, limit: int | None) -> list[list[str]]:
    execution = eval_cfg.get("benchmark_execution")
    if execution is None:
        return [_eval_command(cfg=cfg, model=model, revision=revision, eval_cfg=eval_cfg,
                              output=output, limit=limit)]
    benchmarks = set(eval_cfg.get("benchmarks", []))
    if not isinstance(execution, list) or not execution:
        raise ValueError("benchmark_execution must be a non-empty list")
    commands: list[list[str]] = []
    seen: set[str] = set()
    for item in execution:
        if not isinstance(item, dict):
            raise TypeError("benchmark_execution entries must be mappings")
        dataset, sample_limit = item.get("dataset"), item.get("limit")
        dataset_args = item.get("dataset_args", {})
        if (not isinstance(dataset, str) or dataset not in benchmarks or dataset in seen
                or not isinstance(sample_limit, int) or isinstance(sample_limit, bool) or sample_limit < 1
                or not isinstance(dataset_args, dict)):
            raise ValueError("invalid benchmark_execution entry")
        seen.add(dataset)
        commands.append(_eval_command(cfg=cfg, model=model, revision=revision, eval_cfg=eval_cfg,
                                      output=output, limit=sample_limit, datasets=[dataset],
                                      dataset_args=dataset_args or None))
    if seen != benchmarks:
        raise ValueError("benchmark_execution must cover every configured benchmark exactly once")
    return commands


def cmd_doctor(_: argparse.Namespace) -> int:
    info = {
        "onegpu_posttrain": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ms-swift": _version("ms-swift"),
        "evalscope": _version("evalscope"),
        "torch": _version("torch"),
        "transformers": _version("transformers"),
        "datasets": _version("datasets"),
        "bitsandbytes": _version("bitsandbytes"),
    }
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["vram_gib"] = round(props.total_memory / 1024**3, 2)
    except (ImportError, RuntimeError, OSError):
        info["cuda_available"] = False
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def cmd_validate_lineage(args: argparse.Namespace) -> int:
    result = validate_lineage_root(args.root, require_complete=args.require_complete)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["valid"] else 2


def cmd_validate_mini_data(args: argparse.Namespace) -> int:
    try:
        report = evaluate_mini_readiness(
            args.root, args.profile, args.quality_assessment, getattr(args, "quality_review", None),
        )
        json_path, markdown_path = write_mini_readiness_report(report, args.output)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "invalid_input", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "status": report.overall_status,
        "assessment_id": report.assessment_id,
        "json_report": str(json_path),
        "markdown_report": str(markdown_path),
        "blocker_count": len(report.blockers),
        "warning_count": len(report.warnings),
    }, indent=2, ensure_ascii=False))
    return 0 if report.overall_status == "ready" else 2


def cmd_assess_mini_data_quality(args: argparse.Namespace) -> int:
    try:
        index, index_path = create_quality_assessment_bundle(args.root, args.profile, args.output)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "invalid_input", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "status": "completed",
        "assessment_id": index.assessment_id,
        "dataset_count": len(index.datasets),
        "index": str(index_path),
        "quality_protocol": index.quality_protocol,
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_review_mini_data_quality(args: argparse.Namespace) -> int:
    try:
        index, index_path = create_quality_review_bundle(
            args.root, args.mini_profile, args.quality_assessment, args.review_profile, args.output,
            getattr(args, "decisions", None), getattr(args, "supersedes", None),
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "invalid_input", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "completed", "assessment_id": index["assessment_id"],
                      "dataset_count": len(index["datasets"]), "index": str(index_path),
                      "profile_id": index["profile_id"], "review_outcome": "pending"},
                     indent=2, ensure_ascii=False))
    return 0


def cmd_validate_examples(args: argparse.Namespace) -> int:
    root = Path(args.path)
    results = []
    ok = True
    for filename, stage in EXAMPLE_MAP.items():
        path = root / filename
        if not path.exists():
            results.append({"stage": stage, "path": str(path), "ok": False, "errors": ["missing file"]})
            ok = False
            continue
        r = validate_jsonl(path, stage)
        results.append(r.to_dict())
        ok = ok and r.ok
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if ok else 2


def cmd_cache_source(args: argparse.Namespace) -> int:
    try:
        cfg = _load_yaml(args.config)
        source = cfg["source"]
        identity = SourceIdentity(
            repo=source["repo"],
            config=source.get("config"),
            revision=source["revision"],
            split=args.split,
        )
        manifest_path, manifest, reused = create_source_cache(
            iter_hf_dataset(
                identity.repo,
                config=identity.config,
                revision=identity.revision,
                split=identity.split,
                streaming=True,
            ),
            cache_root=args.cache_dir,
            identity=identity,
            refresh=args.refresh,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "status": "reused" if reused else "cached",
        "manifest": str(manifest_path),
        "cache_key": manifest.cache_key,
        "source": manifest.source.model_dump(mode="json"),
        "row_count": manifest.row_count,
        "byte_count": manifest.byte_count,
        "content_sha256": manifest.content_sha256,
        "refreshed": args.refresh and not reused,
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_build_openr1(args: argparse.Namespace) -> int:
    cfg = _load_yaml(args.config)
    source = cfg["source"]
    tokenizer = _make_tokenizer(args)
    build_cfg = OpenR1BuildConfig(
        prompt_template_id=cfg.get("prompt_template_id", "qwen3-math-v1"),
        seed=int(cfg.get("split_seed", 42)),
        sft_target=args.sft_target,
        alignment_target=args.alignment_target,
        holdout_target=args.holdout_target,
        preference_target=args.preference_target,
        preference_eval_target=args.preference_eval_target,
        grpo_target=args.grpo_target,
        sft_max_length=args.sft_max_length,
        preference_max_length=args.preference_max_length,
        grpo_max_prompt_length=args.grpo_max_prompt_length,
        max_raw_rows=args.max_raw_rows,
    )
    benchmark_index, benchmark_metadata = _benchmark_index(args)
    try:
        source_rows, source_acquisition = _source_iter(args, cfg)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "invalid_source", "error": str(exc)}, ensure_ascii=False))
        return 2
    report = build_openr1_artifacts(
        source_rows,
        output_dir=args.output,
        tokenizer=tokenizer,
        config=build_cfg,
        source_repo=source["repo"],
        source_config=source.get("config"),
        source_revision=source["revision"],
        tokenizer_repo=args.tokenizer_repo,
        tokenizer_revision=args.tokenizer_revision,
        benchmark_index=benchmark_index,
        benchmark_metadata=benchmark_metadata,
        source_acquisition=source_acquisition,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    selected = report["selected"]
    ok = (
        selected["sft"] >= args.sft_target
        and selected["preference"] >= args.preference_target
        and selected["preference_eval"] >= args.preference_eval_target
        and selected["grpo"] >= args.grpo_target
        and selected["holdout"] >= args.holdout_target
        and report["counters"].get("alignment_core_seen", 0) >= args.alignment_target
        and report["overlap"]["train_holdout"] == 0
        and not report["quality_gate_failures"]
    )
    return 0 if ok else 3


def cmd_calibrate_sft_length(args: argparse.Namespace) -> int:
    if args.run_probes and not args.model:
        raise SystemExit("--model is required with --run-probes; without it calibration stays CPU-only.")
    if args.run_probes and not Path(args.model).expanduser().exists():
        raise SystemExit("--model must point to a locally available parent checkpoint; calibration never downloads model weights.")
    if args.probe_steps < 1 or args.probe_steps > 5:
        raise SystemExit("--probe-steps must be between 1 and 5 for a bounded hardware-feasibility probe.")
    try:
        report = calibrate_sft_lengths(
            source=args.source,
            output=args.output,
            sample_limit=args.sample_limit,
            seed=args.seed,
            candidates=args.length or (1024, 2048, 4096),
            tokenizer_repo=args.tokenizer_repo,
            tokenizer_revision=args.tokenizer_revision,
            answer_verification=args.answer_verification,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    if args.run_probes:
        recipe = _load_yaml(args.probe_config)
        base_training = dict(recipe.get("training", {}))
        probe_rows = []
        for candidate in report["candidates"]:
            cap = int(candidate["max_length"])
            probe_dir = output / f"length-{cap}" / "probe"
            if int(candidate["training_examples"]) == 0:
                probe_rows.append({"max_length": cap, "status": "NOT_RUN", "reason": "no verified candidate examples"})
                continue
            training = dict(base_training)
            training.update({
                "max_length": cap,
                "max_steps": args.probe_steps,
                "save_steps": args.probe_steps,
                "save_total_limit": 1,
                "save_only_model": True,
                "split_dataset_ratio": 0,
                "dataset_num_proc": 1,
                "dataloader_num_workers": 0,
                "seed": args.seed,
                "report_to": "none",
            })
            probe_config = probe_dir / "resolved_recipe.yaml"
            probe_config.parent.mkdir(parents=True, exist_ok=False)
            payload = {
                "run_label": f"sft-length-{cap}-s{args.seed}",
                "experiment_id": report["calibration_id"],
                "seed": args.seed,
                "stage": "sft",
                "training": training,
            }
            probe_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            probe_args = argparse.Namespace(
                config=str(probe_config), model=args.model, dataset=candidate["native_path"],
                output=str(probe_dir), merge=False, dry_run=False,
            )
            try:
                code = cmd_run_sft(probe_args)
                run_path = probe_dir / "run.json"
                run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
                execution = run.get("train", {})
                probe_status = execution.get("status", "FAILED" if code else "UNKNOWN")
                probe_rows.append({
                    "max_length": cap,
                    "status": probe_status,
                    "failure_reason": (
                        "out_of_memory" if probe_status == "OOM" else
                        f"training process returned {code}; see stderr log" if probe_status == "FAILED" else None
                    ),
                    "return_code": code,
                    "run_id": run.get("run_id"),
                    "run_json": str(run_path) if run_path.is_file() else None,
                    "dataset_sha256": candidate["content_sha256"],
                    "training_examples": candidate["training_examples"],
                    "wall_seconds": execution.get("wall_seconds"),
                    "gpu_memory_baseline_mib": execution.get("gpu_memory_baseline_mib"),
                    "peak_gpu_memory_mib_sampled": execution.get("peak_gpu_memory_mib_sampled"),
                    "gpu_memory_metric": execution.get("gpu_memory_metric", "unavailable"),
                    "returncode": execution.get("returncode"),
                    "stdout_log": execution.get("stdout_log"),
                    "stderr_log": execution.get("stderr_log"),
                    "environment_path": str(probe_dir / "environment.json"),
                    "environment": run.get("environment", {"status": "unavailable"}),
                })
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, yaml.YAMLError) as exc:
                # Preserve known per-candidate setup failures and continue with later length caps.
                failure = {"max_length": cap, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
                write_json(probe_dir / "probe_failure.json", failure)
                probe_rows.append(failure)
        report["probes"] = probe_rows
        report["interpretation"].append(
            "Short SFT probes are hardware-feasibility checks only; losses and outcomes are not model-quality comparisons."
        )
        write_json(output / "calibration_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_register_sft_calibration(args: argparse.Namespace) -> int:
    try:
        result = register_sft_calibration_candidate(args.manifest)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_derive_sft_length_subset(args: argparse.Namespace) -> int:
    try:
        result = derive_sft_length_subset(args.manifest, max_length=args.max_length, output=args.output)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_repair_cpt_roles(args: argparse.Namespace) -> int:
    try:
        result = repair_legacy_cpt_roles(
            args.manifest, workspace=args.workspace, output_root=args.output_root, apply=args.apply,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc), "execution_started": False}, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_build_finemath(args: argparse.Namespace) -> int:
    cfg = _load_yaml(args.config)
    source = cfg["source"]
    tokenizer = _make_tokenizer(args)
    build_cfg = FineMathBuildConfig(
        train_token_budget=args.train_token_budget,
        holdout_token_budget=args.holdout_token_budget,
        max_seq_tokens=args.max_seq_tokens,
        holdout_permille=args.holdout_permille,
        max_raw_rows=args.max_raw_rows,
    )
    benchmark_index, benchmark_metadata = _benchmark_index(args)
    try:
        source_rows, source_acquisition = _source_iter(args, cfg)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "invalid_source", "error": str(exc)}, ensure_ascii=False))
        return 2
    report = build_finemath_artifacts(
        source_rows,
        output_dir=args.output,
        tokenizer=tokenizer,
        config=build_cfg,
        source_repo=source["repo"],
        source_config=source.get("config"),
        source_revision=source["revision"],
        tokenizer_repo=args.tokenizer_repo,
        tokenizer_revision=args.tokenizer_revision,
        benchmark_index=benchmark_index,
        benchmark_metadata=benchmark_metadata,
        source_acquisition=source_acquisition,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    ok = (
        report["train"]["tokens"] >= args.train_token_budget
        and report["holdout"]["tokens"] >= args.holdout_token_budget
        and report["document_overlap"] == 0
        and not report["quality_gate_failures"]
    )
    return 0 if ok else 3


def cmd_adapt_ms_swift(args: argparse.Namespace) -> int:
    report = adapt_file(args.stage, args.input, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_adapt_registered_dataset(args: argparse.Namespace) -> int:
    report = adapt_registered_dataset(args.manifest, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_build_deepmath_mini_sft(args: argparse.Namespace) -> int:
    contamination_paths = [Path(path) for path in (args.contamination_index or [])]
    benchmark_index = benchmark_index_from_jsonl(contamination_paths) if contamination_paths else None
    report = build_deepmath_mini_sft(
        source=args.source, output_dir=args.output, tokenizer=_make_tokenizer(args),
        benchmark_index=benchmark_index,
        contamination_sources=[{"path": str(path), "sha256": sha256_file(path)} for path in contamination_paths],
        seed=args.seed, max_length=args.max_length, target_examples=args.target,
        tokenizer_repo=args.tokenizer_repo, tokenizer_revision=args.tokenizer_revision,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_plan_model_eval(args: argparse.Namespace) -> int:
    cfg, model, revision, eval_cfg = _build_eval_context(args.config, args.model, args.revision)
    output = Path(args.output)
    commands = _eval_commands(cfg=cfg, model=model, revision=revision, eval_cfg=eval_cfg, output=output, limit=args.limit)
    payload = {
        "run_label": getattr(args, "run_label", None) or cfg.get("run_label", cfg.get("run_id", "model-eval")),
        "stage": args.stage,
        "model": model,
        "model_revision": revision,
        "benchmarks": list(eval_cfg.get("benchmarks", ["gsm8k", "math_500"])),
        "evalscope": {"commands": commands, "work_dir": str(output / "evalscope")},
        "internal_holdout": args.internal_holdout,
        "domain_holdout": args.domain_holdout,
        "protocol_version": eval_cfg.get("protocol_version"),
    }
    write_plan(output / "model_eval_plan.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _evaluation_metric_statuses(
    *, stage: str, evalscope_reports: object, internal_math: object,
    domain_perplexity: object, has_internal_holdout: bool,
    has_domain_holdout: bool, execution_error: str | None,
) -> dict[str, dict[str, object]]:
    """Report measured coverage with observed denominators, never inferred zeros."""
    not_run_reason = execution_error or "评测输入或对应评测器未执行。"
    states: dict[str, dict[str, object]] = {}
    reports = evalscope_reports if isinstance(evalscope_reports, dict) else {}
    for benchmark, metric_id in (("gsm8k", "gsm8k.accuracy"), ("math_500", "math_500.accuracy")):
        report = next((value for name, value in reports.items()
                       if benchmark in str(name).lower() and isinstance(value, dict)), None)
        accuracy = report.get("accuracy", report.get("score")) if isinstance(report, dict) else None
        denominator = report.get("samples") if isinstance(report, dict) else None
        if isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool) and isinstance(denominator, int) and denominator > 0:
            states[metric_id] = {"status": "measured", "value": accuracy, "denominator": denominator,
                                 "denominator_unit": "samples", "reason": "使用评测报告中的实际完成样本数。"}
        else:
            states[metric_id] = {"status": "not_measured", "value": None, "denominator": denominator,
                                 "denominator_unit": "samples",
                                 "reason": not_run_reason if execution_error else f"{benchmark} 未返回有效分数和正样本分母。"}

    internal = internal_math if isinstance(internal_math, dict) else {}
    denominator = internal.get("samples")
    accuracy = internal.get("accuracy")
    if isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool) and isinstance(denominator, int) and denominator > 0:
        states["internal_holdout.accuracy"] = {"status": "measured", "value": accuracy,
                                               "denominator": denominator, "denominator_unit": "samples",
                                               "reason": "使用内部评测实际完成判分样本数。"}
    else:
        states["internal_holdout.accuracy"] = {
            "status": "not_measured", "value": None, "denominator": denominator,
            "denominator_unit": "samples",
            "reason": not_run_reason if execution_error else (
                "未提供 internal holdout。" if not has_internal_holdout
                else "内部评测未产出有效分数和正样本分母。"
            ),
        }

    if stage == "cpt" or has_domain_holdout:
        domain = domain_perplexity if isinstance(domain_perplexity, dict) else {}
        denominator = domain.get("predicted_tokens")
        perplexity = domain.get("perplexity")
        if isinstance(perplexity, (int, float)) and not isinstance(perplexity, bool) and isinstance(denominator, int) and denominator > 0:
            states["domain_ppl.perplexity"] = {
                "status": "measured", "value": perplexity, "denominator": denominator,
                "denominator_unit": "predicted_tokens",
                "samples": domain.get("samples"), "truncated_samples": domain.get("truncated_samples"),
                "reason": "基于 canonical domain holdout 的 token-weighted NLL；以实际预测 token 数为分母。",
            }
        else:
            states["domain_ppl.perplexity"] = {
                "status": "not_measured", "value": None, "denominator": denominator,
                "denominator_unit": "predicted_tokens",
                "reason": not_run_reason if execution_error else (
                    "未提供 CPT domain holdout，未运行 domain PPL。" if not has_domain_holdout
                    else "domain PPL 未产出有效结果和正预测 token 分母。"
                ),
            }
    return states


def cmd_run_model_eval(args: argparse.Namespace) -> int:
    cfg, model, revision, eval_cfg = _build_eval_context(args.config, args.model, args.revision)
    targeted_plan_id = getattr(args, "targeted_plan_id", None)
    targeted_plan_version = getattr(args, "targeted_plan_version", None)
    if (targeted_plan_id is None) != (targeted_plan_version is None):
        raise ValueError("targeted plan ID and version must be supplied together")
    if targeted_plan_id is not None and (
        not re.fullmatch(r"plan-targeted-[a-f0-9]{20}", targeted_plan_id)
        or not isinstance(targeted_plan_version, int) or targeted_plan_version < 1
    ):
        raise ValueError("invalid targeted plan identity")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and ((output / "eval.json").exists() or (output / "evaluation_lineage.json").exists()):
        raise FileExistsError(f"evaluation output already contains a result: {output}; choose a fresh output directory")
    commands = _eval_commands(cfg=cfg, model=model, revision=revision, eval_cfg=eval_cfg, output=output, limit=args.limit)
    resolved_evaluation = {
        "profile_id": (
            "cpt-math-v1" if args.stage == "cpt"
            else "sft-math-eval-v1" if args.stage == "sft"
            else f"{args.stage}-math-eval-v1"
        ),
        "protocol_version": eval_cfg.get("protocol_version", "unknown"),
        "prompt_version": eval_cfg.get("prompt_version"),
        "evalscope": {
            "benchmarks": list(eval_cfg.get("benchmarks", ["gsm8k", "math_500"])),
            "limit_per_subset": args.limit,
            "benchmark_execution": eval_cfg.get("benchmark_execution"),
            "seed": cfg.get("seed", 42),
            "model_args": {"precision": "torch.float16", "device_map": "auto"},
            "generation": eval_cfg.get("generation", {}),
        },
        "internal_math": ({
            "input": str(args.internal_holdout),
            "limit": args.internal_limit,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "answer_parser": "extract_answer+answers_equal:v1",
        } if args.internal_holdout else None),
        "domain_perplexity": ({
            "input": str(args.domain_holdout),
            "limit": args.domain_limit,
            "max_length": args.ppl_max_length,
            "dtype": args.dtype,
        } if args.domain_holdout else None),
        "targeted_plan": ({"plan_id": targeted_plan_id, "plan_version": targeted_plan_version}
                          if targeted_plan_id is not None else None),
    }
    run_label = getattr(args, "run_label", None) or cfg.get("run_label", cfg.get("run_id", f"{args.stage}-evaluation"))
    requested_run_id = getattr(args, "run_id", None)
    if requested_run_id and not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", requested_run_id):
        raise ValueError("invalid evaluation run_id")
    run_id = None if args.dry_run else (requested_run_id or new_run_id("evaluate", label=run_label))
    eval_id = None if args.dry_run else f"eval-{uuid4().hex}"
    started_at = datetime.now().astimezone().isoformat()
    execution_error = None
    optional_errors: list[str] = []
    executions: list[dict[str, object]] = []
    try:
        if not args.dry_run:
            # EvalScope writes review caches into this per-model directory. Create it
            # up front so cache writers do not race directory creation (seen in GRPO
            # MATH-500 runs on Windows).
            review_root = output / "evalscope" / "reviews" / Path(model).name
            review_root.mkdir(parents=True, exist_ok=True)
        for cmd in commands:
            if getattr(args, "stream_output", False):
                executions.append(run_evalscope(cmd, dry_run=args.dry_run, stream_output=True))
            else:
                executions.append(run_evalscope(cmd, dry_run=args.dry_run))
        official_exec = executions[0] if len(executions) == 1 else {"executions": executions}
    except RuntimeError as exc:
        official_exec = {"status": "FAILED", "error": str(exc), "executions": executions}
        execution_error = str(exc)
    result: dict[str, object] = {
        "run_label": run_label,
        "stage": args.stage,
        "model": model,
        "model_revision": revision,
        "resolved_evaluation": resolved_evaluation,
        "official_benchmarks": official_exec,
    }
    if run_id:
        result["run_id"] = run_id
        result["eval_id"] = eval_id
    if not args.dry_run and execution_error is None:
        try:
            result["evalscope_reports"] = collect_evalscope_reports(output / "evalscope")
            reports = result["evalscope_reports"]
            for benchmark in eval_cfg.get("benchmarks", ["gsm8k", "math_500"]):
                report = next((value for name, value in reports.items() if benchmark in str(name).lower()), None)
                if not isinstance(report, dict) or not isinstance(report.get("score"), (int, float)):
                    optional_errors.append(f"missing benchmark score: {benchmark}")
        except (OSError, ValueError) as exc:
            optional_errors.append(f"collect_evalscope_reports: {exc}")
    if args.internal_holdout and not args.dry_run and execution_error is None:
        try:
            generator = TransformersGenerator(
                model,
                revision=revision,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )
            internal = evaluate_internal_math(args.internal_holdout, generator, output / "internal", limit=args.internal_limit)
            result["internal_math"] = internal.to_dict()
            # Release the first model before loading a second for domain PPL.
            del generator
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except (ImportError, RuntimeError) as exc:
                optional_errors.append(f"cuda_cache_cleanup: {exc}")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            optional_errors.append(f"internal_math: {exc}")
    elif args.internal_holdout:
        result["internal_math"] = {"planned": True, "input": args.internal_holdout, "limit": args.internal_limit}
    if args.domain_holdout and not args.dry_run and execution_error is None:
        try:
            ppl = evaluate_cpt_perplexity(
                model=model,
                revision=revision,
                input_path=args.domain_holdout,
                output_path=output / "domain" / "perplexity.json",
                max_length=args.ppl_max_length,
                limit=args.domain_limit,
                dtype=args.dtype,
            )
            result["domain_perplexity"] = ppl.to_dict()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            optional_errors.append(f"domain_perplexity: {exc}")
    elif args.domain_holdout:
        result["domain_perplexity"] = {"planned": True, "input": args.domain_holdout, "limit": args.domain_limit}
    result["metric_statuses"] = _evaluation_metric_statuses(
        stage=args.stage,
        evalscope_reports=result.get("evalscope_reports", {}),
        internal_math=result.get("internal_math"),
        domain_perplexity=result.get("domain_perplexity"),
        has_internal_holdout=bool(args.internal_holdout),
        has_domain_holdout=bool(args.domain_holdout),
        execution_error=execution_error,
    )
    (output / "eval.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.dry_run:
        registered_model = None
        registered_model_id = getattr(args, "registered_model_id", None)
        if registered_model_id:
            from .ui.stage_evaluation import StageEvaluationAdapter
            registered_model, _ = StageEvaluationAdapter(Path.cwd())._registered_base_model(registered_model_id)
            registered_path = StageEvaluationAdapter(Path.cwd())._model_path(registered_model)
            if registered_path != Path(model).resolve(strict=True):
                raise ValueError("评测模型路径与已登记 base model ID 不匹配")
            model_ref = ArtifactRef(id=registered_model.model_id, kind="model", path=str(registered_path),
                                   digest=registered_model.digest, role="evaluation_model")
            model_producer_run_id = registered_model.produced_by_run_id
        else:
            model_ref = _model_lineage_ref(model, revision)
            model_producer_run_id = _registered_model_producer(model, model_ref)
        dataset_refs: list[ArtifactRef] = []
        dataset_ids: list[str] = []
        for benchmark in eval_cfg.get("benchmarks", ["gsm8k", "math_500"]):
            dataset_id = f"benchmark-{benchmark}"
            dataset_ids.append(dataset_id)
            dataset_refs.append(ArtifactRef(
                id=dataset_id, kind="dataset", role="benchmark", availability="unavailable",
                digest=Digest(status="unavailable", reason=(
                    "EvalScope 未暴露此 benchmark 的解析 revision 或内容摘要"
                )),
            ))
        lineage_warnings: list[str] = [
            "benchmark 仅记录 EvalScope 数据集 ID；未捕获解析 revision/内容摘要，评测 lineage 不可完全复现。"
        ] if dataset_refs else []
        incomplete = (model_ref.digest is None or model_ref.digest.status != "available"
                      or any(ref.availability != "available" or ref.digest is None
                             or ref.digest.status != "available" for ref in dataset_refs))
        if not model_producer_run_id and not (registered_model and registered_model.kind == "base"):
            incomplete = True
            lineage_warnings.append("model artifact registration did not identify its producing training/merge run")
        for input_path, input_role in (
            (args.internal_holdout, "internal_holdout"),
            (args.domain_holdout, "cpt_holdout" if args.stage == "cpt" else "domain_holdout"),
        ):
            if input_path:
                ref, evidence_complete, warnings = _dataset_lineage_ref(input_path, role=input_role)
                dataset_refs.append(ref)
                dataset_ids.append(ref.id)
                incomplete = incomplete or not evidence_complete
                lineage_warnings.extend(warnings)
        metric_values: dict[str, dict[str, object]] = {}
        def add_metrics(prefix: str, value: object, denominator: int | None = None,
                        denominator_key: str | None = None) -> None:
            if isinstance(value, dict):
                explicit_denominator = value.get(denominator_key) if denominator_key else None
                samples = (explicit_denominator if isinstance(explicit_denominator, int)
                           else value.get("samples") if isinstance(value.get("samples"), int)
                           else denominator)
                for key, child in value.items():
                    if isinstance(child, (int, float)) and not isinstance(child, bool):
                        metric_values[f"{prefix}.{key}" if prefix else key] = {"value": child, "denominator": samples}
                    elif isinstance(child, dict):
                        add_metrics(f"{prefix}.{key}" if prefix else key, child, samples, denominator_key)
        add_metrics("evalscope", result.get("evalscope_reports", {}), args.limit)
        add_metrics("internal_math", result.get("internal_math", {}))
        add_metrics("domain_perplexity", result.get("domain_perplexity", {}), denominator_key="predicted_tokens")
        protocol_version = str(eval_cfg.get("protocol_version") or "unknown")
        evaluation_status = "failed" if execution_error else ("partial" if optional_errors else "completed")
        evaluation = EvaluationLineage(
            eval_id=eval_id, produced_by_run_id=run_id, model_id=model_ref.id,
            model_produced_by_run_id=model_producer_run_id,
            targeted_plan_id=targeted_plan_id, targeted_plan_version=targeted_plan_version,
            dataset_ids=dataset_ids, dataset_artifacts=dataset_refs,
            protocol_version=protocol_version,
            config_sha256=sha256_json(resolved_evaluation),
            status=evaluation_status, metrics=metric_values,
        )
        eval_run = RunRecord(
            run_id=run_id, run_type="evaluate",
            status="failed" if execution_error else ("partial" if optional_errors else "completed"),
            lineage_status="incomplete" if incomplete else "complete",
            run_label=run_label, started_at=started_at, completed_at=datetime.now().astimezone().isoformat(),
            targeted_plan_id=targeted_plan_id, targeted_plan_version=targeted_plan_version,
            inputs=[model_ref, *dataset_refs],
            outputs=[ArtifactRef(id=eval_id, kind="evaluation", path="eval.json", role="metrics")] if not execution_error else [],
            lineage_warnings=lineage_warnings,
            errors=([execution_error] if execution_error else []) + optional_errors,
        )
        write_json(output / "evaluation_lineage.json", evaluation.model_dump(mode="json"))
        write_json(output / "evaluation_run.json", eval_run.model_dump(mode="json"))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 4 if execution_error or optional_errors else 0


def cmd_run_registered_grpo_eval(args: argparse.Namespace) -> int:
    """Run the shared math evaluation for a registered GRPO model and attach its real rollout evidence."""
    from .lineage import ModelLineage
    from .synthetic.stage_effects import build_stage_effect_report
    from .ui.training_preflight import _find_model, _has_symlink_between

    workspace = Path(args.workspace).expanduser().resolve(strict=True)
    if not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", args.run_id):
        raise ValueError("--run-id must match run-[a-z0-9][a-z0-9-]{0,119}")
    expected_output = f"runs/evaluations/{args.run_id}"
    if args.output != expected_output:
        raise ValueError("输出位置必须是 runs/evaluations/{run_id}")
    output = workspace / expected_output
    if _has_symlink_between(output, workspace) or output.exists() or output.is_symlink():
        raise FileExistsError("Evaluation Run 输出路径不可用或已存在；历史结果不覆盖")

    registry_path, model = _find_model(workspace, args.model_id)
    if model.kind != "merged" or model.stage != "grpo" or not model.adapter_id or not model.produced_by_run_id:
        raise ValueError("GRPO evaluation 只接受具备 merge/adapter lineage 的登记 merged model")
    model_path = Path(model.path or "")
    if not model_path.is_absolute():
        model_path = workspace / model_path
    if _has_symlink_between(model_path, workspace):
        raise ValueError("GRPO merged model 路径不能经过符号链接")
    model_path = model_path.resolve(strict=True)
    model_path.relative_to(workspace)
    model_digest = sha256_directory(model_path) if model_path.is_dir() else sha256_file(model_path)
    if model.digest.status != "available" or model.digest.value != model_digest:
        raise ValueError("GRPO merged model 内容摘要与登记不一致")

    training_root = registry_path.parent.parent
    merge_path = training_root / "merge_runs" / model.produced_by_run_id / "lineage.json"
    if _has_symlink_between(merge_path, workspace) or not merge_path.is_file():
        raise ValueError("GRPO merged model 的 merge Run lineage 缺失")
    merge_run = RunRecord.model_validate_json(merge_path.read_text(encoding="utf-8"))
    if (merge_run.run_id != model.produced_by_run_id or merge_run.run_type != "merge"
            or merge_run.status != "completed"
            or not any(item.id == model.model_id for item in merge_run.outputs)
            or not any(item.id == model.adapter_id for item in merge_run.inputs)):
        raise ValueError("GRPO merge Run 与模型/adapter identity 不匹配")
    adapter_path = registry_path.parent / f"{model.adapter_id}.json"
    if _has_symlink_between(adapter_path, workspace) or not adapter_path.is_file():
        raise ValueError("GRPO training adapter registry 缺失")
    adapter_model = ModelLineage.model_validate_json(adapter_path.read_text(encoding="utf-8"))
    if adapter_model.produced_by_run_id is None or adapter_model.stage != "grpo":
        raise ValueError("GRPO adapter 没有生产训练 Run lineage")
    training_lineage_path = adapter_path.parent.parent / "lineage.json"
    training_result_path = adapter_path.parent.parent / "run.json"
    if (_has_symlink_between(training_lineage_path, workspace) or not training_lineage_path.is_file()
            or _has_symlink_between(training_result_path, workspace) or not training_result_path.is_file()):
        raise ValueError("GRPO training Run evidence 缺失")
    training_run = RunRecord.model_validate_json(training_lineage_path.read_text(encoding="utf-8"))
    training_result = json.loads(training_result_path.read_text(encoding="utf-8"))
    if (training_run.run_id != adapter_model.produced_by_run_id or training_run.run_type != "train"
            or training_run.status != "completed"
            or not any(item.id == model.adapter_id for item in training_run.outputs)
            or training_result.get("run_id") != training_run.run_id):
        raise ValueError("GRPO model training Run 未成功完成或产物身份不匹配")
    targeted_plan_id = getattr(args, "targeted_plan_id", None) or training_run.targeted_plan_id
    targeted_plan_version = getattr(args, "targeted_plan_version", None) or training_run.targeted_plan_version
    if (targeted_plan_id, targeted_plan_version) != (training_run.targeted_plan_id,
                                                       training_run.targeted_plan_version):
        raise ValueError("评测指定的 targeted plan 与训练 Run lineage 不一致")
    targeted_plan = None
    if targeted_plan_id is not None:
        from .data.targeted_stage_plan import load_registered_targeted_stage_plan

        targeted_plan = load_registered_targeted_stage_plan(workspace, targeted_plan_id,
                                                              targeted_plan_version)
        if targeted_plan.target.stage != "grpo":
            raise ValueError("GRPO 训练 Run 关联的 targeted plan 阶段不匹配")
    metrics_path = training_root / "grpo_metrics.json"
    if _has_symlink_between(metrics_path, workspace):
        raise ValueError("GRPO training metrics path cannot contain symbolic links")
    if metrics_path.is_file():
        grpo_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if training_result.get("grpo_metrics") != grpo_metrics:
            raise ValueError("GRPO metrics artifact 与训练 run.json 不一致")
    else:
        grpo_metrics = {"protocol": "grpo-training-metrics-v1", "status": "not_measured",
                        "reason": "producing GRPO Run has no grpo_metrics.json artifact",
                        "metric_statuses": {}}

    forwarded = argparse.Namespace(
        config=str(workspace / "recipes/mini/30-eval-math-v1.yaml"),
        model=str(model_path), revision=None, output=str(output), run_label=f"grpo-eval-{args.model_id}",
        run_id=args.run_id, targeted_plan_id=targeted_plan_id,
        targeted_plan_version=targeted_plan_version, stage="grpo",
        limit=5, internal_holdout=None, internal_limit=20, max_new_tokens=512,
        domain_holdout=None, domain_limit=20, ppl_max_length=1024, dtype="float16",
        dry_run=False, stream_output=True,
    )
    eval_status = cmd_run_model_eval(forwarded)
    eval_path = output / "eval.json"
    if not eval_path.is_file():
        return eval_status
    report = json.loads(eval_path.read_text(encoding="utf-8"))
    latest = grpo_metrics.get("latest") if isinstance(grpo_metrics.get("latest"), dict) else {}
    components = grpo_metrics.get("reward_components") if isinstance(grpo_metrics.get("reward_components"), dict) else {}
    source_statuses = grpo_metrics.get("metric_statuses") if isinstance(grpo_metrics.get("metric_statuses"), dict) else {}
    beta = grpo_metrics.get("beta", training_result.get("training", {}).get("beta"))
    names = {
        "reward": "reward", "reward_std": "reward_std", "frac_reward_zero_std": "frac_reward_zero_std",
        "kl": "kl", "completions_mean_length": "completions/mean_length",
        "completions_clipped_ratio": "completions/clipped_ratio",
    }
    evidence_metrics: dict[str, dict[str, object]] = {}
    evidence_metric_statuses: dict[str, dict[str, object]] = {}
    log_points = grpo_metrics.get("log_points") if isinstance(grpo_metrics.get("log_points"), int) else 0
    for public_name, source_name in names.items():
        key = f"grpo_training.{public_name}"
        source = source_statuses.get(source_name, {})
        value = latest.get(source_name)
        status = ("not_applicable" if public_name == "kl" and beta == 0 else
                  source.get("status", "not_measured") if isinstance(source, dict) else "not_measured")
        denominator = source.get("denominator", log_points) if isinstance(source, dict) else log_points
        reason = source.get("reason") if isinstance(source, dict) else None
        if value is None and status == "measured":
            status, reason = "not_measured", "training metrics artifact did not contain a numeric value"
        evidence_metrics[key] = {"value": value, "denominator": denominator,
                                 "denominator_unit": "trainer_log_points", "status": status,
                                 "source_run_id": training_run.run_id, "reason": reason}
        evidence_metric_statuses[key] = {key_: val for key_, val in evidence_metrics[key].items()
                                         if key_ != "source_run_id"}
    for component in ("onegpu_accuracy", "onegpu_boxed_format"):
        key = f"grpo_training.reward_components.{component}"
        component_data = components.get(component) if isinstance(components.get(component), dict) else {}
        value = component_data.get("latest_mean")
        source = source_statuses.get(f"reward_components.{component}", {})
        status = source.get("status", "not_measured") if isinstance(source, dict) else "not_measured"
        denominator = source.get("denominator", log_points) if isinstance(source, dict) else log_points
        evidence_metrics[key] = {"value": value, "denominator": denominator,
                                 "denominator_unit": "trainer_log_points", "status": status,
                                 "source_run_id": training_run.run_id,
                                 "reason": source.get("reason") if isinstance(source, dict) else None}
        evidence_metric_statuses[key] = {key_: val for key_, val in evidence_metrics[key].items()
                                         if key_ != "source_run_id"}
    report["grpo_training_evidence"] = {
        "training_run_id": training_run.run_id, "merge_run_id": merge_run.run_id,
        "metrics_artifact": str(metrics_path) if metrics_path.is_file() else None,
        "status": grpo_metrics.get("status", "not_measured"), "metrics": evidence_metrics,
    }
    report.setdefault("metric_statuses", {}).update(evidence_metric_statuses)
    report["resolved_evaluation"]["profile_id"] = "grpo-math-eval-v1"
    eval_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lineage_path = output / "evaluation_lineage.json"
    run_path = output / "evaluation_run.json"
    if lineage_path.is_file() and run_path.is_file():
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        lineage.setdefault("metrics", {}).update(evidence_metrics)
        lineage_path.write_text(json.dumps(lineage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        metrics_digest = sha256_file(metrics_path) if metrics_path.is_file() else None
        run.setdefault("inputs", []).append(ArtifactRef(
            id=f"grpo-metrics-{training_run.run_id}", kind="file",
            path=str(metrics_path) if metrics_path.is_file() else None,
            digest=Digest(value=metrics_digest, status="available") if metrics_digest else
                   Digest(status="not_computed", reason="GRPO training metrics artifact not produced"),
            role="grpo_training_diagnostics",
            availability="available" if metrics_digest else "not_computed",
        ).model_dump(mode="json"))
        run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if targeted_plan is not None and eval_status == 0:
            after_lineage = EvaluationLineage.model_validate_json(lineage_path.read_text(encoding="utf-8"))
            if after_lineage.status == "completed":
                before_lineage = _find_targeted_grpo_baseline(workspace, targeted_plan)
                effect = build_stage_effect_report(
                    plan=targeted_plan, training_run=training_run, model_id=model.model_id,
                    after=after_lineage, before=before_lineage,
                )
                effect_path = output / "effect_report.json"
                write_json(effect_path, effect)
                run.setdefault("outputs", []).append(ArtifactRef(
                    id=effect["report_id"], kind="file", path=str(effect_path), role="targeted_effect_report",
                    digest=Digest(value=sha256_file(effect_path), status="available"),
                ).model_dump(mode="json"))
                run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return eval_status


def _find_targeted_grpo_baseline(workspace: Path, plan) -> EvaluationLineage | None:
    """Find only a completed baseline matching the plan's model and evaluation contract."""
    contract = plan.diagnostic_evaluation
    evaluations_root = workspace / "runs" / "evaluations"
    if evaluations_root.is_symlink() or not evaluations_root.is_dir():
        return None
    matches = []
    for path in list(evaluations_root.glob("**/evaluation_lineage.json"))[:2000]:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(workspace):
            continue
        try:
            evaluation = EvaluationLineage.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if (evaluation.status == "completed" and evaluation.model_id == plan.starting_model.model_id
                and contract.dataset_id in evaluation.dataset_ids
                and evaluation.protocol_version == contract.protocol_version
                and evaluation.config_sha256 == contract.config_sha256):
            matches.append(evaluation)
    return matches[0] if len(matches) == 1 else None


def cmd_plan_base_eval(args: argparse.Namespace) -> int:
    # Kept for compatibility; delegates to generic stage evaluation.
    args.stage = "base"
    args.run_label = getattr(args, "run_label", None) or "base-smoke-s42"
    args.revision = None
    args.domain_holdout = getattr(args, "domain_holdout", None)
    args.domain_limit = getattr(args, "domain_limit", None)
    args.ppl_max_length = getattr(args, "ppl_max_length", 1024)
    args.dtype = getattr(args, "dtype", "float16")
    return cmd_plan_model_eval(args)


def cmd_run_base_eval(args: argparse.Namespace) -> int:
    args.stage = "base"
    args.run_label = getattr(args, "run_label", None) or "base-smoke-s42"
    args.revision = None
    return cmd_run_model_eval(args)


def cmd_plan_peft_cpt(args: argparse.Namespace) -> int:
    recipe = _load_yaml(args.config)
    preflight = preflight_mini_cpt(recipe, args.dataset, model=args.model) if isinstance(recipe.get("dataset"), dict) else None
    model_cfg = recipe.get("model_config", "configs/model/qwen3-0.6b-base.yaml")
    model, revision = _resolve_model_config(model_cfg)
    if args.model is not None:
        model, revision = args.model, args.revision
    elif args.revision is not None:
        revision = args.revision
    training = dict(recipe.get("training", {}))
    training.setdefault("seed", recipe.get("seed", 42))
    output = Path(args.output)
    cmd = build_peft_cpt_command(
        model=model,
        model_revision=revision,
        dataset=args.dataset,
        output_dir=output / "train",
        training=training,
    )
    payload = {
        "run_label": recipe.get("run_label", recipe.get("run_id", "cpt-peft-smoke-s42")),
        "stage": "peft_cpt",
        "model": model,
        "model_revision": revision,
        "dataset": str(args.dataset),
        "training": training,
        "train_command": cmd,
        "merge_policy": "latest checkpoint -> explicit merged model directory",
        "environment": capture_environment(),
        "preflight": preflight,
    }
    write_json(output / "train_plan.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_run_peft_cpt(args: argparse.Namespace) -> int:
    recipe = _load_yaml(args.config)
    preflight = (getattr(args, "registered_preflight", None)
                 or (preflight_mini_cpt(recipe, args.dataset, model=args.model)
                     if isinstance(recipe.get("dataset"), dict) else None))
    model_cfg = recipe.get("model_config", "configs/model/qwen3-0.6b-base.yaml")
    model, revision = _resolve_model_config(model_cfg)
    if args.model is not None:
        model, revision = args.model, args.revision
    elif args.revision is not None:
        revision = args.revision
    training = dict(recipe.get("training", {}))
    training.setdefault("seed", recipe.get("seed", 42))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _assert_new_run_directory(output, dry_run=args.dry_run)
    model_ref = None
    if not args.dry_run:
        model_ref = _model_lineage_ref(model)
        _dataset_lineage_ref(args.dataset)
    supplied_run_id = getattr(args, "run_id", None)
    if supplied_run_id is not None and not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", supplied_run_id):
        raise ValueError("--run-id must match run-[a-z0-9][a-z0-9-]{0,119}")
    run_id = None if args.dry_run else (supplied_run_id or new_run_id("train", label=recipe.get("run_label", recipe.get("run_id", "cpt-peft-smoke-s42"))))
    started_at = datetime.now().astimezone().isoformat()
    train_dir = output / "train"
    cmd = build_peft_cpt_command(
        model=model,
        model_revision=revision,
        dataset=args.dataset,
        output_dir=train_dir,
        training=training,
        validation_dataset=getattr(args, "validation_dataset", None),
    )
    environment = capture_environment()
    write_json(output / "environment.json", environment)
    train_exec = run_monitored_command(cmd, log_dir=output / "logs" / "train", dry_run=args.dry_run, allow_failure=True)
    result: dict[str, object] = {
        "run_label": recipe.get("run_label", recipe.get("run_id", "cpt-peft-smoke-s42")),
        "stage": "peft_cpt",
        "model": model,
        "model_revision": revision,
        "dataset": str(args.dataset),
        "training": training,
        "train": train_exec,
        "environment": environment,
        "preflight": preflight,
        **({"training_validation": getattr(args, "validation_metadata", None)}
           if getattr(args, "validation_metadata", None) else {}),
    }
    if args.dry_run:
        result["checkpoint"] = {"planned": True}
        result["merged_model"] = {"planned": bool(args.merge)}
    elif train_exec.get("status") == "PASS":
        checkpoint_selection = _select_training_checkpoint(train_dir, recipe)
        checkpoint = Path(str(checkpoint_selection["selected"]))
        result["checkpoint_selection"] = checkpoint_selection
        result["checkpoint"] = str(checkpoint)
        if args.merge:
            merged_dir = output / "merged" / "M1-CPT"
            merge_exec = _run_merge_with_lineage(output=output, checkpoint=checkpoint, parent_model=model,
                                                 merged_dir=merged_dir, run_label="m1-cpt-merge",
                                                 experiment_id=recipe.get("experiment_id"), stage="cpt", dry_run=args.dry_run)
            result["merge"] = merge_exec
            result["merged_model"] = merge_exec.get("merged_model")
    else:
        result["checkpoint"] = None
        result["merged_model"] = None
    _write_training_run(output=output, result=result, run_id=run_id,
                        run_label=recipe.get("run_label", recipe.get("run_id", "cpt-peft-smoke-s42")),
                        experiment_id=recipe.get("experiment_id"), stage="cpt", model=model,
                        dataset=args.dataset, config_path=args.config,
                        resolved_config={"model": model, "revision": revision, "dataset": str(args.dataset), "training": training,
                                         "preflight": preflight, "usage": (preflight or {}).get("usage"),
                                         "training_validation": getattr(args, "validation_metadata", None)},
                        started_at=started_at, model_ref=model_ref,
                        validation_dataset=getattr(args, "validation_dataset", None),
                        validation_metadata=getattr(args, "validation_metadata", None))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if args.dry_run or train_exec.get("status") == "PASS" else 4


def _prepare_registered_training_validation(*, workspace: Path, stage: str, run_id: str | None,
                                            training_manifest: Path, training_native: str | Path,
                                            recipe: dict, output_root: Path) -> dict:
    """Revalidate the confirmed split identity and materialize ms-swift inputs."""
    from .data.manifest import load_manifest, resolve_canonical_path, sha256_file
    from .ui.training_preflight import _find_manifest
    from .ui.training_validation import (
        apply_native_partition, inspect_registered_validation_dataset, plan_random_split, read_jsonl_records,
    )
    from .ui.stage_profiles import stage_profile_catalog

    ui_metadata = recipe.get("_onegpu_ui") if isinstance(recipe, dict) else None
    validation = ui_metadata.get("validation", {"mode": "none"}) if isinstance(ui_metadata, dict) else {"mode": "none"}
    if not isinstance(validation, dict) or validation.get("mode") not in {"none", "registered", "random_split"}:
        raise ValueError("resolved recipe validation metadata is invalid")
    mode = validation["mode"]
    if mode == "none":
        return {"train_dataset": str(training_native), "validation_dataset": None, "metadata": validation}

    train_manifest = load_manifest(training_manifest)
    train_canonical = resolve_canonical_path(training_manifest, train_manifest)
    if sha256_file(training_manifest) != validation.get("source_manifest_sha256", sha256_file(training_manifest)):
        raise ValueError("training manifest changed after validation preflight")
    if sha256_file(train_canonical) != validation.get("source_content_sha256", sha256_file(train_canonical)):
        raise ValueError("training canonical data changed after validation preflight")

    if mode == "registered":
        validation_dataset_id = validation.get("dataset_id")
        if not isinstance(validation_dataset_id, str) or not validation_dataset_id:
            raise ValueError("registered validation dataset identity is missing")
        profile = next((item for item in stage_profile_catalog(workspace).profiles
                        if item.stage == stage and item.profile_id == recipe.get("_onegpu_ui", {}).get("profile_id")), None)
        allowed_roles = profile.validation_dataset_roles if profile is not None else ["validation", "diagnostic_dev"]
        inspected = inspect_registered_validation_dataset(
            workspace, validation_dataset_id, stage, allowed_roles,
        )
        if (inspected["manifest_sha256"] != validation.get("manifest_sha256")
                or inspected["content_sha256"] != validation.get("content_sha256")):
            raise ValueError("registered validation dataset changed after preflight")
        validation_manifest = _find_manifest(workspace, validation_dataset_id)
        native_output = workspace / "runs" / "dataset-adapters" / f"{run_id}-validation"
        adapted = adapt_registered_dataset(validation_manifest, native_output)
        native_validation = native_output / adapted["native_path"]
        return {"train_dataset": str(training_native), "validation_dataset": str(native_validation),
                "metadata": validation}

    ratio, seed = validation.get("ratio"), validation.get("seed")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("random validation split parameters are invalid")
    plan = plan_random_split(read_jsonl_records(train_canonical), ratio=float(ratio), seed=seed)
    for key in ("algorithm", "split_identity_sha256", "train_count", "validation_count",
                "train_members_sha256", "validation_members_sha256", "membership_manifest_sha256"):
        if plan.get(key) != validation.get(key):
            raise ValueError("deterministic validation split differs from confirmed membership summary")
    if len(read_jsonl_records(training_native)) != train_manifest.sample_count:
        raise ValueError("trainer-native adapter record count does not match registered canonical dataset")
    split = apply_native_partition(training_native, plan, output_root / "validation-split")
    metadata = {**validation, "train_native_sha256": split["train_native_sha256"],
                "validation_native_sha256": split["validation_native_sha256"],
                "membership_manifest_path": split["membership_manifest_path"],
                "membership_manifest_sha256": split["membership_manifest_sha256"]}
    return {"train_dataset": split["train_path"], "validation_dataset": split["validation_path"],
            "metadata": metadata}


def cmd_run_registered_cpt(args: argparse.Namespace) -> int:
    """Preflight an immutable CPT dataset, adapt it, then use the existing monitored runner."""
    workspace = Path(args.workspace).expanduser().resolve(strict=True)
    manifest_arg = Path(args.manifest).expanduser()
    manifest = (manifest_arg if manifest_arg.is_absolute() else workspace / manifest_arg).resolve(strict=True)
    output_arg = Path(args.output).expanduser()
    output = (output_arg if output_arg.is_absolute() else workspace / output_arg).resolve(strict=False)
    manifest.relative_to(workspace)
    output.relative_to(workspace)
    if getattr(args, "run_id", None) and ".onegpu-posttrain" in Path(args.config).parts:
        from .ui.resolved_stage_recipe import verify_materialized_stage_recipe
        verify_materialized_stage_recipe(workspace, args.config, profile_id="cpt-smoke-v1", run_id=args.run_id)
    recipe = _load_yaml(args.config)
    configured_model, configured_revision = _resolve_model_config(
        recipe.get("model_config", "configs/model/qwen3-0.6b-base.yaml"))
    model = args.model or configured_model
    revision = args.revision if args.revision is not None else configured_revision
    adapter_output = workspace / "runs" / "targeted-cpt-adapters" / f"adapter-{uuid4().hex[:12]}"
    prepared = prepare_registered_cpt_training(
        manifest, model=model, revision=revision, adapter_output=adapter_output,
    )
    recipe = _load_yaml(args.config)
    validation = _prepare_registered_training_validation(
        workspace=workspace, stage="cpt", run_id=getattr(args, "run_id", None),
        training_manifest=manifest, training_native=prepared["dataset"],
        recipe=recipe, output_root=adapter_output,
    )
    delegated = argparse.Namespace(
        config=args.config, model=prepared["preflight"]["model_path"],
        revision=revision, dataset=validation["train_dataset"], output=str(output),
        merge=args.merge, dry_run=args.dry_run, run_id=getattr(args, "run_id", None),
        registered_preflight=prepared["preflight"], validation_dataset=validation.get("validation_dataset"),
        validation_metadata=validation.get("metadata"),
    )
    return cmd_run_peft_cpt(delegated)


def cmd_run_registered_preference(args: argparse.Namespace) -> int:
    """Run DPO or SimPO from a verified registered dataset and model IDs only."""
    from .ui.preference_stage_adapter import PreferenceStageAdapter, PreferenceStageRequest
    from .ui.training_preflight import _find_manifest, _find_model

    workspace = Path(args.workspace).expanduser().resolve(strict=True)
    if not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", args.run_id):
        raise ValueError("--run-id must match run-[a-z0-9][a-z0-9-]{0,119}")
    request = PreferenceStageRequest(profile_id=args.profile_id, dataset_id=args.dataset_id,
                                     parent_model_id=args.parent_model_id)
    adapter = PreferenceStageAdapter(workspace)
    summary = adapter.preflight(request, launcher_enabled=True)
    if summary["status"] != "ready_for_cli":
        raise ValueError("Preference 阶段预检阻断：" + "; ".join(summary["blockers"]))
    profile = adapter.profiles[request.profile_id]
    expected_output = f"runs/{profile.stage}/{profile.profile_id}/{args.run_id}"
    if args.output != expected_output:
        raise ValueError("输出位置必须与阶段 profile 和 run_id 一致")
    output = workspace / expected_output
    if output.exists() or output.is_symlink():
        raise FileExistsError("Run 输出位置已存在；历史 Run 不覆盖")

    manifest = _find_manifest(workspace, request.dataset_id)
    _, parent = _find_model(workspace, request.parent_model_id)
    model_path = Path(parent.path or "")
    if not model_path.is_absolute():
        model_path = workspace / model_path
    model_path = model_path.resolve(strict=True)
    model_path.relative_to(workspace)
    native_output = workspace / "runs" / "dataset-adapters" / args.run_id
    adapt_registered_dataset(manifest, native_output)
    native_dataset = native_output / "native" / "preference.jsonl"
    config_path = profile.approved_recipe
    if getattr(args, "config", None):
        from .ui.resolved_stage_recipe import verify_materialized_stage_recipe
        verify_materialized_stage_recipe(workspace, args.config, profile_id=profile.profile_id, run_id=args.run_id)
        config_path = args.config
    recipe = _load_yaml(config_path)
    validation = _prepare_registered_training_validation(
        workspace=workspace, stage=profile.stage, run_id=args.run_id,
        training_manifest=manifest, training_native=native_dataset,
        recipe=recipe, output_root=native_output,
    )
    runner_args = argparse.Namespace(
        config=str(workspace / config_path) if not Path(config_path).is_absolute() else config_path, model=str(model_path),
        dataset=validation["train_dataset"], validation_dataset=validation.get("validation_dataset"),
        validation_metadata=validation.get("metadata"), output=str(output), merge=True, dry_run=False,
        run_id=args.run_id, workspace=str(workspace),
    )
    return cmd_run_dpo(runner_args) if profile.stage == "dpo" else cmd_run_simpo(runner_args)



def cmd_plan_full_cpt_probe(args: argparse.Namespace) -> int:
    recipe = _load_yaml(args.config)
    model_cfg = recipe.get("model_config", "configs/model/qwen3-0.6b-base.yaml")
    model, revision = _resolve_model_config(model_cfg)
    if args.model is not None:
        model, revision = args.model, args.revision
    elif args.revision is not None:
        revision = args.revision
    training = dict(recipe.get("training", {}))
    training.setdefault("seed", recipe.get("seed", 42))
    output = Path(args.output)
    cmd = build_full_cpt_probe_command(
        model=model,
        model_revision=revision,
        dataset=args.dataset,
        output_dir=output / "train",
        training=training,
    )
    payload = {
        "run_label": recipe.get("run_label", recipe.get("run_id", "cpt-full-probe-s42")),
        "stage": "full_cpt_probe",
        "purpose": "feasibility_only",
        "model": model,
        "model_revision": revision,
        "dataset": str(args.dataset),
        "training": training,
        "train_command": cmd,
        "result_semantics": {
            "PASS": "all configured optimizer steps completed",
            "OOM": "CUDA/system allocation failure; valid feasibility result",
            "FAILED": "non-memory execution failure; investigate before interpreting feasibility",
        },
        "environment": capture_environment(),
    }
    write_json(output / "probe_plan.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_run_full_cpt_probe(args: argparse.Namespace) -> int:
    recipe = _load_yaml(args.config)
    model_cfg = recipe.get("model_config", "configs/model/qwen3-0.6b-base.yaml")
    model, revision = _resolve_model_config(model_cfg)
    if args.model is not None:
        model, revision = args.model, args.revision
    elif args.revision is not None:
        revision = args.revision
    training = dict(recipe.get("training", {}))
    training.setdefault("seed", recipe.get("seed", 42))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _assert_new_run_directory(output, dry_run=args.dry_run)
    model_ref = None
    if not args.dry_run:
        model_ref = _model_lineage_ref(model)
        _dataset_lineage_ref(args.dataset)
    run_id = None if args.dry_run else new_run_id("train", label=recipe.get("run_label", recipe.get("run_id", "cpt-full-probe-s42")))
    started_at = datetime.now().astimezone().isoformat()
    train_dir = output / "train"
    cmd = build_full_cpt_probe_command(
        model=model,
        model_revision=revision,
        dataset=args.dataset,
        output_dir=train_dir,
        training=training,
    )
    environment = capture_environment()
    write_json(output / "environment.json", environment)
    execution = run_monitored_command(
        cmd, log_dir=output / "logs" / "train", dry_run=args.dry_run, allow_failure=True
    )
    result: dict[str, object] = {
        "run_label": recipe.get("run_label", recipe.get("run_id", "cpt-full-probe-s42")),
        "stage": "full_cpt_probe",
        "purpose": "feasibility_only",
        "model": model,
        "model_revision": revision,
        "dataset": str(args.dataset),
        "training": training,
        "execution": execution,
        "environment": environment,
    }
    if not args.dry_run and execution.get("status") == "PASS":
        try:
            checkpoint_selection = _select_training_checkpoint(train_dir, recipe)
            checkpoint = Path(str(checkpoint_selection["selected"]))
            result["checkpoint_selection"] = checkpoint_selection
            result["checkpoint"] = str(checkpoint)
        except FileNotFoundError:
            result["checkpoint"] = None
    _write_training_run(output=output, result=result, run_id=run_id,
                        run_label=recipe.get("run_label", recipe.get("run_id", "cpt-full-probe-s42")),
                        experiment_id=recipe.get("experiment_id"), stage="full_cpt_probe", model=model,
                        dataset=args.dataset, config_path=args.config,
                        resolved_config={"model": model, "revision": revision, "dataset": str(args.dataset), "training": training},
                        started_at=started_at, model_ref=model_ref)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    # Diagnostic semantics: OOM is a successful probe execution; generic failures are not.
    return 0 if args.dry_run or execution.get("status") in {"PASS", "OOM"} else 4


def cmd_plan_sft(args: argparse.Namespace) -> int:
    recipe = _load_yaml(args.config)
    training = dict(recipe.get("training", {}))
    training.setdefault("seed", recipe.get("seed", 42))
    policy_context = _policy_training_context(args.dataset, training, "sft")
    output = Path(args.output)
    cmd = build_sft_command(
        model=args.model, dataset=args.dataset, output_dir=output / "train", training=training,
        validation_dataset=getattr(args, "validation_dataset", None),
    )
    payload = {
        "run_label": recipe.get("run_label", recipe.get("run_id", "sft-smoke-s42")),
        "stage": "sft",
        "data_policy_context": policy_context,
        "parent_model": str(args.model),
        "dataset": str(args.dataset),
        "training": training,
        "train_command": cmd,
        "merge_policy": "latest LoRA checkpoint -> merged/M2-SFT",
        "environment": capture_environment(),
    }
    write_json(output / "train_plan.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_run_sft(args: argparse.Namespace) -> int:
    recipe = _load_yaml(args.config)
    supplied_run_id = getattr(args, "run_id", None)
    if supplied_run_id is not None and not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", supplied_run_id):
        raise ValueError("--run-id must match run-[a-z0-9][a-z0-9-]{0,119}")
    targeted_plan_id = getattr(args, "targeted_stage_plan_id", None)
    targeted_plan_version = getattr(args, "targeted_stage_plan_version", None)
    if (targeted_plan_id is None) != (targeted_plan_version is None):
        raise ValueError("targeted stage plan ID and version must be supplied together")
    if targeted_plan_id is not None and (
        not re.fullmatch(r"plan-targeted-[a-f0-9]{20}", targeted_plan_id)
        or not isinstance(targeted_plan_version, int) or targeted_plan_version < 1
    ):
        raise ValueError("invalid targeted stage plan identity")
    training = dict(recipe.get("training", {}))
    training.setdefault("seed", recipe.get("seed", 42))
    ui_config = recipe.get("_onegpu_ui") if isinstance(recipe.get("_onegpu_ui"), dict) else {}
    validation_metadata = (getattr(args, "validation_metadata", None)
                           or ui_config.get("validation") or {})
    policy_context = _policy_training_context(args.dataset, training, "sft")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _assert_new_run_directory(output, dry_run=args.dry_run)
    model_ref = None
    if not args.dry_run:
        model_ref = _model_lineage_ref(args.model)
        _dataset_lineage_ref(args.dataset)
    run_id = None if args.dry_run else (supplied_run_id or new_run_id("train", label=recipe.get("run_label", recipe.get("run_id", "sft-smoke-s42"))))
    started_at = datetime.now().astimezone().isoformat()
    train_dir = output / "train"
    cmd = build_sft_command(
        model=args.model, dataset=args.dataset, output_dir=train_dir, training=training,
        validation_dataset=getattr(args, "validation_dataset", None),
    )
    environment = capture_environment()
    write_json(output / "environment.json", environment)
    execution = run_monitored_command(
        cmd, log_dir=output / "logs" / "train", dry_run=args.dry_run, allow_failure=True
    )
    result: dict[str, object] = {
        "run_label": recipe.get("run_label", recipe.get("run_id", "sft-smoke-s42")),
        "stage": "sft",
        "data_policy_context": policy_context,
        "parent_model": str(args.model),
        "dataset": str(args.dataset),
        "training": training,
        "train": execution,
        "environment": environment,
        **({"training_validation": validation_metadata} if validation_metadata else {}),
        **({"profile_id": ui_config["profile_id"],
            "parameter_overrides": ui_config["parameter_overrides"],
            "training_memory_mode": ui_config.get("training_memory_mode"),
            "effective_config_sha256": ui_config["effective_config_sha256"]}
           if {"profile_id", "parameter_overrides", "effective_config_sha256"} <= ui_config.keys() else {}),
    }
    if targeted_plan_id is not None:
        result["targeted_stage_plan"] = {"plan_id": targeted_plan_id, "plan_version": targeted_plan_version}
    if args.dry_run:
        result["checkpoint"] = {"planned": True}
        result["merged_model"] = {"planned": bool(args.merge)}
    elif execution.get("status") == "PASS":
        checkpoint_selection = _select_training_checkpoint(train_dir, recipe)
        checkpoint = Path(str(checkpoint_selection["selected"]))
        result["checkpoint_selection"] = checkpoint_selection
        result["checkpoint"] = str(checkpoint)
        if args.merge:
            merged_dir = output / "merged" / "M2-SFT"
            merge_exec = _run_merge_with_lineage(output=output, checkpoint=checkpoint, parent_model=args.model,
                                                 merged_dir=merged_dir, run_label="m2-sft-merge",
                                                 experiment_id=recipe.get("experiment_id"), stage="sft", dry_run=args.dry_run)
            result["merge"] = merge_exec
            result["merged_model"] = merge_exec.get("merged_model")
    else:
        result["checkpoint"] = None
        result["merged_model"] = None
    _write_training_run(output=output, result=result, run_id=run_id,
                        run_label=recipe.get("run_label", recipe.get("run_id", "sft-smoke-s42")),
                        experiment_id=recipe.get("experiment_id"), stage="sft", model=args.model,
                        dataset=args.dataset, config_path=args.config,
                        resolved_config={"model": str(args.model), "dataset": str(args.dataset), "training": training,
                                         **({"profile_id": ui_config["profile_id"],
                                             "parameter_overrides": ui_config["parameter_overrides"],
                                             "training_memory_mode": ui_config.get("training_memory_mode"),
                                             "effective_config_sha256": ui_config["effective_config_sha256"]}
                                            if {"profile_id", "parameter_overrides", "effective_config_sha256"} <= ui_config.keys() else {}),
                                         **({"targeted_stage_plan": {"plan_id": targeted_plan_id,
                                                                      "plan_version": targeted_plan_version}}
                                            if targeted_plan_id is not None else {})},
                        started_at=started_at, model_ref=model_ref,
                        validation_dataset=getattr(args, "validation_dataset", None),
                        validation_metadata=validation_metadata,
                        targeted_plan_id=targeted_plan_id, targeted_plan_version=targeted_plan_version)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if args.dry_run or execution.get("status") == "PASS" else 4


def _plan_preference_stage(args: argparse.Namespace, algorithm: str, merged_name: str) -> int:
    recipe = _load_yaml(args.config)
    training = dict(recipe.get("training", {}))
    training.setdefault("seed", recipe.get("seed", 42))
    policy_context = _policy_training_context(args.dataset, training, "preference")
    output = Path(args.output)
    cmd = build_preference_rlhf_command(
        algorithm=algorithm,
        model=args.model,
        dataset=args.dataset,
        output_dir=output / "train",
        training=training,
        validation_dataset=getattr(args, "validation_dataset", None),
    )
    payload = {
        "run_label": recipe.get("run_label", recipe.get("run_id", f"{algorithm}-smoke-s42")),
        "stage": algorithm,
        "data_policy_context": policy_context,
        "parent_model": str(args.model),
        "dataset": str(args.dataset),
        "training": training,
        "train_command": cmd,
        "merge_policy": f"latest LoRA checkpoint -> merged/{merged_name}",
        "environment": capture_environment(),
    }
    write_json(output / "train_plan.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _run_preference_stage(args: argparse.Namespace, algorithm: str, merged_name: str) -> int:
    recipe = _load_yaml(args.config)
    supplied_run_id = getattr(args, "run_id", None)
    if supplied_run_id is not None and not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", supplied_run_id):
        raise ValueError("--run-id must match run-[a-z0-9][a-z0-9-]{0,119}")
    training = dict(recipe.get("training", {}))
    training.setdefault("seed", recipe.get("seed", 42))
    dataset_context = dict(recipe.get("dataset", {}))
    exploration_context = {
        "usage": recipe.get("usage", "unspecified"),
        "actual_record_count": dataset_context.get("actual_records"),
        "count_unit": dataset_context.get("unit"),
        "quality_report": dataset_context.get("quality_report"),
        "quality_status": dataset_context.get("quality_status", "unknown"),
        "official_mini_acceptance": "not_assessed",
    }
    targeted_context = preflight_targeted_training(
        dataset=args.dataset, model=args.model, stage=algorithm,
        workspace=Path(getattr(args, "workspace", Path.cwd())).resolve(), training=training,
    )
    policy_context = _policy_training_context(args.dataset, training, "preference")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_label = recipe.get("run_label", recipe.get("run_id", f"{algorithm}-smoke-s42"))
    _assert_new_run_directory(output, dry_run=args.dry_run)
    model_ref = None
    if not args.dry_run:
        model_ref = _model_lineage_ref(args.model)
        _dataset_lineage_ref(args.dataset)
    run_id = None if args.dry_run else (supplied_run_id or new_run_id("train", label=run_label))
    started_at = datetime.now().astimezone().isoformat()
    train_dir = output / "train"
    cmd = build_preference_rlhf_command(
        algorithm=algorithm,
        model=args.model,
        dataset=args.dataset,
        output_dir=train_dir,
        training=training,
        validation_dataset=getattr(args, "validation_dataset", None),
    )
    environment = capture_environment()
    write_json(output / "environment.json", environment)
    execution = run_monitored_command(
        cmd, log_dir=output / "logs" / "train", dry_run=args.dry_run, allow_failure=True
    )
    result: dict[str, object] = {
        "run_label": run_label,
        "stage": algorithm,
        "exploration_context": exploration_context,
        "targeted_stage_plan": targeted_context,
        "data_policy_context": policy_context,
        "parent_model": str(args.model),
        "dataset": str(args.dataset),
        "training": training,
        "train": execution,
        "environment": environment,
        **({"training_validation": getattr(args, "validation_metadata", None)}
           if getattr(args, "validation_metadata", None) else {}),
    }
    if args.dry_run:
        result["checkpoint"] = {"planned": True}
        result["merged_model"] = {"planned": bool(args.merge)}
    elif execution.get("status") == "PASS":
        checkpoint_selection = _select_training_checkpoint(train_dir, recipe)
        checkpoint = Path(str(checkpoint_selection["selected"]))
        result["checkpoint_selection"] = checkpoint_selection
        result["checkpoint"] = str(checkpoint)
        if args.merge:
            merged_dir = output / "merged" / merged_name
            merge_exec = _run_merge_with_lineage(output=output, checkpoint=checkpoint, parent_model=args.model,
                                                 merged_dir=merged_dir, run_label=f"{algorithm}-merge",
                                                 experiment_id=recipe.get("experiment_id"), stage=algorithm, dry_run=args.dry_run)
            result["merge"] = merge_exec
            result["merged_model"] = merge_exec.get("merged_model")
    else:
        result["checkpoint"] = None
        result["merged_model"] = None
    _write_training_run(output=output, result=result, run_id=run_id,
                        run_label=run_label, experiment_id=recipe.get("experiment_id"), stage=algorithm,
                        model=args.model, dataset=args.dataset, config_path=args.config,
                        resolved_config={"algorithm": algorithm, "model": str(args.model), "dataset": str(args.dataset),
                                         "training": training, "exploration_context": exploration_context,
                                         "targeted_stage_plan": targeted_context,
                                         "training_validation": getattr(args, "validation_metadata", None)},
                        started_at=started_at, model_ref=model_ref,
                        validation_dataset=getattr(args, "validation_dataset", None),
                        validation_metadata=getattr(args, "validation_metadata", None))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    merge_failed = bool(args.merge and execution.get("status") == "PASS"
                        and not (result.get("merged_model") and Path(str(result["merged_model"])).is_dir()))
    return 0 if args.dry_run or (execution.get("status") == "PASS" and not merge_failed) else 4


def cmd_plan_dpo(args: argparse.Namespace) -> int:
    return _plan_preference_stage(args, "dpo", "M3-DPO")


def cmd_run_dpo(args: argparse.Namespace) -> int:
    return _run_preference_stage(args, "dpo", "M3-DPO")


def cmd_plan_simpo(args: argparse.Namespace) -> int:
    return _plan_preference_stage(args, "simpo", "M3-SimPO")


def cmd_run_simpo(args: argparse.Namespace) -> int:
    return _run_preference_stage(args, "simpo", "M3-SimPO")



def _grpo_merged_name(profile: str) -> str:
    return "M3-GRPO-standard" if profile == "standard" else "M3-GRPO-lowmem"


def cmd_plan_grpo(args: argparse.Namespace) -> int:
    recipe = _load_yaml(args.config)
    profile = str(recipe.get("profile", "standard"))
    training = dict(recipe.get("training", {}))
    training.setdefault("seed", recipe.get("seed", 42))
    reward = dict(recipe.get("reward", {"accuracy": 1.0, "format": 0.1}))
    output = Path(args.output)
    cmd = build_grpo_command(
        model=args.model,
        dataset=args.dataset,
        output_dir=output / "train",
        training=training,
        reward=reward,
        external_plugin=args.external_plugin,
        validation_dataset=getattr(args, "validation_dataset", None),
    )
    per_device = int(training.get("per_device_train_batch_size_completions", 1))
    grad_acc = int(training.get("gradient_accumulation_steps", 4))
    max_steps = int(training.get("max_steps", 20))
    payload = {
        "run_label": recipe.get("run_label", recipe.get("run_id", f"grpo-{profile}-smoke-s42")),
        "stage": "grpo",
        "profile": profile,
        "parent_model": str(args.model),
        "dataset": str(args.dataset),
        "training": training,
        "reward": reward,
        "train_command": cmd,
        "budget": {
            "type": "completions",
            "effective_completion_batch_per_optimizer_step": per_device * grad_acc,
            "planned_optimizer_steps": max_steps,
            "planned_training_completion_instances": per_device * grad_acc * max_steps,
            "generation_batch_size": int(training.get("generation_batch_size_completions", per_device * grad_acc)),
            "num_generations": int(training.get("num_generations", 2)),
        },
        "merge_policy": f"PASS -> latest LoRA checkpoint -> merged/{_grpo_merged_name(profile)}",
        "failure_semantics": {
            "PASS": "training completed",
            "OOM": "valid 8GB feasibility outcome",
            "FAILED": "non-memory execution failure; investigate",
        },
        "environment": capture_environment(),
    }
    write_json(output / "train_plan.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_prepare_grpo_calibration(args: argparse.Namespace) -> int:
    manifest = prepare_grpo_calibration(args.config, output_root=args.output_root)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def cmd_summarize_grpo_calibration(args: argparse.Namespace) -> int:
    result = summarize_grpo_calibration(args.manifest, output_dir=args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_compare_evaluation_matrix(args: argparse.Namespace) -> int:
    result = build_evaluation_matrix(args.config, output_dir=args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_run_grpo(args: argparse.Namespace) -> int:
    recipe = _load_yaml(args.config)
    profile = str(recipe.get("profile", "standard"))
    training = dict(recipe.get("training", {}))
    training.setdefault("seed", recipe.get("seed", 42))
    dataset_context = dict(recipe.get("dataset", {}))
    exploration_context = {
        "usage": recipe.get("usage", "unspecified"),
        "actual_record_count": dataset_context.get("actual_records"),
        "count_unit": dataset_context.get("unit"),
        "quality_report": dataset_context.get("quality_report"),
        "quality_status": dataset_context.get("quality_status", "unknown"),
        "official_mini_acceptance": "not_assessed",
    }
    reward = dict(recipe.get("reward", {"accuracy": 1.0, "format": 0.1}))
    targeted_context = preflight_targeted_training(
        dataset=args.dataset, model=args.model, stage="grpo",
        workspace=Path(getattr(args, "workspace", Path.cwd())).resolve(), training=training,
        external_plugin=getattr(args, "external_plugin", None),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    run_label = recipe.get("run_label", recipe.get("run_id", f"grpo-{profile}-smoke-s42"))
    _assert_new_run_directory(output, dry_run=args.dry_run)
    model_ref = None
    if not args.dry_run:
        model_ref = getattr(args, "registered_model_ref", None) or _model_lineage_ref(args.model)
        _dataset_lineage_ref(args.dataset)
    supplied_run_id = getattr(args, "run_id", None)
    if supplied_run_id is not None and not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", supplied_run_id):
        raise ValueError("--run-id must match run-[a-z0-9][a-z0-9-]{0,119}")
    run_id = None if args.dry_run else (supplied_run_id or new_run_id("train", label=run_label))
    started_at = datetime.now().astimezone().isoformat()
    train_dir = output / "train"
    # Keep TensorBoard's asynchronous writer on a run-scoped directory that is
    # guaranteed to exist before MS-Swift initializes its training callbacks.
    logging_dir = output / "tensorboard"
    training["logging_dir"] = str(logging_dir)
    if not args.dry_run:
        logging_dir.mkdir(parents=True, exist_ok=False)
    cmd = build_grpo_command(
        model=args.model,
        dataset=args.dataset,
        output_dir=train_dir,
        training=training,
        reward=reward,
        external_plugin=args.external_plugin,
        validation_dataset=getattr(args, "validation_dataset", None),
    )
    environment = capture_environment()
    write_json(output / "environment.json", environment)
    execution = run_monitored_command(
        cmd,
        log_dir=output / "logs" / "train",
        dry_run=args.dry_run,
        allow_failure=True,
    )
    per_device = int(training.get("per_device_train_batch_size_completions", 1))
    grad_acc = int(training.get("gradient_accumulation_steps", 4))
    max_steps = int(training.get("max_steps", 20))
    result: dict[str, object] = {
        "run_label": run_label,
        "stage": "grpo",
        "profile": profile,
        "exploration_context": exploration_context,
        "targeted_stage_plan": targeted_context,
        "parent_model": str(args.model),
        "dataset": str(args.dataset),
        "training": training,
        "reward": reward,
        "budget": {
            "type": "completions",
            "effective_completion_batch_per_optimizer_step": per_device * grad_acc,
            "planned_optimizer_steps": max_steps,
            "planned_training_completion_instances": per_device * grad_acc * max_steps,
            "generation_batch_size": int(training.get("generation_batch_size_completions", per_device * grad_acc)),
            "num_generations": int(training.get("num_generations", 2)),
        },
        "train": execution,
        "environment": environment,
        **({"training_validation": getattr(args, "validation_metadata", None)}
           if getattr(args, "validation_metadata", None) else {}),
    }
    registered_preflight = getattr(args, "registered_preflight", None)
    if registered_preflight:
        result["reward_contract"] = registered_preflight["datasets"][0].get("reward_contract")
    if args.dry_run:
        result["checkpoint"] = {"planned": True}
        result["merged_model"] = {"planned": bool(args.merge)}
        result["grpo_metrics"] = {"planned": True}
    elif execution.get("status") == "PASS":
        try:
            checkpoint_selection = _select_training_checkpoint(train_dir, recipe)
            checkpoint = Path(str(checkpoint_selection["selected"]))
            result["checkpoint_selection"] = checkpoint_selection
        except (FileNotFoundError, OSError, ValueError) as exc:
            execution["status"] = "FAILED"
            execution["detail"] = f"训练进程报告成功但没有可用 checkpoint：{str(exc)[:240]}"
            result["checkpoint"] = None
            result["merged_model"] = None
            result["grpo_metrics"] = {
                "protocol": "grpo-training-metrics-v1", "status": "not_measured",
                "reason": "training reported PASS but produced no usable checkpoint",
                "metric_statuses": {"reward": {"status": "not_measured", "denominator": 0}},
            }
            write_json(output / "grpo_metrics.json", result["grpo_metrics"])
        else:
            result["checkpoint"] = str(checkpoint)
            try:
                metrics = collect_grpo_training_metrics(
                    train_dir,
                    beta=float(training.get("beta", 0.04)),
                    output_path=output / "grpo_metrics.json",
                )
            except (FileNotFoundError, OSError, ValueError) as exc:
                metrics = {"protocol": "grpo-training-metrics-v1", "status": "not_measured",
                           "reason": f"trainer rollout diagnostics unavailable: {str(exc)[:300]}",
                           "metric_statuses": {"reward": {"status": "not_measured", "denominator": 0}}}
                write_json(output / "grpo_metrics.json", metrics)
            result["grpo_metrics"] = metrics
            if args.merge:
                merged_dir = output / "merged" / _grpo_merged_name(profile)
                merge_exec = _run_merge_with_lineage(output=output, checkpoint=checkpoint, parent_model=args.model,
                                                     merged_dir=merged_dir, run_label=f"grpo-{profile}-merge",
                                                     experiment_id=recipe.get("experiment_id"), stage="grpo", dry_run=args.dry_run)
                result["merge"] = merge_exec
                result["merged_model"] = merge_exec.get("merged_model")
    else:
        result["checkpoint"] = None
        result["merged_model"] = None
        result["grpo_metrics"] = {
            "protocol": "grpo-training-metrics-v1", "status": "not_measured",
            "reason": f"training status was {execution.get('status', 'unknown')}; no rollout metrics are inferred",
            "metric_statuses": {"reward": {"status": "not_measured", "denominator": 0}},
        }
        write_json(output / "grpo_metrics.json", result["grpo_metrics"])
    _write_training_run(output=output, result=result, run_id=run_id,
                        run_label=run_label, experiment_id=recipe.get("experiment_id"), stage="grpo", model=args.model,
                        dataset=args.dataset, config_path=args.config,
                        resolved_config={"profile": profile, "model": str(args.model), "dataset": str(args.dataset),
                                         "training": training, "reward": reward, "exploration_context": exploration_context,
                                         "targeted_stage_plan": targeted_context,
                                         "reward_contract": result.get("reward_contract"),
                                         "training_validation": getattr(args, "validation_metadata", None)},
                        started_at=started_at, model_ref=model_ref,
                        validation_dataset=getattr(args, "validation_dataset", None),
                        validation_metadata=getattr(args, "validation_metadata", None))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    merge_failed = bool(args.merge and execution.get("status") == "PASS"
                        and not (result.get("merged_model") and Path(str(result["merged_model"])).is_dir()))
    return 0 if args.dry_run or (execution.get("status") == "PASS" and not merge_failed) else 4


def cmd_run_registered_grpo(args: argparse.Namespace) -> int:
    """Resolve registered GRPO inputs and invoke only the fixed standard/lowmem recipe."""
    from .ui.grpo_stage_adapter import GRPOStageAdapter, GRPOStageRequest
    from .ui.training_preflight import _find_manifest, _find_model, _has_symlink_between

    workspace = Path(args.workspace).expanduser().resolve(strict=True)
    if not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", args.run_id):
        raise ValueError("--run-id must match run-[a-z0-9][a-z0-9-]{0,119}")
    request = GRPOStageRequest(profile_id=args.profile_id, dataset_id=args.dataset_id,
                               parent_model_id=args.parent_model_id)
    adapter = GRPOStageAdapter(workspace)
    preflight = adapter.preflight(request)
    if preflight["status"] != "ready_for_cli":
        raise ValueError("GRPO 阶段预检阻断：" + "; ".join(preflight["blockers"]))
    profile = adapter.profiles[request.profile_id]
    expected_output = f"runs/grpo/{profile.profile_id}/{args.run_id}"
    if args.output != expected_output:
        raise ValueError("输出位置必须与阶段 profile 和 run_id 一致")
    output = workspace / expected_output
    if _has_symlink_between(output, workspace) or output.exists() or output.is_symlink():
        raise FileExistsError("GRPO Run 输出位置不可用或已存在；历史 Run 不覆盖")
    if not console_command_available("swift"):
        raise RuntimeError("项目 Python 环境未发现 swift 命令；GRPO 训练尚未启动")
    manifest_path = _find_manifest(workspace, request.dataset_id)
    _, parent = _find_model(workspace, request.parent_model_id)
    model_path = Path(parent.path or "")
    if not model_path.is_absolute():
        model_path = workspace / model_path
    model_path = model_path.resolve(strict=True)
    model_path.relative_to(workspace)
    adapter_output = workspace / "runs" / "dataset-adapters" / args.run_id
    if _has_symlink_between(adapter_output, workspace) or adapter_output.exists():
        raise FileExistsError("GRPO trainer-native adapter 输出已存在或含符号链接")
    native_result = adapt_registered_dataset(manifest_path, adapter_output)
    native_dataset = adapter_output / native_result["native_path"]
    config_path = getattr(args, "config", None)
    if config_path:
        from .ui.resolved_stage_recipe import verify_materialized_stage_recipe
        verify_materialized_stage_recipe(workspace, config_path, profile_id=profile.profile_id, run_id=args.run_id)
    recipe_path = (Path(config_path) if config_path and Path(config_path).is_absolute()
                   else workspace / config_path if config_path else workspace / profile.approved_recipe)
    recipe = _load_yaml(recipe_path)
    validation = _prepare_registered_training_validation(
        workspace=workspace, stage="grpo", run_id=args.run_id,
        training_manifest=manifest_path, training_native=native_dataset,
        recipe=recipe, output_root=adapter_output,
    )
    external_plugin = workspace / "plugins/grpo_rewards.py"
    run_args = argparse.Namespace(
        config=str(recipe_path), model=str(model_path), dataset=validation["train_dataset"],
        validation_dataset=validation.get("validation_dataset"), validation_metadata=validation.get("metadata"),
        output=str(output), external_plugin=str(external_plugin), merge=True, dry_run=False,
        workspace=str(workspace), run_id=args.run_id, registered_preflight=preflight,
        registered_model_ref=ArtifactRef(id=parent.model_id, kind="model", path=str(model_path),
                                         digest=parent.digest, role="parent_model"),
    )
    return cmd_run_grpo(run_args)

def cmd_eval_preference(args: argparse.Namespace) -> int:
    if args.dry_run:
        payload = {
            "planned": True,
            "model": args.model,
            "revision": args.revision,
            "input": args.input,
            "output": args.output,
            "max_length": args.max_length,
            "limit": args.limit,
            "metric": "response-mean-logprob-v1",
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    result = evaluate_preference_pairs(
        model_name_or_path=args.model,
        revision=args.revision,
        input_path=args.input,
        output_dir=args.output,
        dtype=args.dtype,
        max_length=args.max_length,
        limit=args.limit,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_run_registered_preference_eval(args: argparse.Namespace) -> int:
    """Evaluate a registered DPO/SimPO model on a disjoint registered holdout."""
    from .ui.preference_evaluation_adapter import (
        PreferenceEvaluationAdapter, PreferenceEvaluationRequest,
    )
    from .ui.training_preflight import _find_model, _has_symlink_between

    workspace = Path(args.workspace).expanduser().resolve(strict=True)
    if not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", args.run_id):
        raise ValueError("--run-id must match run-[a-z0-9][a-z0-9-]{0,119}")
    request = PreferenceEvaluationRequest(profile_id=args.profile_id, model_id=args.model_id,
                                          dataset_id=args.dataset_id)
    adapter = PreferenceEvaluationAdapter(workspace)
    expected_output = f"runs/evaluations/{args.run_id}"
    if args.output != expected_output:
        raise ValueError("输出位置必须是 runs/evaluations/{run_id}")
    output = workspace / expected_output
    if _has_symlink_between(output, workspace):
        raise ValueError("Evaluation Run 输出路径不能包含符号链接")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Evaluation Run 输出已存在；历史评测不会覆盖")
    resolved = adapter.resolve_inputs(request)
    model = resolved["model"]
    manifest = resolved["manifest"]
    _, registered_model = _find_model(workspace, args.model_id)
    model_path = resolved["model_path"]
    canonical_path = resolved["canonical_path"]
    started_at = datetime.now().astimezone().isoformat()
    output.mkdir(parents=True, exist_ok=False)
    eval_id = f"eval-{args.run_id.removeprefix('run-')}"
    metrics: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    preference_payload: dict[str, object] | None = None
    try:
        result = evaluate_preference_pairs(
            model_name_or_path=model_path, input_path=canonical_path,
            output_dir=output / "preference", dtype="float16", max_length=1024, limit=None,
        )
        expected_pairs = manifest.sample_count
        actual_pairs = result.pairs
        metric_status = ("measured" if actual_pairs == expected_pairs else
                         "partial" if actual_pairs > 0 else "not_measured")
        for name, value in (("preference.accuracy", result.policy_pair_accuracy),
                            ("preference.margin", result.policy_pair_margin)):
            metrics[name] = {"value": value, "denominator": actual_pairs,
                             "status": metric_status,
                             "reason": (None if actual_pairs == expected_pairs else
                                        f"scored {actual_pairs}/{expected_pairs} registered pairs")}
        evaluation_status = "completed" if actual_pairs == expected_pairs else (
            "partial" if actual_pairs > 0 else "failed")
        preference_payload = result.to_dict()
        preference_payload["expected_pairs"] = expected_pairs
        preference_payload["metric_status"] = metric_status
        if actual_pairs != expected_pairs:
            errors.append(f"实际完成 {actual_pairs}/{expected_pairs} pairs")
    except Exception as exc:
        evaluation_status = "failed"
        errors.append(f"{type(exc).__name__}: {str(exc)[:500]}")
        metrics = {
            "preference.accuracy": {"value": None, "denominator": 0, "status": "not_measured",
                                    "reason": "preference evaluator failed"},
            "preference.margin": {"value": None, "denominator": 0, "status": "not_measured",
                                  "reason": "preference evaluator failed"},
        }

    shared_metrics = {
        "gsm8k.accuracy": {"status": "not_measured", "reason": "separate shared-math evaluation profile"},
        "math_500.accuracy": {"status": "not_measured", "reason": "separate shared-math evaluation profile"},
    }
    payload = {
        "schema_version": "1.0", "eval_id": eval_id, "run_id": args.run_id,
        "status": evaluation_status, "profile_id": request.profile_id,
        "protocol": "response-mean-logprob-v1", "model_id": model.model_id,
        "dataset_id": manifest.dataset_id, "expected_pairs": manifest.sample_count,
        "preference": preference_payload, "metric_statuses": metrics,
        "shared_task_benchmark": shared_metrics, "errors": errors,
    }
    write_json(output / "eval.json", payload)
    evaluation = EvaluationLineage(
        eval_id=eval_id, produced_by_run_id=args.run_id, model_id=model.model_id,
        model_produced_by_run_id=model.produced_by_run_id,
        dataset_ids=[manifest.dataset_id], protocol_version="response-mean-logprob-v1",
        config_sha256=sha256_json({"profile_id": request.profile_id, "dtype": "float16",
                                   "max_length": 1024, "scoring": "response-mean-logprob-v1"}),
        status=evaluation_status, metrics=metrics,
    )
    model_digest = registered_model.digest
    model_input = ArtifactRef(id=model.model_id, kind="model", path=str(model_path),
                              digest=model_digest, role="evaluation_model")
    dataset_input = ArtifactRef(
        id=manifest.dataset_id, kind="dataset", path=str(resolved["manifest_path"]),
        digest={"value": manifest.hashes["canonical_records_sha256"], "status": "available"},
        role="preference_eval",
    )
    run = RunRecord(
        run_id=args.run_id, run_type="evaluate", status=evaluation_status,
        started_at=started_at, completed_at=datetime.now().astimezone().isoformat(),
        inputs=[model_input, dataset_input],
        outputs=[ArtifactRef(id=eval_id, kind="evaluation", path="eval.json", role="metrics")],
        errors=errors,
    )
    write_json(output / "evaluation_lineage.json", evaluation.model_dump(mode="json"))
    write_json(output / "evaluation_run.json", run.model_dump(mode="json"))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if evaluation_status == "completed" else 4


def cmd_run_registered_choice_eval(args: argparse.Namespace) -> int:
    """Run versioned exact-match scoring on a disjoint registered objective suite."""
    from .data.manifest import validate_manifest_evidence
    from .evaluation.choice_accuracy import (
        CHOICE_SCORING_VERSION,
        evaluate_registered_choice_dataset,
        choice_comparability_key,
    )
    from .evaluation.choice_judge import (
        CHOICE_LLM_JUDGE_PROTOCOL,
        choice_judge_comparability_key,
        choice_judge_progress_line,
        choice_judge_prompt_sha256,
        evaluate_registered_choice_dataset_with_judge,
        load_choice_judge_profile,
    )
    from .ui.stage_evaluation import StageEvaluationAdapter, StageEvaluationRequest
    from .ui.training_preflight import _find_manifest, _has_symlink_between

    workspace = Path(args.workspace).expanduser().resolve(strict=True)
    if not re.fullmatch(r"run-[a-z0-9][a-z0-9-]{0,119}", args.run_id):
        raise ValueError("--run-id must match run-[a-z0-9][a-z0-9-]{0,119}")
    expected_output = f"runs/evaluations/{args.run_id}"
    if args.output != expected_output:
        raise ValueError("输出位置必须是 runs/evaluations/{run_id}")
    output = workspace / expected_output
    if _has_symlink_between(output, workspace) or output.exists():
        raise ValueError("Evaluation Run 输出路径冲突、越界或包含符号链接")
    training_run_id = getattr(args, "training_run_id", None)
    model_id = getattr(args, "model_id", None)
    request = StageEvaluationRequest(training_run_id=training_run_id, model_id=model_id,
                                     profile_id=args.profile_id, dataset_id=args.dataset_id,
                                     scoring_strategy=args.scoring_strategy)
    adapter = StageEvaluationAdapter(workspace)
    preflight = adapter.preflight(request)
    if not preflight.get("ok"):
        raise ValueError("客观题评测启动时复核阻断：" + "; ".join(preflight.get("blockers", [])))
    model, training_run = (adapter._model_for_training_run(training_run_id)
                           if training_run_id else adapter._registered_base_model(model_id))
    manifest_path = _find_manifest(workspace, args.dataset_id)
    manifest = load_manifest(manifest_path)
    evidence = validate_manifest_evidence(manifest_path)
    if evidence:
        raise ValueError("评测数据集登记证据已变化：" + "; ".join(evidence[:3]))
    canonical_path = adapter._canonical_dataset_path(manifest_path, manifest)
    model_path = adapter._model_path(model)
    dataset_hash = manifest.hashes.get("canonical_records_sha256")
    if not dataset_hash or sha256_file(canonical_path) != dataset_hash:
        raise ValueError("评测数据集内容摘要与登记值不一致")
    protocol_version = (CHOICE_LLM_JUDGE_PROTOCOL if args.scoring_strategy == "bailian_llm_judge"
                       else "choice-accuracy-v1")
    judge_profile = load_choice_judge_profile(workspace) if args.scoring_strategy == "bailian_llm_judge" else None
    config = {"profile_id": args.profile_id, "protocol_version": protocol_version,
              "dataset_id": manifest.dataset_id, "dataset_sha256": dataset_hash,
              "sample_count": manifest.sample_count, "max_new_tokens": 16, "temperature": 0.0,
              "scoring": (CHOICE_LLM_JUDGE_PROTOCOL if judge_profile else CHOICE_SCORING_VERSION),
              "scoring_strategy": args.scoring_strategy}
    if judge_profile:
        config.update({"provider_profile_id": judge_profile.profile_id,
                       "provider_model": judge_profile.model, "provider_revision": judge_profile.revision,
                       "judge_prompt_version": "objective-choice-judge-zh-v1",
                       "judge_prompt_sha256": choice_judge_prompt_sha256(),
                       "judge_temperature": 0.0, "judge_max_output_tokens": 256})
        comparison_key = choice_judge_comparability_key(dataset_hash, manifest.sample_count, judge_profile)
    else:
        comparison_key = choice_comparability_key(dataset_hash, manifest.sample_count)
    started_at = datetime.now().astimezone().isoformat()
    output.mkdir(parents=True, exist_ok=False)
    errors: list[str] = []
    try:
        if judge_profile:
            def report_choice_judge_progress(event: dict[str, Any]) -> None:
                print(choice_judge_progress_line(event), flush=True)

            result = evaluate_registered_choice_dataset_with_judge(
                model_path=model_path, canonical_path=canonical_path,
                output_dir=output / "choice_accuracy", workspace=workspace, max_new_tokens=16,
                progress_callback=report_choice_judge_progress,
            )
        else:
            result = evaluate_registered_choice_dataset(
                model_path=model_path, canonical_path=canonical_path,
                output_dir=output / "choice_accuracy", max_new_tokens=16,
            )
        status = ("partial" if judge_profile and result["metrics"]["overall"]["provider_errors"] else
                  "completed" if result.get("sample_count") == manifest.sample_count else "partial")
        metric_statuses = {}
        for question_type, item in result["metrics"].items():
            if judge_profile:
                metric_statuses[f"choice_llm_judge.{question_type}.accuracy"] = {
                    "value": item["accuracy"], "correct": item["correct"],
                    "denominator": item["decided_denominator"], "denominator_unit": "decided_questions",
                    "decision_coverage": item["decision_coverage"], "sample_count": item["denominator"],
                    "undetermined": item["undetermined"], "provider_errors": item["provider_errors"],
                    "status": item["status"], "protocol_version": protocol_version,
                    "comparability_key": comparison_key,
                }
                metric_statuses[f"choice_llm_judge.{question_type}.decision_coverage"] = {
                    "value": item["decision_coverage"], "correct": item["correct"],
                    "denominator": item["denominator"], "denominator_unit": "questions",
                    "status": "measured" if item["denominator"] else "not_measured",
                    "protocol_version": protocol_version, "comparability_key": comparison_key,
                }
            else:
                metric_statuses[f"choice_accuracy.{question_type}"] = {
                    "value": item["accuracy"], "correct": item["correct"],
                    "denominator": item["denominator"], "denominator_unit": "questions",
                    "parse_failures": item["parse_failures"], "status": item["status"],
                    "protocol_version": protocol_version, "comparability_key": comparison_key,
                }
    except Exception as exc:
        status = "failed"
        errors.append(f"{type(exc).__name__}: {str(exc)[:400]}")
        prefix = "choice_llm_judge" if judge_profile else "choice_accuracy"
        metric_statuses = {f"{prefix}.{kind}": {
            "value": None, "correct": 0, "denominator": 0, "denominator_unit": "questions",
            "parse_failures": 0, "status": "not_measured", "reason": "objective evaluator failed",
            "protocol_version": protocol_version, "comparability_key": comparison_key,
        } for kind in ("single_choice", "multiple_choice", "true_false", "overall")}
        result = {"protocol": protocol_version, "sample_count": 0, "metrics": {}}

    eval_id = f"eval-{args.run_id.removeprefix('run-')}"
    payload = {
        "schema_version": "1.0", "eval_id": eval_id, "run_id": args.run_id,
        "status": status, "profile_id": args.profile_id, "protocol_version": protocol_version,
        "scoring_strategy": args.scoring_strategy,
        "model_id": model.model_id, "training_run_id": training_run.run_id if training_run else None,
        "dataset_id": manifest.dataset_id,
        "evaluation_suite": {"dataset_id": manifest.dataset_id, "dataset_sha256": dataset_hash,
                              "manifest_sha256": sha256_file(manifest_path),
                              "sample_count": manifest.sample_count,
                              "protocol_version": protocol_version},
        "comparability": {"key": comparison_key,
                           "note": "仅当评测集 canonical 摘要、样本数及 evaluator protocol 相同时可直接比较；不同 suite 不可横向比较。"},
        ("choice_llm_judge" if judge_profile else "choice_accuracy"):
            {key: value for key, value in result.items() if key != "predictions"},
        "metric_statuses": metric_statuses, "errors": errors,
    }
    write_json(output / "eval.json", payload)
    model_digest = model.digest
    evaluation = EvaluationLineage(
        eval_id=eval_id, produced_by_run_id=args.run_id, model_id=model.model_id,
        model_produced_by_run_id=model.produced_by_run_id, dataset_ids=[manifest.dataset_id],
        dataset_artifacts=[ArtifactRef(id=manifest.dataset_id, kind="dataset", path=str(manifest_path),
                                       digest={"value": dataset_hash, "status": "available"},
                                       role="evaluation_only")],
        protocol_version=protocol_version, config_sha256=sha256_json(config),
        status=status, metrics=metric_statuses,
    )
    inputs = [ArtifactRef(id=model.model_id, kind="model", path=str(model_path),
                          digest=model_digest, role="evaluation_model"),
              ArtifactRef(id=manifest.dataset_id, kind="dataset", path=str(manifest_path),
                          digest={"value": dataset_hash, "status": "available"}, role="evaluation_only")]
    run = RunRecord(
        run_id=args.run_id, run_type="evaluate", status=status, started_at=started_at,
        completed_at=datetime.now().astimezone().isoformat(),
        run_label=(f"{'choice-llm-judge' if judge_profile else 'choice-accuracy'} for {training_run.run_id}"
                   if training_run else
                   f"{'choice-llm-judge' if judge_profile else 'choice-accuracy'} for base model {model.model_id}"),
        inputs=inputs, outputs=[ArtifactRef(id=eval_id, kind="evaluation", path="eval.json", role="metrics")],
        errors=errors,
    )
    write_json(output / "evaluation_lineage.json", evaluation.model_dump(mode="json"))
    write_json(output / "evaluation_run.json", run.model_dump(mode="json"))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status == "completed" else 4


def cmd_compare_alignment(args: argparse.Namespace) -> int:
    result = write_alignment_comparison(
        sft_eval=args.sft_eval,
        dpo_eval=args.dpo_eval,
        simpo_eval=args.simpo_eval,
        sft_preference=args.sft_preference,
        dpo_preference=args.dpo_preference,
        simpo_preference=args.simpo_preference,
        output_dir=args.output,
        dpo_run=getattr(args, "dpo_run", None),
        simpo_run=getattr(args, "simpo_run", None),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0



def cmd_compare_grpo(args: argparse.Namespace) -> int:
    result = write_grpo_comparison(
        sft_eval=args.sft_eval,
        standard_run=args.standard_run,
        standard_eval=args.standard_eval,
        lowmem_run=args.lowmem_run,
        lowmem_eval=args.lowmem_eval,
        output_dir=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_summarize_smoke(args: argparse.Namespace) -> int:
    result = write_smoke_summary(args.run_root, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

def cmd_merge_lora(args: argparse.Namespace) -> int:
    output = Path(args.output)
    model = args.model
    if model is None:
        checkpoint_args = Path(args.checkpoint) / "args.json"
        if not checkpoint_args.is_file():
            raise ValueError("合并 LoRA 时必须提供 --model，或使用包含 args.json 的 Swift checkpoint")
        try:
            model = json.loads(checkpoint_args.read_text(encoding="utf-8")).get("model")
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法从 checkpoint 读取父模型信息：{exc}") from exc
        if not isinstance(model, str) or not model.strip():
            raise ValueError("checkpoint args.json 缺少父模型 model；请显式提供 --model")
    cmd = build_merge_command(checkpoint=args.checkpoint, output_dir=output, model=model)
    result = run_command(cmd, log_dir=output.parent / "merge-logs", dry_run=args.dry_run)
    payload = {"checkpoint": str(args.checkpoint), "merged_model": str(output), "merge": result}
    write_json(output.parent / "merge.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_eval_domain_ppl(args: argparse.Namespace) -> int:
    if args.dry_run:
        payload = {
            "planned": True,
            "model": args.model,
            "revision": args.revision,
            "input": args.input,
            "output": args.output,
            "max_length": args.max_length,
            "limit": args.limit,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    result = evaluate_cpt_perplexity(
        model=args.model,
        revision=args.revision,
        input_path=args.input,
        output_path=args.output,
        max_length=args.max_length,
        limit=args.limit,
        dtype=args.dtype,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0



def cmd_compare_evals(args: argparse.Namespace) -> int:
    result = write_stage_comparison(args.base, args.candidate, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_cpt_effect_report(args: argparse.Namespace) -> int:
    result = write_cpt_effect_report(args.training_run, args.base_eval, args.cpt_eval, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_build_scenario_report(args: argparse.Namespace) -> int:
    definition = load_scenario_definition(args.definition)
    report = build_scenario_report(
        definition=definition,
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        eval_id=args.eval_id,
        model_id=args.model_id,
        protocol_version=args.protocol_version,
        threshold_source=args.threshold_source,
        minimum_samples_required=args.minimum_samples,
        sidecar_path=args.sidecar,
    )
    result = write_scenario_report(report, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_register_targeted_plan(args: argparse.Namespace) -> int:
    plan = load_targeted_plan(args.plan)
    result = save_targeted_plan(plan, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_register_targeted_stage_plan(args: argparse.Namespace) -> int:
    root = Path(args.workspace).expanduser().resolve(strict=True)
    plan_path = Path(args.plan).expanduser()
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    plan = load_targeted_stage_plan(plan_path)
    result = register_targeted_stage_plan(root, plan)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_register_sft_workflow(args: argparse.Namespace) -> int:
    inputs = {
        "plan": args.plan, "profile": args.profile, "template": args.template,
        "protocol": args.output_protocol, "definition": args.definition, "report": args.report,
        "registry": args.role_registry, "train_seeds": args.train_seeds, "policy": args.policy,
        "diagnostic": args.diagnostic, "holdout": args.holdout,
    }
    bundle = register_sft_workflow_bundle(args.workspace, inputs=inputs)
    print(json.dumps(bundle.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def _prepare_targeted_generation_from_args(args: argparse.Namespace):
    return prepare_generation(
        plan_path=args.plan,
        profile_path=args.profile,
        template_path=args.template,
        protocol_path=args.output_protocol,
        definition_path=args.definition,
        report_path=args.report,
        registry_path=args.role_registry,
        train_seed_path=args.train_seeds,
        protected_dataset_paths=args.protected_jsonl,
    )


def cmd_targeted_generation_dry_run(args: argparse.Namespace) -> int:
    setup = _prepare_targeted_generation_from_args(args)
    print(json.dumps(setup.dry_run_summary(), indent=2, ensure_ascii=False))
    return 0


def cmd_generate_targeted_data(args: argparse.Namespace) -> int:
    if not args.execute:
        raise SystemExit("No provider call was made. Pass --execute to explicitly start generation.")
    if not args.output:
        raise SystemExit("--output is required with --execute and must be under a private/ directory")
    setup = _prepare_targeted_generation_from_args(args)
    if setup.profile.adapter == "openai_compatible":
        try:
            load_local_provider_credential(setup.profile.api_key_env or "", Path.cwd() / ".env.local")
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
    result = execute_targeted_generation(
        setup, output_path=args.output, retry_uncertain=args.retry_uncertain,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_compare_cpt_training(args: argparse.Namespace) -> int:
    result = write_training_feasibility_comparison(args.peft, args.full, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

def cmd_serve_ui(args: argparse.Namespace) -> int:
    try:
        from .ui.server import serve
    except ImportError as exc:
        raise SystemExit("Install the optional UI dependencies: pip install -e '.[ui]'") from exc
    serve(
        Path(args.workspace),
        port=args.port,
        host=args.host,
        allow_remote=args.allow_remote,
        enable_policy_editor=args.enable_policy_editor,
        enable_experiment_launcher=getattr(args, "enable_experiment_launcher", False),
        enable_synthetic_generation=getattr(args, "enable_synthetic_generation", False),
    )
    return 0


def _add_common_build_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--input-jsonl", help="Offline/local source JSONL; omit both source inputs to stream from Hugging Face.")
    source.add_argument("--source-cache", help="Verified pinned source cache entry or manifest; no network fallback.")
    p.add_argument("--benchmark-jsonl", action="append", default=[], help="Optional local benchmark JSONL for pre-split decontamination; repeatable.")
    p.add_argument("--benchmark-config", help="Pinned HF benchmark specs used when local benchmark JSONL is not supplied.")
    p.add_argument("--tokenizer-repo", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--tokenizer-revision", default="da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
    p.add_argument("--test-tokenizer", action="store_true", help="Use whitespace tokenization for fixtures/tests only; never for official data.")
    p.add_argument("--max-raw-rows", type=int, default=None)


def _add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="recipes/smoke/00-base-eval.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--model")
    p.add_argument("--revision")
    p.add_argument("--run-label")
    p.add_argument("--run-id", help="Optional stable execution ID (used by supervised local launchers)")
    p.add_argument("--targeted-plan-id", help="Optional targeted stage plan identity to preserve in evaluation lineage")
    p.add_argument("--targeted-plan-version", type=int, help="Immutable targeted stage plan version")
    p.add_argument("--stage", default="model")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--internal-holdout")
    p.add_argument("--internal-limit", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--domain-holdout")
    p.add_argument("--domain-limit", type=int, default=20)
    p.add_argument("--ppl-max-length", type=int, default=1024)
    p.add_argument("--dtype", default="float16")


def _add_targeted_generation_inputs(p: argparse.ArgumentParser) -> None:
    p.add_argument("--plan", required=True, help="Registered generation plan YAML/JSON")
    p.add_argument("--profile", required=True, help="Non-secret provider profile YAML")
    p.add_argument("--template", required=True, help="Versioned prompt template YAML")
    p.add_argument("--output-protocol", required=True, help="Output protocol YAML")
    p.add_argument("--definition", required=True, help="Scenario definition YAML referenced by the plan")
    p.add_argument("--report", required=True, help="Scenario report JSON referenced by the plan")
    p.add_argument("--role-registry", required=True, help="Immutable train/diagnostic/holdout role registry")
    p.add_argument("--train-seeds", required=True, help="Canonical train_seed dataset JSONL")
    p.add_argument(
        "--protected-jsonl", required=True, action="append",
        help="Local diagnostic_dev/final_holdout JSONL used to rebuild contamination index; repeatable",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="onegpu-posttrain")
    sub = p.add_subparsers(dest="command", required=True)
    from .policy_cli import register_policy_commands
    register_policy_commands(sub)

    validate = sub.add_parser("validate-lineage", help="Read-only validation of local lineage records")
    validate.add_argument("--root", required=True, help="Artifact root to inspect")
    validate.add_argument("--require-complete", action="store_true", help="Treat incomplete provenance as an integrity error")
    validate.set_defaults(func=cmd_validate_lineage)

    mini = sub.add_parser("validate-mini-data", help="Evaluate Mini dataset readiness and write JSON/Markdown reports")
    mini.add_argument("--root", required=True, help="Artifact root containing dataset manifests and reports")
    mini.add_argument("--profile", required=True, help="Versioned Mini readiness profile YAML")
    mini.add_argument("--output", help="Report directory (default: <root>/reports/mini-readiness/<assessment_id>)")
    mini.add_argument("--quality-assessment", help="Optional reassessment bundle directory or index.json")
    mini.add_argument("--quality-review", help="Optional stage-quality review bundle directory or index.json")
    mini.set_defaults(func=cmd_validate_mini_data)

    reassess = sub.add_parser(
        "assess-mini-data-quality",
        help="Read Mini datasets and write an additive, digest-bound quality reassessment bundle",
    )
    reassess.add_argument("--root", required=True, help="Artifact root containing dataset manifests and reports")
    reassess.add_argument("--profile", required=True, help="Versioned Mini readiness profile YAML")
    reassess.add_argument("--output", help="Bundle directory (default: <root>/reports/mini-quality-assessments/<assessment_id>)")
    reassess.set_defaults(func=cmd_assess_mini_data_quality)

    review = sub.add_parser(
        "review-mini-data-quality",
        help="Create an offline, digest-bound stage quality review bundle from existing evidence",
    )
    review.add_argument("--root", required=True, help="Artifact root containing local dataset manifests")
    review.add_argument("--mini-profile", required=True, help="Frozen Mini readiness profile YAML")
    review.add_argument("--quality-assessment", required=True, help="Existing additive quality assessment bundle")
    review.add_argument("--review-profile", required=True, help="Versioned stage review profile YAML")
    review.add_argument("--output", help="New review bundle directory (must not already exist)")
    review.add_argument("--decisions", help="Optional human decision JSON; manual checks only, with rationale")
    review.add_argument("--supersedes", help="Optional prior review bundle to link as superseded")
    review.set_defaults(func=cmd_review_mini_data_quality)

    ui = sub.add_parser("serve-ui", help="Serve the local experiment UI (loopback by default)")
    ui.add_argument("--workspace", default=".", help="Project directory containing data/processed, artifacts/ui-demo and runs")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1; remote addresses require --allow-remote)")
    ui.add_argument("--allow-remote", action="store_true", help="Allow remote clients/Host headers; exposes local UI data to reachable clients")
    ui.add_argument("--enable-policy-editor", action="store_true", help="Enable local same-origin policy preview/save endpoints")
    ui.add_argument("--enable-experiment-launcher", action="store_true",
                    help="Enable the fixed Mini SFT 1024-token launcher (loopback only; starts a real training process)")
    ui.add_argument("--enable-synthetic-generation", action="store_true",
                    help="Enable explicit local SFT model-generation jobs (loopback only; may call a paid provider)")
    ui.set_defaults(func=cmd_serve_ui)

    d = sub.add_parser("doctor", help="Inspect local runtime and optional training dependencies")
    d.set_defaults(func=cmd_doctor)

    v = sub.add_parser("validate-example-data", help="Validate the canonical example JSONL files")
    v.add_argument("path")
    v.set_defaults(func=cmd_validate_examples)

    a = sub.add_parser("adapt-ms-swift", help="Convert canonical JSONL to ms-swift native dataset format")
    a.add_argument("--stage", required=True, choices=["cpt", "sft", "preference", "dpo", "simpo", "grpo"])
    a.add_argument("--input", required=True)
    a.add_argument("--output", required=True)
    a.set_defaults(func=cmd_adapt_ms_swift)

    registered_adapter = sub.add_parser(
        "adapt-registered-dataset",
        help="Verify a registered dataset ID/hash and create ms-swift native data with record provenance",
    )
    registered_adapter.add_argument("--manifest", required=True, help="Registered stage dataset manifest")
    registered_adapter.add_argument("--output", required=True, help="New, empty output directory (never overwritten)")
    registered_adapter.set_defaults(func=cmd_adapt_registered_dataset)

    pe = sub.add_parser("plan-base-eval", help="Compatibility alias: create Base evaluation plan")
    _add_eval_args(pe)
    pe.set_defaults(func=cmd_plan_base_eval)

    reval = sub.add_parser("run-base-eval", help="Compatibility alias: run Base evaluation")
    _add_eval_args(reval)
    reval.add_argument("--dry-run", action="store_true")
    reval.set_defaults(func=cmd_run_base_eval)

    pme = sub.add_parser("plan-model-eval", help="Plan unified evaluation for any Base/merged stage model")
    _add_eval_args(pme)
    pme.set_defaults(func=cmd_plan_model_eval)

    rme = sub.add_parser("run-model-eval", help="Run unified EvalScope + internal + optional domain PPL evaluation")
    _add_eval_args(rme)
    rme.add_argument("--registered-model-id", help=argparse.SUPPRESS)
    rme.add_argument("--dry-run", action="store_true")
    rme.add_argument("--stream-output", action="store_true", help=argparse.SUPPRESS)
    rme.set_defaults(func=cmd_run_model_eval)

    rgre = sub.add_parser(
        "run-registered-grpo-eval",
        help="Run shared math benchmarks for a registered GRPO model and attach its producing rollout evidence",
    )
    rgre.add_argument("--workspace", required=True)
    rgre.add_argument("--model-id", required=True)
    rgre.add_argument("--run-id", required=True)
    rgre.add_argument("--output", required=True)
    rgre.add_argument("--targeted-plan-id")
    rgre.add_argument("--targeted-plan-version", type=int)
    rgre.set_defaults(func=cmd_run_registered_grpo_eval)

    pcpt = sub.add_parser("plan-peft-cpt", help="Plan the PEFT-CPT smoke training command")
    pcpt.add_argument("--config", default="recipes/smoke/10-peft-cpt.yaml")
    pcpt.add_argument("--model", help="Optional Base model override (for example, a local directory)")
    pcpt.add_argument("--revision", help="Optional model revision override")
    pcpt.add_argument("--dataset", required=True, help="ms-swift native CPT JSONL")
    pcpt.add_argument("--output", required=True)
    pcpt.set_defaults(func=cmd_plan_peft_cpt)

    rcpt = sub.add_parser("run-peft-cpt", help="Run PEFT-CPT smoke and optionally merge M1")
    rcpt.add_argument("--config", default="recipes/smoke/10-peft-cpt.yaml")
    rcpt.add_argument("--model", help="Optional Base model override (for example, a local directory)")
    rcpt.add_argument("--revision", help="Optional model revision override")
    rcpt.add_argument("--dataset", required=True, help="ms-swift native CPT JSONL")
    rcpt.add_argument("--output", required=True)
    rcpt.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    rcpt.add_argument("--dry-run", action="store_true")
    rcpt.set_defaults(func=cmd_run_peft_cpt)

    rrct = sub.add_parser(
        "run-registered-cpt",
        help="Preflight a registered CPT dataset and run the existing monitored PEFT-CPT pipeline",
    )
    rrct.add_argument("--workspace", default=".", help="Project workspace root; manifest/output must stay inside it")
    rrct.add_argument("--manifest", required=True, help="Registered CPT dataset manifest")
    rrct.add_argument("--config", default="recipes/smoke/10-peft-cpt.yaml")
    rrct.add_argument("--model", help="Optional local base model directory")
    rrct.add_argument("--revision", help="Pinned base model revision when resolving a local HF cache")
    rrct.add_argument("--output", required=True, help="New run output directory inside workspace")
    rrct.add_argument("--run-id", help="Preassigned Run ID from the local Launcher")
    rrct.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    rrct.add_argument("--dry-run", action="store_true")
    rrct.set_defaults(func=cmd_run_registered_cpt)

    rpref = sub.add_parser(
        "run-registered-preference",
        help="Preflight a registered Preference dataset and run an allowlisted DPO/SimPO recipe",
    )
    rpref.add_argument("--workspace", required=True, help="Project workspace root")
    rpref.add_argument("--profile-id", required=True, choices=["dpo-smoke-v1", "simpo-smoke-v1"])
    rpref.add_argument("--dataset-id", required=True, help="Registered Preference dataset ID")
    rpref.add_argument("--parent-model-id", required=True, help="Registered merged SFT model ID")
    rpref.add_argument("--run-id", required=True, help="Preassigned execution identity")
    rpref.add_argument("--output", required=True, help="Fixed stage/profile/run output location")
    rpref.add_argument("--config", help="Resolved recipe materialized by the local Launcher")
    rpref.set_defaults(func=cmd_run_registered_preference)

    fcp = sub.add_parser("plan-full-cpt-probe", help="Plan the 5-step full-parameter CPT feasibility probe")
    fcp.add_argument("--config", default="recipes/smoke/11-full-cpt-probe.yaml")
    fcp.add_argument("--model", help="Optional Base model override (for example, a local directory)")
    fcp.add_argument("--revision", help="Optional model revision override")
    fcp.add_argument("--dataset", required=True, help="ms-swift native CPT JSONL")
    fcp.add_argument("--output", required=True)
    fcp.set_defaults(func=cmd_plan_full_cpt_probe)

    rfcp = sub.add_parser("run-full-cpt-probe", help="Run full-parameter CPT probe; OOM is recorded as a valid result")
    rfcp.add_argument("--config", default="recipes/smoke/11-full-cpt-probe.yaml")
    rfcp.add_argument("--model", help="Optional Base model override (for example, a local directory)")
    rfcp.add_argument("--revision", help="Optional model revision override")
    rfcp.add_argument("--dataset", required=True, help="ms-swift native CPT JSONL")
    rfcp.add_argument("--output", required=True)
    rfcp.add_argument("--dry-run", action="store_true")
    rfcp.set_defaults(func=cmd_run_full_cpt_probe)

    psft = sub.add_parser("plan-sft", help="Plan SFT smoke from merged M1")
    psft.add_argument("--config", default="recipes/smoke/20-sft.yaml")
    psft.add_argument("--model", required=True, help="Merged M1-CPT model directory")
    psft.add_argument("--dataset", required=True, help="ms-swift native SFT JSONL")
    psft.add_argument("--output", required=True)
    psft.set_defaults(func=cmd_plan_sft)

    rsft = sub.add_parser("run-sft", help="Run SFT smoke and optionally merge M2")
    rsft.add_argument("--config", default="recipes/smoke/20-sft.yaml")
    rsft.add_argument("--model", required=True, help="Merged M1-CPT model directory")
    rsft.add_argument("--dataset", required=True, help="ms-swift native SFT JSONL")
    rsft.add_argument("--validation-dataset", help="Optional ms-swift native validation JSONL")
    rsft.add_argument("--output", required=True)
    rsft.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    rsft.add_argument("--dry-run", action="store_true")
    rsft.add_argument("--run-id", help="Optional preassigned execution identity (validated; launcher integration)")
    rsft.add_argument("--targeted-stage-plan-id", help="Optional registered targeted stage plan identity")
    rsft.add_argument("--targeted-stage-plan-version", type=int, help="Immutable targeted stage plan version")
    rsft.set_defaults(func=cmd_run_sft)

    pdpo = sub.add_parser("plan-dpo", help="Plan DPO smoke from merged M2")
    pdpo.add_argument("--config", default="recipes/smoke/30-dpo.yaml")
    pdpo.add_argument("--model", required=True, help="Merged M2-SFT model directory")
    pdpo.add_argument("--dataset", required=True, help="ms-swift native preference JSONL")
    pdpo.add_argument("--output", required=True)
    pdpo.set_defaults(func=cmd_plan_dpo)

    rdpo = sub.add_parser("run-dpo", help="Run DPO smoke and optionally merge M3-DPO")
    rdpo.add_argument("--config", default="recipes/smoke/30-dpo.yaml")
    rdpo.add_argument("--model", required=True, help="Merged M2-SFT model directory")
    rdpo.add_argument("--dataset", required=True, help="ms-swift native preference JSONL")
    rdpo.add_argument("--output", required=True)
    rdpo.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    rdpo.add_argument("--dry-run", action="store_true")
    rdpo.set_defaults(func=cmd_run_dpo)

    psim = sub.add_parser("plan-simpo", help="Plan original SimPO smoke from merged M2")
    psim.add_argument("--config", default="recipes/smoke/31-simpo-original.yaml")
    psim.add_argument("--model", required=True, help="Merged M2-SFT model directory")
    psim.add_argument("--dataset", required=True, help="ms-swift native preference JSONL")
    psim.add_argument("--output", required=True)
    psim.set_defaults(func=cmd_plan_simpo)

    rsim = sub.add_parser("run-simpo", help="Run original SimPO smoke and optionally merge M3-SimPO")
    rsim.add_argument("--config", default="recipes/smoke/31-simpo-original.yaml")
    rsim.add_argument("--model", required=True, help="Merged M2-SFT model directory")
    rsim.add_argument("--dataset", required=True, help="ms-swift native preference JSONL")
    rsim.add_argument("--output", required=True)
    rsim.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    rsim.add_argument("--dry-run", action="store_true")
    rsim.set_defaults(func=cmd_run_simpo)

    pgrpo = sub.add_parser("plan-grpo", help="Plan standard/low-memory GRPO smoke from merged M2")
    pgrpo.add_argument("--config", default="recipes/smoke/40-grpo-standard.yaml")
    pgrpo.add_argument("--model", required=True, help="Merged M2-SFT model directory")
    pgrpo.add_argument("--dataset", required=True, help="ms-swift native GRPO JSONL")
    pgrpo.add_argument("--output", required=True)
    pgrpo.add_argument("--external-plugin", default="plugins/grpo_rewards.py")
    pgrpo.set_defaults(func=cmd_plan_grpo)

    rgrpo = sub.add_parser("run-grpo", help="Run standard/low-memory GRPO smoke; OOM is recorded")
    rgrpo.add_argument("--config", default="recipes/smoke/40-grpo-standard.yaml")
    rgrpo.add_argument("--model", required=True, help="Merged M2-SFT model directory")
    rgrpo.add_argument("--dataset", required=True, help="ms-swift native GRPO JSONL")
    rgrpo.add_argument("--output", required=True)
    rgrpo.add_argument("--external-plugin", default="plugins/grpo_rewards.py")
    rgrpo.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True)
    rgrpo.add_argument("--dry-run", action="store_true")
    rgrpo.set_defaults(func=cmd_run_grpo)

    rrgrpo = sub.add_parser(
        "run-registered-grpo",
        help="Preflight registered GRPO prompts/reward contract and run an allowlisted standard/lowmem recipe",
    )
    rrgrpo.add_argument("--workspace", required=True)
    rrgrpo.add_argument("--profile-id", required=True,
                        choices=["grpo-standard-smoke-v1", "grpo-lowmem-smoke-v1"])
    rrgrpo.add_argument("--dataset-id", required=True)
    rrgrpo.add_argument("--parent-model-id", required=True)
    rrgrpo.add_argument("--run-id", required=True)
    rrgrpo.add_argument("--output", required=True)
    rrgrpo.add_argument("--config", help="Resolved recipe materialized by the local Launcher")
    rrgrpo.set_defaults(func=cmd_run_registered_grpo)

    pcal = sub.add_parser("prepare-grpo-calibration", help="Prepare controlled GRPO completion-length candidates")
    pcal.add_argument("--config", required=True, help="Calibration experiment YAML")
    pcal.add_argument("--output-root", help="Override the experiment output root")
    pcal.set_defaults(func=cmd_prepare_grpo_calibration)

    scal = sub.add_parser("summarize-grpo-calibration", help="Summarize completed/failed GRPO length candidates")
    scal.add_argument("--manifest", required=True, help="calibration_manifest.json")
    scal.add_argument("--output", help="Report output directory; defaults to manifest directory")
    scal.set_defaults(func=cmd_summarize_grpo_calibration)

    em = sub.add_parser("compare-evaluation-matrix", help="Compare unified evaluations with explicit missing/partial/failed states")
    em.add_argument("--config", required=True, help="M2/M3 evaluation matrix YAML")
    em.add_argument("--output", required=True, help="Matrix report output directory")
    em.set_defaults(func=cmd_compare_evaluation_matrix)

    pref = sub.add_parser("eval-preference", help="Evaluate held-out preference pairs with common response log-probability scoring")
    pref.add_argument("--model", required=True)
    pref.add_argument("--revision")
    pref.add_argument("--input", required=True, help="Canonical held-out preference JSONL")
    pref.add_argument("--output", required=True)
    pref.add_argument("--dtype", default="float16")
    pref.add_argument("--max-length", type=int, default=1024)
    pref.add_argument("--limit", type=int)
    pref.add_argument("--dry-run", action="store_true")
    pref.set_defaults(func=cmd_eval_preference)

    rpe = sub.add_parser(
        "run-registered-preference-eval",
        help="Evaluate a registered DPO/SimPO model on a disjoint registered preference_eval holdout",
    )
    rpe.add_argument("--workspace", required=True)
    rpe.add_argument("--profile-id", required=True, choices=["preference-heldout-v1"])
    rpe.add_argument("--model-id", required=True, help="Registered merged DPO/SimPO model ID")
    rpe.add_argument("--dataset-id", required=True, help="Registered preference_eval holdout dataset ID")
    rpe.add_argument("--run-id", required=True)
    rpe.add_argument("--output", required=True, help="Fixed runs/evaluations/{run_id} location")
    rpe.set_defaults(func=cmd_run_registered_preference_eval)

    rce = sub.add_parser(
        "run-registered-choice-eval",
        help="Evaluate a registered merged stage model on a disjoint objective-choice suite",
    )
    rce.add_argument("--workspace", required=True)
    rce.add_argument("--training-run-id")
    rce.add_argument("--model-id", help="Registered external base model for standalone evaluation")
    rce.add_argument("--profile-id", required=True, choices=["choice-accuracy-v1"])
    rce.add_argument("--scoring-strategy", choices=["exact_match", "bailian_llm_judge"], default="exact_match")
    rce.add_argument("--dataset-id", required=True)
    rce.add_argument("--run-id", required=True)
    rce.add_argument("--output", required=True)
    rce.set_defaults(func=cmd_run_registered_choice_eval)

    acmp = sub.add_parser("compare-alignment", help="Compare SFT/DPO/SimPO task and held-out preference metrics")
    acmp.add_argument("--sft-eval", required=True)
    acmp.add_argument("--dpo-eval", required=True)
    acmp.add_argument("--simpo-eval", required=True)
    acmp.add_argument("--sft-preference", required=True)
    acmp.add_argument("--dpo-preference", required=True)
    acmp.add_argument("--simpo-preference", required=True)
    acmp.add_argument("--dpo-run", help="Optional DPO run directory/lineage.json for matchedness verification")
    acmp.add_argument("--simpo-run", help="Optional SimPO run directory/lineage.json for matchedness verification")
    acmp.add_argument("--output", required=True)
    acmp.set_defaults(func=cmd_compare_alignment)

    gcmp = sub.add_parser("compare-grpo", help="Compare M2, GRPO-standard and GRPO-lowmem task/RL metrics")
    gcmp.add_argument("--sft-eval", required=True)
    gcmp.add_argument("--standard-run", required=True)
    gcmp.add_argument("--standard-eval")
    gcmp.add_argument("--lowmem-run", required=True)
    gcmp.add_argument("--lowmem-eval")
    gcmp.add_argument("--output", required=True)
    gcmp.set_defaults(func=cmd_compare_grpo)

    smsum = sub.add_parser("summarize-smoke", help="Build one summary table for the entire smoke pipeline")
    smsum.add_argument("--run-root", default="runs/smoke")
    smsum.add_argument("--output", required=True)
    smsum.set_defaults(func=cmd_summarize_smoke)

    ml = sub.add_parser("merge-lora", help="Merge an ms-swift LoRA checkpoint into a standalone model")
    ml.add_argument("--checkpoint", required=True)
    ml.add_argument("--model", help="Exact base model used to train this adapter; defaults to checkpoint args.json")
    ml.add_argument("--output", required=True)
    ml.add_argument("--dry-run", action="store_true")
    ml.set_defaults(func=cmd_merge_lora)

    ppl = sub.add_parser("eval-domain-ppl", help="Evaluate token-weighted perplexity on canonical CPT holdout")
    ppl.add_argument("--model", required=True)
    ppl.add_argument("--revision")
    ppl.add_argument("--input", required=True)
    ppl.add_argument("--output", required=True)
    ppl.add_argument("--max-length", type=int, default=1024)
    ppl.add_argument("--limit", type=int)
    ppl.add_argument("--dtype", default="float16")
    ppl.add_argument("--dry-run", action="store_true")
    ppl.set_defaults(func=cmd_eval_domain_ppl)

    cmp = sub.add_parser("compare-evals", help="Compare two eval.json artifacts and write stage comparison reports")
    cmp.add_argument("--base", required=True)
    cmp.add_argument("--candidate", required=True)
    cmp.add_argument("--output", required=True)
    cmp.set_defaults(func=cmd_compare_evals)

    cpt_effect = sub.add_parser(
        "report-cpt-effect",
        help="Compare Base vs CPT evaluations on the same holdout and include observed training resources",
    )
    cpt_effect.add_argument("--training-run", required=True, help="CPT training run directory containing run.json and lineage.json")
    cpt_effect.add_argument("--base-eval", required=True, help="Base evaluation eval.json")
    cpt_effect.add_argument("--cpt-eval", required=True, help="CPT-model evaluation eval.json")
    cpt_effect.add_argument("--output", required=True, help="New JSON effect-report path")
    cpt_effect.set_defaults(func=cmd_cpt_effect_report)

    scenario = sub.add_parser(
        "build-scenario-report",
        help="Aggregate per-example predictions and fixed labels into a scenario diagnosis report",
    )
    scenario.add_argument("--definition", required=True, help="Versioned scenario definition YAML")
    scenario.add_argument("--dataset", required=True, help="Canonical evaluation dataset JSONL")
    scenario.add_argument("--predictions", required=True, help="Per-example prediction JSONL")
    scenario.add_argument("--eval-id", required=True)
    scenario.add_argument("--model-id", required=True)
    scenario.add_argument("--protocol-version", required=True)
    scenario.add_argument("--threshold-source", required=True, help="Where the diagnostic threshold was preregistered")
    scenario.add_argument("--minimum-samples", type=int, required=True)
    scenario.add_argument("--sidecar", help="Optional fixed problem_id-to-label JSONL for sidecar definitions")
    scenario.add_argument("--output", required=True, help="New JSON report path; existing paths are not overwritten")
    scenario.set_defaults(func=cmd_build_scenario_report)

    targeted_plan = sub.add_parser(
        "register-targeted-plan",
        help="Validate and register a targeted-data plan locally; does not call a provider or start training",
    )
    targeted_plan.add_argument("--plan", required=True, help="Versioned targeted-generation plan YAML/JSON")
    targeted_plan.add_argument("--output", required=True, help="New registration JSON path; existing paths are not overwritten")
    targeted_plan.set_defaults(func=cmd_register_targeted_plan)

    shared_stage_plan = sub.add_parser(
        "register-targeted-stage-plan",
        help="Register an immutable cross-stage CPT/SFT/Preference/GRPO plan under artifacts/targeted-plans",
    )
    shared_stage_plan.add_argument("--workspace", default=".")
    shared_stage_plan.add_argument("--plan", required=True, help="Shared versioned targeted stage plan YAML/JSON")
    shared_stage_plan.set_defaults(func=cmd_register_targeted_stage_plan)

    sft_workflow = sub.add_parser(
        "register-sft-workflow",
        help="Validate and immutably register all local inputs for the explicit SFT generation UI workflow",
    )
    sft_workflow.add_argument("--workspace", default=".", help="Project workspace root")
    sft_workflow.add_argument("--plan", required=True)
    sft_workflow.add_argument("--profile", required=True)
    sft_workflow.add_argument("--template", required=True)
    sft_workflow.add_argument("--output-protocol", required=True)
    sft_workflow.add_argument("--definition", required=True)
    sft_workflow.add_argument("--report", required=True)
    sft_workflow.add_argument("--role-registry", required=True)
    sft_workflow.add_argument("--train-seeds", required=True)
    sft_workflow.add_argument("--policy", required=True)
    sft_workflow.add_argument("--diagnostic", required=True)
    sft_workflow.add_argument("--holdout", required=True)
    sft_workflow.set_defaults(func=cmd_register_sft_workflow)

    targeted_dry_run = sub.add_parser(
        "dry-run-targeted-generation",
        help="Validate targeted generation identities and contamination; no provider call or credential lookup",
    )
    _add_targeted_generation_inputs(targeted_dry_run)
    targeted_dry_run.set_defaults(func=cmd_targeted_generation_dry_run)

    targeted_generate = sub.add_parser(
        "generate-targeted-data",
        help="Generate targeted candidates only when explicitly passed --execute",
    )
    _add_targeted_generation_inputs(targeted_generate)
    targeted_generate.add_argument(
        "--output", help="Private raw-response JSONL path under artifacts/synthetic/<plan_id>/private/",
    )
    targeted_generate.add_argument("--execute", action="store_true", help="Explicitly authorize provider requests")
    targeted_generate.add_argument(
        "--retry-uncertain", action="store_true",
        help="Retry requests that may already have been billed; use only after reviewing the ledger",
    )
    targeted_generate.set_defaults(func=cmd_generate_targeted_data)

    cptcmp = sub.add_parser("compare-cpt-training", help="Compare PEFT-CPT and Full-CPT probe resource outcomes")
    cptcmp.add_argument("--peft", required=True, help="PEFT-CPT run.json")
    cptcmp.add_argument("--full", required=True, help="Full-CPT probe run.json")
    cptcmp.add_argument("--output", required=True)
    cptcmp.set_defaults(func=cmd_compare_cpt_training)

    cache = sub.add_parser("cache-source", help="Materialize a pinned Hugging Face dataset split as a verified local JSONL cache")
    cache.add_argument("--config", required=True, help="Dataset source YAML (repo/config/revision)")
    cache.add_argument("--split", default="train", help="Upstream dataset split to cache (default: train)")
    cache.add_argument("--cache-dir", default="data/raw/cache", help="Local cache root (default: data/raw/cache)")
    cache.add_argument("--refresh", action="store_true", help="Fetch the full split again and atomically update the cache manifest")
    cache.set_defaults(func=cmd_cache_source)

    o = sub.add_parser("build-openr1", help="Build SFT/preference/GRPO/holdout canonical data from OpenR1")
    _add_common_build_args(o)
    o.set_defaults(func=cmd_build_openr1)
    o.add_argument("--sft-target", type=int, default=200)
    o.add_argument("--alignment-target", type=int, default=300)
    o.add_argument("--holdout-target", type=int, default=100)
    o.add_argument("--preference-target", type=int, default=200)
    o.add_argument("--preference-eval-target", type=int, default=50)
    o.add_argument("--grpo-target", type=int, default=200)
    o.add_argument("--sft-max-length", type=int, default=1024)
    o.add_argument("--preference-max-length", type=int, default=1024)
    o.add_argument("--grpo-max-prompt-length", type=int, default=512)

    sft_cal = sub.add_parser(
        "calibrate-sft-length",
        help="Build offline exploratory SFT length candidates from a local DeepMath Parquet cache",
    )
    sft_cal.add_argument("--source", required=True, help="Local DeepMath Parquet file or directory; no network fallback")
    sft_cal.add_argument("--output", required=True, help="New calibration output directory; existing directories are never overwritten")
    sft_cal.add_argument("--sample-limit", type=int, default=2000, help="Deterministically sampled source-question rows")
    sft_cal.add_argument("--seed", type=int, default=42)
    sft_cal.add_argument("--length", type=int, action="append", default=[], help="Candidate max length; repeatable (default: 1024, 2048, 4096)")
    sft_cal.add_argument("--tokenizer-repo", default="Qwen/Qwen3-0.6B-Base")
    sft_cal.add_argument("--tokenizer-revision", default="da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
    sft_cal.add_argument(
        "--answer-verification", choices=("exact", "semantic"), default="exact",
        help="Candidate answer rule: exact normalized match by default; semantic uses bounded local math-verify batches",
    )
    sft_cal.add_argument("--run-probes", action="store_true", help="Explicitly run bounded GPU SFT probes; otherwise CPU-only")
    sft_cal.add_argument("--model", help="Local parent checkpoint; required with --run-probes")
    sft_cal.add_argument("--probe-config", default="recipes/smoke/20-sft.yaml", help="Base recipe for bounded probe controls")
    sft_cal.add_argument("--probe-steps", type=int, default=3, help="Probe update steps (1-5)")
    sft_cal.set_defaults(func=cmd_calibrate_sft_length)

    sft_register = sub.add_parser(
        "register-sft-calibration",
        help="Register an existing local SFT calibration candidate with verified dataset lineage",
    )
    sft_register.add_argument("--manifest", required=True, help="Existing calibration candidate manifest.json")
    sft_register.set_defaults(func=cmd_register_sft_calibration)

    sft_derive = sub.add_parser(
        "derive-sft-length-subset",
        help="Derive an immutable shorter SFT cohort from a registered local dataset without rescanning source data",
    )
    sft_derive.add_argument("--manifest", required=True, help="Registered parent SFT dataset manifest YAML")
    sft_derive.add_argument("--max-length", type=int, required=True, help="Maximum measured rendered token length")
    sft_derive.add_argument("--output", required=True, help="New immutable child dataset artifact directory")
    sft_derive.set_defaults(func=cmd_derive_sft_length_subset)

    cpt_repair = sub.add_parser(
        "repair-cpt-roles",
        help="Preview, then explicitly derive a role-labeled child from a verified legacy FineMath CPT dataset",
    )
    cpt_repair.add_argument("--manifest", required=True, help="Hash-verified legacy FineMath CPT manifest")
    cpt_repair.add_argument("--workspace", default=".", help="Project workspace root")
    cpt_repair.add_argument("--output-root", default="data/processed/cpt-role-repair",
                            help="Workspace-relative root for the new derived dataset")
    cpt_repair.add_argument("--apply", action="store_true",
                            help="Create and register a new immutable dataset; default is read-only preview")
    cpt_repair.set_defaults(func=cmd_repair_cpt_roles)

    f = sub.add_parser("build-finemath", help="Build CPT train/holdout canonical data from FineMath")
    _add_common_build_args(f)
    f.set_defaults(func=cmd_build_finemath)
    f.add_argument("--train-token-budget", type=int, default=500_000)
    f.add_argument("--holdout-token-budget", type=int, default=50_000)
    f.add_argument("--max-seq-tokens", type=int, default=1024)
    f.add_argument("--holdout-permille", type=int, default=50)

    deepmath_mini = sub.add_parser(
        "build-deepmath-mini-sft",
        help="Build a new offline Mini SFT cohort from the pinned local DeepMath snapshot",
    )
    deepmath_mini.add_argument("--source", required=True, help="Local HF snapshot directory; no network fallback")
    deepmath_mini.add_argument("--output", required=True, help="New immutable artifact directory")
    deepmath_mini.add_argument("--contamination-index", action="append", help="Local JSONL benchmark/holdout index; repeatable")
    deepmath_mini.add_argument("--seed", type=int, default=42)
    deepmath_mini.add_argument("--max-length", type=int, default=2048)
    deepmath_mini.add_argument("--target", type=int, default=5000)
    deepmath_mini.add_argument("--tokenizer-repo", default="Qwen/Qwen3-0.6B-Base")
    deepmath_mini.add_argument("--tokenizer-revision", default="da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
    deepmath_mini.add_argument("--test-tokenizer", action="store_true", help="Testing only; never use for Mini artifacts")
    deepmath_mini.set_defaults(func=cmd_build_deepmath_mini_sft)
    return p


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))

if __name__ == "__main__":
    main()
