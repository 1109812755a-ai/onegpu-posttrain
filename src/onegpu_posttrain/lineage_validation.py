"""Read-only validation for local file-based experiment lineage."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .data.manifest import DatasetManifest, validate_manifest_evidence
from .lineage import (
    EvaluationLineage,
    ModelLineage,
    RunRecord,
    load_versioned_payload,
    sha256_directory,
    sha256_file,
)


def validate_lineage_root(root: str | Path, *, require_complete: bool = False) -> dict[str, Any]:
    """Validate known local lineage records without changing any artifact."""
    base = Path(root).resolve(strict=True)
    project_root = next(
        (parent for parent in (base, *base.parents) if (parent / "pyproject.toml").is_file()),
        None,
    )

    artifact_index: dict[str, list[Path]] = {}
    artifact_index_ready = False

    def index_artifacts() -> None:
        nonlocal artifact_index_ready
        if artifact_index_ready:
            return
        names = {"lineage.json", "evaluation_run.json", "evaluation_lineage.json", "model_artifacts"}
        for current, directories, files in os.walk(base, topdown=True, followlinks=False):
            current_path = Path(current)
            relative = current_path.relative_to(base)
            entries = [*directories, *files]
            for name in entries:
                candidate = current_path / name
                if name in names:
                    artifact_index.setdefault(name, []).append(candidate)
                if current_path.name == "manifests":
                    artifact_index.setdefault("manifests/*", []).append(candidate)

            # Prune irrelevant trees before descending; checking path parts
            # after rglob still traverses potentially huge Python/JS installs.
            directories[:] = [name for name in directories if name not in {".git", ".venv", "node_modules"}]
            # Pytest basetemp directories are fixtures, not project artifacts.
            # Ignore only top-level scratch trees when scanning the repo root.
            if project_root == base and not relative.parts:
                directories[:] = [name for name in directories if not name.startswith((".pytest-tmp", ".tmp-pytest"))]
        for paths in artifact_index.values():
            paths.sort()
        artifact_index_ready = True

    def artifact_paths(pattern: str):
        index_artifacts()
        yield from artifact_index.get(pattern, [])
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    run_ids: dict[str, str] = {}
    run_records: dict[str, tuple[Path, RunRecord]] = {}
    run_raw_records: dict[str, dict[str, Any]] = {}
    dataset_records: dict[str, tuple[Path, DatasetManifest]] = {}
    model_records: dict[str, tuple[Path, ModelLineage]] = {}
    artifact_definitions: dict[str, list[str]] = defaultdict(list)
    output_producers: dict[str, list[str]] = defaultdict(list)
    graph: dict[str, set[str]] = defaultdict(set)
    evaluations: list[tuple[Path, EvaluationLineage]] = []

    def issue(target: list[dict[str, str]], code: str, path: Path, message: str) -> None:
        target.append({"code": code, "path": path.relative_to(base).as_posix(), "message": message})

    for path in artifact_paths("lineage.json"):
        if any(part in {".git", ".venv", "node_modules"} for part in path.parts):
            continue
        if path.is_symlink():
            issue(warnings, "symlink_skipped", path, "符号链接不作为本地血缘输入读取。")
            continue
        try:
            payload, version = load_versioned_payload(path, legacy_keys={"run_id", "stage", "inputs"})
            if version == "legacy":
                issue(warnings, "legacy", path, "旧版运行记录可读取，但不按 v1 血缘解释。")
                continue
            if version != "v1":
                issue(warnings, "incomplete", path, "未带版本的运行记录，证据不完整。")
                continue
            record = RunRecord.model_validate(payload)
            run_records.setdefault(record.run_id, (path, record))
            run_raw_records.setdefault(record.run_id, payload)
            if record.run_id in run_ids:
                issue(errors, "duplicate_run_id", path, f"run_id 与 {run_ids[record.run_id]} 重复：{record.run_id}")
            else:
                run_ids[record.run_id] = path.relative_to(base).as_posix()
            if record.lineage_status == "incomplete":
                issue(errors if require_complete else warnings, "incomplete", path,
                      "运行血缘不完整：" + ("；".join(record.lineage_warnings) or "缺少可验证的 manifest / 外部证据。"))
            for artifact in record.outputs:
                output_producers[artifact.id].append(path.relative_to(base).as_posix())
            for artifact in [*record.inputs, *record.outputs]:
                if artifact.availability == "unavailable" or (artifact.digest and artifact.digest.status != "available"):
                    issue(warnings, "unavailable_evidence", path, f"产物 {artifact.id} 的外部摘要或内容不可用。")
                if not artifact.path:
                    continue
                ref = Path(artifact.path)
                if ref.is_absolute():
                    candidate = ref
                else:
                    candidates = [parent / ref for parent in list(path.parents)[:5]]
                    if project_root is not None:
                        candidates.append(project_root / ref)
                    candidate = next((item for item in candidates if item.exists()), candidates[0])
                try:
                    resolved = candidate.resolve(strict=False)
                    resolved.relative_to(base)
                except (OSError, ValueError):
                    issue(warnings, "external_or_missing_path", path, f"引用路径不在根目录内或不可读取：{artifact.path}")
                    continue
                if not resolved.exists():
                    if artifact.availability == "unavailable" or (artifact.digest and artifact.digest.status != "available"):
                        issue(warnings, "external_or_missing_path", path, f"外部或不可用引用未在本地解析：{artifact.path}")
                    else:
                        issue(errors, "dangling_local_ref", path, f"本地引用文件不存在：{artifact.path}")
                    continue
                digest = artifact.digest
                if digest and digest.status == "available":
                    actual = sha256_directory(resolved) if resolved.is_dir() else sha256_file(resolved)
                    if actual != digest.value:
                        issue(errors, "digest_mismatch", path, f"引用产物摘要不匹配：{artifact.path}")
        except (OSError, ValueError, ValidationError, TypeError) as exc:
            issue(errors, "invalid_run", path, f"运行血缘无效：{exc}")

    for path in artifact_paths("evaluation_run.json"):
        if path.is_symlink():
            issue(warnings, "symlink_skipped", path, "符号链接不作为本地血缘输入读取。")
            continue
        try:
            record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
            run_records.setdefault(record.run_id, (path, record))
            run_raw_records.setdefault(record.run_id, json.loads(path.read_text(encoding="utf-8")))
            if record.lineage_status == "incomplete":
                issue(errors if require_complete else warnings, "incomplete", path,
                      "评测运行血缘不完整：" + ("；".join(record.lineage_warnings)
                                                or "缺少可验证的外部评测数据证据。"))
            if record.run_id in run_ids:
                issue(errors, "duplicate_run_id", path, f"run_id 与 {run_ids[record.run_id]} 重复：{record.run_id}")
            else:
                run_ids[record.run_id] = path.relative_to(base).as_posix()
            for artifact in [*record.inputs, *record.outputs]:
                if artifact.availability == "unavailable" or (artifact.digest and artifact.digest.status != "available"):
                    issue(warnings, "unavailable_evidence", path, f"产物 {artifact.id} 的外部摘要或内容不可用。")
                if not artifact.path:
                    continue
                ref = Path(artifact.path)
                candidates = [ref] if ref.is_absolute() else [parent / ref for parent in list(path.parents)[:5]]
                if not ref.is_absolute() and project_root is not None:
                    candidates.append(project_root / ref)
                resolved = next((item.resolve(strict=False) for item in candidates if item.exists()), candidates[0].resolve(strict=False))
                try:
                    resolved.relative_to(base)
                except ValueError:
                    issue(warnings, "external_or_missing_path", path, f"引用路径不在根目录内：{artifact.path}")
                    continue
                if not resolved.exists():
                    issue(errors, "dangling_local_ref", path, f"本地引用文件不存在：{artifact.path}")
                    continue
                if artifact.digest and artifact.digest.status == "available":
                    actual = sha256_directory(resolved) if resolved.is_dir() else sha256_file(resolved)
                    if actual != artifact.digest.value:
                        issue(errors, "digest_mismatch", path, f"引用产物摘要不匹配：{artifact.path}")
            for artifact in record.outputs:
                output_producers[artifact.id].append(path.relative_to(base).as_posix())
        except (OSError, ValueError, ValidationError) as exc:
            issue(errors, "invalid_run", path, f"评测 Run 格式无效：{exc}")

    for path in artifact_paths("evaluation_lineage.json"):
        if path.is_symlink():
            issue(warnings, "symlink_skipped", path, "符号链接不作为本地血缘输入读取。")
            continue
        try:
            evaluation = EvaluationLineage.model_validate_json(path.read_text(encoding="utf-8"))
            artifact_definitions[evaluation.eval_id].append(path.relative_to(base).as_posix())
            evaluations.append((path, evaluation))
            if evaluation.produced_by_run_id not in run_ids:
                issue(errors, "dangling_run", path, f"评测结果引用了不存在的 Run：{evaluation.produced_by_run_id}")
        except (OSError, ValueError, ValidationError) as exc:
            issue(errors, "invalid_evaluation", path, f"评测血缘无效：{exc}")

    for path in artifact_paths("manifests/*"):
        if path.is_symlink():
            issue(warnings, "symlink_skipped", path, "符号链接不作为本地 manifest 输入读取。")
            continue
        if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else yaml.safe_load(path.read_text(encoding="utf-8"))
            manifest = DatasetManifest.model_validate(raw)
            dataset_records.setdefault(manifest.dataset_id, (path, manifest))
            artifact_definitions[manifest.dataset_id].append(path.relative_to(base).as_posix())
            for detail in validate_manifest_evidence(path):
                issue(errors, "manifest_evidence", path, detail)
            if manifest.quality_report:
                artifact_definitions[manifest.quality_report.report_id].append(path.relative_to(base).as_posix())
            lineage = raw.get("lineage") if isinstance(raw, dict) else None
            if isinstance(lineage, dict):
                for parent in lineage.get("parents", []):
                    graph[manifest.dataset_id].add(parent)
                producer = lineage.get("produced_by_run_id")
                if producer and producer not in run_ids:
                    issue(errors, "dangling_run", path, f"数据集引用了不存在的 Run：{producer}")
        except (OSError, ValueError, ValidationError, TypeError) as exc:
            issue(errors, "invalid_manifest", path, f"数据 manifest 无效：{exc}")

    for directory in artifact_paths("model_artifacts"):
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink():
                issue(warnings, "symlink_skipped", path, "符号链接不作为本地模型血缘输入读取。")
                continue
            try:
                model = ModelLineage.model_validate_json(path.read_text(encoding="utf-8"))
                model_records.setdefault(model.model_id, (path, model))
                artifact_definitions[model.model_id].append(path.relative_to(base).as_posix())
                for parent in model.parent_model_ids:
                    graph[model.model_id].add(parent)
            except (OSError, ValueError, ValidationError) as exc:
                issue(errors, "invalid_model", path, f"模型血缘无效：{exc}")

    for path, evaluation in evaluations:
        if evaluation.model_id not in artifact_definitions:
            issue(warnings, "unresolved_model_id", path, f"评测模型只有 ID 或外部定义不可得：{evaluation.model_id}")
        for dataset_id in evaluation.dataset_ids:
            if dataset_id not in artifact_definitions and not dataset_id.startswith("benchmark-"):
                issue(warnings, "unresolved_dataset_id", path, f"评测数据集只有 ID 或外部 manifest 不可得：{dataset_id}")

    for artifact_id, locations in artifact_definitions.items():
        if len(locations) > 1:
            issue(errors, "duplicate_artifact_id", base / locations[-1],
                  f"artifact ID {artifact_id} 重复定义：{', '.join(locations)}")
    for artifact_id, locations in output_producers.items():
        if len(locations) > 1:
            issue(errors, "duplicate_artifact_id", base / locations[-1],
                  f"artifact ID {artifact_id} 被多个 Run 产出：{', '.join(locations)}")

    # Close references by stable IDs and recorded hashes; never infer by display name.
    for dataset_id, (manifest_path, manifest) in dataset_records.items():
        if not manifest.lineage:
            target = errors if require_complete else warnings
            issue(target, "dataset_lineage_missing", manifest_path,
                  f"已登记数据集缺少显式构建 lineage：{dataset_id}")
            continue
        canonical_digest = manifest.hashes.get("canonical_records_sha256")
        if manifest.lineage.content_sha256 != canonical_digest:
            issue(errors, "dataset_lineage_digest_mismatch", manifest_path,
                  f"数据集 {dataset_id} 的 lineage content SHA-256 与 manifest 不一致。")
        if manifest.composition is not None:
            declared = manifest.composition.parents
            lineage_parent_ids = manifest.lineage.parents
            if [parent.dataset_id for parent in declared] != lineage_parent_ids:
                issue(errors, "dataset_composition_parent_mismatch", manifest_path,
                      "组合 manifest 的 parent IDs 与数据集 lineage 不一致。")
            for parent in declared:
                found = dataset_records.get(parent.dataset_id)
                if found is None:
                    issue(errors, "dataset_composition_parent_missing", manifest_path,
                          f"组合数据集 {dataset_id} 找不到父数据集 manifest：{parent.dataset_id}")
                    continue
                parent_path, parent_manifest = found
                expected_parent_digest = parent_manifest.hashes.get("canonical_records_sha256")
                if expected_parent_digest != parent.content_sha256:
                    issue(errors, "dataset_composition_parent_digest_mismatch", manifest_path,
                          f"组合数据集 {dataset_id} 记录的父数据摘要与 manifest 不一致：{parent.dataset_id}")
                if parent.quality_report_id and (parent_manifest.quality_report is None
                        or parent_manifest.quality_report.report_id != parent.quality_report_id):
                    issue(errors, "dataset_composition_parent_quality_mismatch", manifest_path,
                          f"组合数据集 {dataset_id} 记录的父质量报告不一致：{parent.dataset_id}")
                if not parent_path.is_file():
                    issue(errors, "dataset_composition_parent_manifest_missing", manifest_path,
                          f"父数据集 manifest 不可读取：{parent.dataset_id}")
        producer = run_records.get(manifest.lineage.produced_by_run_id)
        if producer is None:
            target = errors if require_complete else warnings
            issue(target, "dangling_dataset_producer", manifest_path,
                  f"数据集 {dataset_id} 的 producing Run 不在本地审计根中：{manifest.lineage.produced_by_run_id}")
        elif producer[1].run_type != "data_build":
            issue(errors, "dataset_producer_type_mismatch", manifest_path,
                  f"数据集 {dataset_id} 的 producer 不是 data_build Run。")
        elif not any(ref.kind == "dataset" and ref.id == dataset_id and ref.digest
                     and ref.digest.status == "available" and ref.digest.value == canonical_digest
                     for ref in producer[1].outputs):
            issue(errors, "dataset_producer_output_mismatch", manifest_path,
                  f"数据集 {dataset_id} 的构建 Run 未声明同摘要输出。")
        elif manifest.composition is not None:
            expected_parents = {item.dataset_id: item.content_sha256 for item in manifest.composition.parents}
            actual_parents = {
                ref.id: ref.digest.value for ref in producer[1].inputs
                if ref.kind == "dataset" and ref.role == "parent_dataset" and ref.digest
                and ref.digest.status == "available"
            }
            if actual_parents != expected_parents:
                issue(errors, "dataset_composition_run_inputs_mismatch", manifest_path,
                      "组合数据集的 build Run 未以完整 ID/摘要引用全部父数据集。")

    for run_path, run in run_records.values():
        if run.run_type == "train" and run.status == "completed":
            missing_config = []
            if not run.recipe_path:
                missing_config.append("approved recipe path")
            if not run.resolved_config_sha256:
                missing_config.append("resolved config SHA-256")
            if missing_config:
                target = errors if require_complete else warnings
                issue(target, "training_config_identity_missing", run_path,
                      "成功训练 Run 缺少配置身份：" + "、".join(missing_config))
        for ref in run.inputs:
            if ref.kind == "dataset":
                if run.run_type == "train" and not ref.role:
                    issue(errors, "dataset_input_role_missing", run_path,
                          f"训练 Run 输入数据 {ref.id} 未记录 train/eval 等用途 role。")
                if ref.id in dataset_records:
                    manifest_path, manifest = dataset_records[ref.id]
                    dataset_digest = manifest.hashes.get("canonical_records_sha256")
                    if ref.digest is None or ref.digest.status != "available":
                        issue(errors, "dataset_input_digest_missing", run_path,
                              f"Run 输入数据 {ref.id} 未记录可用 canonical SHA-256。")
                    elif dataset_digest != ref.digest.value:
                        issue(errors, "dataset_input_digest_mismatch", run_path,
                              f"Run 输入数据 {ref.id} 的 SHA-256 与登记 manifest 不一致：{manifest_path.relative_to(base).as_posix()}")
                    expected_role = manifest.metadata.get("dataset_role") or manifest.metadata.get("role") or manifest.metadata.get("split")
                    if (run.run_type == "train" and isinstance(expected_role, str) and ref.role
                            and ref.role != expected_role):
                        issue(errors, "dataset_input_role_mismatch", run_path,
                              f"Run 输入数据 {ref.id} 声明 role={ref.role}，manifest role={expected_role}。")
                elif not ref.id.startswith("benchmark-"):
                    # data-build and quality runs may retain external source IDs that
                    # intentionally have no local DatasetManifest. Training/evaluation
                    # inputs, in contrast, must resolve to registered data or a benchmark.
                    target = (errors if require_complete and run.status == "completed"
                              and run.run_type in {"train", "evaluate"} else warnings)
                    issue(target, "unresolved_dataset_id", run_path,
                          f"Run 输入数据只有 ID、没有本地 manifest：{ref.id}")
            if ref.kind == "model" and ref.id in model_records:
                _, model = model_records[ref.id]
                if ref.digest is None or ref.digest.status != "available":
                    target = errors if require_complete else warnings
                    issue(target, "model_input_digest_missing", run_path,
                          f"Run 输入模型 {ref.id} 未记录可用 SHA-256。")
                elif model.digest.status != "available" or ref.digest.value != model.digest.value:
                    issue(errors, "model_input_digest_mismatch", run_path,
                          f"Run 输入模型 {ref.id} 的 SHA-256 与模型登记不一致。")
        for ref in run.outputs:
            if ref.kind == "dataset" and ref.id in dataset_records:
                manifest_path, manifest = dataset_records[ref.id]
                if not manifest.lineage or manifest.lineage.produced_by_run_id != run.run_id:
                    issue(errors, "dataset_producer_mismatch", run_path,
                          f"Run 输出数据 {ref.id} 的 manifest 未指向当前 producing Run：{manifest_path.relative_to(base).as_posix()}")
                expected_digest = manifest.hashes.get("canonical_records_sha256")
                if (ref.digest is None or ref.digest.status != "available"
                        or ref.digest.value != expected_digest):
                    issue(errors, "dataset_output_digest_mismatch", run_path,
                          f"Run 输出数据 {ref.id} 的 SHA-256 与 manifest 不一致。")
            if ref.kind != "model" or ref.id not in model_records:
                if ref.kind == "model" and run.status == "completed":
                    target = errors if require_complete else warnings
                    issue(target, "unresolved_model_output", run_path,
                          f"成功 Run 声明的模型产物没有本地模型登记：{ref.id}")
                continue
            _, model = model_records[ref.id]
            if model.produced_by_run_id != run.run_id:
                issue(errors, "model_producer_mismatch", run_path,
                      f"Run 输出模型 {ref.id} 的 producing run 声明不一致。")
            if (ref.digest and ref.digest.status == "available" and model.digest.status == "available"
                    and ref.digest.value != model.digest.value):
                issue(errors, "model_output_digest_mismatch", run_path,
                      f"Run 输出模型 {ref.id} 的 SHA-256 与模型登记不一致。")

    for model_path, model in model_records.values():
        if model.digest.status == "available" and model.path:
            supplied = Path(model.path)
            candidate = supplied if supplied.is_absolute() else base / supplied
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(base)
                actual = sha256_directory(resolved) if resolved.is_dir() else sha256_file(resolved)
                if actual != model.digest.value:
                    issue(errors, "model_digest_mismatch", model_path, f"模型登记摘要与实际内容不一致：{model.model_id}")
            except (OSError, ValueError):
                issue(errors, "model_path_unavailable", model_path,
                      f"模型登记路径无法在审计根目录内解析：{model.model_id}")
        if model.produced_by_run_id:
            producer = run_records.get(model.produced_by_run_id)
            if producer is None:
                issue(errors, "dangling_model_producer", model_path,
                      f"模型引用的 producing Run 不存在：{model.produced_by_run_id}")
            else:
                if not any(ref.kind == "model" and ref.id == model.model_id for ref in producer[1].outputs):
                    issue(errors, "model_not_declared_output", model_path,
                          f"producing Run 未声明产出模型 {model.model_id}。")
                if producer[1].status != "completed":
                    issue(errors, "model_producer_not_completed", model_path,
                          f"模型 {model.model_id} 的 producing Run 状态为 {producer[1].status}，不能视为成功模型产物。")
                declared_parents = {ref.id for ref in producer[1].inputs if ref.kind == "model"}
                lineage_parents = set(model.parent_model_ids)
                if model.adapter_id:
                    lineage_parents.add(model.adapter_id)
                if lineage_parents != declared_parents:
                    issue(errors, "model_parent_input_mismatch", model_path,
                          f"模型 {model.model_id} 登记的 parent IDs 与 producing Run 的模型输入不一致。")
            if model.digest.status == "available" and producer is not None:
                output_ref = next((ref for ref in producer[1].outputs
                                   if ref.kind == "model" and ref.id == model.model_id), None)
                if (output_ref is None or output_ref.digest is None or output_ref.digest.status != "available"
                        or output_ref.digest.value != model.digest.value):
                    issue(errors, "model_producer_digest_mismatch", model_path,
                          f"模型 {model.model_id} 的登记摘要与 producing Run 输出摘要不一致。")
        for parent_id in model.parent_model_ids:
            if parent_id not in model_records:
                issue(warnings, "unresolved_model_parent", model_path,
                      f"模型 {model.model_id} 的父模型仅有 ID、没有本地登记：{parent_id}")

    for eval_path, evaluation in evaluations:
        model_entry = model_records.get(evaluation.model_id)
        if model_entry is None:
            target = errors if require_complete else warnings
            issue(target, "unresolved_model_id", eval_path,
                  f"评测引用的模型没有本地登记：{evaluation.model_id}")
        elif (evaluation.model_produced_by_run_id is not None
              and evaluation.model_produced_by_run_id != model_entry[1].produced_by_run_id):
            issue(errors, "evaluation_model_producer_mismatch", eval_path,
                  f"评测声明的模型生产 Run 与登记模型不一致：{evaluation.model_id}")
        eval_run = run_records.get(evaluation.produced_by_run_id)
        if eval_run is not None:
            model_inputs = [ref for ref in eval_run[1].inputs if ref.kind == "model" and ref.id == evaluation.model_id]
            if not model_inputs:
                issue(errors, "evaluation_model_input_missing", eval_path,
                      f"评测 Run 未声明消费 lineage 中的模型：{evaluation.model_id}")
            elif model_entry is not None:
                model_digest = model_entry[1].digest
                eval_digest = model_inputs[0].digest
                if (eval_digest is None or eval_digest.status != "available"
                        or model_digest.status != "available"):
                    target = errors if require_complete else warnings
                    issue(target, "evaluation_model_digest_mismatch", eval_path,
                          f"评测 Run 或登记模型缺少可用摘要：{evaluation.model_id}")
                elif eval_digest.value != model_digest.value:
                    issue(errors, "evaluation_model_digest_mismatch", eval_path,
                          f"评测 Run 消费的模型摘要与登记模型不一致：{evaluation.model_id}")
            raw_eval = run_raw_records.get(evaluation.produced_by_run_id, {})
            declared_training_run = raw_eval.get("parent_run_id")
            if (declared_training_run and evaluation.model_produced_by_run_id
                    and declared_training_run != evaluation.model_produced_by_run_id):
                issue(errors, "evaluation_training_run_mismatch", eval_path,
                      "评测 Run 的 parent_run_id 与评测模型的 producing Run 不一致。")
            for dataset_id in evaluation.dataset_ids:
                if not any(ref.kind == "dataset" and ref.id == dataset_id for ref in eval_run[1].inputs):
                    issue(errors, "evaluation_dataset_input_missing", eval_path,
                          f"评测 Run 未声明消费 lineage 中的数据集：{dataset_id}")

    for source, parents in graph.items():
        for parent in parents:
            if parent not in artifact_definitions:
                issue(warnings, "unresolved_parent_id", base, f"{source} 的上游仅有 ID，未找到本地定义：{parent}")

    visiting: set[str] = set()
    visited: set[str] = set()
    cycle_nodes: set[str] = set()

    def visit(node: str, trail: list[str]) -> None:
        if node in visiting:
            cycle_nodes.update(trail[trail.index(node):] if node in trail else [node])
            return
        if node in visited:
            return
        visiting.add(node)
        for parent in graph.get(node, set()):
            visit(parent, [*trail, node])
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node, [])
    if cycle_nodes:
        issue(errors, "lineage_cycle", base, "血缘图存在环：" + ", ".join(sorted(cycle_nodes)))

    return {
        "root": str(base), "valid": not errors, "errors": errors, "warnings": warnings,
        "summary": {"runs": len(run_ids), "artifacts": len(artifact_definitions),
                    "datasets": len(dataset_records), "models": len(model_records),
                    "evaluations": len(evaluations), "errors": len(errors), "warnings": len(warnings)},
    }
