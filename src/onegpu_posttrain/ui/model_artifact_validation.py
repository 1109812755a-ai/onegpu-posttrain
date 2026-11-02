"""Shared validation for registered model artifacts and their producer runs."""

from __future__ import annotations

from pathlib import Path

from ..lineage import ModelLineage, RunRecord


def validate_model_producing_run(workspace: Path, registry_path: Path, model: ModelLineage) -> RunRecord | None:
    """Resolve and validate a producer run, including nested merge-run records."""
    if not model.produced_by_run_id:
        return None

    root = workspace.resolve(strict=True)
    registry = registry_path.resolve(strict=True)
    registry.relative_to(root)
    search_root = registry.parent.parent.resolve(strict=True)
    search_root.relative_to(root)
    matches: list[RunRecord] = []
    for candidate in search_root.rglob("lineage.json"):
        if candidate.is_symlink():
            continue
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            run = RunRecord.model_validate_json(resolved.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if run.run_id == model.produced_by_run_id:
            matches.append(run)

    if len(matches) != 1:
        raise ValueError("父模型 producing run lineage 缺失或不唯一")
    run = matches[0]
    if run.status != "completed":
        raise ValueError("父模型 producing run 尚未成功完成")
    outputs = [item for item in run.outputs if item.id == model.model_id and item.kind == "model"]
    if len(outputs) != 1:
        raise ValueError("父模型 producing run 未唯一声明该模型输出")
    output_digest = outputs[0].digest
    if (model.digest.status == "available" and output_digest is not None
            and output_digest.status == "available" and model.digest.value != output_digest.value):
        raise ValueError("父模型登记摘要与 producing run 输出摘要不一致")
    return run
