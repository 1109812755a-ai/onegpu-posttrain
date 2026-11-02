from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .stage_compare import _metrics


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _evalscope_denominator(payload: dict[str, Any], eval_path: Path, benchmark: str) -> int | None:
    report = ((payload.get("evalscope_reports") or {}).get(benchmark) or {})
    report_path_value = report.get("report_path")
    if not isinstance(report_path_value, str) or not report_path_value:
        return None
    report_path = Path(report_path_value)
    candidates = [report_path, eval_path.parent / report_path]
    for candidate in candidates:
        report_payload = _load(candidate)
        if report_payload is None:
            continue
        metrics = report_payload.get("metrics") or []
        if metrics and isinstance(metrics[0], dict):
            count = metrics[0].get("num")
            if isinstance(count, int):
                return count
    return None


def _project_root(config_path: str | Path) -> Path:
    resolved = Path(config_path).resolve()
    for parent in resolved.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return resolved.parent


def _resolve_reference(value: str, base: Path) -> Path:
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else base / candidate).resolve(strict=False)


def _one_model(
    spec: dict[str, Any], expected_metrics: list[str], limits: dict[str, Any], project_root: Path,
) -> dict[str, Any]:
    configured_eval_path = Path(spec["eval_path"])
    eval_path = configured_eval_path
    if not eval_path.is_absolute():
        eval_path = project_root / eval_path
    eval_path = eval_path.resolve(strict=False)
    payload = _load(eval_path)
    lineage = _load(eval_path.with_name("evaluation_lineage.json"))
    run = _load(eval_path.with_name("evaluation_run.json"))
    metrics = _metrics(payload or {})
    issues: list[dict[str, str]] = []
    mismatches = False

    def issue(code: str, message: str, *, mismatch: bool = False) -> None:
        nonlocal mismatches
        issues.append({"code": code, "message": message})
        mismatches = mismatches or mismatch

    if payload is None:
        issue("metrics_missing", "评测指标文件缺失。")
    if run is None:
        issue("run_missing", "评测运行记录缺失。")
    if lineage is None:
        issue("lineage_missing", "评测 lineage 文件缺失。")

    run_id = (run or {}).get("run_id")
    eval_id = (lineage or {}).get("eval_id")
    lineage_run_id = (lineage or {}).get("produced_by_run_id")
    if run and not run_id:
        issue("run_id_missing", "评测运行记录缺少 run_id。")
    if lineage and not eval_id:
        issue("eval_id_missing", "评测 lineage 缺少 eval_id。")
    if run_id and lineage_run_id and run_id != lineage_run_id:
        issue("run_id_mismatch", "评测运行记录与 lineage 的 run_id 不一致。", mismatch=True)
    elif run_id and not lineage_run_id:
        issue("lineage_run_id_missing", "评测 lineage 缺少 produced_by_run_id。")

    payload_run_id = (payload or {}).get("run_id")
    payload_eval_id = (payload or {}).get("eval_id")
    if payload_run_id and run_id and payload_run_id != run_id:
        issue("metrics_run_id_mismatch", "指标文件与评测运行记录的 run_id 不一致。", mismatch=True)
    elif payload is not None and not payload_run_id:
        issue("metrics_run_id_missing", "指标文件未记录 run_id。")
    if payload_eval_id and eval_id and payload_eval_id != eval_id:
        issue("metrics_eval_id_mismatch", "指标文件与 lineage 的 eval_id 不一致。", mismatch=True)
    elif payload is not None and not payload_eval_id:
        issue("metrics_eval_id_missing", "指标文件未记录 eval_id。")

    configured_model_path = spec.get("model_path")
    run_inputs = (run or {}).get("inputs")
    model_inputs = [item for item in run_inputs if isinstance(item, dict) and item.get("kind") == "model"] if isinstance(run_inputs, list) else []
    input_model_ids = {item.get("id") for item in model_inputs if isinstance(item.get("id"), str)}
    lineage_model_id = (lineage or {}).get("model_id")
    if lineage and not lineage_model_id:
        issue("model_id_missing", "评测 lineage 缺少 model_id。")
    if run and not model_inputs:
        issue("run_model_input_missing", "评测运行记录未标明输入模型。")
    if lineage_model_id and model_inputs and lineage_model_id not in input_model_ids:
        issue("model_id_mismatch", "评测 lineage 模型 ID 与运行记录输入模型不一致。", mismatch=True)
    if configured_model_path and model_inputs:
        configured = _resolve_reference(str(configured_model_path), project_root)
        input_paths = [item.get("path") for item in model_inputs if isinstance(item.get("path"), str) and item.get("path")]
        if input_paths and not any(_resolve_reference(value, project_root) == configured for value in input_paths):
            issue("model_path_mismatch", "配置模型路径与评测运行记录的输入模型路径不一致。", mismatch=True)
        elif not input_paths:
            issue("model_path_unavailable", "评测运行记录未提供输入模型路径。")

    run_status = (run or {}).get("status")
    lineage_status = (lineage or {}).get("status")
    expected_lineage_status = {"completed": "completed", "partial": "partial", "failed": "failed", "oom": "failed"}.get(run_status)
    if run and run_status not in {"completed", "partial", "failed", "oom"}:
        issue("run_status_invalid", "评测运行记录状态缺失或不受支持。")
    if lineage and not lineage_status:
        issue("lineage_status_missing", "评测 lineage 缺少 status。")
    elif expected_lineage_status and lineage_status and expected_lineage_status != lineage_status:
        issue("status_mismatch", "评测运行记录与 lineage 的终态不一致。", mismatch=True)

    outputs = (run or {}).get("outputs")
    evaluation_outputs = [item for item in outputs if isinstance(item, dict) and item.get("kind") == "evaluation"] if isinstance(outputs, list) else []
    if run and not evaluation_outputs:
        issue("evaluation_output_missing", "评测运行记录未列出评测指标产物。")
    elif evaluation_outputs:
        matching_output = False
        for output in evaluation_outputs:
            output_id = output.get("id")
            output_path = output.get("path")
            id_matches = bool(eval_id and output_id == eval_id)
            path_matches = bool(output_path and _resolve_reference(str(output_path), eval_path.parent) == eval_path.resolve(strict=False))
            if id_matches and path_matches:
                matching_output = True
                break
        if not matching_output:
            issue("evaluation_output_mismatch", "评测运行记录输出与当前指标文件的 eval_id 或路径不一致。", mismatch=True)

    integrity_status = "mismatch" if mismatches else ("incomplete" if issues else "verified")
    if payload is None:
        status = "missing"
    elif run_status in {"failed", "oom"} or lineage_status == "failed":
        status = "failed"
    else:
        present = all(metrics.get(name) is not None for name in expected_metrics)
        status = "completed" if present and run_status == "completed" and lineage_status == "completed" and integrity_status == "verified" else "partial"
    lineage_ref = None
    if lineage:
        lineage_ref = {
            "eval_id": eval_id,
            "produced_by_run_id": lineage_run_id,
            "model_id": lineage_model_id,
            "dataset_ids": lineage.get("dataset_ids", []),
            "protocol_version": lineage.get("protocol_version"),
            "config_sha256": lineage.get("config_sha256"),
            "status": lineage.get("status"),
        }
    lineage_metrics = (lineage or {}).get("metrics") or {}
    samples = {
        "gsm8k": {
            "requested": limits.get("gsm8k"),
            "observed": (lineage_metrics.get("evalscope.gsm8k.score") or {}).get("denominator")
            or _evalscope_denominator(payload or {}, eval_path, "gsm8k"),
        },
        "math_500": {
            "requested": limits.get("math_500_total"),
            "observed": (lineage_metrics.get("evalscope.math_500.score") or {}).get("denominator")
            or _evalscope_denominator(payload or {}, eval_path, "math_500"),
        },
        "internal_holdout": {
            "requested": limits.get("internal_holdout"),
            "observed": ((payload or {}).get("internal_math") or {}).get("samples"),
        },
        "domain_ppl": {
            "requested": limits.get("domain_ppl"),
            "observed": ((payload or {}).get("domain_perplexity") or {}).get("samples"),
        },
    }
    return {
        "model_id": spec["model_id"],
        "model_path": spec["model_path"],
        "eval_path": str(configured_eval_path),
        "status": status,
        "metrics": metrics,
        "samples": samples,
        "lineage": lineage_ref,
        "evaluation_run_id": run_id,
        "provenance": {"status": integrity_status, "issues": issues},
        "errors": (run or {}).get("errors", []),
    }


def build_evaluation_matrix(config_path: str | Path, *, output_dir: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    expected_metrics = list(config.get("expected_metrics", ["gsm8k", "math_500", "internal_acc"]))
    limits = config.get("limits", {})
    project_root = _project_root(config_path)
    models = [_one_model(spec, expected_metrics, limits, project_root) for spec in config["models"]]
    baseline_id = str(config["baseline_model_id"])
    baseline = next((model for model in models if model["model_id"] == baseline_id), None)
    if baseline is None:
        raise ValueError(f"baseline_model_id {baseline_id!r} is not in models")

    populated = [model for model in models if model["lineage"]]
    protocol_versions = {model["lineage"]["protocol_version"] for model in populated}
    config_hashes = {model["lineage"]["config_sha256"] for model in populated}
    dataset_id_sets = {tuple(sorted(model["lineage"]["dataset_ids"])) for model in populated}
    expected_protocol = config.get("expected_protocol_version")
    protocol_consistency = {
        "expected_protocol_version": expected_protocol,
        "protocol_versions": sorted(protocol_versions),
        "protocol_version_consistent": len(protocol_versions) == 1 and (not expected_protocol or protocol_versions == {expected_protocol}),
        "config_hashes": sorted(config_hashes),
        "config_hash_consistent": len(config_hashes) == 1,
        "dataset_id_sets": [list(values) for values in sorted(dataset_id_sets)],
        "dataset_ids_consistent": len(dataset_id_sets) == 1,
    }
    all_completed = all(model["status"] == "completed" for model in models)
    provenance_valid = all(model.get("provenance", {}).get("status") == "verified" for model in models)
    comparable = all_completed and provenance_valid and all((
        protocol_consistency["protocol_version_consistent"],
        protocol_consistency["config_hash_consistent"],
        protocol_consistency["dataset_ids_consistent"],
    ))
    for model in models:
        delta = {}
        for name in expected_metrics:
            candidate_value = model["metrics"].get(name)
            baseline_value = baseline["metrics"].get(name)
            delta[name] = (
                float(candidate_value) - float(baseline_value)
                if isinstance(candidate_value, (int, float)) and isinstance(baseline_value, (int, float))
                else None
            )
        model["delta_vs_baseline"] = delta

    result = {
        "protocol": config.get("matrix_protocol", "unified-model-evaluation-matrix-v1"),
        "baseline_model_id": baseline_id,
        "requested_limits": limits,
        "expected_metrics": expected_metrics,
        "comparability": {
            "status": "comparable" if comparable else "partial_or_inconsistent",
            "protocol_consistency": protocol_consistency,
            "limitations": [] if comparable else [
                "至少一个模型评测缺失/部分/失败、来源血缘不完整/不匹配，或评测协议、配置哈希、数据集 ID 不一致；缺失指标保持 null。"
            ],
        },
        "models": models,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation_matrix.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    headers = ["Model", "Status", *expected_metrics]
    lines = [
        "# Unified M2/M3 Evaluation Matrix", "",
        f"Baseline: `{baseline_id}`; comparability: `{result['comparability']['status']}`.", "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for model in models:
        values = [model["model_id"], model["status"]]
        values.extend("N/A" if model["metrics"].get(name) is None else f"{model['metrics'][name]:.6g}" for name in expected_metrics)
        lines.append("| " + " | ".join(values) + " |")
    if result["comparability"]["limitations"]:
        lines.extend(["", "## Comparability notes", "", *[f"- {note}" for note in result["comparability"]["limitations"]]])
    lines.append("")
    (out / "evaluation_matrix.md").write_text("\n".join(lines), encoding="utf-8")
    return result
