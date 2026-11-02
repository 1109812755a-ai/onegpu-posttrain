from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    reports = payload.get("evalscope_reports") or {}
    internal = payload.get("internal_math") or {}
    domain = payload.get("domain_perplexity") or {}
    return {
        "gsm8k": (reports.get("gsm8k") or {}).get("score"),
        "math_500": (reports.get("math_500") or {}).get("score"),
        "internal_acc": internal.get("accuracy"),
        "domain_ppl": domain.get("perplexity"),
    }


def _evaluation_lineage(eval_path: str | Path) -> dict[str, Any] | None:
    sidecar = Path(eval_path).with_name("evaluation_lineage.json")
    if not sidecar.is_file():
        return None
    try:
        return _load(sidecar)
    except (OSError, ValueError):
        return None


def compare_eval_files(base_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    base = _load(base_path)
    candidate = _load(candidate_path)
    bm = _metrics(base)
    cm = _metrics(candidate)
    base_lineage = _evaluation_lineage(base_path)
    candidate_lineage = _evaluation_lineage(candidate_path)
    delta: dict[str, float | None] = {}
    for k in bm:
        b, c = bm[k], cm[k]
        delta[k] = (float(c) - float(b)) if isinstance(b, (int, float)) and isinstance(c, (int, float)) else None
    warnings: list[str] = []
    comparable = True
    if not base_lineage or not candidate_lineage:
        warnings.append("至少一个评测缺少有效 evaluation_lineage；旧版分数仍展示，但协议可比性无法确认。")
        comparable = False
    else:
        if base_lineage.get("protocol_version") != candidate_lineage.get("protocol_version"):
            warnings.append("评测协议版本不一致；delta 仅作描述，不应直接比较。")
            comparable = False
        if set(base_lineage.get("dataset_ids", [])) != set(candidate_lineage.get("dataset_ids", [])):
            warnings.append("评测数据集集合不同（例如留出集或清洗版本不同）；delta 是描述性差异，不代表清洗带来因果收益。")
    return {
        "base": {"run_id": base.get("run_id"), "stage": base.get("stage"), "model": base.get("model"), "metrics": bm},
        "candidate": {"run_id": candidate.get("run_id"), "stage": candidate.get("stage"), "model": candidate.get("model"), "metrics": cm},
        "delta_candidate_minus_base": delta,
        "comparability": {"status": "comparable" if comparable else "warning", "warnings": warnings},
        "notes": {
            "domain_ppl": "Lower is better; therefore a negative delta is an improvement.",
            "accuracy_metrics": "Higher is better; positive deltas are improvements.",
            "causality": "指标 delta 是描述性对比；除非通过受控实验设计隔离清洗策略等变量，否则不构成因果结论 (not a causal claim).",
        },
    }


def write_stage_comparison(base_path: str | Path, candidate_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    result = compare_eval_files(base_path, candidate_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage_comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    base = result["base"]["metrics"]
    cand = result["candidate"]["metrics"]
    delta = result["delta_candidate_minus_base"]
    def fmt(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    rows = []
    for metric in ("gsm8k", "math_500", "internal_acc", "domain_ppl"):
        rows.append(f"| `{metric}` | {fmt(base.get(metric))} | {fmt(cand.get(metric))} | {fmt(delta.get(metric))} |")
    warnings = result["comparability"]["warnings"]
    warning_block = ["## 可比性提示", ""]
    warning_block.extend([f"- {warning}" for warning in warnings] or ["- 已找到两侧评测血缘；请结合协议和实验控制条件解读。"])
    warning_block.extend(["", f"> {result['notes']['causality']}"])
    md = "\n".join([
        "# Stage Comparison",
        "",
        "> Candidate minus Base. Accuracy: higher is better. Domain PPL: lower is better.",
        "",
        "| Metric | Base | Candidate | Delta |",
        "|---|---:|---:|---:|",
        *rows,
        "",
        *warning_block,
        "",
    ])
    (out / "stage_comparison.md").write_text(md, encoding="utf-8")
    return result
