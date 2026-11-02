from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _execution(run: dict[str, Any]) -> dict[str, Any]:
    return dict(run.get("execution") or run.get("train") or {})


def write_training_feasibility_comparison(
    peft_run: str | Path,
    full_run: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    peft = _load(peft_run)
    full = _load(full_run)
    pe = _execution(peft)
    fe = _execution(full)
    result = {
        "peft": {
            "status": pe.get("status", "UNKNOWN"),
            "wall_seconds": pe.get("wall_seconds"),
            "peak_gpu_memory_mib_sampled": pe.get("peak_gpu_memory_mib_sampled"),
            "delta_gpu_memory_mib_sampled": pe.get("delta_gpu_memory_mib_sampled"),
        },
        "full": {
            "status": fe.get("status", "UNKNOWN"),
            "wall_seconds": fe.get("wall_seconds"),
            "peak_gpu_memory_mib_sampled": fe.get("peak_gpu_memory_mib_sampled"),
            "delta_gpu_memory_mib_sampled": fe.get("delta_gpu_memory_mib_sampled"),
        },
        "interpretation": (
            "Full-CPT is technically feasible under the probe profile."
            if fe.get("status") == "PASS"
            else "Full-CPT OOM is a valid feasibility result; keep PEFT-CPT as the official 8GB path."
            if fe.get("status") == "OOM"
            else "Full-CPT probe did not produce a feasibility conclusion; inspect the failure logs."
        ),
        "memory_metric_note": "nvidia-smi memory.used is system-level sampled memory on GPU0, not process-exclusive allocator peak.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cpt_feasibility.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# CPT Feasibility Comparison",
        "",
        "| Path | Status | Wall seconds | Peak GPU memory MiB (sampled) | Delta MiB (sampled) |",
        "|---|---|---:|---:|---:|",
    ]
    for label, item in (("PEFT-CPT", result["peft"]), ("Full-CPT Probe", result["full"])):
        md.append(
            f"| {label} | {item['status']} | {item['wall_seconds'] if item['wall_seconds'] is not None else '-'} | "
            f"{item['peak_gpu_memory_mib_sampled'] if item['peak_gpu_memory_mib_sampled'] is not None else '-'} | "
            f"{item['delta_gpu_memory_mib_sampled'] if item['delta_gpu_memory_mib_sampled'] is not None else '-'} |"
        )
    md += ["", f"> {result['interpretation']}", "", f"Note: {result['memory_metric_note']}"]
    (out / "cpt_feasibility.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return result
