"""Offline, exploratory SFT sequence-length calibration for local DeepMath Parquet."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import statistics
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..adapters.ms_swift import adapt_sft
from ..evaluation.answers import extract_answer, normalize_answer
from ..evaluation.verification import (
    AnswerVerificationResult,
    verification_configuration,
    verification_summary,
    verify_answer_pairs,
)
from ..schemas import Message, SFTRecord
from .io import write_jsonl
from .normalize import problem_id
from .tokenization import HuggingFaceTokenizer, TokenizerBackend

DEFAULT_LENGTH_CANDIDATES = (1024, 2048, 4096)
DEFAULT_TOKENIZER_REPO = "Qwen/Qwen3-0.6B-Base"
DEFAULT_TOKENIZER_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
REQUIRED_COLUMNS = (
    "question", "final_answer", "r1_solution_1", "r1_solution_2", "r1_solution_3",
)
PROTOCOL_VERSION = "sft-length-calibration-v2"


class CalibrationError(ValueError):
    """Raised when an offline calibration input or protocol is invalid."""


class CandidateCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_length: int = Field(gt=0)
    trace_slots: int = Field(ge=0)
    nonempty_traces: int = Field(ge=0)
    length_eligible_traces: int = Field(ge=0)
    length_retention_rate: float | None = None
    eligible_source_problems: int = Field(ge=0)
    training_examples: int = Field(ge=0)
    answer_agreement_examples: int = Field(ge=0)
    answer_agreement_rate: float | None = None
    rendered_length_tokens: dict[str, int | float | None]
    content_sha256: str
    canonical_path: str
    native_path: str
    manifest_path: str


class SFTLengthCalibrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.1"] = "1.1"
    protocol_version: Literal["sft-length-calibration-v2"] = PROTOCOL_VERSION
    calibration_id: str
    status: Literal["exploratory"] = "exploratory"
    created_at: str
    source: dict[str, Any]
    sampling: dict[str, Any]
    tokenizer: dict[str, Any]
    quality_evidence: dict[str, Any]
    candidates: list[CandidateCalibration]
    interpretation: list[str]
    probes: list[dict[str, Any]] = Field(default_factory=list)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_snapshot_revision(root: Path) -> str:
    parts = root.resolve().parts
    try:
        index = parts.index("snapshots")
        return parts[index + 1]
    except (ValueError, IndexError):
        return f"local-{_sha256_bytes(str(root.resolve()).encode())[:16]}"


def _parquet_inventory(root: str | Path) -> tuple[Path, list[dict[str, Any]], str]:
    path = Path(root).expanduser().resolve()
    if not path.exists():
        raise CalibrationError(
            f"DeepMath 本地缓存不存在：{path}。请先将数据缓存到本机，再重试；本命令不会访问网络或自动下载。"
        )
    files = [path] if path.is_file() and path.suffix.lower() == ".parquet" else sorted(path.rglob("*.parquet"))
    files = [file for file in files if file.is_file()]
    if not files:
        raise CalibrationError(f"本地路径中没有 Parquet 文件：{path}；不会尝试联网下载。")
    inventory = []
    for file in files:
        try:
            relative = file.relative_to(path).as_posix() if path.is_dir() else file.name
        except ValueError:
            relative = file.name
        inventory.append({"path": relative, "size_bytes": file.stat().st_size, "sha256": _sha256_file(file), "absolute_path": file})
    identity = [{key: value for key, value in item.items() if key != "absolute_path"} for item in inventory]
    fingerprint = _sha256_bytes(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    return path, inventory, fingerprint


def _iter_parquet_rows(inventory: list[dict[str, Any]]) -> Iterable[tuple[str, int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise CalibrationError("读取本地 Parquet 需要 pyarrow；请安装项目 data extra。") from exc

    for item in inventory:
        file = item["absolute_path"]
        parquet = pq.ParquetFile(file)
        columns = set(parquet.schema_arrow.names)
        missing = sorted(set(REQUIRED_COLUMNS) - columns)
        if missing:
            raise CalibrationError(f"Parquet 缺少 DeepMath 必需列 {missing}：{item['path']}")
        row_index = 0
        selected_columns = [name for name in REQUIRED_COLUMNS if name in columns]
        for batch in parquet.iter_batches(batch_size=256, columns=selected_columns):
            for row in batch.to_pylist():
                record_id = _sha256_bytes(f"{item['sha256']}:{row_index}".encode())
                yield record_id, row_index, row
                row_index += 1


def _sample_rows(
    inventory: list[dict[str, Any]], *, limit: int, seed: int,
) -> tuple[list[dict[str, Any]], int]:
    """Read all shards and retain the lowest deterministic hash ranks in O(limit) rows."""
    if limit <= 0:
        raise CalibrationError("sample limit 必须大于 0。")
    heap: list[tuple[int, str, dict[str, Any]]] = []
    total = 0
    for record_id, row_index, row in _iter_parquet_rows(inventory):
        total += 1
        rank = int(_sha256_bytes(f"{seed}:{record_id}".encode()), 16)
        row_ref = {"source_row_id": record_id, "source_row_index": row_index, "row": row}
        entry = (-rank, record_id, row_ref)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    sample = [entry[2] for entry in sorted(heap, key=lambda entry: (-entry[0], entry[1]))]
    return sample, total


def _percentile(values: list[int], p: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def _length_distribution(values: list[int]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check_new_output(output: str | Path) -> Path:
    requested = Path(output).expanduser()
    path = requested if requested.is_absolute() else Path.cwd() / requested
    resolved = path.resolve(strict=False)
    if resolved.exists():
        raise FileExistsError(f"校准输出必须是全新路径，避免覆盖已有产物：{path}")
    repo_root = Path.cwd().resolve()
    official_mini = (repo_root / "data" / "processed" / "mini-v1").resolve(strict=False)
    try:
        resolved.relative_to(official_mini)
    except ValueError:
        return path
    raise CalibrationError(f"校准产物不能写入官方 Mini 路径：{official_mini}")


def calibrate_sft_lengths(
    *,
    source: str | Path,
    output: str | Path,
    sample_limit: int = 2000,
    seed: int = 42,
    candidates: tuple[int, ...] | list[int] = DEFAULT_LENGTH_CANDIDATES,
    tokenizer_repo: str = DEFAULT_TOKENIZER_REPO,
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
    tokenizer: TokenizerBackend | None = None,
    answer_verification: Literal["exact", "semantic"] = "exact",
) -> dict[str, Any]:
    """Build offline, non-authoritative candidate manifests and a calibration report."""
    caps = tuple(int(cap) for cap in candidates)
    if not caps or any(cap <= 0 for cap in caps) or len(set(caps)) != len(caps):
        raise CalibrationError("length candidates 必须是非空、唯一且为正整数的集合。")
    if answer_verification not in {"exact", "semantic"}:
        raise CalibrationError("answer_verification must be 'exact' or 'semantic'.")
    output_path = _check_new_output(output)
    source_path, inventory, source_digest = _parquet_inventory(source)
    sample, source_row_count = _sample_rows(inventory, limit=sample_limit, seed=seed)
    if not sample:
        raise CalibrationError("DeepMath 本地 Parquet 没有可用记录。")

    tokenizer_backend = tokenizer or HuggingFaceTokenizer(
        tokenizer_repo, tokenizer_revision, local_files_only=True,
    )
    template = getattr(getattr(tokenizer_backend, "tokenizer", None), "chat_template", None)
    template_sha = _sha256_bytes(str(template or "").encode("utf-8"))
    rendered: list[dict[str, Any]] = []
    total_slots = len(sample) * 3
    nonempty = 0
    answer_extractable = 0
    exact_agreement = 0
    answer_unparseable = 0
    reference_answer_missing = 0
    parsed_answer_mismatch = 0
    all_lengths: list[int] = []
    pending_semantic: list[tuple[dict[str, Any], str, str]] = []

    for sampled in sample:
        row = sampled["row"]
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        gold = str(row.get("final_answer") or "").strip()
        pid = problem_id(question)
        traces_for_problem = []
        for index in range(1, 4):
            field = f"r1_solution_{index}"
            trace = str(row.get(field) or "").strip()
            if not trace:
                continue
            nonempty += 1
            answer = extract_answer(trace)
            if answer is not None:
                answer_extractable += 1
            else:
                answer_unparseable += 1
            if not gold:
                reference_answer_missing += 1
            agreement = bool(answer is not None and gold and normalize_answer(answer) == normalize_answer(gold))
            exact_agreement += int(agreement)
            parsed_answer_mismatch += int(answer is not None and bool(gold) and not agreement)
            record_id = f"dm_{sampled['source_row_id'][:20]}_r{index}"
            messages = [Message(role="user", content=question), Message(role="assistant", content=trace)]
            tokens = tokenizer_backend.count_messages([message.model_dump() for message in messages])
            all_lengths.append(tokens)
            traces_for_problem.append({
                "id": record_id,
                "trace_index": index,
                "trace": trace,
                "tokens": tokens,
                "messages": messages,
                "agreement": agreement,
                "verification_status": (
                    "unparseable" if answer is None else "unverified" if not gold or not agreement else "correct"
                ),
                "verification_method": (
                    "extract-answer:v1" if answer is None else "no-reference" if not gold else
                    "normalize-answer:v1" if agreement else "semantic-verification-disabled"
                ),
            })
            trace_entry = traces_for_problem[-1]
            if answer_verification == "semantic" and answer is not None and gold and not agreement:
                pending_semantic.append((trace_entry, answer, gold))
        rendered.append({
            "source_row_id": sampled["source_row_id"],
            "problem_id": pid,
            "question": question,
            "gold": gold,
            "row": row,
            "traces": traces_for_problem,
        })

    if answer_verification == "semantic" and pending_semantic:
        semantic_results = verify_answer_pairs([
            (prediction, reference) for _, prediction, reference in pending_semantic
        ])
        for (trace, _, _), result in zip(pending_semantic, semantic_results, strict=True):
            trace["verification_status"] = result.status
            trace["verification_method"] = result.method
            trace["verifier"] = result.verifier
            trace["verifier_version"] = result.verifier_version

    trace_verification_results = []
    for problem in rendered:
        for trace in problem["traces"]:
            if trace["verification_status"] == "correct" and trace["agreement"]:
                trace_verification_results.append(AnswerVerificationResult(
                    status="correct", method="normalize-answer:v1", verifier="builtin", verifier_version="1",
                ))
            else:
                trace_verification_results.append(AnswerVerificationResult(
                    status=trace["verification_status"], method=trace["verification_method"],
                    verifier=trace.get("verifier"), verifier_version=trace.get("verifier_version"),
                ))
    verification_evidence = verification_summary(trace_verification_results)

    output_path.mkdir(parents=True, exist_ok=False)
    candidate_reports: list[CandidateCalibration] = []
    candidate_manifest_refs: list[dict[str, Any]] = []
    for cap in caps:
        eligible_traces = [trace for problem in rendered for trace in problem["traces"] if trace["tokens"] <= cap]
        records: list[dict[str, Any]] = []
        exact_examples = 0
        eligible_problems = 0
        for problem in rendered:
            eligible = [
                trace for trace in problem["traces"]
                if trace["tokens"] <= cap and (
                    trace["agreement"] if answer_verification == "exact"
                    else trace["verification_status"] == "correct"
                )
            ]
            if not eligible:
                continue
            eligible_problems += 1
            chosen = min(eligible, key=lambda trace: (trace["tokens"], trace["trace_index"]))
            exact_examples += int(chosen["agreement"])
            record = SFTRecord(
                id=chosen["id"],
                raw_problem=problem["question"],
                messages=chosen["messages"],
                prompt_template_id="qwen3-math-v1",
                metadata={
                    "problem_id": problem["problem_id"],
                    "source_record_id": f"deepmath:{problem['source_row_id']}",
                    "source_trace_id": f"deepmath:{problem['source_row_id']}:r{chosen['trace_index']}",
                    "rendered_tokens": chosen["tokens"],
                    "reference_answer": problem["gold"],
                    "answer_agreement": "exact_normalized_match" if chosen["agreement"] else "not_exact",
                    "answer_verification_mode": answer_verification,
                    "answer_verification_status": chosen["verification_status"],
                    "answer_verification_method": chosen["verification_method"],
                    "verifier": chosen.get("verifier", "builtin" if chosen["agreement"] else None),
                    "verifier_version": chosen.get("verifier_version", "1" if chosen["agreement"] else None),
                    "semantic_verification": (
                        "not_measured" if answer_verification == "exact" else chosen["verification_status"]
                    ),
                    "source_snapshot_sha256": source_digest,
                },
            ).model_dump(mode="json")
            records.append(record)

        candidate_dir = output_path / f"length-{cap}"
        canonical_path = candidate_dir / "canonical" / "sft.jsonl"
        native_path = candidate_dir / "native" / "sft.jsonl"
        write_jsonl(canonical_path, records)
        write_jsonl(native_path, (adapt_sft(record) for record in records))
        content_digest = _sha256_file(canonical_path)
        native_digest = _sha256_file(native_path)
        manifest = {
            "schema_version": "1.1",
            "dataset_id": f"ds-sft-length-{cap}-{source_digest[:12]}-{seed}-{answer_verification}",
            "status": "exploratory",
            "stage": "sft",
            "source": {
                "name": "zwhe99/DeepMath-103K",
                "local_path": str(source_path),
                "revision": _local_snapshot_revision(source_path),
                "snapshot_sha256": source_digest,
                "source_rows": source_row_count,
            },
            "sampling": {
                "method": "lowest_sha256_rank_per_source_row",
                "seed": seed,
                "requested_source_rows": sample_limit,
                "selected_source_rows": len(sample),
                "source_row_denominator": source_row_count,
                "source_row_ids_sha256": _sha256_bytes("\n".join(sorted(item["source_row_id"] for item in sample)).encode()),
            },
            "eligibility": {
                "max_rendered_tokens": cap,
                "token_count_method": "pinned tokenizer + final user/assistant chat-template rendering",
                "overlength_action": "reject; never truncate",
                "length_eligible_trace_count": len(eligible_traces),
                "nonempty_trace_denominator": nonempty,
                "training_record_rule": (
                    "per source row, shortest length-eligible trace whose extracted final answer exactly matches normalized final_answer"
                    if answer_verification == "exact" else
                    "per source row, shortest length-eligible trace with a correct typed final-answer verification outcome"
                ),
                "answer_verification_mode": answer_verification,
            },
            "tokenizer": {"repo": tokenizer_repo, "revision": tokenizer_revision, "chat_template_sha256": template_sha},
            "quality_evidence": {
                "length_eligibility": "measured",
                "answer_agreement": "exact normalized string match only; not a semantic correctness verifier",
                "answer_extractable_count": answer_extractable,
                "answer_agreement_count": exact_agreement,
                "answer_unparseable_count": answer_unparseable,
                "reference_answer_missing_count": reference_answer_missing,
                "parsed_answer_mismatch_count": parsed_answer_mismatch,
                "semantic_verification": "not_measured" if answer_verification == "exact" else "measured; final-answer equivalence only",
                "answer_verification": verification_evidence,
                "quality_approval": "not_evaluated",
            },
            "counts": {
                "trace_slots": total_slots,
                "nonempty_traces": nonempty,
                "length_eligible_traces": len(eligible_traces),
                "source_problems_with_eligible_trace": sum(bool([t for t in item["traces"] if t["tokens"] <= cap]) for item in rendered),
                "answer_agreement_training_examples": exact_examples,
                "verification_selected_training_examples": len(records),
            },
            "hashes": {"canonical_sha256": content_digest, "native_sha256": native_digest},
            "paths": {"canonical": "canonical/sft.jsonl", "native": "native/sft.jsonl"},
        }
        manifest_path = candidate_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        candidate_manifest_refs.append({"max_length": cap, "path": str(manifest_path), "sha256": _sha256_file(manifest_path)})
        rate = len(eligible_traces) / nonempty if nonempty else None
        candidate_reports.append(CandidateCalibration(
            max_length=cap,
            trace_slots=total_slots,
            nonempty_traces=nonempty,
            length_eligible_traces=len(eligible_traces),
            length_retention_rate=rate,
            eligible_source_problems=sum(bool([t for t in item["traces"] if t["tokens"] <= cap]) for item in rendered),
            training_examples=len(records),
            answer_agreement_examples=exact_examples,
            answer_agreement_rate=exact_examples / eligible_problems if eligible_problems else None,
            rendered_length_tokens=_length_distribution(all_lengths),
            content_sha256=content_digest,
            canonical_path=str(canonical_path),
            native_path=str(native_path),
            manifest_path=str(manifest_path),
        ))

    report = SFTLengthCalibrationReport(
        calibration_id=f"sft-length-{source_digest[:12]}-s{seed}-{answer_verification}",
        created_at=datetime.now(timezone.utc).isoformat(),
        source={
            "name": "zwhe99/DeepMath-103K",
            "local_path": str(source_path),
            "revision": _local_snapshot_revision(source_path),
            "snapshot_sha256": source_digest,
            "parquet_files": [{key: value for key, value in item.items() if key != "absolute_path"} for item in inventory],
            "row_count": source_row_count,
        },
        sampling={
            "method": "lowest_sha256_rank_per_source_row",
            "seed": seed,
            "requested_source_rows": sample_limit,
            "selected_source_rows": len(sample),
            "selected_source_row_ids_sha256": _sha256_bytes("\n".join(sorted(item["source_row_id"] for item in sample)).encode()),
            "trace_slots_per_selected_row": 3,
        },
        tokenizer={"repo": tokenizer_repo, "revision": tokenizer_revision, "chat_template_sha256": template_sha,
                   "local_files_only": tokenizer is None, "name": getattr(tokenizer_backend, "name", type(tokenizer_backend).__name__)},
        quality_evidence={
            "length_eligibility": "measured",
            "answer_extraction": "measured using project extract_answer helper",
            "answer_agreement": "exact normalized string agreement with final_answer; not semantic verification",
            "answer_extractable_count": answer_extractable,
            "answer_unparseable_count": answer_unparseable,
            "reference_answer_missing_count": reference_answer_missing,
            "parsed_answer_mismatch_count": parsed_answer_mismatch,
            "answer_agreement_count": exact_agreement,
            "semantic_verification": "not_measured" if answer_verification == "exact" else "measured; final-answer equivalence only",
            "answer_verification_mode": answer_verification,
            "answer_verification": verification_evidence,
            "verifier_coverage": (
                verification_evidence["coverage"] if answer_verification == "semantic" else "not_measured"
            ),
            "verifier_configuration": verification_configuration(),
            "quality_approval": "not_evaluated",
        },
        candidates=candidate_reports,
        interpretation=[
            "This is an exploratory deterministic sample, not a population estimate or official Mini dataset.",
            "Length eligibility does not establish semantic correctness.",
            (
                "Exact normalized answer agreement is a conservative screening signal, not a semantic verifier."
                if answer_verification == "exact" else
                "Semantic verification applies only to extracted final-answer equivalence and does not assess reasoning quality."
            ),
            "No model-quality or causal improvement claim can be inferred from this report.",
        ],
    ).model_dump(mode="json")
    report["candidate_manifests"] = candidate_manifest_refs
    _write_json(output_path / "calibration_report.json", report)
    return report
