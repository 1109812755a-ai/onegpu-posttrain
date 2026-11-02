"""Bounded, auditable final-answer equivalence verification."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import multiprocessing
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .answers import normalize_answer

VerificationStatus = Literal[
    "correct", "incorrect", "unparseable", "unsupported", "unverified", "timeout", "verifier_error",
]
VERIFICATION_PROTOCOL = "math-answer-verification-v1"
DEFAULT_BATCH_SIZE = 32
DEFAULT_BATCH_TIMEOUT_SECONDS = 60.0
MAX_ANSWER_CHARS = 4096


class AnswerVerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: VerificationStatus
    method: str
    verifier: str | None = None
    verifier_version: str | None = None
    detail: str | None = None


class AnswerPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction: str
    reference: str


def _verification_identity() -> tuple[str | None, str | None]:
    try:
        version = importlib.metadata.version("math-verify")
    except importlib.metadata.PackageNotFoundError:
        return None, None
    return "math-verify", version


def verification_configuration(*, batch_size: int = DEFAULT_BATCH_SIZE,
                               timeout_seconds: float = DEFAULT_BATCH_TIMEOUT_SECONDS) -> dict[str, Any]:
    return {
        "protocol": VERIFICATION_PROTOCOL,
        "verifier": _verification_identity(),
        "batch_size": batch_size,
        "batch_timeout_seconds": timeout_seconds,
        "max_answer_chars": MAX_ANSWER_CHARS,
        "math_verify_internal_timeout": "disabled; supervised by parent batch deadline",
    }


def _math_verify_worker(pairs: list[dict[str, str]], connection: Any) -> None:
    """Spawn-safe worker; math-verify's own per-expression multiprocessing timeout is disabled."""
    try:
        from math_verify import parse, verify  # type: ignore[import-not-found]
    except ImportError:
        connection.send({"kind": "unavailable"})
        connection.close()
        return

    outcomes: list[str] = []
    for pair in pairs:
        prediction = pair["prediction"]
        reference = pair["reference"]
        if max(len(prediction), len(reference)) > MAX_ANSWER_CHARS:
            outcomes.append("unsupported")
            continue
        try:
            reference_parsed = parse(reference, parsing_timeout=None, raise_on_error=True)
            prediction_parsed = parse(prediction, parsing_timeout=None, raise_on_error=True)
            if not reference_parsed or not prediction_parsed:
                outcomes.append("unparseable")
                continue
            outcomes.append("correct" if verify(
                reference_parsed, prediction_parsed, timeout_seconds=None, raise_on_error=True,
            ) else "incorrect")
        except (ValueError, TypeError, ArithmeticError):
            outcomes.append("unparseable")
        except Exception:  # noqa: BLE001 - worker failures are explicit evidence outcomes.
            outcomes.append("verifier_error")
    try:
        connection.send({"kind": "results", "outcomes": outcomes})
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


def _run_bounded_batch(pairs: list[AnswerPair], timeout_seconds: float) -> list[VerificationStatus]:
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_math_verify_worker,
        args=([pair.model_dump(mode="json") for pair in pairs], send),
    )
    try:
        process.start()
        send.close()
        if not receive.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            return ["timeout"] * len(pairs)
        response = receive.recv()
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join()
        if process.exitcode not in (0, None):
            return ["verifier_error"] * len(pairs)
        if response.get("kind") == "unavailable":
            return ["unverified"] * len(pairs)
        outcomes = response.get("outcomes")
        if response.get("kind") != "results" or not isinstance(outcomes, list) or len(outcomes) != len(pairs):
            return ["verifier_error"] * len(pairs)
        accepted = {"correct", "incorrect", "unparseable", "unsupported", "unverified", "timeout", "verifier_error"}
        return [outcome if outcome in accepted else "verifier_error" for outcome in outcomes]
    except (OSError, EOFError, RuntimeError, ValueError):
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)
        return ["verifier_error"] * len(pairs)
    finally:
        receive.close()
        try:
            send.close()
        except OSError:
            pass
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(timeout=5)


def verify_answer_pairs(
    pairs: Sequence[tuple[str, str]],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout_seconds: float = DEFAULT_BATCH_TIMEOUT_SECONDS,
) -> list[AnswerVerificationResult]:
    """Verify ordered (prediction, reference) pairs with bounded, isolated batches."""
    if batch_size < 1 or timeout_seconds <= 0:
        raise ValueError("batch_size and timeout_seconds must be positive")
    verifier, verifier_version = _verification_identity()
    results: list[AnswerVerificationResult | None] = [None] * len(pairs)
    pending: list[tuple[int, AnswerPair]] = []
    for index, (prediction, reference) in enumerate(pairs):
        if not isinstance(prediction, str) or not isinstance(reference, str) or not reference.strip():
            results[index] = AnswerVerificationResult(
                status="unsupported", method="answer-verification:v1",
                detail="Prediction/reference is missing or not text.",
            )
        elif normalize_answer(prediction) == normalize_answer(reference):
            results[index] = AnswerVerificationResult(
                status="correct", method="normalize-answer:v1", verifier="builtin", verifier_version="1",
            )
        elif max(len(prediction), len(reference)) > MAX_ANSWER_CHARS:
            results[index] = AnswerVerificationResult(
                status="unsupported", method="answer-verification:v1",
                detail=f"Answer exceeds {MAX_ANSWER_CHARS} characters.",
            )
        elif verifier is None:
            results[index] = AnswerVerificationResult(
                status="unverified", method="math-verify:v1", detail="Optional math-verify dependency is unavailable.",
            )
        else:
            pending.append((index, AnswerPair(prediction=prediction, reference=reference)))

    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        outcomes = _run_bounded_batch([pair for _, pair in chunk], timeout_seconds)
        for (index, _), status in zip(chunk, outcomes, strict=True):
            results[index] = AnswerVerificationResult(
                status=status, method=VERIFICATION_PROTOCOL, verifier=verifier,
                verifier_version=verifier_version,
            )
    return [result or AnswerVerificationResult(
        status="verifier_error", method=VERIFICATION_PROTOCOL, verifier=verifier,
        verifier_version=verifier_version, detail="Missing worker result.",
    ) for result in results]


def verification_summary(results: Sequence[AnswerVerificationResult]) -> dict[str, Any]:
    """Return deterministic counts and coverage for a batch of results."""
    from collections import Counter

    counts = Counter(result.status for result in results)
    decided = counts["correct"] + counts["incorrect"]
    return {
        "protocol": VERIFICATION_PROTOCOL,
        "verifier": next((result.verifier for result in results if result.verifier not in {None, "builtin"}), None),
        "verifier_version": next((result.verifier_version for result in results if result.verifier not in {None, "builtin"}), None),
        "configuration": verification_configuration(),
        "denominator": len(results),
        "decided_count": decided,
        "coverage": decided / len(results) if results else 0.0,
        "counts": dict(sorted(counts.items())),
        "claim_limit": "Final-answer equivalence only; not reasoning correctness or overall dataset quality.",
    }
