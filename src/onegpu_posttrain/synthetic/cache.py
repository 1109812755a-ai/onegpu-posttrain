"""Content-addressed, hash-verified private provider-response cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from ..lineage import sha256_json
from .providers import GenerationResponse


def response_cache_path(plan_dir: Path, request_sha256: str) -> Path:
    # Keep Windows paths short; the full request hash remains in the cache payload and ledger.
    return plan_dir / "private" / "responses" / f"{request_sha256[:24]}.json"


def write_response_cache(plan_dir: Path, request_sha256: str, response: GenerationResponse) -> tuple[str, str]:
    target = response_cache_path(plan_dir, request_sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError("response cache entry already exists without a matching completed ledger record")
    response_payload = response.model_dump(mode="json")
    response_sha256 = sha256_json(response_payload)
    payload = {
        "schema_version": "1.0",
        "request_sha256": request_sha256,
        "response_sha256": response_sha256,
        "response": response_payload,
    }
    temporary = target.parent / f".{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target.relative_to(plan_dir).as_posix(), response_sha256


def load_response_cache(plan_dir: Path, relative_path: str, request_sha256: str,
                        expected_response_sha256: str) -> GenerationResponse:
    target = (plan_dir / relative_path).resolve()
    try:
        target.relative_to(plan_dir.resolve())
    except ValueError:
        raise ValueError("response cache path escaped its plan directory") from None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        response_payload = payload["response"]
        if payload.get("request_sha256") != request_sha256:
            raise ValueError
        if payload.get("response_sha256") != expected_response_sha256:
            raise ValueError
        if sha256_json(response_payload) != expected_response_sha256:
            raise ValueError
        return GenerationResponse.model_validate(response_payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise ValueError("completed response cache is missing, malformed, or hash-mismatched") from None


def load_response_cache_by_key(plan_dir: Path, request_sha256: str) -> tuple[GenerationResponse, str, str] | None:
    target = response_cache_path(plan_dir, request_sha256)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        response_payload = payload["response"]
        response_digest = payload["response_sha256"]
        if payload.get("request_sha256") != request_sha256 or sha256_json(response_payload) != response_digest:
            raise ValueError
        response = GenerationResponse.model_validate(response_payload)
        return response, target.relative_to(plan_dir).as_posix(), response_digest
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        raise ValueError("orphan response cache is malformed or hash-mismatched") from None
