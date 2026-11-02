"""Content-verified local copies of pinned dataset source splits."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .io import read_jsonl


class SourceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str = Field(min_length=1)
    config: str | None = None
    revision: str = Field(min_length=1)
    split: str = Field(min_length=1)

    def normalized(self) -> dict[str, str | None]:
        return {
            "repo": self.repo.strip(),
            "config": self.config.strip() if self.config and self.config.strip() else None,
            "revision": self.revision.strip(),
            "split": self.split.strip(),
        }

    @property
    def cache_key(self) -> str:
        canonical = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SourceCacheManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: SourceIdentity
    status: Literal["complete"]
    data_file: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    byte_count: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str


def _entry_key(identity: SourceIdentity) -> str:
    # Leave headroom for long Windows workspace paths; the complete key is in the manifest.
    return identity.cache_key[:24]


def source_cache_dir(cache_root: str | Path, identity: SourceIdentity) -> Path:
    return Path(cache_root) / _entry_key(identity)


def _manifest_path(path: str | Path) -> Path:
    path = Path(path)
    return path / "manifest.json" if path.is_dir() else path


def validate_source_cache(
    path: str | Path,
    *,
    expected: SourceIdentity | None = None,
) -> SourceCacheManifest:
    """Load and verify cache metadata, source identity, row count, and content digest."""
    manifest_path = _manifest_path(path).resolve(strict=True)
    if not manifest_path.is_file():
        raise ValueError(f"source cache manifest is not a file: {manifest_path}")
    try:
        manifest = SourceCacheManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid source cache manifest {manifest_path}: {exc}") from exc

    if manifest.cache_key != manifest.source.cache_key:
        raise ValueError("source cache identity hash does not match its manifest source fields")
    if manifest.cache_key[:24] != manifest_path.parent.name:
        raise ValueError("source cache entry directory does not match its manifest identity")
    if expected is not None and manifest.source.normalized() != expected.normalized():
        raise ValueError(
            "source cache identity mismatch: "
            f"expected {expected.normalized()}, found {manifest.source.normalized()}"
        )

    entry_dir = manifest_path.parent
    data_path = (entry_dir / manifest.data_file).resolve(strict=True)
    try:
        data_path.relative_to(entry_dir.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("source cache data file escapes its cache entry directory") from exc
    if not data_path.is_file():
        raise ValueError(f"source cache data file is not a file: {data_path}")

    content_hash = hashlib.sha256()
    byte_count = 0
    row_count = 0
    with data_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            content_hash.update(chunk)
            byte_count += len(chunk)
            row_count += chunk.count(b"\n")
    if byte_count != manifest.byte_count:
        raise ValueError(f"source cache byte count mismatch: expected {manifest.byte_count}, found {byte_count}")
    if row_count != manifest.row_count:
        raise ValueError(f"source cache row count mismatch: expected {manifest.row_count}, found {row_count}")
    if content_hash.hexdigest() != manifest.content_sha256:
        raise ValueError("source cache content SHA256 mismatch")
    return manifest


def create_source_cache(
    rows: Iterable[Mapping[str, Any]],
    *,
    cache_root: str | Path,
    identity: SourceIdentity,
    refresh: bool = False,
) -> tuple[Path, SourceCacheManifest, bool]:
    """Materialize a complete row iterator; return (manifest path, manifest, reused)."""
    entry_dir = source_cache_dir(cache_root, identity)
    manifest_path = entry_dir / "manifest.json"
    if manifest_path.exists():
        try:
            current = SourceCacheManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot safely refresh invalid source cache manifest {manifest_path}: {exc}") from exc
        if current.source.normalized() != identity.normalized() or current.cache_key != identity.cache_key:
            raise ValueError("source cache entry-key collision: existing manifest belongs to a different source")
        if not refresh:
            cached = validate_source_cache(manifest_path, expected=identity)
            return manifest_path, cached, True

    entry_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".source-", suffix=".partial", dir=entry_dir)
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    row_count = 0
    byte_count = 0
    try:
        with os.fdopen(fd, "wb") as stream:
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    raise TypeError(f"source row {index} is not a JSON object")
                encoded = (json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                stream.write(encoded)
                digest.update(encoded)
                row_count += 1
                byte_count += len(encoded)
            stream.flush()
            os.fsync(stream.fileno())

        content_sha256 = digest.hexdigest()
        blob_path = None
        for prefix_length in (12, 20, 32, 64):
            candidate = entry_dir / f"r-{content_sha256[:prefix_length]}.jsonl"
            if not candidate.exists():
                blob_path = candidate
                break
            if candidate.stat().st_size == byte_count and _sha256_file(candidate) == content_sha256:
                blob_path = candidate
                temp_path.unlink()
                break
        if blob_path is None:
            raise ValueError("unable to allocate a unique content-addressed source cache blob")
        if temp_path.exists():
            os.replace(temp_path, blob_path)
        blob_name = blob_path.name

        manifest = SourceCacheManifest(
            cache_key=identity.cache_key,
            source=identity,
            status="complete",
            data_file=blob_name,
            row_count=row_count,
            byte_count=byte_count,
            content_sha256=content_sha256,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        manifest_fd, manifest_temp_name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=entry_dir)
        manifest_temp = Path(manifest_temp_name)
        try:
            with os.fdopen(manifest_fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(manifest.model_dump_json(indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(manifest_temp, manifest_path)
        finally:
            manifest_temp.unlink(missing_ok=True)
        return manifest_path, manifest, False
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_source_cache(path: str | Path, *, expected: SourceIdentity | None = None) -> Iterator[dict[str, Any]]:
    manifest_path = _manifest_path(path)
    manifest = validate_source_cache(manifest_path, expected=expected)
    yield from read_jsonl(manifest_path.parent / manifest.data_file)


def open_source_cache(
    path: str | Path,
    *,
    expected: SourceIdentity,
) -> tuple[SourceCacheManifest, Iterator[dict[str, Any]], str]:
    """Verify a cache and return its manifest, rows, and manifest SHA256."""
    manifest_path = _manifest_path(path)
    manifest = validate_source_cache(manifest_path, expected=expected)
    return manifest, read_jsonl(manifest_path.parent / manifest.data_file), _sha256_file(manifest_path)
