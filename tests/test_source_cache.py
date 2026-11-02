from __future__ import annotations

import pytest

from onegpu_posttrain.data.source_cache import (
    SourceIdentity,
    create_source_cache,
    iter_source_cache,
    validate_source_cache,
)


def _identity(revision: str = "abc123") -> SourceIdentity:
    return SourceIdentity(repo="org/dataset", config="default", revision=revision, split="train")


def test_source_cache_materializes_and_verifies_rows(tmp_path):
    identity = _identity()
    manifest_path, manifest, reused = create_source_cache(
        ({"id": n, "text": f"row-{n}"} for n in range(3)),
        cache_root=tmp_path,
        identity=identity,
    )

    assert not reused
    assert manifest.status == "complete"
    assert manifest.row_count == 3
    assert manifest.byte_count > 0
    assert len(manifest.content_sha256) == 64
    assert validate_source_cache(manifest_path, expected=identity) == manifest
    assert list(iter_source_cache(manifest_path, expected=identity)) == [
        {"id": n, "text": f"row-{n}"} for n in range(3)
    ]


def test_source_cache_reuses_verified_entry_without_consuming_rows(tmp_path):
    identity = _identity()
    manifest_path, _, _ = create_source_cache([{"id": 1}], cache_root=tmp_path, identity=identity)

    def must_not_be_consumed():
        raise AssertionError("cache hit should not consume or fetch source rows")
        yield {}

    reused_path, manifest, reused = create_source_cache(
        must_not_be_consumed(), cache_root=tmp_path, identity=identity
    )
    assert reused
    assert reused_path == manifest_path
    assert manifest.row_count == 1


def test_source_cache_allows_a_complete_empty_split(tmp_path):
    manifest_path, manifest, _ = create_source_cache([], cache_root=tmp_path, identity=_identity())
    assert manifest.row_count == 0
    assert manifest.byte_count == 0
    assert list(iter_source_cache(manifest_path)) == []


def test_source_cache_interruption_does_not_publish_or_leave_partial_file(tmp_path):
    def interrupted():
        yield {"id": 1}
        raise OSError("network interrupted")

    with pytest.raises(OSError, match="network interrupted"):
        create_source_cache(interrupted(), cache_root=tmp_path, identity=_identity())

    entry = tmp_path / _identity().cache_key
    assert not (entry / "manifest.json").exists()
    assert not list(entry.glob("*.partial"))


def test_failed_refresh_preserves_prior_complete_cache(tmp_path):
    identity = _identity()
    manifest_path, original, _ = create_source_cache([{"id": "old"}], cache_root=tmp_path, identity=identity)
    old_manifest = manifest_path.read_bytes()

    def interrupted_refresh():
        yield {"id": "new"}
        raise OSError("refresh failed")

    with pytest.raises(OSError, match="refresh failed"):
        create_source_cache(interrupted_refresh(), cache_root=tmp_path, identity=identity, refresh=True)

    assert manifest_path.read_bytes() == old_manifest
    assert validate_source_cache(manifest_path, expected=identity) == original
    assert list(iter_source_cache(manifest_path)) == [{"id": "old"}]


def test_source_cache_rejects_identity_mismatch_and_tampered_blob(tmp_path):
    identity = _identity()
    manifest_path, manifest, _ = create_source_cache([{"id": 1}], cache_root=tmp_path, identity=identity)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_source_cache(manifest_path, expected=_identity("different"))

    data_path = manifest_path.parent / manifest.data_file
    data_path.write_bytes(b'{"id":2}\n')
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_source_cache(manifest_path)
